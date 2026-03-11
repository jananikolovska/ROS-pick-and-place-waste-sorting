import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from collections import deque
import time
import os
from datetime import datetime


class CameraNode(Node):
    """
    Captures live video from a USB camera and publishes frames to ROS2 topic.
    Uses /dev/video8 device at 1080x720 resolution, 30 FPS.
    """
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('compute_metrics', True)
        self.declare_parameter('video_device', '/dev/video8')
        self.declare_parameter('topic_name', '/camera_frame')
        self.declare_parameter('limit_runtime', True)
        self.declare_parameter('runtime_limit_sec', 5*60)

        self.compute_metrics = self.get_parameter('compute_metrics').get_parameter_value().bool_value
        self.video_device = self.get_parameter('video_device').get_parameter_value().string_value
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.limit_runtime = self.get_parameter('limit_runtime').get_parameter_value().bool_value
        self.runtime_limit_sec = self.get_parameter('runtime_limit_sec').get_parameter_value().integer_value

        self.publisher = self.create_publisher(Image, self.topic_name, 10)
        self.bridge = CvBridge()

        # Open the camera
        self.cap = cv2.VideoCapture(self.video_device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera at {self.video_device}")
            return

        # Metrics state
        self._frames_published: int = 0
        _fps_window_size = 90
        self._fps_timestamps: deque = deque(maxlen=_fps_window_size)
        self._metrics_log = None

        self._start_time = time.monotonic()
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        if self.compute_metrics:
            # Open log file
            metrics_dir = 'camera_metrics'
            os.makedirs(metrics_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_path = os.path.join(metrics_dir, f'metrics_camera_{ts}.txt')
            try:
                self._metrics_log = open(log_path, 'w')
                self.get_logger().info(f'[METRICS] Logging to file: {log_path}')
            except OSError as e:
                self.get_logger().warning(f'[METRICS] Could not open log file: {e}')

            self._metrics_timer = self.create_timer(5.0, self.print_metrics)
            self.get_logger().info('[METRICS] Camera metrics enabled (printed every 5 s).')

    def timer_callback(self):
        # Check runtime limit
        if self.limit_runtime:
            elapsed = time.monotonic() - self._start_time
            if elapsed > self.runtime_limit_sec:
                self.get_logger().info(f"Runtime limit of {self.runtime_limit_sec} seconds reached. Shutting down camera node.")
                rclpy.shutdown()
                return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to read frame")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.publisher.publish(msg)

        if self.compute_metrics:
            self._frames_published += 1
            self._fps_timestamps.append(time.monotonic())

    def print_metrics(self):
        if not self.compute_metrics:
            return
        fps = 0.0
        if len(self._fps_timestamps) >= 2:
            window_s = self._fps_timestamps[-1] - self._fps_timestamps[0]
            if window_s > 0.0:
                fps = (len(self._fps_timestamps) - 1) / window_s

        msg = (
            f'\n========== Camera Node Metrics ==========\n'
            f'  Frames published : {self._frames_published}\n'
            f'  Publish FPS      : {fps:.2f} fps  '
            f'(last {len(self._fps_timestamps)}-frame window)\n'
            f'========================================='
        )
        self.get_logger().info(msg)

        if self._metrics_log:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._metrics_log.write(f'[{ts}]\n{msg}\n\n')
            self._metrics_log.flush()

    def destroy_node(self):
        if self.compute_metrics:
            self.get_logger().info('[METRICS] === Final Camera Metrics on Shutdown ===')
            self.print_metrics()
            if self._metrics_log:
                self._metrics_log.close()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

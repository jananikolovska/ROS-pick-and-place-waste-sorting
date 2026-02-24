import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    """
    Captures live video from a USB camera and publishes frames to ROS2 topic.
    Uses /dev/video8 device at 1080x720 resolution, 30 FPS.
    """
    def __init__(self):
        super().__init__('camera_node')

        self.publisher = self.create_publisher(Image, '/camera_frame', 10)
        self.bridge = CvBridge()

        # Open the Logitech BRIO
        self.cap = cv2.VideoCapture('/dev/video8', cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera at /dev/video0")
            return

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)  # 30 FPS

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to read frame")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)

    def destroy_node(self):
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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import helper_nodes

class SaveNode(Node):
    """
    Saves annotated video frames as PNG files to disk with logo overlay.
    Creates numbered sequence of detection result images for analysis.
    """
    def __init__(self):
        super().__init__('save_node')

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera_frame_annotated',
            self.listener_callback,
            10
        )
        
        # Load logo (as RGBA)
        try:
            pkg_path = os.path.dirname(helper_nodes.__file__)
            logo_path = os.path.join(pkg_path, 'resource', 'logo.png')
            logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
            if logo is None:
                raise FileNotFoundError("Logo image not found or unreadable.")
            self.logo = logo
        except Exception as e:
            self.get_logger().error(f"Failed to load logo: {e}")
            self.logo = None

        # Create results directory
        self.results_dir = os.path.join(pkg_path, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.frame_count = 0

    def overlay_logo(self, frame, logo):
        """Overlay RGBA logo on BGR frame (bottom-right)."""
        if logo is None:
            return frame

        # Resize logo if larger than frame
        h, w = frame.shape[:2]
        lh, lw = logo.shape[:2]
        scale = min(w / (3 * lw), h / (3 * lh), 1.0)
        logo = cv2.resize(logo, (0, 0), fx=scale, fy=scale)

        # Split logo into channels
        if logo.shape[2] == 4:
            b, g, r, a = cv2.split(logo)
            alpha = a.astype(float) / 255.0
            logo_rgb = cv2.merge((b, g, r))
        else:
            logo_rgb = logo
            alpha = np.ones((logo.shape[0], logo.shape[1]), dtype=float)

        # Compute position
        y1 = h - logo.shape[0] - 10
        y2 = y1 + logo.shape[0]
        x1 = w - logo.shape[1] - 10
        x2 = x1 + logo.shape[1]

        # Clip if too big
        if x1 < 0 or y1 < 0:
            return frame

        roi = frame[y1:y2, x1:x2].astype(float)
        for c in range(3):
            roi[..., c] = roi[..., c] * (1 - alpha) + logo_rgb[..., c] * alpha

        frame[y1:y2, x1:x2] = roi.astype(np.uint8)
        return frame

    def listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = self.overlay_logo(frame, self.logo)

            # Save the frame as a PNG file
            os.makedirs("/home/ubuntu/voyager-sdk/ros2_ws/results", exist_ok=True)
            file_path = os.path.join("/home/ubuntu/voyager-sdk/ros2_ws/results", f"frame_{self.frame_count:06d}.png")
            
            cv2.imwrite(file_path, frame)
            self.get_logger().info(f"Saved frame to {file_path}")
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Save error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SaveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
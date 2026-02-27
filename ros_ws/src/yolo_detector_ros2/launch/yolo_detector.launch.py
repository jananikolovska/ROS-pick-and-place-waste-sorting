"""
Launch file for YOLO Detector Node

This launch file starts the YOLO object detection node with configurable parameters.

Usage:
    # With default parameters
    ros2 launch yolo_detector_ros2 yolo_detector.launch.py
    
    # With custom parameters
    ros2 launch yolo_detector_ros2 yolo_detector.launch.py \
        model_path:=yolov8s_best.pt \
        every_n_frames:=10 \
        conf_thres:=0.5 \
        keep_largest_only:=true \
        save_results:=false \
        debug_prints:=false

Parameters:
    image_topic: Input camera topic (default: /camera_frame)
    detections_topic: Output Int32 topic for class ID (default: /detections)
    model_path: YOLO model weights file (default: yolo_detector_ros2/yolov8s_best.pt)
    every_n_frames: Process every Nth frame to reduce load (default: 5)
    conf_thres: Detection confidence threshold 0.0-1.0 (default: 0.25)
    iou_thres: IoU threshold for NMS 0.0-1.0 (default: 0.45)
    save_dir: Directory for annotated images (default: yolo_results)
    crop_ratio: Crop edges before detection (default: 0.2 = 20% from each side)
    max_bbox_ratio: Filter detections larger than this (default: 0.9 = 90% of image)
    stable_frames_required: Detections needed for stability (default: 10)
    variance_threshold: Max bbox variance for stability in pixels (default: 15.0)
    destabilize_threshold: Movement distance to trigger re-detection in pixels (default: 50.0)
    keep_largest_only: Keep only largest object if multiple detected (default: true)
    save_results: Save annotated images to disk (default: true)
    debug_prints: Enable verbose debug output (default: true)
    publish_interval: Seconds between re-publishing stable detections (default: 0.25)

Class Mapping:
    0 = glass
    1 = metal
    2 = paper/cardboard
    3 = plastic
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for YOLO detector node."""
    
    return LaunchDescription([
        # ============================================================
        # CAMERA TOPICS
        # ============================================================
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera_frame",
            description="Topic to subscribe to for camera frames"
        ),
        
        DeclareLaunchArgument(
            "detections_topic",
            default_value="/detections",
            description="Topic to publish class ID as Int32 (0=glass, 1=metal, 2=paper, 3=plastic)"
        ),
        
        # ============================================================
        # YOLO MODEL CONFIGURATION
        # ============================================================
        DeclareLaunchArgument(
            "model_path",
            default_value="yolo_detector_ros2/yolov8s_best.pt",
            description="Path to YOLO model weights file (e.g., yolov8s.pt, yolov8n.pt, custom trained)"
        ),
        
        DeclareLaunchArgument(
            "conf_thres",
            default_value="0.25",
            description="Confidence threshold for detections (0.0 - 1.0). Higher = fewer false positives"
        ),
        
        DeclareLaunchArgument(
            "iou_thres",
            default_value="0.45",
            description="IoU threshold for non-maximum suppression (0.0 - 1.0)"
        ),
        
        # ============================================================
        # PROCESSING CONFIGURATION
        # ============================================================
        DeclareLaunchArgument(
            "every_n_frames",
            default_value="5",
            description="Process every N-th frame to reduce computational load (5 = ~6 FPS from 30 FPS)"
        ),
        
        DeclareLaunchArgument(
            "crop_ratio",
            default_value="0.2",
            description="Crop ratio to remove outer edges (0.2 = remove 20% from each side, keep center 60%)"
        ),
        
        DeclareLaunchArgument(
            "max_bbox_ratio",
            default_value="0.9",
            description="Maximum bbox size relative to image (0.9 = filter detections larger than 90%)"
        ),
        
        # ============================================================
        # PLACEMENT DETECTION PARAMETERS
        # ============================================================
        DeclareLaunchArgument(
            "stable_frames_required",
            default_value="10",
            description="Number of consecutive detections required for stability (~1.7s at every_n_frames=5)"
        ),
        
        DeclareLaunchArgument(
            "variance_threshold",
            default_value="15.0",
            description="Maximum bbox position variance (pixels) to consider object stable (handles jitter)"
        ),
        
        DeclareLaunchArgument(
            "destabilize_threshold",
            default_value="50.0",
            description="Object movement (pixels) to trigger destabilization and stop publishing"
        ),
        
        # ============================================================
        # DETECTION FILTERING
        # ============================================================
        DeclareLaunchArgument(
            "keep_largest_only",
            default_value="true",
            description="Keep only the largest object if multiple detected (recommended: true)"
        ),
        
        # ============================================================
        # OUTPUT CONFIGURATION
        # ============================================================
        DeclareLaunchArgument(
            "save_results",
            default_value="true",
            description="Save annotated images when object is first placed (true/false)"
        ),
        
        DeclareLaunchArgument(
            "save_dir",
            default_value="yolo_results",
            description="Directory where annotated images are saved"
        ),
        
        DeclareLaunchArgument(
            "publish_interval",
            default_value="0.25",
            description="Seconds between re-publishing class ID for stable objects (0.25 = 4 times/sec)"
        ),
        
        # ============================================================
        # DEBUG CONFIGURATION
        # ============================================================
        DeclareLaunchArgument(
            "debug_prints",
            default_value="true",
            description="Enable verbose debug output (true/false). Set to false for clean operation"
        ),
        
        # ============================================================
        # YOLO DETECTOR NODE
        # ============================================================
        Node(
            package="yolo_detector_ros2",
            executable="yolo_detector_node",
            name="yolo_detector",
            output="screen",
            parameters=[
                {"image_topic": LaunchConfiguration("image_topic")},
                {"detections_topic": LaunchConfiguration("detections_topic")},
                {"model_path": LaunchConfiguration("model_path")},
                {"every_n_frames": LaunchConfiguration("every_n_frames")},
                {"conf_thres": LaunchConfiguration("conf_thres")},
                {"iou_thres": LaunchConfiguration("iou_thres")},
                {"save_dir": LaunchConfiguration("save_dir")},
                {"crop_ratio": LaunchConfiguration("crop_ratio")},
                {"max_bbox_ratio": LaunchConfiguration("max_bbox_ratio")},
                {"stable_frames_required": LaunchConfiguration("stable_frames_required")},
                {"variance_threshold": LaunchConfiguration("variance_threshold")},
                {"destabilize_threshold": LaunchConfiguration("destabilize_threshold")},
                {"keep_largest_only": LaunchConfiguration("keep_largest_only")},
                {"save_results": LaunchConfiguration("save_results")},
                {"debug_prints": LaunchConfiguration("debug_prints")},
                {"publish_interval": LaunchConfiguration("publish_interval")},
            ]
        ),
    ])

from launch import LaunchDescription
from launch_ros.actions import Node

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='recycle_inference_node_cpp',
            executable='inference_cpp',
            name='recycle_ax_ros2_cpp_inference_node',
            output='screen',
            parameters=[
                {
                    # --- Model / hardware ---
                    'model_name': 'yolov8s-recycle',
                    'aipu_cores': 4,
                    'mean': [0.485, 0.456, 0.406],
                    'stddev': [0.229, 0.224, 0.225],

                    # --- Topics ---
                    'input_topic': '/camera_frame',
                    'annotated_topic': '/camera_frame_annotated',  # annotated image with bounding boxes
                    'output_topic': '/detections_topic',           # detection strings (label + confidence + bbox)
                    'class_detections_topic': '/detections',       # Int32: 0=glass 1=metal 2=paper 3=plastic

                    # --- Topic publish toggles ---
                    'publish_annotated_image': True,   # toggle /camera_frame_annotated
                    'publish_detection_strings': True, # toggle /detections_topic
                    'publish_class_id': True,          # toggle /detections (Int32)

                    # --- Model output format ---
                    # num_classes is auto-derived from model_info.json labels at startup;
                    # the value set here is overridden and kept only as a fallback default.
                    'num_classes': 4,          # recycling model: glass/metal/paper/plastic (overridden by label count)
                    'has_objectness': False,    # True = YOLOv5 (shape [..., num_classes+5]), False = YOLOv8 (shape [..., num_classes+4, ...])

                    # --- Inference thresholds ---
                    'confidence_threshold': 0.10,
                    'nms_threshold': 0.45,

                    # --- Frame processing ---
                    'every_n_frames': 5,        # process every N-th frame
                    'crop_ratio': 0.2,          # fraction to crop from each side

                    # --- Detection filtering ---
                    'max_bbox_ratio': 0.9,      # discard boxes larger than this fraction of the image
                    'keep_largest_only': True,  # keep only the biggest detection per frame

                    # --- Stable placement detection ---
                    'stable_frames_required': 10,   # consecutive detections needed to call it stable
                    'variance_threshold': 15.0,     # max centre-position std-dev (px) to count as stable
                    'destabilize_threshold': 50.0,  # centre movement (px) that resets tracking

                    # --- Publishing ---
                    'publish_interval': 0.25,   # seconds between re-publishes while object is stable

                    # --- Output ---
                    'save_results': True,
                    'save_dir': 'yolo_results',
                }
            ]
        )
    ])

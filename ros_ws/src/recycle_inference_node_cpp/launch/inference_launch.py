from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    compute_metrics_arg = DeclareLaunchArgument(
        'compute_metrics',
        default_value='true',
        description='Enable performance metrics (latency, FPS, counters). Printed every 5 s and on shutdown.'
    )

    return LaunchDescription([
        compute_metrics_arg,
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
                    'num_classes': 4,
                    'has_objectness': False,

                    # --- Inference thresholds ---
                    'confidence_threshold': 0.10,
                    'nms_threshold': 0.45,

                    # --- Frame processing ---
                    'crop_ratio': 0.2,

                    # --- Detection filtering ---
                    'max_bbox_ratio': 0.9,
                    'keep_largest_only': True,

                    # --- Stable placement detection ---
                    'stable_frames_required': 20,
                    'variance_threshold': 15.0,
                    'destabilize_threshold': 30.0,
                    'false_detection_frames': 7,

                    # --- Publishing ---
                    'publish_interval': 0.25,

                    # --- Output ---
                    'save_results': True,
                    'save_dir': 'yolo_results',

                },
                {'compute_metrics': ParameterValue(LaunchConfiguration('compute_metrics'), value_type=bool)}
            ]
        )
    ])

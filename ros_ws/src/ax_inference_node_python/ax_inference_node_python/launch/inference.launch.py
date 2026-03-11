from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ax_inference_node_python',
            executable='inference_python',
            name='axros2_python_inference_node',
            output='screen',
            parameters=[
                {
                    'model_name': 'yolov8s-coco',
                    'aipu_cores': 4,
                    'input_topic': '/camera_frame',
                    'output_topic': '/detections_topic',
                    'confidence_threshold': 0.25,
                    'nms_threshold': 0.45,
                    'mean': [0.485, 0.456, 0.406],
                    'stddev': [0.229, 0.224, 0.225],
                    'publish_annotated': False,
                    # --- Metrics ---
                    'compute_metrics': True,
                    'save_dir': '~/ros2_metrics',
                }
            ]
        )
    ])

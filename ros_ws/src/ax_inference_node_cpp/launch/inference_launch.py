from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ax_inference_node_cpp',
            executable='inference_cpp',
            name='axros2_cpp_inference_node',
            output='screen',
            parameters=[
                {
                    'model_name': 'yolov5n-v7-coco-onnx',
                    'aipu_cores': 4,
                    'input_topic': '/camera_frame',
                    'output_topic': '/detections_topic',
                    'confidence_threshold': 0.10,
                    'nms_threshold': 0.45,
                    'publish_annotated': False,
                    'mean': [0.485, 0.456, 0.406],
                    'stddev': [0.229, 0.224, 0.225],
                    # --- Metrics ---
                    'compute_metrics': True,
                    'save_dir': '~/ros2_metrics',
                }
            ]
        )
    ])

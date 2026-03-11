from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video8',
            description='Path to the video device (e.g., /dev/video8)'
        ),
        DeclareLaunchArgument(
            'topic_name',
            default_value='/camera_frame',
            description='Topic name to publish camera frames'
        ),
        DeclareLaunchArgument(
            'limit_runtime',
            default_value='true',
            description='Whether to limit how long the camera runs'
        ),
        DeclareLaunchArgument(
            'runtime_limit_sec',
            default_value=str(5*60),
            description='Time limit in seconds (default: 5 min)'
        ),
        Node(
            package='helper_nodes',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[{
                'video_device': LaunchConfiguration('video_device'),
                'topic_name': LaunchConfiguration('topic_name'),
                'limit_runtime': LaunchConfiguration('limit_runtime'),
                'runtime_limit_sec': LaunchConfiguration('runtime_limit_sec'),
            }]
        )
    ])

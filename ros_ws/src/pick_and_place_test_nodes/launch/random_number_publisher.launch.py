from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    random_number_publisher_node = Node(
        package="pick_and_place_test_nodes",
        executable="random_number_publisher",
        output="screen"
    )

    return LaunchDescription([random_number_publisher_node])

import os

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ur_moveit_config.launch_common import load_yaml
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():

  # Declare launch argument for detections topic
  detections_topic_arg = DeclareLaunchArgument(
      "detections_topic",
      default_value="/detections",
      description="Topic to subscribe for detection results (Int32 class IDs from YOLO detector)"
  )

  compute_metrics_arg = DeclareLaunchArgument(
      "compute_metrics",
      default_value="true",
      description="Enable performance metrics logging (latency, counters). Printed every 5 s and on shutdown."
  )

  robot_description_content = Command(
      [
          PathJoinSubstitution([FindExecutable(name="xacro")]),
          " ",
          PathJoinSubstitution(
              [FindPackageShare('pick_and_place_description'),
              "urdf", "robot_arm.xacro"]
          ),
          " ",
          "use_gazebo:=",
          "true",
          " ",
          "use_ignition:=",
          "false",
          " ",
          "use_fake_hardware:=",
          "false"
      ]
  )
  robot_description = {"robot_description": robot_description_content}

  bringup_dir = get_package_share_directory('pick_and_place_moveit_config')
  srdf_path = os.path.join(bringup_dir, 'srdf', 'robot_arm.srdf')
  srdf = open(srdf_path).read()

  robot_description_semantic = {"robot_description_semantic": srdf}

  robot_description_kinematics = PathJoinSubstitution(
      [FindPackageShare('pick_and_place_moveit_config'), "config", "kinematics.yaml"]
  )
  
  move_group_demo = Node(
      package="pick_and_place_test_nodes",
      executable="test_pick_and_place",
      output="screen",
      parameters=[
          robot_description,
          robot_description_semantic,
          robot_description_kinematics,
          {"use_sim_time": True},
          {"detections_topic": LaunchConfiguration("detections_topic")},
          {"compute_metrics": ParameterValue(LaunchConfiguration("compute_metrics"), value_type=bool)}
      ],
  )

  return LaunchDescription([detections_topic_arg, compute_metrics_arg, move_group_demo])

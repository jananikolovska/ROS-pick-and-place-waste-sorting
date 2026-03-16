#!/bin/bash
# Source this script in every terminal before running ROS 2 or inference nodes

export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
cd ~/voyager-sdk
source venv/bin/activate
source /opt/ros/humble/setup.bash
cd ~/ROS-pick-and-place-waste-sorting/ros_ws
source install/setup.bash
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.10/site-packages:$PYTHONPATH

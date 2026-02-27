# pick_and_place
repo to showcase pick and place with moveit2 and perception pipeline

## Overview

This is a ROS 2 project that demonstrates an autonomous robotic pick-and-place system with intelligent sorting capabilities. The system uses MoveIt2 for motion planning and Gazebo for physics simulation. When triggered by external messages, the robot automatically spawns colored boxes in a designated pickup zone, picks them up at a center location, and places them in one of three drop-off zones based on the box color (green, red, or yellow). After placing each box, it returns to the home position and is ready for the next cycle.

The system is fully event-driven and message-based, making it easy to integrate with higher-level planning or perception systems. All pick-and-place operations are performed automatically in separate threads to ensure non-blocking execution.

## Summary of Key Changes

This repository has been enhanced with a message-driven pick-and-place system:

- **Dynamic Box Spawning**: Boxes are now spawned dynamically by the pick and place node instead of being pre-loaded in the world, allowing for repeated cycles
- **Refactored Pick-and-Place Logic**: The core pick-and-place workflow (spawn → pick → move → place → delete) is now encapsulated in a reusable function
- **Message-Driven Execution**: The pick and place node subscribes to a random number topic and executes a complete cycle for each received message
- **Random Number Publisher**: A separate node publishes random numbers (0-3) every 20 seconds to trigger pick-and-place cycles
- **Threaded Execution**: Pick-and-place cycles run in separate threads to avoid blocking the main ROS 2 executor, enabling proper async/await patterns
- **Three-Color Sorting System**: The robot now sorts boxes by color based on the received message:
  - **Green boxes** (messages 0, 1): Placed at -60° position
  - **Red boxes** (message 2): Placed at -120° position
  - **Yellow boxes** (message 3): Placed at 60° position
- **Visual Ground Markers**: The Gazebo world includes colored floor markers (green, red, yellow, and pink for spawn) to visually indicate drop-off zones and the pickup location
- **Multi-Cycle Support**: The system seamlessly handles multiple sequential pick-and-place operations with automatic box cleanup between cycles

## Usage
``` ros2 launch pick_and_place_description start_control.launch.py ```

## To launch the model with test environment in gazebo
``` ros2 launch pick_and_place_simulation start_simulation.launch.py ```

## To launch moveit
``` ros2 launch pick_and_place_moveit_config start_moveit.launch.py ```

## To run pick and place demo without vision
``` ros2 launch pick_and_place_test_nodes start_pick_and_place.launch.py ```

## To run the random number publisher (triggers pick and place cycles)
``` ros2 run pick_and_place_test_nodes random_number_publisher ```

## Cleaning up processes
It's good practice to kill running ROS 2, Gazebo, MoveIt, and RViz processes before rebuilding the project to avoid conflicts. Run this before `colcon build`:
```
pkill -f gzserver
pkill -f gzclient
pkill -f moveit
pkill -f rviz
pkill -f ros2

conda deactivate
source install/setup.bash
```

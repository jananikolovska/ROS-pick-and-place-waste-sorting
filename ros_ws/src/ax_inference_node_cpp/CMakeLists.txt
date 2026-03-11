cmake_minimum_required(VERSION 3.12)
project(ax_inference_node_cpp VERSION 0.1.0)

# Set C++ standard and build options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)


# Voyager SDK / Axelera AI runtime
find_package(PkgConfig REQUIRED)
pkg_check_modules(axruntime REQUIRED axruntime)
pkg_check_modules(opencv4 REQUIRED opencv4)
pkg_check_modules(axstreamer REQUIRED axstreamer)

# Create the executable
add_executable(inference_cpp src/inference_node.cpp)

# Compiler flags (warnings, suppressions, etc.)
target_compile_options(inference_cpp PRIVATE
    -O3
    -mcpu=native
    -Wall
    -Werror
    -Wno-sign-compare
    -Wno-unused-variable
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
)

# Include paths for headers
target_include_directories(inference_cpp PRIVATE
    ${axruntime_INCLUDE_DIRS}/include
    ${opencv4_INCLUDE_DIRS}
    ${axstreamer_INCLUDE_DIRS}
    /home/jetson/voyager-sdk/operators/onnxruntime/onnxruntime-linux-aarch64-1.17.1/include
)

# Library search paths
target_link_directories(inference_cpp PUBLIC
    ${axruntime_LIBRARY_DIRS}
    ${opencv4_LIBRARY_DIRS}
    ${axstreamer_LIBRARY_DIRS}
    /home/jetson/voyager-sdk/operators/onnxruntime/onnxruntime-linux-aarch64-1.17.1/lib
)

# Link libraries
target_link_libraries(inference_cpp
    ${axruntime_LIBRARIES}
    ${opencv4_LIBRARIES}
    ${axstreamer_LIBRARIES}
    onnxruntime
)

# Add ROS 2 dependencies
ament_target_dependencies(inference_cpp rclcpp OpenCV sensor_msgs cv_bridge)

# Install the executable into lib/
install(TARGETS inference_cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()

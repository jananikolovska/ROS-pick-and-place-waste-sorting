# YOLO Detector ROS2 Package

A ROS2 package for real-time object detection and classification using YOLOv8, specifically designed for waste sorting applications in the pick-and-place system.

## Overview

This package provides computer vision capabilities for detecting and classifying waste objects using a trained YOLOv8 model. It processes camera frames, detects objects, tracks their placement, and publishes classification results to enable automated sorting.

## Features

- **Real-time Detection**: Processes camera frames using YOLOv8 for object detection
- **Waste Classification**: Classifies objects into 4 categories: glass, metal, paper/cardboard, and plastic
- **Placement Detection**: State machine that detects when objects are placed (stable) vs. picked up (destabilized)
- **Smart Publishing**: 
  - Publishes detection once when object becomes stable
  - Re-publishes every 0.25s while object remains stable
  - Stops publishing when object is moved
- **Configurable Filtering**: 
  - Process every N-th frame for performance
  - Filter out oversized detections
  - Keep only largest object (optional)
- **Debug Control**: Optional debug prints for monitoring detection pipeline
- **Image Saving**: Optional saving of annotated detection images

## Class Mapping

The detector publishes `std_msgs/Int32` messages with the following class IDs:

| Class ID | Material        | Pick-and-Place Action |
|----------|-----------------|----------------------|
| 0        | Glass           | Green bin (60°)      |
| 1        | Metal           | Green bin (60°)      |
| 2        | Paper/Cardboard | Red bin (120°)       |
| 3        | Plastic         | Yellow bin (-60°)    |

## Topics

### Subscribed Topics
- `/camera_frame` (sensor_msgs/Image): Camera image stream

### Published Topics
- `/detections` (std_msgs/Int32): Detected object class ID (0-3)

## Launch Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_topic` | string | `/camera_frame` | Input camera topic |
| `detections_topic` | string | `/detections` | Output detection topic |
| `model_path` | string | `yolo_detector_ros2/yolov8s_best.pt` | Path to YOLOv8 model weights |
| `every_n_frames` | int | 5 | Process every N-th frame |
| `conf_thres` | float | 0.25 | Confidence threshold (0.0-1.0) |
| `iou_thres` | float | 0.45 | IoU threshold for NMS |
| `crop_ratio` | float | 0.2 | Crop ratio from each edge |
| `max_bbox_ratio` | float | 0.9 | Filter bboxes larger than this ratio |
| `stable_frames_required` | int | 10 | Frames needed for stable detection |
| `variance_threshold` | float | 15.0 | Max variance (pixels) for stability |
| `destabilize_threshold` | float | 50.0 | Movement (pixels) to trigger re-detection |
| `keep_largest_only` | bool | true | Keep only largest object if multiple detected |
| `save_results` | bool | true | Save annotated images |
| `debug_prints` | bool | true | Enable debug printing |
| `publish_interval` | float | 0.25 | Re-publish interval (seconds) |
| `save_dir` | string | `yolo_results` | Directory for saved images |

## Usage

### Basic Launch
```bash
ros2 launch yolo_detector_ros2 yolo_detector.launch.py
```

### Launch with Custom Parameters
```bash
ros2 launch yolo_detector_ros2 yolo_detector.launch.py \
    detections_topic:=/my_detections \
    conf_thres:=0.35 \
    keep_largest_only:=true \
    debug_prints:=false
```

### Integration with Pick-and-Place System

1. **Start the simulation**:
   ```bash
   ros2 launch pick_and_place_simulation start_simulation.launch.py
   ```

2. **Start MoveIt**:
   ```bash
   ros2 launch pick_and_place_moveit_config start_moveit.launch.py
   ```

3. **Start robot control**:
   ```bash
   ros2 launch pick_and_place_description start_control.launch.py
   ```

4. **Start YOLO detector**:
   ```bash
   ros2 launch yolo_detector_ros2 yolo_detector.launch.py
   ```

5. **Start pick-and-place node**:
   ```bash
   ros2 launch pick_and_place_test_nodes start_pick_and_place.launch.py
   ```

## Detection State Machine

The detector uses a 3-state machine for robust placement detection:

1. **DETECTING**: Tracking object, waiting for stability
   - Requires `stable_frames_required` consecutive detections
   - Checks bbox variance is below `variance_threshold`
   - Transitions to STABLE when conditions met

2. **STABLE**: Object placed and stable
   - Publishes detection immediately on transition
   - Re-publishes every `publish_interval` seconds
   - Monitors for destabilization (movement)
   - Transitions to DESTABILIZED if object moves

3. **DESTABILIZED**: Object was moved
   - Stops publishing
   - Clears tracking history
   - Transitions back to DETECTING for fresh detection

## Output Messages

The node prints status messages for all publishing events:

```
[PUBLISH] Int32: 3 (plastic) - Track #0
[RE-PUBLISH] Int32: 3 (plastic) - Track #0
[RE-PUBLISH] Int32: 3 (plastic) - Track #0
[STOP PUBLISH] Int32: 3 (plastic) - Track #0 (object moved)
```

## Model Training

The package uses a custom-trained YOLOv8 model (`yolov8s_best.pt`) for waste classification. The model was trained on a dataset of waste objects in various lighting and background conditions.

## Dependencies

- ROS2 Humble
- Python 3
- OpenCV (cv2)
- ultralytics (YOLOv8)
- cv_bridge
- sensor_msgs
- std_msgs
- vision_msgs

## File Structure

```
yolo_detector_ros2/
├── README.md                          # This file
├── package.xml                        # ROS2 package manifest
├── setup.py                          # Python package setup
├── yolov8s_best.pt                   # Trained YOLO model weights
├── launch/
│   └── yolo_detector.launch.py       # Launch file with all parameters
├── yolo_detector_ros2/
│   ├── __init__.py
│   └── yolo_detector_node.py         # Main detection node
└── resource/
    └── yolo_detector_ros2
```

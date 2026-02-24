# How to Run the YOLO Detector

This guide explains how to run the YOLO object detection node in your ROS2 workspace.

## Prerequisites

1. **Build the workspace** (if not already done):
   ```bash
   cd /home/jana/TEZA/pick_and_place
   colcon build
   ```

2. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

## Running the Detector

### Method 1: With Video Stream Node (Simulated Camera)

This method uses a pre-recorded video to simulate camera input.

**Terminal 1 - Start Video Streamer:**
```bash
source install/setup.bash
ros2 run helper_nodes video_stream_node
```

**Terminal 2 - Start YOLO Detector:**
```bash
source install/setup.bash
ros2 run yolo_detector_ros2 yolo_detector_node
```

### Method 2: With Real Camera

If you have a camera node publishing to `/camera_frame`:

```bash
source install/setup.bash
ros2 run yolo_detector_ros2 yolo_detector_node
```

### Method 3: With Custom Parameters

You can override the default parameters:

```bash
source install/setup.bash
ros2 run yolo_detector_ros2 yolo_detector_node \
  --ros-args \
  -p image_topic:=/my_camera \
  -p model_path:=yolov8s.pt \
  -p every_n_frames:=10 \
  -p conf_thres:=0.5 \
  -p save_dir:=my_results
```

## Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_topic` | `/camera_frame` | Topic to subscribe for camera frames |
| `detections_topic` | `/detections` | Topic to publish detection results |
| `model_path` | `yolov8n.pt` | Path to YOLO model weights (yolov8n.pt, yolov8s.pt, etc.) |
| `every_n_frames` | `30` | Process every N-th frame (1 = process all frames) |
| `conf_thres` | `0.25` | Confidence threshold for detections (0.0 - 1.0) |
| `iou_thres` | `0.45` | IoU threshold for non-maximum suppression (0.0 - 1.0) |
| `save_dir` | `yolo_results` | Directory where annotated images are saved |

## Understanding the Debug Output

The node outputs extensive debug information:

### Initialization Phase
```
================================================================================
DEBUG: Starting YoloDetectorNode initialization
================================================================================
DEBUG: Declaring ROS2 parameters...
DEBUG: Loading parameter values...
DEBUG: Loading YOLO model from: yolov8n.pt
DEBUG: ✓ YOLO model loaded successfully!
```

### Frame Processing Phase
```
>>> Frame #30 received
DEBUG: [Frame 30] Callback triggered
DEBUG: Image encoding: bgr8
DEBUG: Image dimensions: 640x480
DEBUG: ✓ Processing this frame (frame % 30 == 0)
------------------------------------------------------------
DEBUG: Converting ROS Image message to OpenCV BGR format...
DEBUG: ✓ Conversion successful
DEBUG: OpenCV image shape: (480, 640, 3)
------------------------------------------------------------
DEBUG: Starting YOLO inference...
DEBUG: ✓ Inference completed in 0.123 seconds
DEBUG: Number of detections: 3
------------------------------------------------------------
DEBUG: Saving annotated image...
DEBUG: ✓ Image saved successfully
------------------------------------------------------------
DEBUG: Building Detection2DArray message...
DEBUG:   Detection #1:
DEBUG:     Class ID: 0
DEBUG:     Confidence: 0.850
DEBUG:     Center: (320.5, 240.8)
DEBUG:     Size: 150.2 x 200.3
DEBUG: Publishing 3 detections to /detections
✓ Published 3 detections
```

## Viewing Results

### 1. Check Annotated Images

Annotated images with bounding boxes are saved to the `yolo_results` directory:
```bash
ls -lh yolo_results/
```

Each file is named: `frame_XXXXXX_TIMESTAMP.jpg`

### 2. Visualize in RViz2

You can visualize the detections in RViz2:
```bash
rviz2
```

Then add a topic visualization for `/detections`

### 3. Echo Detection Messages

To see raw detection data in the terminal:
```bash
source install/setup.bash
ros2 topic echo /detections
```

### 4. Check Topics

List all active topics:
```bash
ros2 topic list
```

Check the detection message structure:
```bash
ros2 interface show vision_msgs/msg/Detection2DArray
```

## Troubleshooting

### Issue: "No module named 'ultralytics'"
**Solution:** Install YOLO:
```bash
pip install ultralytics
```

### Issue: "Failed to open video file"
**Solution:** Check the video path in the video_stream_node:
```bash
# The path is relative to where you run the node
# Make sure ../media/waste_test_video.mp4 exists
ls ../media/waste_test_video.mp4
```

### Issue: "cv_bridge AttributeError: _ARRAY_API not found"
**Solution:** Downgrade NumPy:
```bash
pip install 'numpy<2'
```

### Issue: No frames being processed
**Solution:** Check if the camera topic is publishing:
```bash
ros2 topic hz /camera_frame
```

### Issue: Too slow / High CPU usage
**Solution:** Increase `every_n_frames` parameter:
```bash
ros2 run yolo_detector_ros2 yolo_detector_node --ros-args -p every_n_frames:=60
```

## Performance Tips

1. **Use lighter YOLO models** for faster inference:
   - `yolov8n.pt` - Nano (fastest, least accurate)
   - `yolov8s.pt` - Small
   - `yolov8m.pt` - Medium
   - `yolov8l.pt` - Large (slowest, most accurate)

2. **Process fewer frames** by increasing `every_n_frames` (e.g., 60 or 90)

3. **Increase confidence threshold** to reduce false positives: `-p conf_thres:=0.5`

4. **Use GPU** if available (YOLO will automatically use CUDA if PyTorch with CUDA support is installed)

## Complete Example Session

```bash
# Terminal 1 - Source and start video stream
cd /home/jana/TEZA/pick_and_place
source install/setup.bash
ros2 run helper_nodes video_stream_node

# Terminal 2 - Source and start detector
cd /home/jana/TEZA/pick_and_place
source install/setup.bash
ros2 run yolo_detector_ros2 yolo_detector_node

# Terminal 3 - Monitor detections (optional)
cd /home/jana/TEZA/pick_and_place
source install/setup.bash
ros2 topic echo /detections
```

## Stopping the Nodes

Press `Ctrl+C` in each terminal to gracefully shutdown the nodes.

You'll see debug output like:
```
DEBUG: Keyboard interrupt received (Ctrl+C)
DEBUG: Shutting down gracefully...
DEBUG: Destroying node...
DEBUG: ✓ Node destroyed
DEBUG: ✓ ROS2 shutdown complete
```

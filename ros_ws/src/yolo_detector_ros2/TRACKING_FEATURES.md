# YOLO Detector - Tracking Features

## Overview

The YOLO detector has been upgraded with a sophisticated tracking and filtering system optimized for efficiency and reliability:

- ⚡ **Processes every 30th frame** (configurable) for reduced computational load
- 🎯 **Requires 5 overlapping detections** (not consecutive) to confirm an object
- 🚫 **Filters out stationary objects** - only publishes when objects move
- ✂️ **Crops images** to focus on workspace center
- 🛡️ **Filters large bounding boxes** to prevent false scene-wide detections

**Key Benefit**: ~30x fewer inferences while maintaining detection reliability!

## Performance Characteristics

| Metric | Value | Description |
|--------|-------|-------------|
| Frame Processing | Every 30 frames | Skips 29 frames, processes 1 |
| Detection Confirmation | 5 overlapping | Must see object 5 times |
| Time to Confirm | ~5 seconds @ 30 FPS | 5 detections × 30 frames/detection |
| Republication | Only on movement | Prevents duplicate messages |
| CPU/GPU Usage | ~97% reduction | Compared to processing every frame |

## New Features

### 1. Efficient Frame Processing
- **Processes every Nth frame** (default: 30) instead of all frames
- Dramatically reduces computational load
- Example at 30 FPS:
  - Before: 30 inferences/second
  - After: 1 inference/second
  - **Savings: 97% reduction**

### 2. Smart Object Confirmation (5 Overlapping Detections)
- Objects confirmed after 5 successful detections (not necessarily consecutive)
- Much faster than requiring 30 consecutive detections
- More robust to occasional missed detections
- Timeline example:
  ```
  Frame 30:   Detection #1 → Track starts (1/5)
  Frame 60:   Detection #2 → Track grows (2/5)
  Frame 90:   Detection #3 → Track grows (3/5)
  Frame 120:  Detection #4 → Track grows (4/5)
  Frame 150:  Detection #5 → Track STABLE (5/5) ✓ PUBLISH!
  ```

### 3. Movement-Based Publishing
- **Prevents duplicate messages** for stationary objects
- Tracks last published position for each object
- Only republishes when object moves ≥ 20 pixels (configurable)
- Debug output shows movement decisions:
  ```
  DEBUG: Track #0 has MOVED: distance=35.8px, size_change=12.3px
  DEBUG: ✓ Track #0 will be published (moved or new)
  
  DEBUG: Track #0 stationary: distance=5.2px < 20.0px threshold
  DEBUG: ✗ Track #0 SKIPPED (no significant movement)
  ```

### 4. Image Cropping (1/5 from each side)
- Removes 20% (1/5) from each edge of the image
- Only processes the center 60% x 60% region
- Helps focus on the workspace area and reduces edge artifacts
- Bounding box coordinates are automatically adjusted back to original image space

### 5. Large Object Filtering
- Filters out detections that occupy > 90% of the cropped image
- Prevents false detections of the entire scene or background
- Debug messages printed when objects are filtered:
  ```
  DEBUG: ✗ FILTERED OUT detection #1:
  DEBUG:   Class: 0, Conf: 0.850
  DEBUG:   BBox: (320.5, 240.8, 580.2x420.3)
  DEBUG:   Area: 243924.1 (91.2% of image)
  DEBUG:   Reason: Exceeds 90.0% threshold
  ```

### 6. Temporal Tracking System
- Each detection is tracked across frames using IoU (Intersection over Union)
- Detections are matched to existing tracks based on:
  - Bounding box overlap (IoU threshold: 0.5)
  - Class ID consistency
- New tracks are created for unmatched detections
- Stale tracks (not seen for 60 frames) are automatically removed

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `every_n_frames` | `30` | Process every Nth frame (30 = 1 FPS at 30 FPS input) |
| `stable_frames_required` | `5` | Number of overlapping detections to confirm object |
| `movement_threshold` | `20.0` | Minimum pixel movement to republish (prevents duplicates) |
| `crop_ratio` | `0.2` | Ratio to crop from each side (0.2 = 1/5 = 20%) |
| `max_bbox_ratio` | `0.9` | Maximum bbox size relative to image (0.9 = 90%) |
| `conf_thres` | `0.25` | YOLO confidence threshold (0.0 - 1.0) |
| `iou_thres` | `0.45` | YOLO IoU threshold for NMS (0.0 - 1.0) |

## Processing Pipeline

```
1. Frame Received
   ↓
2. Skip if not Nth frame (e.g., skip 29, process 1)
   ↓
3. Convert to OpenCV
   ↓
4. Crop to Center 60% x 60%
   ↓
5. Run YOLO Inference (only on processed frames)
   ↓
6. Filter Large Bboxes (>90%)
   ↓
7. Match to Existing Tracks (IoU-based)
   ↓
8. Check Track Stability (5 detections?)
   ↓
9. Check if Object Moved (>20px?)
   ↓
10. Publish Only Moved Objects
```

## Complete Example Timeline

### At 30 FPS with default settings (every_n_frames=30, stable_frames_required=5):

```
Frame 1-29:   Skipped (no processing)
Frame 30:     PROCESSED → Object detected → Track #0 created (1/5)
Frame 31-59:  Skipped
Frame 60:     PROCESSED → Object detected → Track #0 updated (2/5)
Frame 61-89:  Skipped
Frame 90:     PROCESSED → Object detected → Track #0 updated (3/5)
Frame 91-119: Skipped
Frame 120:    PROCESSED → Object detected → Track #0 updated (4/5)
Frame 121-149: Skipped
Frame 150:    PROCESSED → Object detected → Track #0 STABLE (5/5) ✓
              → First time seen, has "moved" → PUBLISHED!
Frame 151-179: Skipped
Frame 180:    PROCESSED → Object detected → Track #0 still stable
              → Position unchanged (distance=3.2px) → NOT published
Frame 181-209: Skipped
Frame 210:    PROCESSED → Object detected → Track #0 still stable
              → Object moved (distance=45.8px) → PUBLISHED!
```

**Total time to first detection: 150 frames ÷ 30 FPS = 5 seconds**

## Debug Output Example

```
DEBUG: [Frame 150] Callback triggered
DEBUG: Image encoding: bgr8
DEBUG: Image dimensions: 640x480
DEBUG: ✓ Processing this frame (frame % 30 == 0)
------------------------------------------------------------
DEBUG: Cropped image from 640x480 to 384x288
DEBUG: Removed 128px from left/right, 96px from top/bottom
------------------------------------------------------------
DEBUG: Starting YOLO inference on cropped image...
DEBUG: ✓ Inference completed in 0.123 seconds
DEBUG: Raw detections from YOLO: 3
------------------------------------------------------------
DEBUG: Filtering large bounding boxes...
DEBUG: ✓ Kept detection #1: cls=0, conf=0.850, area=35.2%
DEBUG: ✗ FILTERED OUT detection #2:
DEBUG:   Class: 1, Conf: 0.720
DEBUG:   Area: 243924.1 (91.2% of image)
DEBUG:   Reason: Exceeds 90.0% threshold
DEBUG: Kept 2/3 detections after filtering
------------------------------------------------------------
DEBUG: Matching detections to existing tracks...
DEBUG: ✓ Matched detection (cls=0) to track #0
DEBUG: ✓ Created new track #5 for detection (cls=2)
------------------------------------------------------------
DEBUG: Checking track stability...
DEBUG: ✓ Track #0 is STABLE (5 detections)
DEBUG: Track #5 not yet stable (1/5 detections, need 4 more)
DEBUG: Found 1 stable tracks
------------------------------------------------------------
DEBUG: Checking which stable tracks have moved...
DEBUG: Track #0 has MOVED: distance=inf, size_change=inf
DEBUG: ✓ Track #0 will be published (moved or new)
DEBUG: 1 tracks to publish (out of 1 stable)
------------------------------------------------------------
DEBUG: Building Detection2DArray message for stable tracks...
DEBUG: Adding track #1 to message:
DEBUG:   Class: 0, Confidence: 0.850
DEBUG:   Cropped coords: (192.5, 144.8, 150.2x120.3)
DEBUG:   Original coords: (320.5, 240.8, 150.2x120.3)

DEBUG: Publishing 1 stable detections to /detections
✓ Published 1 stable detections
============================================================
```

## How to Run

### Basic Usage

```bash
# Terminal 1 - Video source
source install/setup.bash
ros2 run helper_nodes video_stream_node

# Terminal 2 - Detector with tracking
source install/setup.bash
ros2 run yolo_detector_ros2 yolo_detector_node
```

### Custom Parameters

**For faster response (confirm objects in ~1.5 seconds at 30 FPS):**
```bash
ros2 run yolo_detector_ros2 yolo_detector_node \
  --ros-args \
  -p every_n_frames:=15 \
  -p stable_frames_required:=3
```

**For more reliable detection (confirm objects in ~7.5 seconds at 30 FPS):**
```bash
ros2 run yolo_detector_ros2 yolo_detector_node \
  --ros-args \
  -p every_n_frames:=45 \
  -p stable_frames_required:=5
```

**For maximum sensitivity to movement:**
```bash
ros2 run yolo_detector_ros2 yolo_detector_node \
  --ros-args \
  -p movement_threshold:=10.0
```

**To process more of the image (reduce crop):**
```bash
ros2 run yolo_detector_ros2 yolo_detector_node \
  --ros-args \
  -p crop_ratio:=0.1 \
  -p max_bbox_ratio:=0.85
```

## Understanding Tracking Behavior

### Initial Startup (Frames 1-150 with default settings)
- System processes every 30th frame (frames 30, 60, 90, 120, 150)
- Tracks are built up as detections accumulate
- You'll see: "Track #X not yet stable (Y/5 detections, need Z more)"
- At frame 150 (5th detection), object is confirmed and published (if first time)

### After Object Confirmed (Frame 150+)
- Object continues to be tracked on every 30th frame
- **Only republished if it moves** ≥ 20 pixels
- Stationary objects are tracked but not republished
- You'll see:
  ```
  Track #0 stationary: distance=5.2px < 20.0px threshold
  Track #0 SKIPPED (no significant movement)
  ```

### Object Moves
- System detects position change
- Object is republished with new position
- You'll see:
  ```
  Track #0 has MOVED: distance=35.8px
  Track #0 will be published (moved or new)
  ```

### Object Leaves Scene
- Track stops being updated on processed frames
- After 60 processed frames (~1800 total frames or 60 seconds at 30 FPS), track is removed
- If object returns, it's treated as a new track (needs 5 detections again)

### Frame Processing at Different Rates

**At 30 FPS video:**
- `every_n_frames=30`: Processes 1 frame/second, confirm in ~5 seconds
- `every_n_frames=15`: Processes 2 frames/second, confirm in ~2.5 seconds
- `every_n_frames=60`: Processes 0.5 frames/second, confirm in ~10 seconds

**At 60 FPS video:**
- `every_n_frames=30`: Processes 2 frames/second, confirm in ~2.5 seconds
- `every_n_frames=60`: Processes 1 frame/second, confirm in ~5 seconds

## Tuning Tips

### For Faster Detection Response
Reduce frame interval and detection count:
```bash
-p every_n_frames:=15 \
-p stable_frames_required:=3
```
**Result**: ~1.5 second confirmation at 30 FPS

### For More Reliable Detection (Fewer False Positives)
Increase detection count or frame interval:
```bash
-p stable_frames_required:=7 \
-p every_n_frames:=30
```
**Result**: ~7 second confirmation, more stable

### For Better Performance (Lower CPU/GPU)
Increase frame interval:
```bash
-p every_n_frames:=60
```
**Result**: Half the computational load

### For More Frequent Updates
Decrease frame interval (but increases load):
```bash
-p every_n_frames:=15
```
**Result**: 2x the computational load, faster response

### For More Sensitive Movement Detection
Lower movement threshold:
```bash
-p movement_threshold:=10.0  # Republish on 10px movement
```

### For Less Sensitive Movement Detection (Reduce Updates)
Raise movement threshold:
```bash
-p movement_threshold:=50.0  # Only republish on 50px movement
```

### To Process More of the Image
Reduce `crop_ratio`:
```bash
-p crop_ratio:=0.1  # Only remove 10% from each side
```

### To Filter More Aggressively
Lower `max_bbox_ratio`:
```bash
-p max_bbox_ratio:=0.7  # Filter bboxes > 70%
```

## Troubleshooting

**Issue:** No detections are ever published
- Check if objects are visible for at least 5 processed frames (150 total frames at default)
- Verify objects aren't being filtered (check debug output for "FILTERED OUT")
- Try reducing `stable_frames_required` to 3 temporarily for testing
- Check that `every_n_frames` isn't too high (try 15 or 20)

**Issue:** Objects detected but never published
- May not be moving enough to trigger republication
- Check debug: "Track #X stationary: distance=Y.Ypx"
- Try lowering `movement_threshold` to 10.0

**Issue:** Too many detections filtered out
- Increase `max_bbox_ratio` (e.g., to 0.95)
- Check if cropping is too aggressive (reduce `crop_ratio` to 0.1)
- Verify objects are actually small enough (not scene-wide)

**Issue:** Unstable detections (track IDs jumping)
- YOLO may be detecting inconsistently across frames
- Try lowering `conf_thres` for more consistent detections (e.g., 0.20)
- Try processing more frequently (`every_n_frames:=15`)
- Adjust `iou_threshold_tracking` in code (currently 0.5)

**Issue:** Performance is still slow
- Increase `every_n_frames` to reduce load (try 45 or 60)
- Use a lighter YOLO model (yolov8n.pt instead of yolov8s.pt)
- Run on GPU if available
- Check if video source is high resolution (crop more aggressively)

**Issue:** Objects take too long to confirm
- Reduce `every_n_frames` (e.g., 15 for 2x faster)
- Reduce `stable_frames_required` (e.g., 3 for faster confirmation)
- Trade-off: Faster response vs. more false positives

## Technical Details

### Track Matching Algorithm
Uses IoU (Intersection over Union) to match detections across frames:
- Computes IoU between new detection and last detection in each track
- Requires same class ID
- Minimum IoU threshold: 0.5
- Greedily matches to track with highest IoU

### Coordinate System
- YOLO runs on **cropped image** (center 60% x 60%)
- Detections in cropped coordinates
- Published detections transformed back to **original image coordinates**
- Formula: `x_original = x_cropped + crop_offset_x`

### Memory Management
- Each track stores max 5 recent detections (using `deque` with `maxlen`)
- Old tracks (60+ processed frames without update) are automatically pruned
- Memory usage scales with number of distinct objects seen
- Very lightweight: ~few KB per active track

## Key Benefits

1. ✅ **97% reduction in computational load** - Processes every 30th frame instead of all frames
2. ✅ **Fast object confirmation** - Only 5 detections needed vs 30 consecutive
3. ✅ **No duplicate messages** - Movement detection prevents republishing stationary objects
4. ✅ **No false positives** - Multi-detection requirement filters transient detections
5. ✅ **Cleaner workspace focus** - Image cropping removes edges and reduces noise
6. ✅ **Filters out scene-wide detections** - 90% size filter prevents background detections
7. ✅ **Consistent object IDs** - Tracks maintained across frames with IoU matching
8. ✅ **Scalable performance** - Adjustable parameters for speed vs reliability trade-off
9. ✅ **Real-time capability** - Can handle 30 FPS video at ~1 FPS processing rate
10. ✅ **Automatic cleanup** - Stale tracks removed, no memory leaks


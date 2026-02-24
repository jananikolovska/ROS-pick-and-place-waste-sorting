#!/usr/bin/env python3
"""
YOLO Detector ROS2 Node
=======================
This node subscribes to camera frames, runs YOLO object detection,
saves annotated images, and publishes detection results.

Author: Your Name
Date: February 23, 2026
"""

import os
import time
from collections import defaultdict, deque

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose, ObjectHypothesis, Pose2D, Point2D

import cv2
import numpy as np
from ultralytics import YOLO


class YoloDetectorNode(Node):
    """
    YOLO Object Detection Node for ROS2
    
    This node:
    1. Subscribes to image stream from a camera topic
    2. Processes every N-th frame with YOLO model
    3. Saves annotated images to disk
    4. Publishes detections in vision_msgs format
    """
    
    def __init__(self):
        """Initialize the YOLO detector node with parameters and setup ROS connections."""
        super().__init__("yolo_detector_node")
        
        print("\n" + "="*80)
        print("DEBUG: Starting YoloDetectorNode initialization")
        print("="*80)

        # ============================================================
        # STEP 1: Declare ROS2 Parameters
        # ============================================================
        # These can be overridden via launch files or command line
        print("DEBUG: Declaring ROS2 parameters...")
        
        self.declare_parameter("image_topic", "/camera_frame")
        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("model_path", "yolo_detector_ros2/yolov8s_best.pt")
        self.declare_parameter("every_n_frames", 5)  # Process every 5th frame (6 FPS)
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("save_dir", "yolo_results")
        self.declare_parameter("crop_ratio", 0.2)  # Crop 20% (1/5) from each side
        self.declare_parameter("max_bbox_ratio", 0.9)  # Filter bboxes > 90% of image
        self.declare_parameter("stable_frames_required", 10)  # Require 10 overlapping detections (~1.7s)
        self.declare_parameter("variance_threshold", 15.0)  # Max bbox position variance (pixels) for stability
        self.declare_parameter("destabilize_threshold", 50.0)  # Movement (pixels) to mark as destabilized
        print("DEBUG: All parameters declared successfully")

        # ============================================================
        # STEP 2: Load Parameter Values
        # ============================================================
        print("DEBUG: Loading parameter values...")
        
        # Topic to subscribe for incoming camera frames
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        print(f"DEBUG: image_topic = '{self.image_topic}'")
        
        # Topic to publish detection results
        self.detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        print(f"DEBUG: detections_topic = '{self.detections_topic}'")
        
        # Path to YOLO model weights file (e.g., yolov8n.pt, yolov8s.pt)
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        print(f"DEBUG: model_path = '{self.model_path}'")
        
        # Process only every N-th frame to reduce computational load
        self.every_n_frames = self.get_parameter("every_n_frames").get_parameter_value().integer_value
        print(f"DEBUG: every_n_frames = {self.every_n_frames}")
        
        # Confidence threshold for detections (0.0 - 1.0)
        self.conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        print(f"DEBUG: conf_thres = {self.conf_thres}")
        
        # Crop ratio for image preprocessing
        self.crop_ratio = self.get_parameter("crop_ratio").get_parameter_value().double_value
        print(f"DEBUG: crop_ratio = {self.crop_ratio} (removing {self.crop_ratio*100}% from each side)")
        
        # Maximum bbox size relative to image (filter larger ones)
        self.max_bbox_ratio = self.get_parameter("max_bbox_ratio").get_parameter_value().double_value
        print(f"DEBUG: max_bbox_ratio = {self.max_bbox_ratio} (filtering bboxes > {self.max_bbox_ratio*100}%)")
        
        # Number of stable frames required before publishing
        self.stable_frames_required = self.get_parameter("stable_frames_required").get_parameter_value().integer_value
        print(f"DEBUG: stable_frames_required = {self.stable_frames_required} detections")
        
        # Maximum bbox position variance for stability
        self.variance_threshold = self.get_parameter("variance_threshold").get_parameter_value().double_value
        print(f"DEBUG: variance_threshold = {self.variance_threshold} pixels (for twitching)")
        
        # Movement threshold to mark object as destabilized
        self.destabilize_threshold = self.get_parameter("destabilize_threshold").get_parameter_value().double_value
        print(f"DEBUG: destabilize_threshold = {self.destabilize_threshold} pixels (for re-detection)")
        
        # IoU threshold for non-maximum suppression (0.0 - 1.0)
        self.iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        print(f"DEBUG: iou_thres = {self.iou_thres}")
        
        # Directory where annotated images will be saved
        self.save_dir = self.get_parameter("save_dir").get_parameter_value().string_value
        print(f"DEBUG: save_dir = '{self.save_dir}'")

        # ============================================================
        # STEP 3: Create Output Directory
        # ============================================================
        print(f"DEBUG: Creating output directory: {os.path.abspath(self.save_dir)}")
        os.makedirs(self.save_dir, exist_ok=True)
        print("DEBUG: Output directory ready")

        # ============================================================
        # STEP 4: Load YOLO Model
        # ============================================================
        print(f"DEBUG: Loading YOLO model from: {self.model_path}")
        print("DEBUG: This may take a few seconds...")
        try:
            self.model = YOLO(self.model_path)
            print("DEBUG: ✓ YOLO model loaded successfully!")
        except Exception as e:
            print(f"DEBUG: ✗ FAILED to load YOLO model: {e}")
            raise

        # ============================================================
        # STEP 5: Setup ROS2 Communication
        # ============================================================
        print("DEBUG: Setting up ROS2 publishers and subscribers...")
        
        # CvBridge converts between ROS Image messages and OpenCV images
        self.bridge = CvBridge()
        print("DEBUG: CvBridge initialized")
        
        # Counter to track frame numbers
        self.frame_count = 0
        print("DEBUG: Frame counter initialized to 0")
        
        # ============================================================
        # STEP 5b: Setup Detection Tracking System
        # ============================================================
        print("DEBUG: Initializing detection tracking system...")
        
        # Track detection history for temporal consistency
        # Format: {object_id: deque of (frame_num, bbox, class_id, conf)}
        self.detection_tracks = defaultdict(lambda: deque(maxlen=self.stable_frames_required))
        self.next_object_id = 0
        self.iou_threshold_tracking = 0.5  # IoU threshold for matching detections
        
        # Object states for placement detection state machine
        # States: "DETECTING" -> "STABLE" -> "DESTABILIZED" -> "DETECTING"
        # Format: {track_id: state_string}
        self.object_states = {}
        
        # Track last stable bbox position for destabilization detection
        # Format: {track_id: (xc, yc, w, h)}
        self.last_stable_bbox = {}
        
        # Store crop parameters
        self.crop_offset_x = 0
        self.crop_offset_y = 0
        
        print(f"DEBUG: Tracking system initialized (IoU threshold: {self.iou_threshold_tracking})")
        print(f"DEBUG: Placement detection enabled: variance_threshold={self.variance_threshold}px, destabilize_threshold={self.destabilize_threshold}px")

        # Publisher for detection results (vision_msgs/Detection2DArray)
        self.pub = self.create_publisher(Detection2DArray, self.detections_topic, 10)
        print(f"DEBUG: Publisher created on topic: {self.detections_topic}")
        
        # Subscriber for camera frames (sensor_msgs/Image)
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        print(f"DEBUG: Subscriber created on topic: {self.image_topic}")

        # ============================================================
        # Initialization Complete - Display Summary
        # ============================================================
        print("\n" + "="*80)
        print("YOLO DETECTOR NODE - INITIALIZATION COMPLETE")
        print("="*80)
        self.get_logger().info("YOLO PLACEMENT DETECTOR node started.")
        self.get_logger().info(f"Subscribing: {self.image_topic}")
        self.get_logger().info(f"Publishing:  {self.detections_topic}")
        self.get_logger().info(f"Model:       {self.model_path}")
        self.get_logger().info(f"Processing:  Every {self.every_n_frames} frames (~{30/self.every_n_frames:.0f} FPS)")
        self.get_logger().info(f"Crop ratio:  {self.crop_ratio} (1/5 from each side)")
        self.get_logger().info(f"Max bbox:    {self.max_bbox_ratio} (filtering larger)")
        self.get_logger().info(f"Placement detection: {self.stable_frames_required} stable detections (~{self.stable_frames_required * self.every_n_frames / 30:.1f}s)")
        self.get_logger().info(f"Variance threshold: {self.variance_threshold}px (smooths twitching)")
        self.get_logger().info(f"Destabilize threshold: {self.destabilize_threshold}px (for re-detection)")
        self.get_logger().info(f"Save dir:    {os.path.abspath(self.save_dir)}")
        print("="*80)
        print("DEBUG: Waiting for incoming frames...")
        print("="*80 + "\n")

    def compute_iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1, box2: (xc, yc, w, h) center coordinates and dimensions
            
        Returns:
            float: IoU score (0.0 - 1.0)
        """
        # Convert center format to corner format
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        # Compute intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height
        
        # Compute union area
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        # Compute IoU
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def match_detection_to_track(self, bbox, class_id):
        """
        Match a new detection to an existing track based on IoU and class.
        
        Args:
            bbox: (xc, yc, w, h) bounding box
            class_id: int class identifier
            
        Returns:
            int or None: matched track ID, or None if no match
        """
        best_iou = 0
        best_track_id = None
        
        for track_id, track_history in self.detection_tracks.items():
            if len(track_history) == 0:
                continue
                
            # Get most recent detection in this track
            last_frame, last_bbox, last_class, last_conf = track_history[-1]
            
            # Must be same class
            if last_class != class_id:
                continue
            
            # Compute IoU with last detection
            iou = self.compute_iou(bbox, last_bbox)
            
            if iou > best_iou and iou >= self.iou_threshold_tracking:
                best_iou = iou
                best_track_id = track_id
        
        return best_track_id

    def compute_bbox_variance(self, track_id):
        """
        Compute position variance of bounding box centers to detect twitching.
        
        Args:
            track_id: int track identifier
            
        Returns:
            float: Standard deviation of center positions (pixels)
        """
        track_history = self.detection_tracks[track_id]
        
        if len(track_history) < 2:
            return 0.0
        
        # Extract center positions
        centers = [(bbox[0], bbox[1]) for (frame, bbox, cls, conf) in track_history]
        centers_array = np.array(centers)
        
        # Calculate std deviation of x and y separately, then take mean
        x_std = np.std(centers_array[:, 0])
        y_std = np.std(centers_array[:, 1])
        
        # Return average variance
        variance = (x_std + y_std) / 2.0
        return variance

    def is_track_stable(self, track_id):
        """
        Check if a track has been stable for the required number of detections
        with low bbox position variance (handles twitching).
        
        Args:
            track_id: int track identifier
            
        Returns:
            bool: True if track has enough detections AND low position variance
        """
        track_history = self.detection_tracks[track_id]
        
        # Need enough detections
        if len(track_history) < self.stable_frames_required:
            return False
        
        # Check bbox variance (should be low for stable placement)
        variance = self.compute_bbox_variance(track_id)
        is_stable = variance < self.variance_threshold
        
        if not is_stable:
            print(f"DEBUG: Track #{track_id} has high variance ({variance:.1f}px) - still moving/twitching")
        
        return is_stable

    def has_object_destabilized(self, track_id, current_bbox):
        """
        Check if a previously stable object has moved significantly (destabilized).
        
        Args:
            track_id: int track identifier
            current_bbox: (xc, yc, w, h) current bounding box in cropped coords
            
        Returns:
            bool: True if object has moved beyond destabilize_threshold
        """
        # If no stable bbox recorded, can't be destabilized
        if track_id not in self.last_stable_bbox:
            return False
        
        # Get last stable position
        last_bbox = self.last_stable_bbox[track_id]
        
        # Calculate position change (center point movement)
        xc_curr, yc_curr, w_curr, h_curr = current_bbox
        xc_last, yc_last, w_last, h_last = last_bbox
        
        # Euclidean distance between centers
        distance = np.sqrt((xc_curr - xc_last)**2 + (yc_curr - yc_last)**2)
        
        # Also check size change (could indicate different detection)
        size_change = abs(w_curr - w_last) + abs(h_curr - h_last)
        
        has_moved = distance >= self.destabilize_threshold or size_change >= self.destabilize_threshold
        
        if has_moved:
            print(f"DEBUG: Track #{track_id} DESTABILIZED: distance={distance:.1f}px, size_change={size_change:.1f}px (threshold={self.destabilize_threshold}px)")
        
        return has_moved

    def image_cb(self, msg: Image):
        """
        Callback function triggered every time a new camera frame arrives.
        
        Placement Detection Pipeline:
        1. Skip frames (process every 5th frame for 6 FPS)
        2. Convert ROS Image to OpenCV format
        3. Crop image by 1/5 on each side (keep center 60%)
        4. Run YOLO inference on cropped image
        5. Filter out large bounding boxes (> 90% of cropped image)
        6. Match detections to existing tracks
        7. Track bbox variance to detect stable placement (handles twitching)
        8. State machine: DETECTING → STABLE → DESTABILIZED → DETECTING
        9. Publish ONLY when object first becomes stable (placed)
        10. Allow re-detection after object is moved (destabilized)
        
        Args:
            msg (Image): ROS2 Image message from the camera
        """
        # ============================================================
        # STEP 1: Frame Counter and Logging
        # ============================================================
        self.frame_count += 1
        self.get_logger().info(f">>> Frame #{self.frame_count} received")
        print(f"\nDEBUG: [Frame {self.frame_count}] Callback triggered")
        print(f"DEBUG: Image encoding: {msg.encoding}")
        print(f"DEBUG: Image dimensions: {msg.width}x{msg.height}")

        # ============================================================
        # STEP 2: Frame Skipping Logic
        # ============================================================
        # Only process every N-th frame to reduce computational load
        if self.frame_count % int(self.every_n_frames) != 0:
            skip_reason = f"Skipping frame (processing every {self.every_n_frames} frames)"
            print(f"DEBUG: {skip_reason}")
            return
        
        print(f"DEBUG: ✓ Processing this frame (frame % {self.every_n_frames} == 0)")
        print("-" * 60)

        # ============================================================
        # STEP 3: Convert ROS Image to OpenCV Format
        # ============================================================
        print("DEBUG: Converting ROS Image message to OpenCV BGR format...")
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            print(f"DEBUG: ✓ Conversion successful, shape: {frame_bgr.shape}")
        except Exception as e:
            print(f"DEBUG: ✗ Conversion FAILED: {e}")
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # ============================================================
        # STEP 4: Crop Image by 1/5 on Each Side
        # ============================================================
        h, w = frame_bgr.shape[:2]
        crop_w = int(w * self.crop_ratio)
        crop_h = int(h * self.crop_ratio)
        
        # Store crop offset for later bbox coordinate adjustment
        self.crop_offset_x = crop_w
        self.crop_offset_y = crop_h
        
        # Crop to center region
        cropped_frame = frame_bgr[crop_h:h-crop_h, crop_w:w-crop_w]
        crop_h_new, crop_w_new = cropped_frame.shape[:2]
        
        print(f"DEBUG: Cropped image from {w}x{h} to {crop_w_new}x{crop_h_new}")
        print(f"DEBUG: Removed {crop_w}px from left/right, {crop_h}px from top/bottom")
        print("-" * 60)

        # ============================================================
        # STEP 5: Run YOLO Inference on Cropped Image
        # ============================================================
        print("DEBUG: Starting YOLO inference on cropped image...")
        inference_start = time.time()
        
        try:
            results_list = self.model.predict(
                source=cropped_frame,
                conf=float(self.conf_thres),
                iou=float(self.iou_thres),
                verbose=False
            )
            inference_time = time.time() - inference_start
            print(f"DEBUG: ✓ Inference completed in {inference_time:.3f} seconds")
            
            if not results_list:
                print("DEBUG: ✗ No results returned from YOLO model")
                return
            
            r0 = results_list[0]
            print(f"DEBUG: Raw detections from YOLO: {len(r0.boxes) if r0.boxes is not None else 0}")
            
        except Exception as e:
            print(f"DEBUG: ✗ YOLO inference FAILED: {e}")
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        # ============================================================
        # STEP 6: Filter Large Bounding Boxes
        # ============================================================
        print("-" * 60)
        print("DEBUG: Filtering large bounding boxes...")
        
        boxes = r0.boxes
        if boxes is None or len(boxes) == 0:
            print("DEBUG: No detections in this frame")
            return

        # Extract detection data
        xywh = boxes.xywh.cpu().numpy()  # (xc, yc, w, h) in cropped image coords
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        
        # Calculate image area for filtering
        cropped_area = crop_w_new * crop_h_new
        max_bbox_area = cropped_area * self.max_bbox_ratio
        
        # Filter detections
        filtered_detections = []
        for i, ((xc, yc, w, h), conf, cls_id) in enumerate(zip(xywh, confs, clss)):
            bbox_area = w * h
            bbox_ratio = bbox_area / cropped_area
            
            if bbox_area > max_bbox_area:
                print(f"DEBUG: ✗ FILTERED OUT detection #{i+1}:")
                print(f"DEBUG:   Class: {cls_id}, Conf: {conf:.3f}")
                print(f"DEBUG:   BBox: ({xc:.1f}, {yc:.1f}, {w:.1f}x{h:.1f})")
                print(f"DEBUG:   Area: {bbox_area:.1f} ({bbox_ratio*100:.1f}% of image)")
                print(f"DEBUG:   Reason: Exceeds {self.max_bbox_ratio*100}% threshold")
                self.get_logger().warn(f"Filtered large bbox: {bbox_ratio*100:.1f}% of image (cls={cls_id})")
                continue
            
            filtered_detections.append((xc, yc, w, h, conf, cls_id))
            print(f"DEBUG: ✓ Kept detection #{i+1}: cls={cls_id}, conf={conf:.3f}, area={bbox_ratio*100:.1f}%")
        
        print(f"DEBUG: Kept {len(filtered_detections)}/{len(xywh)} detections after filtering")
        
        if len(filtered_detections) == 0:
            print("DEBUG: All detections were filtered out")
            return

        # ============================================================
        # STEP 7: Match Detections to Tracks
        # ============================================================
        print("-" * 60)
        print("DEBUG: Matching detections to existing tracks...")
        
        matched_tracks = set()
        new_detections = []
        
        for (xc, yc, w, h, conf, cls_id) in filtered_detections:
            bbox = (xc, yc, w, h)
            
            # Try to match to existing track
            track_id = self.match_detection_to_track(bbox, cls_id)
            
            if track_id is not None:
                # Matched to existing track
                print(f"DEBUG: ✓ Matched detection (cls={cls_id}) to track #{track_id}")
                self.detection_tracks[track_id].append((self.frame_count, bbox, cls_id, conf))
                matched_tracks.add(track_id)
            else:
                # No match, create new track
                new_track_id = self.next_object_id
                self.next_object_id += 1
                print(f"DEBUG: ✓ Created new track #{new_track_id} for detection (cls={cls_id})")
                self.detection_tracks[new_track_id].append((self.frame_count, bbox, cls_id, conf))
                matched_tracks.add(new_track_id)

        # Remove old tracks (not seen in last 60 frames)
        tracks_to_remove = []
        for track_id, track_history in self.detection_tracks.items():
            if len(track_history) > 0:
                last_frame = track_history[-1][0]
                if self.frame_count - last_frame > 60:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            print(f"DEBUG: Removing stale track #{track_id}")
            del self.detection_tracks[track_id]
            # Also remove from state tracking
            if track_id in self.object_states:
                del self.object_states[track_id]
                print(f"DEBUG: Cleared state for track #{track_id}")
            if track_id in self.last_stable_bbox:
                del self.last_stable_bbox[track_id]
                print(f"DEBUG: Cleared stable bbox for track #{track_id}")

        # ============================================================
        # STEP 8: Placement Detection State Machine
        # ============================================================
        print("-" * 60)
        print("DEBUG: Running placement detection state machine...")
        
        tracks_to_publish = []
        
        for track_id in matched_tracks:
            track_history = self.detection_tracks[track_id]
            frame_num, bbox, cls_id, conf = track_history[-1]
            
            # Initialize state if new track
            if track_id not in self.object_states:
                self.object_states[track_id] = "DETECTING"
                print(f"DEBUG: Track #{track_id} initialized in DETECTING state")
            
            current_state = self.object_states[track_id]
            print(f"DEBUG: Track #{track_id} current state: {current_state}")
            
            # --- STATE: DETECTING ---
            if current_state == "DETECTING":
                # Check if object has stabilized
                if self.is_track_stable(track_id):
                    variance = self.compute_bbox_variance(track_id)
                    print(f"DEBUG: ✓ Track #{track_id} became STABLE! (variance={variance:.1f}px < {self.variance_threshold}px)")
                    print(f"DEBUG: ✓ Object placed successfully - publishing detection!")
                    
                    # Transition to STABLE state
                    self.object_states[track_id] = "STABLE"
                    self.last_stable_bbox[track_id] = bbox
                    
                    # Publish this placement detection
                    tracks_to_publish.append(track_id)
                else:
                    track_len = len(track_history)
                    frames_needed = self.stable_frames_required - track_len
                    print(f"DEBUG: Track #{track_id} still detecting ({track_len}/{self.stable_frames_required}, need {frames_needed} more or lower variance)")
            
            # --- STATE: STABLE ---
            elif current_state == "STABLE":
                # Check if object has been moved/destabilized
                if self.has_object_destabilized(track_id, bbox):
                    print(f"DEBUG: ✓ Track #{track_id} has been moved - transitioning to DESTABILIZED")
                    self.object_states[track_id] = "DESTABILIZED"
                    # Clear track history to start fresh detection
                    self.detection_tracks[track_id].clear()
                else:
                    print(f"DEBUG: Track #{track_id} remains STABLE (not publishing - already reported)")
            
            # --- STATE: DESTABILIZED ---
            elif current_state == "DESTABILIZED":
                # Start detecting again
                print(f"DEBUG: Track #{track_id} in DESTABILIZED state - resetting to DETECTING")
                self.object_states[track_id] = "DETECTING"
                # History was already cleared when it destabilized
        
        print(f"DEBUG: {len(tracks_to_publish)} new placements detected")

        # ============================================================
        # STEP 9: Build Detection Message (Only Newly Placed Objects)
        # ============================================================
        if len(tracks_to_publish) == 0:
            print("DEBUG: No new placements detected - nothing to publish")
            return
        
        print("-" * 60)
        print("DEBUG: Building Detection2DArray message for newly placed objects...")
        
        det_array = Detection2DArray()
        det_array.header = msg.header

        for track_id in tracks_to_publish:
            track_history = self.detection_tracks[track_id]
            
            # Use most recent detection from this track
            frame_num, bbox, cls_id, conf = track_history[-1]
            xc, yc, w, h = bbox
            
            # Adjust coordinates back to original (uncropped) image space
            xc_original = xc + self.crop_offset_x
            yc_original = yc + self.crop_offset_y
            
            print(f"DEBUG: Adding track #{track_id} to message:")
            print(f"DEBUG:   Class: {cls_id}, Confidence: {conf:.3f}")
            print(f"DEBUG:   Cropped coords: ({xc:.1f}, {yc:.1f}, {w:.1f}x{h:.1f})")
            print(f"DEBUG:   Original coords: ({xc_original:.1f}, {yc_original:.1f}, {w:.1f}x{h:.1f})")
            
            # Create Detection2D message
            det = Detection2D()
            det.header = msg.header

            # Create BoundingBox2D with original image coordinates
            bbox_msg = BoundingBox2D()
            
            # Pose2D has a position field (Point2D) with x, y
            bbox_msg.center.position.x = float(xc_original)
            bbox_msg.center.position.y = float(yc_original)
            bbox_msg.center.theta = 0.0  # No rotation
            
            bbox_msg.size_x = float(w)
            bbox_msg.size_y = float(h)
            det.bbox = bbox_msg

            # Create ObjectHypothesisWithPose
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(cls_id)  # class_id is a string
            hyp.hypothesis.score = float(conf)
            det.results.append(hyp)

            det_array.detections.append(det)

        # ============================================================
        # STEP 10: Publish Placement Detections
        # ============================================================
        print(f"\nDEBUG: Publishing {len(det_array.detections)} placement detections to {self.detections_topic}")
        self.pub.publish(det_array)
        self.get_logger().info(f"✓ Published {len(det_array.detections)} new object placements")
        print("="*60 + "\n")
        
        # ============================================================
        # STEP 11: Save Annotated Image (Optional)
        # ============================================================
        try:
            annotated_bgr = r0.plot()
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            if stamp_ns == 0:
                stamp_ns = int(time.time() * 1e9)
            out_path = os.path.join(
                self.save_dir,
                f"frame_{self.frame_count:06d}_{stamp_ns}.jpg"
            )
            cv2.imwrite(out_path, annotated_bgr)
            print(f"DEBUG: Saved annotated image: {out_path}")
        except Exception as e:
            print(f"DEBUG: Could not save annotated image: {e}")



def main():
    """
    Main entry point for the YOLO detector node.
    
    Initializes the ROS2 system, creates the detector node,
    and spins until interrupted by user (Ctrl+C).
    """
    print("\n" + "="*80)
    print("YOLO DETECTOR NODE - STARTING UP")
    print("="*80)
    print("DEBUG: Initializing ROS2...")
    
    rclpy.init()
    print("DEBUG: ✓ ROS2 initialized")
    
    print("DEBUG: Creating YoloDetectorNode instance...")
    node = YoloDetectorNode()
    print("DEBUG: ✓ Node created successfully")
    
    try:
        print("DEBUG: Starting node spin (entering event loop)...")
        print("DEBUG: Press Ctrl+C to shutdown")
        print("="*80)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("DEBUG: Keyboard interrupt received (Ctrl+C)")
        print("DEBUG: Shutting down gracefully...")
        print("="*80)
    finally:
        print("DEBUG: Destroying node...")
        node.destroy_node()
        print("DEBUG: ✓ Node destroyed")
        
        print("DEBUG: Shutting down ROS2...")
        rclpy.shutdown()
        print("DEBUG: ✓ ROS2 shutdown complete")
        print("="*80)
        print("YOLO DETECTOR NODE - SHUTDOWN COMPLETE")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
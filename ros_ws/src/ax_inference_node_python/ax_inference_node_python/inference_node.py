#!/usr/bin/env python3
import rclpy
import rclpy.time
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import numpy as np
import cv2
from pathlib import Path
import json
import onnxruntime as ort
import time
from collections import deque
import os
import datetime

from axelera.runtime import Context

#### Helper functions for preprocessing and postprocessing ####

def execute_onnx_postprocess(
    onnx_session: ort.InferenceSession,
    onnx_io_binding: ort.IOBinding,
    onnx_input_buffers: list[np.ndarray],
    inputs_list: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Execute ONNX postprocessing using preallocated I/O binding and input buffers.   
    """
    if len(inputs_list) != len(onnx_input_buffers):
        raise ValueError(f"Expected {len(onnx_input_buffers)} inputs, but got {len(inputs_list)}.")

    # Copy new data into the preallocated buffers
    for i, arr in enumerate(inputs_list):
        np.copyto(onnx_input_buffers[i], arr, casting='same_kind')

    # Run with the provided binding (zero-copy execution)
    onnx_session.run_with_iobinding(onnx_io_binding)

    # Copy outputs to CPU (allocates NumPy arrays)
    onnx_outputs = onnx_io_binding.copy_outputs_to_cpu()

    return onnx_outputs


def extract_bounding_boxes(predictions: np.ndarray, has_objectness: bool, confidence_threshold: float):
    """
    Vectorized extraction of bounding boxes, confidences, and class IDs.
    """
    N = predictions.shape[0]
    if has_objectness:
        class_scores = predictions[:, 5:]  # shape (N, num_classes)
        class_ids = np.argmax(class_scores, axis=1)
        confidences = predictions[:, 4] * class_scores[np.arange(N), class_ids]
    else:
        class_scores = predictions[:, 4:]  # shape (N, num_classes)
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(N), class_ids]

    # Apply confidence threshold
    mask = confidences > confidence_threshold
    if not np.any(mask):
        return [], [], []

    # Select only valid predictions
    preds = predictions[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    x_center = preds[:, 0]
    y_center = preds[:, 1]
    width = preds[:, 2]
    height = preds[:, 3]

    # Convert to xyxy
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    boxes = np.stack([x1, y1, x2, y2], axis=1).tolist()
    confidences = confidences.tolist()
    class_ids = class_ids.tolist()

    
    return boxes, confidences, class_ids

def postprocess_model_output(
    onnx_session: ort.InferenceSession,
    onnx_input_names: list[str],
    onnx_io_binding: ort.IOBinding,
    onnx_input_buffers: list[np.ndarray],
    inputs_list: list[np.ndarray],
    confidence_threshold: float,
    nms_threshold: float
) -> tuple[str, list[tuple[int, float, list[float]]]]:
    """
    Postprocess ONNX model results to extract final detections.

    Args:
        onnx_session: The ONNX inference session.
        onnx_input_names: List of input names for the ONNX model.
        onnx_io_binding: Pre-initialized ONNX I/O binding for zero-copy execution.
        onnx_input_buffers: Preallocated NumPy input buffers.
        inputs_list: List of input arrays for the ONNX model.
        confidence_threshold: Minimum confidence threshold for filtering.
        nms_threshold: IoU threshold for Non-Maximum Suppression.

    Returns:
        (box_type, detections) where detections = (class_id, confidence, (x1, y1, x2, y2))
    """
    # Run ONNX postprocessing using existing session
    onnx_results = execute_onnx_postprocess(onnx_session, onnx_io_binding, onnx_input_buffers, inputs_list)
    
    final_detections = []
    box_type = None

    for i, result in enumerate(onnx_results):
        if result.ndim == 3 and result.shape[2] == 85:  # YOLOv5 format
            box_type = 'xyxy'
            predictions = result[0]  # shape: (N, 85)
            has_objectness = True
        elif result.ndim == 3 and result.shape[1] == 84:  # YOLOv8 format
            box_type = 'xyxy'
            predictions = result[0].T  # shape: (N, 84)
            has_objectness = False
        else:
            raise ValueError(f"Unexpected result shape: {result.shape}. Expected (1, N, 85) or (1, 84, N).")

        # Parse predictions
        boxes, confidences, class_ids = extract_bounding_boxes(predictions, has_objectness, confidence_threshold)
        
        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append((class_ids[i], confidences[i], boxes[i]))

    return box_type, final_detections

### ROS2 Node ###

# ROS2 Node
class AxeleraYoloInference(Node):
    def __init__(self):
        super().__init__('axelera_camera_classifier')
        
        self.current_frame = None
        self.frame_id = 0
        
        # Declare ROS parameters
        self.declare_parameter('model_name', '')
        self.declare_parameter('aipu_cores', 4)
        self.declare_parameter('input_topic', '/camera_frame')
        self.declare_parameter('output_topic', '/detections_topic')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('mean', [0.485, 0.456, 0.406])
        self.declare_parameter('stddev', [0.229, 0.224, 0.225])
        self.declare_parameter('publish_annotated', False)
        self.declare_parameter('compute_metrics', True)
        self.declare_parameter('save_dir', '~/ros2_metrics')

        # Load parameters
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value  
        aipu_cores = self.get_parameter('aipu_cores').get_parameter_value().integer_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.nms_threshold = self.get_parameter('nms_threshold').get_parameter_value().double_value
        self.mean = self.get_parameter('mean').get_parameter_value().double_array_value
        self.stddev = self.get_parameter('stddev').get_parameter_value().double_array_value
        self.publish_annotated = self.get_parameter('publish_annotated').get_parameter_value().bool_value
        self.compute_metrics = self.get_parameter('compute_metrics').get_parameter_value().bool_value
        self.save_dir = os.path.expanduser(
            self.get_parameter('save_dir').get_parameter_value().string_value)

        model_path = Path(f'../build/{self.model_name}/{self.model_name}/1/model.json')
        labels_path = Path(f'../build/{self.model_name}/{self.model_name}/model_info.json')
        self.onnx_model_path = Path(f'../build/{self.model_name}/{self.model_name}/1/postprocess_graph.onnx')
                    
        if model_path.is_dir():
            model_path = model_path / "model.json"

        # Load labels dynamically from the JSON file
        self.labels = self.load_labels(Path(labels_path))

        # Load model and runtime
        self.ctx = Context()
        self.model = self.ctx.load_model(model_path)

        self.input_infos = self.model.inputs()
        self.output_infos = self.model.outputs()

        self.batch_size = self.input_infos[0].shape[0]
        self.inputs = [np.zeros(t.shape, dtype=np.int8) for t in self.input_infos]
        self.outputs = [np.zeros(t.shape, dtype=np.int8) for t in self.output_infos]

        connection = self.ctx.device_connect(None, self.batch_size)
        self.instance = connection.load_model_instance(
            self.model,
            num_sub_devices=self.batch_size,
            aipu_cores=aipu_cores
        )

        self.bridge = CvBridge()

        # Load ONNX model session
        self.onnx_session = ort.InferenceSession(str(self.onnx_model_path))
        self.onnx_input_names = [input_meta.name for input_meta in self.onnx_session.get_inputs()]
        
        # --- PRECOMPUTED PREPROCESSING VALUES ---
        # Extract and precompute values that don't change between frames
        input_info = self.input_infos[0]
        self.model_height = input_info.unpadded_shape[1]  # Height from tensor info
        self.model_width = input_info.unpadded_shape[2]   # Width from tensor info
        self.mean_array = np.array(self.mean, dtype=np.float32)
        self.stddev_array = np.array(self.stddev, dtype=np.float32)
        self.tensor_scale = input_info.scale
        self.tensor_zero_point = input_info.zero_point
        self.tensor_padding = input_info.padding[1:]  # Skip batch dimension padding

        # Fused normalize+quantize coefficients (like C++):
        #   quantized = round(pixel_rgb_uint8 * mul + add), then clip to [-128, 127]
        inv_scale = 1.0 / self.tensor_scale
        self._quant_mul = ((1.0 / (self.stddev_array * 255.0)) * inv_scale).reshape(1, 1, 3)
        self._quant_add = ((-self.mean_array / self.stddev_array) * inv_scale + self.tensor_zero_point).reshape(1, 1, 3)

        # Preallocated image buffer (reused every frame)
        self._preproc_buf = np.zeros((self.model_height, self.model_width, 3), dtype=np.uint8)

        # Precomputed dequantization parameters per output head
        self._dequant_params = []
        for info in self.output_infos:
            slices = tuple(slice(b, -e if e else None) for b, e in info.padding)
            self._dequant_params.append((slices, info.zero_point, info.scale))

        # --- Optimized ONNX setup ---
        self.onnx_input_metas = self.onnx_session.get_inputs()
        self.onnx_output_metas = self.onnx_session.get_outputs()

        # Precompute input shapes (replace -1 with 1)
        self.onnx_input_shapes = [
            tuple(1 if dim == -1 or dim is None else dim for dim in meta.shape)
            for meta in self.onnx_input_metas
        ]

        # Preallocate CPU input buffers
        self.onnx_input_buffers = [
            np.empty(shape, dtype=np.float32)
            for shape in self.onnx_input_shapes
        ]

        # Preallocate I/O binding
        self.onnx_io_binding = self.onnx_session.io_binding()

        # Bind all inputs (CPU binding; zero-copy)
        for name, arr in zip(self.onnx_input_names, self.onnx_input_buffers):
            self.onnx_io_binding.bind_input(
                name=name,
                device_type='cpu',
                device_id=0,
                element_type=np.float32,
                shape=arr.shape,
                buffer_ptr=arr.__array_interface__['data'][0]
            )

        # Bind all outputs to CPU (ONNX allocates them automatically)
        for output_meta in self.onnx_output_metas:
            self.onnx_io_binding.bind_output(output_meta.name, 'cpu', 0)


        # ROS2 publishers and subscribers
        if self.publish_annotated:
            annotated_topic = input_topic + "_annotated"
            self.image_pub = self.create_publisher(Image, annotated_topic, 10)
        self.subscription = self.create_subscription(Image, input_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(String, output_topic, 10)

        # --- Metrics setup ---
        _W = 100  # rolling window size
        self.frames_received = 0
        self.frames_processed = 0
        self._processed_ts: deque = deque(maxlen=60)   # for FPS
        self._mw_callback_queue = deque(maxlen=_W)     # DDS + scheduling delay (ms)
        self._mw_preprocess    = deque(maxlen=_W)      # resize + norm + quant (ms)
        self._mw_aipu          = deque(maxlen=_W)      # axr_run_model_instance (ms)
        self._mw_dequant       = deque(maxlen=_W)      # int8→float32 + unpad (ms)
        self._mw_nms           = deque(maxlen=_W)      # ONNX NMS postprocess (ms)
        self._mw_e2e           = deque(maxlen=_W)      # full callback→publish (ms)
        self._mw_detections    = deque(maxlen=_W)      # detections per frame
        # All-frame storage (never evicts) — mirrors C++ all_* vectors
        self._all_callback_queue: list[float] = []
        self._all_preprocess:    list[float] = []
        self._all_aipu:          list[float] = []
        self._all_dequant:       list[float] = []
        self._all_nms:           list[float] = []
        self._all_e2e:           list[float] = []
        self._all_detections:    list[float] = []
        # Window snapshots: summary stats captured every _W frames
        self._metric_window = _W
        self._window_snapshots = {
            'callback_queue': [], 'preprocess': [], 'aipu': [],
            'dequant': [], 'nms': [], 'e2e': [], 'detections': []
        }
        if self.compute_metrics:
            os.makedirs(self.save_dir, exist_ok=True)
            self._metrics_print_timer = self.create_timer(5.0, self._log_metrics)

        self.get_logger().info(f"Axelera inference node started with model: {self.model_name}")

    def load_labels(self, labels_path: Path) -> list[str]:
        """Load labels from a JSON file."""
        try:
            with labels_path.open('r') as f:
                data = json.load(f) 
                labels = data.get("labels", [])
                if not labels:
                    self.get_logger().warning(f"No labels found in {labels_path}")               
                return labels
        except FileNotFoundError:
            self.get_logger().error(f"Labels file not found: {labels_path}")
            return []
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Error parsing JSON in {labels_path}: {e}")
            return []
        except Exception as e:
            self.get_logger().error(f"Unexpected error loading labels: {e}")
            return []    

    def _preprocess_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Preprocess: resize with aspect ratio, fused normalize+quantize, pad."""
        h, w = frame.shape[:2]
        scale_factor = min(self.model_width / w, self.model_height / h)
        resized_w = int(w * scale_factor)
        resized_h = int(h * scale_factor)
        x_offset = (self.model_width - resized_w) // 2
        y_offset = (self.model_height - resized_h) // 2

        # Reuse preallocated buffer
        buf = self._preproc_buf
        buf[:] = 0
        resized = cv2.resize(frame, (resized_w, resized_h))
        buf[y_offset:y_offset+resized_h, x_offset:x_offset+resized_w] = resized

        # BGR→RGB view (no copy) + fused normalize+quantize in-place
        rgb = buf[:, :, ::-1]
        tmp = rgb.astype(np.float32)
        tmp *= self._quant_mul
        tmp += self._quant_add
        np.round(tmp, out=tmp)
        np.clip(tmp, -128, 127, out=tmp)
        quantized = tmp.astype(np.int8)

        padded = np.pad(quantized, self.tensor_padding, mode="constant",
                        constant_values=self.tensor_zero_point)

        if self.batch_size > 1:
            padded = np.repeat(padded[np.newaxis, ...], self.batch_size, axis=0)
        else:
            padded = padded[np.newaxis, ...]

        return padded, scale_factor, x_offset, y_offset

    def image_callback(self, msg: Image):
        try:
            t0 = time.perf_counter()
            if self.compute_metrics:
                self.frames_received += 1
                try:
                    msg_stamp_ns = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds
                    now_ns = self.get_clock().now().nanoseconds
                    cq_ms = (now_ns - msg_stamp_ns) / 1e6
                    if cq_ms >= 0.0:
                        self._mw_callback_queue.append(cq_ms)
                        self._all_callback_queue.append(cq_ms)
                except Exception:
                    pass

            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.publish_annotated:
                self.current_frame = frame.copy()

            t1 = time.perf_counter()

            # --- Preprocess ---
            processed_input, scale, x_offset, y_offset = self._preprocess_frame(frame)
            self.inputs[0][:] = processed_input

            t2 = time.perf_counter()

            # --- AIPU inference (synchronous) ---
            self.instance.run(self.inputs, self.outputs)

            t3 = time.perf_counter()

            # --- Dequantize model outputs (in-place ops) ---
            outs = []
            for i, (slices, zp, sc) in enumerate(self._dequant_params):
                sliced = self.outputs[i][slices]
                dequant = sliced.astype(np.float32)
                dequant -= zp
                dequant *= sc
                outs.append(dequant.transpose(0, 3, 1, 2))

            t4 = time.perf_counter()

            # --- ONNX postprocess + NMS ---
            box_type, detections = postprocess_model_output(
                self.onnx_session, self.onnx_input_names,
                self.onnx_io_binding, self.onnx_input_buffers,
                outs,
                confidence_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold
            )

            t5 = time.perf_counter()

            # --- Annotate & publish image (optional) ---
            if self.publish_annotated:
                self.scale = scale
                self.x_offset = x_offset
                self.y_offset = y_offset
                annotated = self.plot_detections(detections, box_type)
                msg_out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                self.image_pub.publish(msg_out)

            # --- Publish detections ---
            if detections:
                det_msg = String()
                det_msg.data = "\n".join(
                    f"Detection: {self.labels[class_id]} ({confidence:.1f}%) at ({x1:.1f}, {y1:.1f}, {w:.1f}, {h:.1f})"
                    for class_id, confidence, (x1, y1, w, h) in detections
                )
                self.publisher.publish(det_msg)

            t6 = time.perf_counter()

            # --- Metrics tracking ---
            if self.compute_metrics:
                d_pre  = (t2 - t1) * 1000
                d_aipu = (t3 - t2) * 1000
                d_deq  = (t4 - t3) * 1000
                d_nms  = (t5 - t4) * 1000
                d_e2e  = (t6 - t0) * 1000
                d_det  = float(len(detections))

                # Rolling window (for live log display)
                self._mw_preprocess.append(d_pre)
                self._mw_aipu.append(d_aipu)
                self._mw_dequant.append(d_deq)
                self._mw_nms.append(d_nms)
                self._mw_e2e.append(d_e2e)
                self._mw_detections.append(d_det)

                # All-frame storage (never evicts)
                self._all_preprocess.append(d_pre)
                self._all_aipu.append(d_aipu)
                self._all_dequant.append(d_deq)
                self._all_nms.append(d_nms)
                self._all_e2e.append(d_e2e)
                self._all_detections.append(d_det)

                self.frames_processed += 1
                self._processed_ts.append(t6)

                # Snapshot window summary every _metric_window frames
                W = self._metric_window
                if self.frames_processed % W == 0:
                    wend = self.frames_processed
                    wstart = wend - W
                    for key, all_list in [
                        ('preprocess', self._all_preprocess),
                        ('aipu',       self._all_aipu),
                        ('dequant',    self._all_dequant),
                        ('nms',        self._all_nms),
                        ('e2e',        self._all_e2e),
                        ('detections', self._all_detections),
                        ('callback_queue', self._all_callback_queue),
                    ]:
                        count = min(len(all_list), W)
                        if count == 0:
                            continue
                        self._window_snapshots[key].append(
                            self._compute_window_stats(wstart, wend, all_list[-count:])
                        )

        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")
    
    def plot_detections(self, detections: list[tuple[int, float, list[float]]], box_type: str) -> np.ndarray:
        """
        Plot detections on the original frame with scaling and padding reversal.
        """
        frame = self.current_frame.copy()

        for detection in detections:
            class_id, confidence, box = detection
            label = self.labels[class_id]

            if len(box) != 4:
                raise ValueError("Bounding box should have 4 values")

            if box_type == 'xywh':
                x, y, w, h = box
                x1 = float(x)
                y1 = float(y)
                x2 = x1 + float(w)
                y2 = y1 + float(h)
            elif box_type == 'xyxy':
                x1, y1, x2, y2 = map(float, box)
            else:
                raise ValueError(f"Unsupported box_format '{box_type}', use 'xyxy' or 'xywh'.")

            # Map from model coordinates (640x640) back to original frame space
            x1 = (x1 - self.x_offset) / self.scale
            y1 = (y1 - self.y_offset) / self.scale
            x2 = (x2 - self.x_offset) / self.scale
            y2 = (y2 - self.y_offset) / self.scale

            # Ensure integers
            x1, y1, x2, y2 = map(lambda v: int(round(v)), [x1, y1, x2, y2])

            # Clamp to frame size to avoid drawing outside image
            h_frame, w_frame = frame.shape[:2]
            x1 = max(0, min(x1, w_frame - 1))
            y1 = max(0, min(y1, h_frame - 1))
            x2 = max(0, min(x2, w_frame - 1))
            y2 = max(0, min(y2, h_frame - 1))

            # Draw center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            color = (0, 255, 0) if confidence >= 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {confidence:.0%}"
            cv2.putText(frame, text, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), radius=3, color=(0, 0, 255), thickness=-1)

        # Change the font color for the model name to black
        cv2.putText(frame, f"Model used: {self.model_name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame
    
    # ------------------------------------------------------------------ #
    #  Metrics helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_window_stats(frame_start: int, frame_end: int,
                              data: list[float]) -> dict:
        """Compute summary stats for a window of data, matching C++ WindowStats."""
        arr = np.asarray(data, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return {'frame_start': frame_start, 'frame_end': frame_end,
                    'mean': 0.0, 'stddev': 0.0, 'p50': 0.0, 'p95': 0.0,
                    'min': 0.0, 'max': 0.0}
        sorted_arr = np.sort(arr)
        return {
            'frame_start': frame_start,
            'frame_end':   frame_end,
            'mean':   float(np.mean(arr)),
            'stddev': float(np.std(arr)),
            'p50':    float(sorted_arr[min(int(np.ceil(0.50 * n)) - 1, n - 1)]),
            'p95':    float(sorted_arr[min(int(np.ceil(0.95 * n)) - 1, n - 1)]),
            'min':    float(sorted_arr[0]),
            'max':    float(sorted_arr[-1]),
        }

    @staticmethod
    def _vector_stats_json(data: list[float]) -> dict | None:
        """Compute overall stats from an all-frame list, matching C++ vector_stats_json."""
        if not data:
            return None
        arr = np.asarray(data, dtype=np.float64)
        n = len(arr)
        sorted_arr = np.sort(arr)
        return {
            'mean':   float(np.mean(arr)),
            'stddev': float(np.std(arr)),
            'p50':    float(sorted_arr[min(int(np.ceil(0.50 * n)) - 1, n - 1)]),
            'p95':    float(sorted_arr[min(int(np.ceil(0.95 * n)) - 1, n - 1)]),
            'min':    float(sorted_arr[0]),
            'max':    float(sorted_arr[-1]),
            'count':  n,
        }

    @staticmethod
    def _stats(d: deque):
        """Return (mean_ms, std_ms, p95_ms) for a latency deque."""
        if not d:
            return float('nan'), float('nan'), float('nan')
        arr = np.asarray(d, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr)), float(np.percentile(arr, 95))

    def _compute_fps(self) -> float:
        ts = self._processed_ts
        if len(ts) < 2:
            return 0.0
        dt = ts[-1] - ts[0]
        return (len(ts) - 1) / dt if dt > 0 else 0.0

    def _format_metrics(self) -> str:
        dropped = self.frames_received - self.frames_processed
        drop_pct = 100.0 * dropped / max(self.frames_received, 1)
        fps = self._compute_fps()
        W = self._mw_e2e.maxlen

        def row(label: str, d: deque) -> str:
            m, s, p95 = self._stats(d)
            return f"  {label:<32} {m:8.2f} ± {s:6.2f}   p95: {p95:8.2f} ms"

        det_mean, det_std, _ = self._stats(self._mw_detections)
        det_line = (f"  {'Detections per frame':<32} {det_mean:8.1f} ± {det_std:6.1f}   "
                    f"(rolling last {W} frames)")
        lines = [
            "=" * 70,
            f"[Python Inference Node]  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Model            : {self.model_name}",
            f"  Frames received  : {self.frames_received}",
            f"  Frames processed : {self.frames_processed}",
            f"  Frames dropped   : {dropped}  ({drop_pct:.1f}%)",
            f"  Processed FPS    : {fps:.2f}",
            det_line,
            f"  Rolling window   : last {W} frames  |  latency = mean ± std",
            "-" * 70,
            row("Callback queue delay (DDS)", self._mw_callback_queue),
            row("Preprocessing (resize+norm+quant)", self._mw_preprocess),
            row("AIPU inference", self._mw_aipu),
            row("Dequantization (int8 → float32)", self._mw_dequant),
            row("ONNX NMS postprocess", self._mw_nms),
            row("End-to-end (callback → publish)", self._mw_e2e),
            "=" * 70,
        ]
        return "\n".join(lines)

    def _log_metrics(self):
        if not self.compute_metrics:
            return
        self.get_logger().info("\n" + self._format_metrics())

    def _save_metrics(self):
        if not self.compute_metrics:
            return
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # --- Human-readable text report ---
        text = self._format_metrics()
        txt_path = os.path.join(self.save_dir, f"python_{self.model_name}_{ts}.txt")
        try:
            with open(txt_path, 'w') as f:
                f.write(text + "\n")
            self.get_logger().info(f"Text metrics saved → {txt_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save text metrics: {e}")

        # --- Comprehensive JSON (mirrors C++ JSON structure) ---
        dropped = max(self.frames_received - self.frames_processed, 0)
        root = {
            'model':            self.model_name,
            'frames_received':  self.frames_received,
            'frames_processed': self.frames_processed,
            'frames_dropped':   dropped,
            'fps':              self._compute_fps(),
            'metric_window':    self._metric_window,
            'per_frame': {
                'callback_queue_ms': self._all_callback_queue,
                'preprocess_ms':     self._all_preprocess,
                'aipu_ms':           self._all_aipu,
                'dequant_ms':        self._all_dequant,
                'nms_ms':            self._all_nms,
                'e2e_ms':            self._all_e2e,
                'detections':        self._all_detections,
            },
            'window_snapshots': self._window_snapshots,
            'overall': {
                'callback_queue_ms': self._vector_stats_json(self._all_callback_queue),
                'preprocess_ms':     self._vector_stats_json(self._all_preprocess),
                'aipu_ms':           self._vector_stats_json(self._all_aipu),
                'dequant_ms':        self._vector_stats_json(self._all_dequant),
                'nms_ms':            self._vector_stats_json(self._all_nms),
                'e2e_ms':            self._vector_stats_json(self._all_e2e),
                'detections':        self._vector_stats_json(self._all_detections),
            },
        }
        json_path = os.path.join(self.save_dir, f"python_{self.model_name}_{ts}.json")
        try:
            with open(json_path, 'w') as f:
                json.dump(root, f, indent=2)
            self.get_logger().info(f"JSON metrics saved → {json_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save JSON metrics: {e}")

    # ------------------------------------------------------------------ #

    def destroy_node(self):
        if self.compute_metrics:
            self._log_metrics()
            self._save_metrics()
        self.ctx.__exit__(None, None, None)
        super().destroy_node()

# Main entry point
def main(args=None):
    # run_prebuild()
    rclpy.init(args=args)
    node = AxeleraYoloInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down inference node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()

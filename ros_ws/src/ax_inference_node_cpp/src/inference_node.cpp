#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <iostream>
#include <typeinfo>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>

#include "axruntime/axruntime.hpp"
#include "opencv2/opencv.hpp"
#include <onnxruntime_cxx_api.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <optional>

#include <tuple>
#include <algorithm>
#include <utility>
#include <cassert>
#include <deque>
#include <numeric>
#include <cmath>
#include <ctime>
#include <iomanip>

#include <nlohmann/json.hpp>


struct Detection {
    int class_id;
    float confidence;
    cv::Rect2f box;
};

// Model parameters
constexpr auto DEFAULT_LABELS = "ax_datasets/labels/coco.names";

using namespace std::string_literals;

// Helper to compute unpadded shape
std::vector<size_t> compute_unpadded_shape(const size_t* dims,
                                           const size_t (*padding)[2],
                                           size_t ndim) {
    std::vector<size_t> unpadded;
    unpadded.reserve(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        size_t pad_left = padding[i][0];
        size_t pad_right = padding[i][1];
        assert(dims[i] >= (pad_left + pad_right));  // safety check
        unpadded.push_back(dims[i] - pad_left - pad_right);
    }

    return unpadded;
}

size_t get_flat_index_NHWC(size_t n, size_t h, size_t w, size_t c,
                           size_t H, size_t W, size_t C) {
    return ((n * H + h) * W + w) * C + c;
}

size_t get_flat_index_NCHW(size_t n, size_t c, size_t h, size_t w,
                           size_t C, size_t H, size_t W) {
    return ((n * C + c) * H + h) * W + w;
}

std::vector<float> transpose_NHWC_to_NCHW(const std::vector<float>& input,
                                          size_t N, size_t H, size_t W, size_t C) {
    assert(input.size() == N * H * W * C && "Input size mismatch");

    std::vector<float> output(N * C * H * W);
    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t c = 0; c < C; ++c) {
                    size_t in_idx = get_flat_index_NHWC(n, h, w, c, H, W, C);
                    size_t out_idx = get_flat_index_NCHW(n, c, h, w, C, H, W);
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
    return output;
}

// Tile size for the cache-blocked NHWC→NCHW transpose below.
// 16×16 float tiles = 1 KB per tile side; two tiles fit comfortably in the
// 32 KB L1D cache of a Cortex-A78AE core (Jetson Orin AGX).
static constexpr size_t kTransposeTile = 16;

// Out-of-place, cache-blocked NHWC→NCHW transpose.
//   src : float[HW][C]  — row C is innermost (NHWC, contiguous along C)
//   dst : float[C][HW]  — row HW is innermost (NCHW, contiguous along HW)
//   HW  : N * H * W  (batch + spatial dims flattened)
//
// Naive NHWC→NCHW stores scatter across dst[c * HW + hw], placing consecutive
// channel writes H*W*4 bytes apart — far beyond a cache line.  Processing in
// kTransposeTile × kTransposeTile blocks keeps the active src and dst sub-
// matrices in L1 throughout the inner loop, eliminating that cache thrashing.



std::vector<Ort::Value> execute_onnx_postprocess(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names,
    const std::vector<std::vector<float>>& inputs_list) {
    std::vector<Ort::Value> input_tensors;
    size_t num_model_inputs = session.GetInputCount();

    if (inputs_list.size() != num_model_inputs)
        throw std::invalid_argument("Number of inputs does not match model input count.");

    for (size_t i = 0; i < num_model_inputs; ++i) {
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();

        size_t actual_size = inputs_list[i].size();
        for (auto& dim : input_shape)
            if (dim == -1) dim = static_cast<int64_t>(actual_size);

        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            allocator.GetInfo(),
            const_cast<float*>(inputs_list[i].data()),
            actual_size,
            input_shape.data(),
            input_shape.size()
        );

        input_tensors.push_back(std::move(tensor));
    }


    try {
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );

        return output_tensors;
    }
    catch (const Ort::Exception& e) {
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;
        throw; // rethrow or handle gracefully
    }

    return input_tensors;
}

std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>>
extract_bounding_boxes(
    const std::vector<std::vector<float>>& predictions,
    bool has_objectness,
    float confidence_threshold) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (const auto& pred : predictions) {
        float conf;
        int cls_id;

        if (has_objectness) {
            float obj_conf = pred[4];  // objectness
            // Find class with highest confidence
            float max_cls_conf = 0.0f;
            int max_cls_id = -1;
            for (int c = 5; c < 85; ++c) {
                if (pred[c] > max_cls_conf) {
                    max_cls_conf = pred[c];
                    max_cls_id = c - 5;
                }
            }
            conf = obj_conf * max_cls_conf;
            cls_id = max_cls_id;
        } else {
            // Find class with highest class score
            float max_cls_conf = 0.0f;
            int max_cls_id = -1;
            for (int c = 4; c < 84; ++c) {
                if (pred[c] > max_cls_conf) {
                    max_cls_conf = pred[c];
                    max_cls_id = c - 4;
                }
            }
            conf = max_cls_conf;
            cls_id = max_cls_id;
        }

        if (conf < confidence_threshold)
            continue;

        // Convert from center+size format to corner coordinates
        float x_center = pred[0];
        float y_center = pred[1];
        float width = pred[2];
        float height = pred[3];
        float x1 = x_center - width / 2.0f;
        float y1 = y_center - height / 2.0f;
        float x2 = x_center + width / 2.0f;
        float y2 = y_center + height / 2.0f;

        // Fix vexing parse by using braces instead of parentheses
        cv::Rect box{cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2))};
        boxes.push_back(box);
        confidences.push_back(conf);
        class_ids.push_back(cls_id);
    }

    return {boxes, confidences, class_ids};
}

std::tuple<std::string, std::vector<Detection>>
postprocess_model_output(
    Ort::Session& onnx_session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names,
    const std::vector<std::vector<float>>& inputs_list,
    float confidence_threshold,
    float nms_threshold,
    rclcpp::Logger logger) {
    // Run ONNX postprocess
    auto onnx_results = execute_onnx_postprocess(
        onnx_session,
        allocator,
        input_names,
        output_names,
        inputs_list
    );

    std::vector<Detection> final_detections;
    std::string box_type;

    for (size_t ri = 0; ri < onnx_results.size(); ++ri) {
        const auto& result = onnx_results[ri];
        auto shape_info = result.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = shape_info.GetShape();
        const float* data = result.GetTensorData<float>();


        int64_t N = 0, stride = 0;
        bool has_objectness = false;

        if (shape.size() == 3 && shape[2] == 85) {
            // YOLOv5
            N = shape[1];
            stride = 85;
            has_objectness = true;
            box_type = "xyxy";
        }
        else if (shape.size() == 3 && shape[1] == 84) {
            // YOLOv8
            N = shape[2];
            stride = 84;
            has_objectness = false;
            box_type = "xyxy";
        } else {
            throw std::runtime_error("Unexpected result shape");
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        boxes.reserve(N);
        confidences.reserve(N);
        class_ids.reserve(N);

        // --- Parallelized parsing ---
        cv::parallel_for_(cv::Range(0, static_cast<int>(N)), [&](const cv::Range& range) {
            std::vector<cv::Rect> local_boxes;
            std::vector<float> local_confs;
            std::vector<int> local_classes;

            local_boxes.reserve(range.size());
            local_confs.reserve(range.size());
            local_classes.reserve(range.size());

            for (int i = range.start; i < range.end; ++i) {
                float x, y, w, h;
                const float* class_scores;

                if (has_objectness) {
                    // YOLOv5: [1, N, 85] format
                    const float* p = data + i * stride;
                    x = p[0];
                    y = p[1];
                    w = p[2];
                    h = p[3];
                    float obj = p[4];
                    class_scores = p + 5;

                    int best_class = 0;
                    float best_score = 0.0f;

                    for (int c = 0; c < 80; ++c) {  // 80 classes for COCO
                        float conf = obj * class_scores[c];
                        if (conf > best_score) {
                            best_score = conf;
                            best_class = c;
                        }
                    }

                    if (best_score >= confidence_threshold) {
                        int left = static_cast<int>(x - w / 2);
                        int top = static_cast<int>(y - h / 2);
                        int right = static_cast<int>(x + w / 2);
                        int bottom = static_cast<int>(y + h / 2);

                        local_boxes.emplace_back(left, top, right - left, bottom - top);
                        local_confs.push_back(best_score);
                        local_classes.push_back(best_class);
                    }
                } else {
                    // YOLOv8: [1, 84, N] format - channels are transposed
                    x = data[0 * N + i];  // Channel 0 for x coordinates
                    y = data[1 * N + i];  // Channel 1 for y coordinates
                    w = data[2 * N + i];  // Channel 2 for width
                    h = data[3 * N + i];  // Channel 3 for height

                    int best_class = 0;
                    float best_score = 0.0f;

                    // Class scores start from channel 4
                    for (int c = 0; c < 80; ++c) {  // 80 classes for COCO
                        float class_conf = data[(4 + c) * N + i];
                        if (class_conf > best_score) {
                            best_score = class_conf;
                            best_class = c;
                        }
                    }

                    if (best_score >= confidence_threshold) {
                        int left = static_cast<int>(x - w / 2);
                        int top = static_cast<int>(y - h / 2);
                        int right = static_cast<int>(x + w / 2);
                        int bottom = static_cast<int>(y + h / 2);

                        local_boxes.emplace_back(left, top, right - left, bottom - top);
                        local_confs.push_back(best_score);
                        local_classes.push_back(best_class);
                    }
                }
            }

            // Append local results (thread-safe with mutex or per-thread collection)
            static std::mutex mtx;
            std::lock_guard<std::mutex> lock(mtx);
            boxes.insert(boxes.end(), local_boxes.begin(), local_boxes.end());
            confidences.insert(confidences.end(), local_confs.begin(), local_confs.end());
            class_ids.insert(class_ids.end(), local_classes.begin(), local_classes.end());
        });

        // --- Apply Non-Maximum Suppression ---
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices);

        final_detections.reserve(final_detections.size() + indices.size());
        for (int idx : indices) {
            Detection det;
            det.class_id = class_ids[idx];
            det.confidence = confidences[idx];
            det.box = boxes[idx];
            final_detections.push_back(det);
        }
    }

    return {box_type, final_detections};
}

std::tuple<float, int, int> preprocess_frame(
    const cv::Mat& frame,
    const axrTensorInfo& info,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& stddev,
    cv::Mat& padded_buffer,
    int8_t* quantized_ptr)
{
    const int height = info.dims[1];
    const int width = info.dims[2];
    const int channels = info.dims[3];

    const auto [y_pad_left, y_pad_right] = info.padding[1];
    const auto [x_pad_left, x_pad_right] = info.padding[2];

    const int unpadded_height = height - y_pad_left - y_pad_right;
    const int unpadded_width  = width  - x_pad_left - x_pad_right;

    // Resize scale
    float scale = std::min(
        static_cast<float>(unpadded_width) / frame.cols,
        static_cast<float>(unpadded_height) / frame.rows);

    int resized_width  = static_cast<int>(frame.cols * scale);
    int resized_height = static_cast<int>(frame.rows * scale);

    int x_offset = (unpadded_width  - resized_width)  / 2;
    int y_offset = (unpadded_height - resized_height) / 2;

    // Resize directly into padded buffer ROI
    cv::Mat roi = padded_buffer(cv::Rect(x_offset, y_offset, resized_width, resized_height));
    cv::resize(frame, roi, roi.size(), 0, 0, cv::INTER_LINEAR);

    // Clear padding
    if (y_offset > 0) padded_buffer.rowRange(0, y_offset).setTo(cv::Scalar(0,0,0));
    if (y_offset + resized_height < height) padded_buffer.rowRange(y_offset + resized_height, height).setTo(cv::Scalar(0,0,0));
    if (x_offset > 0) padded_buffer.colRange(0, x_offset).setTo(cv::Scalar(0,0,0));
    if (x_offset + resized_width < width) padded_buffer.colRange(x_offset + resized_width, width).setTo(cv::Scalar(0,0,0));

    // --- Quantize directly into raw pointer ---
    const float inv255 = 1.0f / 255.0f;
    const float inv_scale = 1.0f / info.scale;
    float mul[3], add[3];
    for (int c = 0; c < 3; ++c) {
        float norm_mul = (1.0f / stddev[c]);
        float norm_add = (-mean[c] / stddev[c]);
        mul[c] = norm_mul * inv_scale * inv255;
        add[c] = norm_add * inv_scale + info.zero_point;
    }

    const int HW = height * width;
    const uint8_t* src = padded_buffer.ptr<uint8_t>();
    for (int i = 0; i < HW; ++i) {
        const int src_idx = i * 3;
        const int dst_idx = i * channels;

        float r = src[src_idx + 2];
        float g = src[src_idx + 1];
        float b = src[src_idx + 0];

        float q0 = std::clamp(r * mul[0] + add[0], -128.f, 127.f);
        float q1 = std::clamp(g * mul[1] + add[1], -128.f, 127.f);
        float q2 = std::clamp(b * mul[2] + add[2], -128.f, 127.f);

        quantized_ptr[dst_idx + 0] = static_cast<int8_t>(std::lrintf(q0));
        quantized_ptr[dst_idx + 1] = static_cast<int8_t>(std::lrintf(q1));
        quantized_ptr[dst_idx + 2] = static_cast<int8_t>(std::lrintf(q2));
    }

    return {scale, x_offset, y_offset};
}

cv::Mat plot_detections(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::string& box_type,
    const std::vector<std::string>& labels,
    const std::string& model_name,
    float scale,
    int x_offset,
    int y_offset) {
    cv::Mat annotated_frame = frame.clone();

    for (const auto& detection : detections) {
        int class_id = detection.class_id;
        float confidence = detection.confidence;
        cv::Rect2f box = detection.box;

        std::string label = (class_id >= 0 && class_id < labels.size()) ? labels[class_id] : "Unknown";

        // Reverse scaling and padding transformations
        float x1 = (box.x - x_offset) / scale;
        float y1 = (box.y - y_offset) / scale;
        float x2 = ((box.x + box.width) - x_offset) / scale;
        float y2 = ((box.y + box.height) - y_offset) / scale;

        // Ensure integers and clamp to frame size
        int x1_int = std::max(0, std::min(static_cast<int>(std::round(x1)), frame.cols - 1));
        int y1_int = std::max(0, std::min(static_cast<int>(std::round(y1)), frame.rows - 1));
        int x2_int = std::max(0, std::min(static_cast<int>(std::round(x2)), frame.cols - 1));
        int y2_int = std::max(0, std::min(static_cast<int>(std::round(y2)), frame.rows - 1));

        // Draw bounding box
        cv::Scalar color = (confidence >= 0.5) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(annotated_frame, cv::Point(x1_int, y1_int), cv::Point(x2_int, y2_int), color, 2);

        // Add label and confidence
        std::ostringstream text;
        text << label << " " << static_cast<int>(confidence * 100) << "%";
        cv::putText(annotated_frame, text.str(), cv::Point(x1_int, std::max(0, y1_int - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

        // Draw center point
        int cx = (x1_int + x2_int) / 2;
        int cy = (y1_int + y2_int) / 2;
        cv::circle(annotated_frame, cv::Point(cx, cy), 3, cv::Scalar(0, 0, 255), -1);
    }

    // Add model name
    cv::putText(annotated_frame, "Model used: " + model_name, cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    return annotated_frame;
}

std_msgs::msg::String create_detection_message(
    const std::vector<Detection>& detections,
    const std::vector<std::string>& labels) {
    std_msgs::msg::String msg;
    std::ostringstream ss;

    for (const auto& detection : detections) {
        int class_id = detection.class_id;
        float confidence = detection.confidence;
        cv::Rect2f box = detection.box;

        std::string label = (class_id >= 0 && class_id < labels.size()) ? labels[class_id] : "Unknown";

        ss << "Detection: " << label << " (" << confidence * 100 << "%) at ("
           << box.x << ", " << box.y << ", " << box.width << ", " << box.height << ")\n";
    }

    msg.data = ss.str();
    return msg;
}

auto read_labels(const std::string& path) -> std::vector<std::string> {
    std::vector<std::string> labels;

    // Open the JSON file
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + path);
    }

    // Parse JSON
    nlohmann::json j;
    file >> j;

    // Extract "labels" array
    if (j.contains("labels") && j["labels"].is_array()) {
        for (const auto& label : j["labels"]) {
            if (label.is_string()) {
                labels.push_back(label.get<std::string>());
            }
        }
    } else {
        throw std::runtime_error("JSON file does not contain 'labels' array: " + path);
    }
      return labels;
}
void logger(void* arg, axrLogLevel level, const char* msg) {
    (void)arg;
    (void)level;
}

// Define a ROS 2 Node class
class AxeleraYoloInference : public rclcpp::Node {
public:
    AxeleraYoloInference() : Node("axelera_yolo_inference") {
        RCLCPP_INFO(this->get_logger(), "Starting Axelera YOLO Inference Node...");

        // Declare parameters with default values
        this->declare_parameter("model_name", "");
        this->declare_parameter("aipu_cores", 4);
        this->declare_parameter("input_topic", "/camera_frame");
        this->declare_parameter("output_topic", "/detections_topic");
        this->declare_parameter("confidence_threshold", 0.25);
        this->declare_parameter("nms_threshold", 0.45);
        this->declare_parameter("mean", std::vector<double>{0.485, 0.456, 0.406});
        this->declare_parameter("stddev", std::vector<double>{0.229, 0.224, 0.225});
        this->declare_parameter("publish_annotated", false);
        this->declare_parameter("compute_metrics", true);
        this->declare_parameter("save_dir", std::string("~/ros2_metrics"));

        // Get parameter values
        model_name_ = this->get_parameter("model_name").as_string();
        aipu_cores_ = this->get_parameter("aipu_cores").as_int();
        input_topic_ = this->get_parameter("input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
        nms_threshold_ = this->get_parameter("nms_threshold").as_double();

        // Precompute mean and stddev for optimized preprocessing
        auto mean_param = this->get_parameter("mean").as_double_array();
        auto stddev_param = this->get_parameter("stddev").as_double_array();
        publish_annotated_ = this->get_parameter("publish_annotated").as_bool();
        compute_metrics_ = this->get_parameter("compute_metrics").as_bool();
        {
            std::string raw = this->get_parameter("save_dir").as_string();
            if (!raw.empty() && raw[0] == '~') {
                const char* home = std::getenv("HOME");
                save_dir_ = home ? std::string(home) + raw.substr(1) : raw;
            } else {
                save_dir_ = raw;
            }
        }

        // Convert to arrays for faster access
        for (size_t i = 0; i < 3 && i < mean_param.size(); ++i) {
            mean_[i] = static_cast<float>(mean_param[i]);
        }
        for (size_t i = 0; i < 3 && i < stddev_param.size(); ++i) {
            stddev_[i] = static_cast<float>(stddev_param[i]);
        }

        // Load labels using default path
        const auto root = std::getenv("AXELERA_FRAMEWORK");
        std::string model_info_path = "../build/" + model_name_ + "/" + model_name_ + "/model_info.json";
        this->labels_ = read_labels(model_info_path);
        onnx_model_path_ = "../build/" + model_name_ + "/" + model_name_ + "/1/postprocess_graph.onnx";

        // Initialize publishers and subscribers with parametric topic names
        if (publish_annotated_) {
            std::string annotated_topic = input_topic_ + "_annotated";
            image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(annotated_topic, 10);
        }
        detection_pub_ = this->create_publisher<std_msgs::msg::String>(output_topic_, 10);
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            input_topic_, 10, std::bind(&AxeleraYoloInference::image_callback, this, std::placeholders::_1));

        // Initialize runtime context
        ctx_ = axr::to_ptr(axr_create_context());
        if (!ctx_) {
            throw std::runtime_error("Failed to create runtime context");
        }
        axr_set_logger(ctx_.get(), AXR_LOG_WARNING, logger, nullptr);

        // Load model using parametric model name
        std::string model_json_path = "../build/" + model_name_ + "/" + model_name_ + "/1/model.json";
        model_ = axr_load_model(ctx_.get(), model_json_path.c_str());
        if (!model_) {
            throw std::runtime_error("Failed to load model from path: " + model_json_path);
        }

        // Get model information
        auto inputs = axr_num_model_inputs(model_);
        for (size_t n = 0; n != inputs; ++n) {
            input_infos_.push_back(axr_get_model_input(model_, n));
        }
        auto outputs = axr_num_model_outputs(model_);
        for (size_t n = 0; n != outputs; ++n) {
            output_infos_.push_back(axr_get_model_output(model_, n));
        }
        const auto batch_size = input_infos_[0].dims[0];

        // Connect to device
        connection_ = axr_device_connect(ctx_.get(), nullptr, batch_size, nullptr);
        if (!connection_) {
            throw std::runtime_error("Failed to connect to device");
        }

        // Create model instance with parametric aipu_cores
        const auto props = "input_dmabuf=0;num_sub_devices=" + std::to_string(batch_size)
                           + ";aipu_cores=" + std::to_string(aipu_cores_);
        auto properties = axr_create_properties(ctx_.get(), props.c_str());
        instance_ = axr_load_model_instance(connection_, model_, properties);
        if (!instance_) {
            throw std::runtime_error("Failed to create model instance");
        }

        // Prepare buffers
        input_args_.resize(inputs);
        output_args_.resize(outputs);
        input_data_.resize(inputs);
        output_data_.resize(outputs);

        for (int n = 0; n != inputs; ++n) {
            input_data_[n] = std::make_unique<std::int8_t[]>(axr_tensor_size(&input_infos_[n]));
            input_args_[n].ptr = input_data_[n].get();
            input_args_[n].fd = 0;
            input_args_[n].offset = 0;
        }
        for (int n = 0; n != outputs; ++n) {
            output_data_[n] = std::make_unique<std::int8_t[]>(axr_tensor_size(&output_infos_[n]));
            output_args_[n].ptr = output_data_[n].get();
            output_args_[n].fd = 0;
            output_args_[n].offset = 0;
        }

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        onnx_session_ = std::make_unique<Ort::Session>(env_, onnx_model_path_.c_str(), session_options);
        allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();

        // Query input/output count from the ONNX model
        size_t onnx_input_count = onnx_session_->GetInputCount();
        size_t onnx_output_count = onnx_session_->GetOutputCount();

        for (size_t i = 0; i < onnx_input_count; ++i) {
            Ort::AllocatedStringPtr name = onnx_session_->GetInputNameAllocated(i, *allocator_);
            input_names_.push_back(name.get());
            owned_input_names_.push_back(std::move(name));
        }

        for (size_t i = 0; i < onnx_output_count; ++i) {
            Ort::AllocatedStringPtr name = onnx_session_->GetOutputNameAllocated(i, *allocator_);
            output_names_.push_back(name.get());
            owned_output_names_.push_back(std::move(name));
        }

        // Precompute preprocessing parameters from input tensor info
        const auto& input_info = input_infos_[0];
        input_height_ = input_info.dims[1];
        input_width_ = input_info.dims[2];
        input_channels_ = input_info.dims[3];
        y_pad_left_ = input_info.padding[1][0];
        y_pad_right_ = input_info.padding[1][1];
        x_pad_left_ = input_info.padding[2][0];
        x_pad_right_ = input_info.padding[2][1];
        unpadded_height_ = input_height_ - y_pad_left_ - y_pad_right_;
        unpadded_width_ = input_width_ - x_pad_left_ - x_pad_right_;

        // Precompute normalization arrays
        normalization_mul_[0] = 1.0f / stddev_[0];
        normalization_mul_[1] = 1.0f / stddev_[1];
        normalization_mul_[2] = 1.0f / stddev_[2];
        normalization_add_[0] = -mean_[0] / stddev_[0];
        normalization_add_[1] = -mean_[1] / stddev_[1];
        normalization_add_[2] = -mean_[2] / stddev_[2];

        // Store quantization parameters
        input_scale_ = input_info.scale;
        input_zero_point_ = input_info.zero_point;
        quantized_buffer_size_ = input_height_ * input_width_ * input_channels_;

        // Initialize preallocated padded image buffer
        padded_buffer_ = cv::Mat(input_height_, input_width_, CV_8UC3);

        // Preallocated buffer for quantized data to avoid repeated allocations
        quantized_buffer_.resize(quantized_buffer_size_);

        // Precompute output processing parameters
        const auto& output_info = output_infos_[0];
        output_unpadded_shape_ = compute_unpadded_shape(output_info.dims, output_info.padding, output_info.ndims);
        output_N_ = output_unpadded_shape_[0];
        output_H_ = output_unpadded_shape_[1];
        output_W_ = output_unpadded_shape_[2];
        output_C_ = output_unpadded_shape_[3];
        output_padded_H_ = output_info.dims[1];
        output_padded_W_ = output_info.dims[2];
        output_padded_C_ = output_info.dims[3];
        output_input_stride_n_ = output_padded_H_ * output_padded_W_ * output_padded_C_;
        output_input_stride_h_ = output_padded_W_ * output_padded_C_;
        output_input_stride_w_ = output_padded_C_;
        output_output_stride_n_ = output_C_ * output_H_ * output_W_;
        output_output_stride_c_ = output_H_ * output_W_;
        output_output_stride_h_ = output_W_;
        output_pad_n_ = output_info.padding[0][0];
        output_pad_h_ = output_info.padding[1][0];
        output_pad_w_ = output_info.padding[2][0];
        output_pad_c_ = output_info.padding[3][0];
        output_scale_ = output_info.scale;
        output_zero_point_f_ = static_cast<float>(output_info.zero_point);
        output_result_size_ = output_N_ * output_C_ * output_H_ * output_W_;

        // Precomputed parameters for all outputs
        for (const auto& output_info : output_infos_) {
            OutputParams params;
            params.unpadded_shape = compute_unpadded_shape(output_info.dims, output_info.padding, output_info.ndims);
            params.N = params.unpadded_shape[0];
            params.H = params.unpadded_shape[1];
            params.W = params.unpadded_shape[2];
            params.C = params.unpadded_shape[3];
            params.padded_H = output_info.dims[1];
            params.padded_W = output_info.dims[2];
            params.padded_C = output_info.dims[3];
            params.input_stride_n = params.padded_H * params.padded_W * params.padded_C;
            params.input_stride_h = params.padded_W * params.padded_C;
            params.input_stride_w = params.padded_C;
            params.output_stride_n = params.C * params.H * params.W;
            params.output_stride_c = params.H * params.W;
            params.output_stride_h = params.W;
            params.pad_n = output_info.padding[0][0];
            params.pad_h = output_info.padding[1][0];
            params.pad_w = output_info.padding[2][0];
            params.pad_c = output_info.padding[3][0];
            params.scale = output_info.scale;
            params.zero_point_f = static_cast<float>(output_info.zero_point);
            params.result_size = params.N * params.C * params.H * params.W;
            all_output_params_.push_back(params);
        }

        // Pre-allocate dequantized output buffers (avoids per-frame alloc + zero-init)
        dequant_buffers_.resize(all_output_params_.size());
        for (size_t i = 0; i < all_output_params_.size(); ++i) {
            dequant_buffers_[i].resize(all_output_params_[i].result_size);
        }

        // --- Metrics ---
        if (compute_metrics_) {
            std::filesystem::create_directories(save_dir_);
            metrics_timer_ = this->create_wall_timer(
                std::chrono::seconds(5),
                [this]() { log_metrics(); });
        }

        RCLCPP_INFO(this->get_logger(), "Axelera YOLO Inference Node initialized successfully.");
    }

    ~AxeleraYoloInference() {
        if (compute_metrics_) {
            log_metrics();
            save_metrics();
        }
        instance_ = nullptr;
        connection_ = nullptr;
        model_ = nullptr;
        ctx_ = nullptr;
    }

private:
    std::vector<axrTensorInfo> input_infos_;
    std::vector<axrTensorInfo> output_infos_;
    std::vector<axrArgument> input_args_;
    std::vector<axrArgument> output_args_;
    std::vector<std::unique_ptr<std::int8_t[]>> input_data_;
    std::vector<std::unique_ptr<std::int8_t[]>> output_data_;
    std::shared_ptr<axrContext> ctx_;
    axrModel* model_ = nullptr;
    axrConnection* connection_ = nullptr;
    axrModelInstance* instance_ = nullptr;

    std::vector<std::string> labels_;  // Class variable for labels
    std::string onnx_model_path_;

    std::unique_ptr<Ort::Session> onnx_session_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
    Ort::Env env_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<Ort::AllocatedStringPtr> owned_input_names_;
    std::vector<Ort::AllocatedStringPtr> owned_output_names_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr detection_pub_;

    // Parametric member variables
    std::string model_name_;
    int aipu_cores_;
    std::string input_topic_;
    std::string output_topic_;
    float confidence_threshold_;
    float nms_threshold_;
    bool publish_annotated_;
    std::array<float, 3> mean_;
    std::array<float, 3> stddev_;

    // Precomputed preprocessing parameters (moved from preprocess_frame)
    size_t input_height_;
    size_t input_width_;
    size_t input_channels_;
    size_t y_pad_left_, y_pad_right_;
    size_t x_pad_left_, x_pad_right_;
    size_t unpadded_height_, unpadded_width_;
    std::array<float, 3> normalization_mul_;
    std::array<float, 3> normalization_add_;
    float input_scale_;
    float input_zero_point_;
    size_t quantized_buffer_size_;

    // Preallocated buffer for padded image
    cv::Mat padded_buffer_;

    // Preallocated buffer for quantized data to avoid repeated allocations
    std::vector<std::int8_t> quantized_buffer_;

    // Precomputed output processing parameters as simple variables (assuming single output for simplicity)
    std::vector<size_t> output_unpadded_shape_;
    size_t output_N_, output_H_, output_W_, output_C_;
    size_t output_padded_H_, output_padded_W_, output_padded_C_;
    size_t output_input_stride_n_, output_input_stride_h_, output_input_stride_w_;
    size_t output_output_stride_n_, output_output_stride_c_, output_output_stride_h_;
    size_t output_pad_n_, output_pad_h_, output_pad_w_, output_pad_c_;
    float output_scale_, output_zero_point_f_;
    size_t output_result_size_;

    // Precomputed parameters for all outputs
    struct OutputParams {
        std::vector<size_t> unpadded_shape;
        size_t N, H, W, C;
        size_t padded_H, padded_W, padded_C;
        size_t input_stride_n, input_stride_h, input_stride_w;
        size_t output_stride_n, output_stride_c, output_stride_h;
        size_t pad_n, pad_h, pad_w, pad_c;
        float scale, zero_point_f;
        size_t result_size;
    };
    std::vector<OutputParams> all_output_params_;

    // Pre-allocated dequantized output buffers (one per head, reused every frame)
    std::vector<std::vector<float>> dequant_buffers_;

    // --- Metrics ---
    bool compute_metrics_{false};
    std::string save_dir_;
    uint64_t frames_received_{0};
    uint64_t frames_processed_{0};
    static constexpr size_t kMetricWindow = 100;
    static constexpr size_t kFpsWindow    = 60;
    std::deque<double> mw_callback_queue_; // DDS + scheduling delay (ms)
    std::deque<double> mw_preprocess_;     // resize + norm + quant (ms)
    std::deque<double> mw_aipu_;           // axr_run_model_instance (ms)
    std::deque<double> mw_dequant_;        // int8 → float32 (ms)
    std::deque<double> mw_nms_;            // ONNX NMS postprocess (ms)
    std::deque<double> mw_e2e_;            // full callback → publish (ms)
    std::deque<double> mw_detections_;     // detections per frame
    std::deque<std::chrono::steady_clock::time_point> processed_ts_;
    rclcpp::TimerBase::SharedPtr metrics_timer_;

    // All-frame storage (every frame, no eviction)
    std::vector<double> all_callback_queue_;
    std::vector<double> all_preprocess_;
    std::vector<double> all_aipu_;
    std::vector<double> all_dequant_;
    std::vector<double> all_nms_;
    std::vector<double> all_e2e_;
    std::vector<double> all_detections_;

    // Window snapshots: summary stats captured every kMetricWindow frames
    struct WindowStats {
        uint64_t frame_start;  // first frame index in this window
        uint64_t frame_end;    // last frame index (exclusive)
        double mean, stddev, p50, p95, min_val, max_val;
    };
    struct AllWindowSnapshots {
        std::vector<WindowStats> callback_queue;
        std::vector<WindowStats> preprocess;
        std::vector<WindowStats> aipu;
        std::vector<WindowStats> dequant;
        std::vector<WindowStats> nms;
        std::vector<WindowStats> e2e;
        std::vector<WindowStats> detections;
    } window_snapshots_;

    // ---------------------------------------------------------------- //
    //  Metrics helpers                                                   //
    // ---------------------------------------------------------------- //

    template <typename D>
    static double ms_dur(D d) {
        return std::chrono::duration<double, std::milli>(d).count();
    }

    static void push_metric(std::deque<double>& d, double val) {
        if (d.size() >= kMetricWindow) d.pop_front();
        d.push_back(val);
    }

    // Compute summary stats from a range of doubles
    template <typename Iter>
    static WindowStats compute_window_stats(uint64_t frame_start, uint64_t frame_end,
                                            Iter begin, Iter end) {
        WindowStats ws{};
        ws.frame_start = frame_start;
        ws.frame_end   = frame_end;
        size_t n = static_cast<size_t>(std::distance(begin, end));
        if (n == 0) return ws;

        double sum = std::accumulate(begin, end, 0.0);
        ws.mean = sum / static_cast<double>(n);
        double sq = 0.0;
        for (auto it = begin; it != end; ++it) {
            double d = *it - ws.mean;
            sq += d * d;
        }
        ws.stddev = std::sqrt(sq / static_cast<double>(n));

        std::vector<double> sorted(begin, end);
        std::sort(sorted.begin(), sorted.end());
        ws.min_val = sorted.front();
        ws.max_val = sorted.back();
        ws.p50 = sorted[std::min(static_cast<size_t>(std::ceil(0.50 * n)) - 1, n - 1)];
        ws.p95 = sorted[std::min(static_cast<size_t>(std::ceil(0.95 * n)) - 1, n - 1)];
        return ws;
    }

    // Convert a WindowStats to a JSON object
    static nlohmann::json stats_to_json(const WindowStats& ws) {
        return {
            {"frame_start", ws.frame_start}, {"frame_end", ws.frame_end},
            {"mean", ws.mean}, {"stddev", ws.stddev},
            {"p50", ws.p50}, {"p95", ws.p95},
            {"min", ws.min_val}, {"max", ws.max_val}
        };
    }

    // Compute overall stats from an all-frame vector
    static nlohmann::json vector_stats_json(const std::vector<double>& v) {
        if (v.empty()) return nullptr;
        auto ws = compute_window_stats(0, v.size(), v.begin(), v.end());
        auto j = stats_to_json(ws);
        j.erase("frame_start");
        j.erase("frame_end");
        j["count"] = v.size();
        return j;
    }

    static std::string format_row(const std::string& label,
                                   const std::deque<double>& d) {
        std::ostringstream os;
        os << "  " << std::left << std::setw(34) << label;
        if (d.empty()) {
            os << "  N/A";
        } else {
            double sum = std::accumulate(d.begin(), d.end(), 0.0);
            double mean = sum / static_cast<double>(d.size());
            double sq = 0.0;
            for (double v : d) sq += (v - mean) * (v - mean);
            double stddev = std::sqrt(sq / static_cast<double>(d.size()));

            std::vector<double> sorted(d.begin(), d.end());
            std::sort(sorted.begin(), sorted.end());
            size_t idx = std::min(
                static_cast<size_t>(std::ceil(0.95 * static_cast<double>(sorted.size()))) - 1,
                sorted.size() - 1);
            double p95 = sorted[idx];

            os << std::fixed << std::setprecision(2)
               << std::right << std::setw(8) << mean
               << " \u00b1 " << std::setw(6) << stddev
               << "   p95: " << std::setw(8) << p95 << " ms";
        }
        return os.str();
    }

    double compute_fps() const {
        if (processed_ts_.size() < 2) return 0.0;
        double dt = std::chrono::duration<double>(
            processed_ts_.back() - processed_ts_.front()).count();
        return dt > 0.0 ? static_cast<double>(processed_ts_.size() - 1) / dt : 0.0;
    }

    std::string format_metrics() const {
        uint64_t dropped = frames_received_ > frames_processed_
                           ? frames_received_ - frames_processed_ : 0;
        double drop_pct = frames_received_ > 0
                          ? 100.0 * dropped / frames_received_ : 0.0;

        std::time_t t = std::time(nullptr);
        std::tm tm_buf{};
        localtime_r(&t, &tm_buf);
        char time_str[32];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);

        // Detections per frame summary
        std::string det_line;
        {
            double det_mean = 0.0, det_std = 0.0;
            if (!mw_detections_.empty()) {
                for (double v : mw_detections_) det_mean += v;
                det_mean /= static_cast<double>(mw_detections_.size());
                for (double v : mw_detections_) det_std += (v - det_mean) * (v - det_mean);
                det_std = std::sqrt(det_std / static_cast<double>(mw_detections_.size()));
            }
            std::ostringstream ds;
            ds << "  " << std::left << std::setw(34) << "Detections per frame"
               << std::fixed << std::setprecision(1)
               << std::right << std::setw(8) << det_mean
               << " \u00b1 " << std::setw(6) << det_std
               << "   (rolling last " << kMetricWindow << " frames)";
            det_line = ds.str();
        }

        std::ostringstream oss;
        oss << std::string(70, '=') << "\n"
            << "[C++ Inference Node]  " << time_str << "\n"
            << "  Model            : " << model_name_ << "\n"
            << "  Frames received  : " << frames_received_ << "\n"
            << "  Frames processed : " << frames_processed_ << "\n"
            << "  Frames dropped   : " << dropped
            << "  (" << std::fixed << std::setprecision(1) << drop_pct << "%)\n"
            << "  Processed FPS    : " << std::fixed << std::setprecision(2)
            << compute_fps() << "\n"
            << det_line << "\n"
            << "  Rolling window   : last " << kMetricWindow
            << " frames  |  latency = mean \u00b1 std\n"
            << std::string(70, '-') << "\n"
            << format_row("Callback queue delay (DDS)",       mw_callback_queue_) << "\n"
            << format_row("Preprocessing (resize+norm+quant)", mw_preprocess_)    << "\n"
            << format_row("AIPU inference",                    mw_aipu_)          << "\n"
            << format_row("Dequantization (int8 \u2192 float32)",  mw_dequant_)     << "\n"
            << format_row("ONNX NMS postprocess",              mw_nms_)           << "\n"
            << format_row("End-to-end (callback \u2192 publish)",  mw_e2e_)         << "\n"
            << std::string(70, '=') << "\n";
        return oss.str();
    }

    void log_metrics() {
        RCLCPP_INFO(this->get_logger(), "\n%s", format_metrics().c_str());
    }

    void save_metrics() {
        std::time_t t = std::time(nullptr);
        std::tm tm_buf{};
        localtime_r(&t, &tm_buf);
        char ts_str[32];
        std::strftime(ts_str, sizeof(ts_str), "%Y%m%d_%H%M%S", &tm_buf);

        // --- Still save human-readable text report ---
        std::string txt_path = save_dir_ + "/cpp_" + model_name_ + "_" + ts_str + ".txt";
        {
            std::ofstream f(txt_path);
            if (f) {
                f << format_metrics();
                RCLCPP_INFO(this->get_logger(), "Text metrics saved -> %s", txt_path.c_str());
            }
        }

        // --- Save comprehensive JSON ---
        std::string json_path = save_dir_ + "/cpp_" + model_name_ + "_" + ts_str + ".json";

        uint64_t dropped = frames_received_ > frames_processed_
                           ? frames_received_ - frames_processed_ : 0;

        // Helper lambda: convert a vector of WindowStats to a JSON array
        auto snapshots_json = [](const std::vector<WindowStats>& snaps) {
            nlohmann::json arr = nlohmann::json::array();
            for (const auto& ws : snaps) arr.push_back(stats_to_json(ws));
            return arr;
        };

        nlohmann::json root;
        root["model"]            = model_name_;
        root["frames_received"]  = frames_received_;
        root["frames_processed"] = frames_processed_;
        root["frames_dropped"]   = dropped;
        root["fps"]              = compute_fps();
        root["metric_window"]    = kMetricWindow;

        // Per-frame raw data
        root["per_frame"]["callback_queue_ms"] = all_callback_queue_;
        root["per_frame"]["preprocess_ms"]     = all_preprocess_;
        root["per_frame"]["aipu_ms"]           = all_aipu_;
        root["per_frame"]["dequant_ms"]        = all_dequant_;
        root["per_frame"]["nms_ms"]            = all_nms_;
        root["per_frame"]["e2e_ms"]            = all_e2e_;
        root["per_frame"]["detections"]        = all_detections_;

        // Window snapshots (every kMetricWindow frames)
        root["window_snapshots"]["callback_queue_ms"] = snapshots_json(window_snapshots_.callback_queue);
        root["window_snapshots"]["preprocess_ms"]     = snapshots_json(window_snapshots_.preprocess);
        root["window_snapshots"]["aipu_ms"]           = snapshots_json(window_snapshots_.aipu);
        root["window_snapshots"]["dequant_ms"]        = snapshots_json(window_snapshots_.dequant);
        root["window_snapshots"]["nms_ms"]            = snapshots_json(window_snapshots_.nms);
        root["window_snapshots"]["e2e_ms"]            = snapshots_json(window_snapshots_.e2e);
        root["window_snapshots"]["detections"]        = snapshots_json(window_snapshots_.detections);

        // Overall (all-frame) aggregate stats
        root["overall"]["callback_queue_ms"] = vector_stats_json(all_callback_queue_);
        root["overall"]["preprocess_ms"]     = vector_stats_json(all_preprocess_);
        root["overall"]["aipu_ms"]           = vector_stats_json(all_aipu_);
        root["overall"]["dequant_ms"]        = vector_stats_json(all_dequant_);
        root["overall"]["nms_ms"]            = vector_stats_json(all_nms_);
        root["overall"]["e2e_ms"]            = vector_stats_json(all_e2e_);
        root["overall"]["detections"]        = vector_stats_json(all_detections_);

        std::ofstream jf(json_path);
        if (jf) {
            jf << root.dump(2);
            RCLCPP_INFO(this->get_logger(), "JSON metrics saved -> %s", json_path.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to save JSON metrics: %s", json_path.c_str());
        }
    }

    std::vector<std::vector<float>>& process_outputs(
    const std::vector<std::unique_ptr<std::int8_t[]>>& outputs) {

    for (size_t i = 0; i < outputs.size(); ++i) {
        const int8_t* __restrict__ raw = outputs[i].get();
        const auto& p = all_output_params_[i];
        float* __restrict__ result_ptr = dequant_buffers_[i].data();

        const float scale = p.scale;
        const float zp    = p.zero_point_f;
        const size_t HW   = p.H * p.W;
        const size_t C    = p.C;

        // Fused tiled dequantize + NHWC→NCHW transpose.
        // Tile kTileW spatial positions at a time so the working set
        // (kTileW × C × 4 bytes ≈ 5 KB for C=84) stays in L1.
        static constexpr size_t kTileW = 16;  // 1 cache line of floats

        cv::parallel_for_(
            cv::Range(0, static_cast<int>(p.N * p.H)),
            [&](const cv::Range& range) {
                alignas(64) float tile[kTileW * 256];  // ≤ 16 KB on stack

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                const float32x4_t v_scale = vdupq_n_f32(scale);
                const float32x4_t v_zp    = vdupq_n_f32(zp);
#endif

                for (int nh = range.start; nh < range.end; ++nh) {
                    const size_t n = static_cast<size_t>(nh) / p.H;
                    const size_t h = static_cast<size_t>(nh) % p.H;

                    const size_t in_row = (n + p.pad_n) * p.input_stride_n
                                        + (h + p.pad_h) * p.input_stride_h;
                    const size_t out_n  = n * p.output_stride_n;
                    const size_t out_hw = h * p.W;

                    for (size_t w0 = 0; w0 < p.W; w0 += kTileW) {
                        const size_t tw = std::min(kTileW, p.W - w0);

                        // ── Dequantize kTileW × C int8 → L1 tile ──
                        for (size_t dw = 0; dw < tw; ++dw) {
                            const int8_t* __restrict__ src = raw + in_row
                                + (w0 + dw + p.pad_w) * p.input_stride_w
                                + p.pad_c;
                            float* __restrict__ td = tile + dw * C;

                            size_t c = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                            for (; c + 16 <= C; c += 16) {
                                int8x16_t vi   = vld1q_s8(src + c);
                                int16x8_t lo16 = vmovl_s8(vget_low_s8(vi));
                                int16x8_t hi16 = vmovl_s8(vget_high_s8(vi));
                                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                                vst1q_f32(td + c,      vmulq_f32(vsubq_f32(f0, v_zp), v_scale));
                                vst1q_f32(td + c + 4,  vmulq_f32(vsubq_f32(f1, v_zp), v_scale));
                                vst1q_f32(td + c + 8,  vmulq_f32(vsubq_f32(f2, v_zp), v_scale));
                                vst1q_f32(td + c + 12, vmulq_f32(vsubq_f32(f3, v_zp), v_scale));
                            }
#endif
                            for (; c < C; ++c) {
                                td[c] = (static_cast<float>(src[c]) - zp) * scale;
                            }
                        }

                        // ── Transpose tile → NCHW output ──
                        for (size_t c = 0; c < C; ++c) {
                            float* __restrict__ dst = result_ptr + out_n + c * HW + out_hw + w0;
                            for (size_t dw = 0; dw < tw; ++dw) {
                                dst[dw] = tile[dw * C + c];
                            }
                        }
                    }
                }
            }
        );
    }

    return dequant_buffers_;
}

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            auto t0 = std::chrono::steady_clock::now();
            if (compute_metrics_) {
                frames_received_++;
                rclcpp::Time msg_stamp(msg->header.stamp);
                double cq_ms = (this->get_clock()->now() - msg_stamp).seconds() * 1000.0;
                if (cq_ms >= 0.0) {
                    push_metric(mw_callback_queue_, cq_ms);
                    all_callback_queue_.push_back(cq_ms);
                }
            }
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            auto t1 = std::chrono::steady_clock::now();
            process_image(frame, t0, t1);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

   void process_image(const cv::Mat& frame,
                   std::chrono::steady_clock::time_point t0,
                   std::chrono::steady_clock::time_point t1) {
    // --- Preprocess directly into the accelerator input buffer ---
    auto [scale, x_offset, y_offset] = preprocess_frame(
    frame, input_infos_[0], mean_, stddev_, padded_buffer_,
    reinterpret_cast<int8_t*>(input_args_[0].ptr));

    auto t2 = std::chrono::steady_clock::now();

    // --- Model inference ---
    if (axr_run_model_instance(instance_, input_args_.data(), input_args_.size(),
                               output_args_.data(), output_args_.size()) != AXR_SUCCESS) {
        RCLCPP_ERROR(this->get_logger(), "Failed to run model instance");
        return;
    }

    auto t3 = std::chrono::steady_clock::now();

    // --- Dequantize model outputs ---
    auto dequantized_outputs = this->process_outputs(output_data_);


    auto t4 = std::chrono::steady_clock::now();

    // --- Postprocess model outputs ---
    auto [box_type, detections] = postprocess_model_output(
        *onnx_session_, *allocator_, input_names_, output_names_,
        dequantized_outputs, confidence_threshold_, nms_threshold_,
        this->get_logger());

    auto t5 = std::chrono::steady_clock::now();

    // --- Annotate & publish image (optional) ---
    if (publish_annotated_) {
        cv::Mat annotated_frame = plot_detections(
            frame, detections, box_type, labels_, model_name_, scale, x_offset, y_offset);
        auto msg_out = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", annotated_frame).toImageMsg();
        msg_out->header.stamp = this->now();
        msg_out->header.frame_id = "camera_frame";
        image_pub_->publish(*msg_out);
    }

    // --- Publish detections ---
    if (!detections.empty()) {
        auto detection_msg = create_detection_message(detections, labels_);
        detection_pub_->publish(detection_msg);
    }

    auto t6 = std::chrono::steady_clock::now();

    // --- Metrics tracking ---
    if (compute_metrics_) {
        double d_preprocess = ms_dur(t2 - t1);
        double d_aipu       = ms_dur(t3 - t2);
        double d_dequant    = ms_dur(t4 - t3);
        double d_nms        = ms_dur(t5 - t4);
        double d_e2e        = ms_dur(t6 - t0);
        double d_det        = static_cast<double>(detections.size());

        // Rolling window (for live log display)
        push_metric(mw_preprocess_, d_preprocess);
        push_metric(mw_aipu_,       d_aipu);
        push_metric(mw_dequant_,    d_dequant);
        push_metric(mw_nms_,        d_nms);
        push_metric(mw_e2e_,        d_e2e);
        push_metric(mw_detections_, d_det);

        // All-frame storage (never evicts)
        all_preprocess_.push_back(d_preprocess);
        all_aipu_.push_back(d_aipu);
        all_dequant_.push_back(d_dequant);
        all_nms_.push_back(d_nms);
        all_e2e_.push_back(d_e2e);
        all_detections_.push_back(d_det);

        frames_processed_++;

        // Snapshot window summary every kMetricWindow frames
        if (frames_processed_ % kMetricWindow == 0) {
            uint64_t wend = frames_processed_;
            uint64_t wstart = wend - kMetricWindow;
            auto snap = [&](const std::vector<double>& all, std::vector<WindowStats>& out) {
                size_t count = std::min(all.size(), kMetricWindow);
                if (count == 0) return;
                auto begin = all.end() - static_cast<ptrdiff_t>(count);
                auto end   = all.end();
                out.push_back(compute_window_stats(wstart, wend, begin, end));
            };
            snap(all_preprocess_,     window_snapshots_.preprocess);
            snap(all_aipu_,           window_snapshots_.aipu);
            snap(all_dequant_,        window_snapshots_.dequant);
            snap(all_nms_,            window_snapshots_.nms);
            snap(all_e2e_,            window_snapshots_.e2e);
            snap(all_detections_,     window_snapshots_.detections);
            snap(all_callback_queue_, window_snapshots_.callback_queue);
        }

        if (processed_ts_.size() >= kFpsWindow) processed_ts_.pop_front();
        processed_ts_.push_back(t6);
    }
}
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AxeleraYoloInference>());
    rclcpp::shutdown();
    return 0;
}

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
#include <std_msgs/msg/int32.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>

#include "axruntime/axruntime.hpp"
#include "opencv2/opencv.hpp"
#include <onnxruntime_cxx_api.h>
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
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <set>

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
    size_t pad_left  = padding[i][0];
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

std::vector<std::vector<float>> process_outputs(
    const std::vector<std::unique_ptr<std::int8_t[]>>& outputs,
    const std::vector<axrTensorInfo>& output_infos,
    rclcpp::Logger logger) {
    std::vector<std::vector<float>> outs;
    outs.reserve(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& info = output_infos[i];
        const int8_t* output = outputs[i].get();

        const auto& padded_shape = info.dims;
        const auto& padding = info.padding;

        auto unpadded_shape = compute_unpadded_shape(padded_shape, padding, info.ndims);

        const size_t N = unpadded_shape[0];
        const size_t H = unpadded_shape[1];
        const size_t W = unpadded_shape[2];
        const size_t C = unpadded_shape[3];

        const size_t padded_H = padded_shape[1];
        const size_t padded_W = padded_shape[2];
        const size_t padded_C = padded_shape[3];

        const size_t input_stride_n = padded_H * padded_W * padded_C;
        const size_t input_stride_h = padded_W * padded_C;
        const size_t input_stride_w = padded_C;

        const size_t output_stride_n = C * H * W;
        const size_t output_stride_c = H * W;
        const size_t output_stride_h = W;

        const size_t pad_n = padding[0][0];
        const size_t pad_h = padding[1][0];
        const size_t pad_w = padding[2][0];
        const size_t pad_c = padding[3][0];

        const float scale = info.scale;
        const float zero_point_f = static_cast<float>(info.zero_point);

        std::vector<float> result(N * C * H * W);
        float* result_ptr = result.data();

        for (size_t n = 0; n < N; ++n) {
            const size_t n_in_base = (n + pad_n) * input_stride_n;
            const size_t n_out_base = n * output_stride_n;

            for (size_t h = 0; h < H; ++h) {
                const size_t h_in_base = n_in_base + (h + pad_h) * input_stride_h;

                for (size_t w = 0; w < W; ++w) {
                    const size_t w_in_base = h_in_base + (w + pad_w) * input_stride_w;
                    const size_t w_out_base = n_out_base + h * output_stride_h + w;

                    const int8_t* input_row = output + w_in_base + pad_c;

                    size_t c = 0;
                    for (; c + 3 < C; c += 4) {
                        const int8_t v0 = input_row[c];
                        const int8_t v1 = input_row[c + 1];
                        const int8_t v2 = input_row[c + 2];
                        const int8_t v3 = input_row[c + 3];

                        result_ptr[w_out_base + c * output_stride_c] = (v0 - zero_point_f) * scale;
                        result_ptr[w_out_base + (c + 1) * output_stride_c] = (v1 - zero_point_f) * scale;
                        result_ptr[w_out_base + (c + 2) * output_stride_c] = (v2 - zero_point_f) * scale;
                        result_ptr[w_out_base + (c + 3) * output_stride_c] = (v3 - zero_point_f) * scale;
                    }
                    for (; c < C; ++c) {
                        const int8_t value = input_row[c];
                        result_ptr[w_out_base + c * output_stride_c] = (value - zero_point_f) * scale;
                    }
                }
            }
        }

        outs.push_back(std::move(result));
    }

    return outs;
}

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
    } catch (const Ort::Exception& e) {
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;
        throw;
    }

    return input_tensors;
}

std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>>
extract_bounding_boxes(
    const std::vector<std::vector<float>>& predictions,
    bool has_objectness,
    int num_classes,
    float confidence_threshold) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (const auto& pred : predictions) {
        float conf;
        int cls_id;

        if (has_objectness) {
            float obj_conf = pred[4];
            float max_cls_conf = 0.0f;
            int max_cls_id = -1;
            for (int c = 5; c < num_classes + 5; ++c) {
                if (pred[c] > max_cls_conf) {
                    max_cls_conf = pred[c];
                    max_cls_id = c - 5;
                }
            }
            conf = obj_conf * max_cls_conf;
            cls_id = max_cls_id;
        } else {
            float max_cls_conf = 0.0f;
            int max_cls_id = -1;
            for (int c = 4; c < num_classes + 4; ++c) {
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

        float x_center = pred[0];
        float y_center = pred[1];
        float width = pred[2];
        float height = pred[3];
        float x1 = x_center - width / 2.0f;
        float y1 = y_center - height / 2.0f;
        float x2 = x_center + width / 2.0f;
        float y2 = y_center + height / 2.0f;

        cv::Rect box{cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2))};
        boxes.push_back(box);
        confidences.push_back(conf);
        class_ids.push_back(cls_id);
    }

    return {boxes, confidences, class_ids};
}

std::vector<Detection>
postprocess_model_output(
    Ort::Session& onnx_session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names,
    const std::vector<std::vector<float>>& inputs_list,
    float confidence_threshold,
    float nms_threshold,
    int num_classes,
    bool has_objectness,
    rclcpp::Logger logger) {
    auto onnx_results = execute_onnx_postprocess(
        onnx_session, allocator, input_names, output_names, inputs_list);

    std::vector<Detection> final_detections;

    for (const auto& result : onnx_results) {
	    auto shape_info = result.GetTensorTypeAndShapeInfo();
	    std::vector<int64_t> shape = shape_info.GetShape();
	    const float* data = result.GetTensorData<float>();

	    const int64_t stride = has_objectness ? (num_classes + 5) : (num_classes + 4);

	    int64_t N = 0;

	    if (has_objectness) {
		// YOLOv5-style: [batch, N, stride]
		if (shape.size() != 3 || shape[2] != stride)
		    throw std::runtime_error("Unexpected result shape for YOLOv5 (has_objectness=true)");
		N = shape[1];
	    } else {
		// YOLOv8-style: [batch, stride, N]
		if (shape.size() != 3 || shape[1] != stride)
		    throw std::runtime_error("Unexpected result shape for YOLOv8 (has_objectness=false)");
		N = shape[2];
	    }
	
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        boxes.reserve(N); confidences.reserve(N); class_ids.reserve(N);

        cv::parallel_for_(cv::Range(0, static_cast<int>(N)), [&](const cv::Range& range) {
            std::vector<cv::Rect> local_boxes;
            std::vector<float> local_confs;
            std::vector<int> local_classes;
            local_boxes.reserve(range.size());
            local_confs.reserve(range.size());
            local_classes.reserve(range.size());

            for (int i = range.start; i < range.end; ++i) {
                float x, y, w, h;
                if (has_objectness) {
                    const float* p = data + i * stride;
                    x = p[0]; y = p[1]; w = p[2]; h = p[3];
                    float obj = p[4];
                    int best_class = 0; float best_score = 0.0f;
                    for (int c = 0; c < num_classes; ++c) {
                        float conf = obj * p[5 + c];
                        if (conf > best_score) { best_score = conf; best_class = c; }
                    }
                    if (best_score >= confidence_threshold) {
                        local_boxes.emplace_back(int(x-w/2), int(y-h/2), int(w), int(h));
                        local_confs.push_back(best_score);
                        local_classes.push_back(best_class);
                    }
                } else {
                    x = data[0*N+i]; y = data[1*N+i]; w = data[2*N+i]; h = data[3*N+i];
                    int best_class = 0; float best_score = 0.0f;
                    for (int c = 0; c < num_classes; ++c) {
                        float cc = data[(4+c)*N+i];
                        if (cc > best_score) { best_score = cc; best_class = c; }
                    }
                    if (best_score >= confidence_threshold) {
                        local_boxes.emplace_back(int(x-w/2), int(y-h/2), int(w), int(h));
                        local_confs.push_back(best_score);
                        local_classes.push_back(best_class);
                    }
                }
            }

            static std::mutex mtx;
            std::lock_guard<std::mutex> lock(mtx);
            boxes.insert(boxes.end(), local_boxes.begin(), local_boxes.end());
            confidences.insert(confidences.end(), local_confs.begin(), local_confs.end());
            class_ids.insert(class_ids.end(), local_classes.begin(), local_classes.end());
        });

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

    return final_detections;
}

std::tuple<std::vector<std::int8_t>&, float, int, int> preprocess_frame(
    const cv::Mat& frame,
    const axrTensorInfo& info,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& stddev,
    cv::Mat& padded_buffer,
    std::vector<std::int8_t>& quantized_buffer) {
    const auto height = info.dims[1];
    const auto width = info.dims[2];
    const auto channels = info.dims[3];
    const auto [y_pad_left, y_pad_right] = info.padding[1];
    const auto [x_pad_left, x_pad_right] = info.padding[2];
    const auto unpadded_height = height - y_pad_left - y_pad_right;
    const auto unpadded_width = width - x_pad_left - x_pad_right;

    float scale = std::min(static_cast<float>(unpadded_width) / frame.cols,
                           static_cast<float>(unpadded_height) / frame.rows);
    int resized_width = static_cast<int>(frame.cols * scale);
    int resized_height = static_cast<int>(frame.rows * scale);
    int x_offset = (unpadded_width - resized_width) / 2;
    int y_offset = (unpadded_height - resized_height) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);

    padded_buffer.setTo(cv::Scalar(0, 0, 0));
    resized.copyTo(padded_buffer(cv::Rect(x_offset, y_offset, resized_width, resized_height)));

    const size_t total_size = height * width * channels;
    if (quantized_buffer.size() != total_size)
        quantized_buffer.resize(total_size, static_cast<std::int8_t>(std::clamp(info.zero_point, -128, 127)));

    const float inv_scale = 1.0f / info.scale;
    const float inv_255 = 1.0f / 255.0f;
    const float combined_mul[3] = {
        (1.0f / stddev[0]) * inv_scale * inv_255,
        (1.0f / stddev[1]) * inv_scale * inv_255,
        (1.0f / stddev[2]) * inv_scale * inv_255
    };
    const float combined_add[3] = {
        (-mean[0] / stddev[0]) * inv_scale + info.zero_point,
        (-mean[1] / stddev[1]) * inv_scale + info.zero_point,
        (-mean[2] / stddev[2]) * inv_scale + info.zero_point
    };

    const uint8_t* src_ptr = padded_buffer.ptr<uint8_t>();
    int8_t* dst_ptr = quantized_buffer.data();

    for (int y = 0; y < static_cast<int>(height); ++y) {
        for (int x = 0; x < static_cast<int>(width); ++x) {
            const size_t pixel_idx = (y * width + x) * 3;
            for (int c = 0; c < 3; ++c) {
                float qf = src_ptr[pixel_idx + (2 - c)] * combined_mul[c] + combined_add[c];
                int8_t q = static_cast<int8_t>(std::round(std::clamp(qf, -128.0f, 127.0f)));
                dst_ptr[(y * width + x) * channels + c] = q;
            }
        }
    }

    return {quantized_buffer, scale, x_offset, y_offset};
}

cv::Mat plot_detections(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::string& box_type,
    const std::vector<std::string>& labels,
    const std::string& model_name,
    float scale,
    int x_offset,
    int y_offset)
{
    cv::Mat annotated_frame = frame.clone();

    for (const auto& detection : detections) {
        int class_id = detection.class_id;
        float confidence = detection.confidence;
        cv::Rect2f box = detection.box;

        std::string label = (class_id >= 0 && class_id < (int)labels.size()) ? labels[class_id] : "Unknown";

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
    const std::vector<std::string>& labels)
{
    std_msgs::msg::String msg;
    std::ostringstream ss;

    for (const auto& detection : detections) {
        int class_id = detection.class_id;
        float confidence = detection.confidence;
        cv::Rect2f box = detection.box;
        
        std::string label = (class_id >= 0 && class_id < (int)labels.size()) ? labels[class_id] : "Unknown";

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

void logger(void *arg, axrLogLevel level, const char *msg)
{
    (void) arg;
    (void) level;
}

// Define a ROS 2 Node class
class AxeleraYoloInference : public rclcpp::Node
{
public:
    AxeleraYoloInference() : Node("axelera_yolo_inference")
    {
        RCLCPP_INFO(this->get_logger(), "Starting Axelera YOLO Inference Node...");

        // Declare parameters with default values
        this->declare_parameter("model_name", "yolo11x-coco-onnx");
        this->declare_parameter("aipu_cores", 4);
        this->declare_parameter("input_topic", "/camera_frame");
        this->declare_parameter("output_topic", "/detections_topic");
        this->declare_parameter("confidence_threshold", 0.25);
        this->declare_parameter("nms_threshold", 0.45);
        this->declare_parameter("mean", std::vector<double>{0.485, 0.456, 0.406});
        this->declare_parameter("stddev", std::vector<double>{0.229, 0.224, 0.225});
        this->declare_parameter("num_classes", 80);
        this->declare_parameter("has_objectness", false);
        this->declare_parameter("box_type", "xyxy");
        this->declare_parameter("annotated_topic", "");
        this->declare_parameter("publish_annotated_image", true);
        this->declare_parameter("publish_detection_strings", true);
        this->declare_parameter("publish_class_id", true);
        this->declare_parameter("every_n_frames", 5);
        this->declare_parameter("crop_ratio", 0.2);
        this->declare_parameter("max_bbox_ratio", 0.9);
        this->declare_parameter("stable_frames_required", 10);
        this->declare_parameter("variance_threshold", 15.0);
        this->declare_parameter("destabilize_threshold", 50.0);
        this->declare_parameter("keep_largest_only", true);
        this->declare_parameter("save_results", true);
        this->declare_parameter("publish_interval", 0.25);
        this->declare_parameter("class_detections_topic", "/detections");
        this->declare_parameter("save_dir", "yolo_results");

        // Get parameter values
        model_name_ = this->get_parameter("model_name").as_string();
        aipu_cores_ = this->get_parameter("aipu_cores").as_int();
        input_topic_ = this->get_parameter("input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
        nms_threshold_ = this->get_parameter("nms_threshold").as_double();
        
        auto mean_param = this->get_parameter("mean").as_double_array();
        auto stddev_param = this->get_parameter("stddev").as_double_array();
        
        // Convert to arrays
        for (size_t i = 0; i < 3 && i < mean_param.size(); ++i) {
            mean_[i] = static_cast<float>(mean_param[i]);
        }
        for (size_t i = 0; i < 3 && i < stddev_param.size(); ++i) {
            stddev_[i] = static_cast<float>(stddev_param[i]);
        }

        std::string annotated_topic_param = this->get_parameter("annotated_topic").as_string();
        every_n_frames_   = this->get_parameter("every_n_frames").as_int();
        crop_ratio_       = static_cast<float>(this->get_parameter("crop_ratio").as_double());
        max_bbox_ratio_   = static_cast<float>(this->get_parameter("max_bbox_ratio").as_double());
        stable_frames_required_ = this->get_parameter("stable_frames_required").as_int();
        variance_threshold_    = static_cast<float>(this->get_parameter("variance_threshold").as_double());
        destabilize_threshold_ = static_cast<float>(this->get_parameter("destabilize_threshold").as_double());
        keep_largest_only_ = this->get_parameter("keep_largest_only").as_bool();
        save_results_      = this->get_parameter("save_results").as_bool();
        publish_interval_  = static_cast<float>(this->get_parameter("publish_interval").as_double());
        class_detections_topic_ = this->get_parameter("class_detections_topic").as_string();
        save_dir_          = this->get_parameter("save_dir").as_string();
        publish_annotated_image_    = this->get_parameter("publish_annotated_image").as_bool();
        publish_detection_strings_  = this->get_parameter("publish_detection_strings").as_bool();
        publish_class_id_           = this->get_parameter("publish_class_id").as_bool();
        num_classes_     = static_cast<int>(this->get_parameter("num_classes").as_int());
        has_objectness_  = this->get_parameter("has_objectness").as_bool();
        box_type_        = this->get_parameter("box_type").as_string();

        RCLCPP_INFO(this->get_logger(), "Parameters loaded:");
        RCLCPP_INFO(this->get_logger(), "  Model name: %s", model_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "  AIPU cores: %d", aipu_cores_);
        RCLCPP_INFO(this->get_logger(), "  Input topic: %s", input_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Output topic: %s", output_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Confidence threshold: %.2f", confidence_threshold_);
        RCLCPP_INFO(this->get_logger(), "  NMS threshold: %.2f", nms_threshold_);

        // Load labels from model_info.json and derive num_classes from the actual model
        const auto root = std::getenv("AXELERA_FRAMEWORK");
        std::string model_info_path = "../build/" + model_name_ + "/" + model_name_ + "/model_info.json";
        this->labels_ = read_labels(model_info_path);
        num_classes_ = static_cast<int>(labels_.size());

        // Build class_id_map_ from the actual loaded labels (not hardcoded)
        for (int i = 0; i < num_classes_; ++i) {
            class_id_map_[i] = labels_[i];
        }

        RCLCPP_INFO(this->get_logger(), "========================================");
        RCLCPP_INFO(this->get_logger(), "  Model     : %s", model_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Labels    : %s", model_info_path.c_str());
        RCLCPP_INFO(this->get_logger(), "  Classes   : %d", num_classes_);
        RCLCPP_INFO(this->get_logger(), "  Class map :");
        for (int i = 0; i < num_classes_; ++i) {
            RCLCPP_INFO(this->get_logger(), "    [%d] -> %s", i, class_id_map_[i].c_str());
        }
        RCLCPP_INFO(this->get_logger(), "========================================");

        // Initialize publishers and subscribers with parametric topic names
        std::string annotated_topic = annotated_topic_param.empty() ? input_topic_ + "_annotated" : annotated_topic_param;
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(annotated_topic, 10);
        detection_pub_ = this->create_publisher<std_msgs::msg::String>(output_topic_, 10);
        class_pub_ = this->create_publisher<std_msgs::msg::Int32>(class_detections_topic_, 10);
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

        for (int n = 0; n != (int)inputs; ++n) {
            input_data_[n] = std::make_unique<std::int8_t[]>(axr_tensor_size(&input_infos_[n]));
            input_args_[n].ptr = input_data_[n].get();
            input_args_[n].fd = 0;
            input_args_[n].offset = 0;
        }
        for (int n = 0; n != (int)outputs; ++n) {
            output_data_[n] = std::make_unique<std::int8_t[]>(axr_tensor_size(&output_infos_[n]));
            output_args_[n].ptr = output_data_[n].get();
            output_args_[n].fd = 0;
            output_args_[n].offset = 0;
        }

        // Initialize persistent ONNX session (once at startup, not per-frame)
        onnx_model_path_ = "../build/" + model_name_ + "/" + model_name_ + "/1/postprocess_graph.onnx";
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        onnx_session_ = std::make_unique<Ort::Session>(env_, onnx_model_path_.c_str(), session_options);
        allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();

        for (size_t i = 0; i < onnx_session_->GetInputCount(); ++i) {
            Ort::AllocatedStringPtr name = onnx_session_->GetInputNameAllocated(i, *allocator_);
            input_names_.push_back(name.get());
            owned_input_names_.push_back(std::move(name));
        }
        for (size_t i = 0; i < onnx_session_->GetOutputCount(); ++i) {
            Ort::AllocatedStringPtr name = onnx_session_->GetOutputNameAllocated(i, *allocator_);
            output_names_.push_back(name.get());
            owned_output_names_.push_back(std::move(name));
        }

        // Preallocate preprocessing buffers (reused every frame)
        const auto& input_info = input_infos_[0];
        input_height_   = input_info.dims[1];
        input_width_    = input_info.dims[2];
        input_channels_ = input_info.dims[3];
        padded_buffer_  = cv::Mat(input_height_, input_width_, CV_8UC3);
        quantized_buffer_.resize(input_height_ * input_width_ * input_channels_);

        // Precompute output dequantization parameters for member process_outputs
        for (const auto& out_info : output_infos_) {
            OutputParams params;
            params.unpadded_shape = compute_unpadded_shape(out_info.dims, out_info.padding, out_info.ndims);
            params.N = params.unpadded_shape[0];
            params.H = params.unpadded_shape[1];
            params.W = params.unpadded_shape[2];
            params.C = params.unpadded_shape[3];
            params.padded_H = out_info.dims[1];
            params.padded_W = out_info.dims[2];
            params.padded_C = out_info.dims[3];
            params.input_stride_n = params.padded_H * params.padded_W * params.padded_C;
            params.input_stride_h = params.padded_W * params.padded_C;
            params.input_stride_w = params.padded_C;
            params.output_stride_n = params.C * params.H * params.W;
            params.output_stride_c = params.H * params.W;
            params.output_stride_h = params.W;
            params.pad_n = out_info.padding[0][0];
            params.pad_h = out_info.padding[1][0];
            params.pad_w = out_info.padding[2][0];
            params.pad_c = out_info.padding[3][0];
            params.scale = out_info.scale;
            params.zero_point_f = static_cast<float>(out_info.zero_point);
            params.result_size = params.N * params.C * params.H * params.W;
            all_output_params_.push_back(params);
        }

        RCLCPP_INFO(this->get_logger(), "Axelera YOLO Inference Node initialized successfully.");
    }

    ~AxeleraYoloInference()
    {
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
    std::array<float, 3> mean_;
    std::array<float, 3> stddev_;

    // Preallocated preprocessing buffers (reused every frame)
    size_t input_height_ = 0;
    size_t input_width_ = 0;
    size_t input_channels_ = 0;
    cv::Mat padded_buffer_;
    std::vector<std::int8_t> quantized_buffer_;

    // Precomputed output dequantization parameters
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

    // Model parameters
    int num_classes_;
    bool has_objectness_;
    std::string box_type_;

    // Placement detection parameters
    int every_n_frames_;
    float crop_ratio_;
    float max_bbox_ratio_;
    int stable_frames_required_;
    float variance_threshold_;
    float destabilize_threshold_;
    bool keep_largest_only_;
    bool save_results_;
    float publish_interval_;
    std::string class_detections_topic_;
    std::string save_dir_;
    bool publish_annotated_image_;
    bool publish_detection_strings_;
    bool publish_class_id_;

    // Publishers
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr class_pub_;

    // Class name mapping
    std::unordered_map<int, std::string> class_id_map_;

    // Tracking state
    int frame_count_ = 0;
    int next_object_id_ = 0;
    float iou_threshold_tracking_ = 0.5f;
    // Each track entry: (frame_num, bbox as xywh-center, class_id, confidence)
    std::unordered_map<int, std::deque<std::tuple<int, cv::Vec4f, int, float>>> detection_tracks_;
    std::unordered_map<int, std::string> object_states_;
    std::unordered_map<int, cv::Vec4f> last_stable_bbox_;
    std::unordered_map<int, std::pair<int, std::chrono::steady_clock::time_point>> last_published_objects_;

    // ----------------------------------------------------------------
    // Tracking helpers
    // ----------------------------------------------------------------

    // Optimised member process_outputs — uses precomputed OutputParams
    std::vector<std::vector<float>> process_outputs(
        const std::vector<std::unique_ptr<std::int8_t[]>>& outputs) {
        std::vector<std::vector<float>> outs(outputs.size());
        cv::parallel_for_(cv::Range(0, static_cast<int>(outputs.size())),
                          [&](const cv::Range& range) {
                              for (int i = range.start; i < range.end; ++i) {
                                  const int8_t* output = outputs[i].get();
                                  const auto& p = all_output_params_[i];
                                  std::vector<float> result(p.result_size);
                                  float* rp = result.data();
                                  for (size_t n = 0; n < p.N; ++n) {
                                      const size_t n_in = (n + p.pad_n) * p.input_stride_n;
                                      const size_t n_out = n * p.output_stride_n;
                                      for (size_t h = 0; h < p.H; ++h) {
                                          const size_t h_in = n_in + (h + p.pad_h) * p.input_stride_h;
                                          const size_t h_out = n_out + h * p.output_stride_h;
                                          for (size_t w = 0; w < p.W; ++w) {
                                              const size_t w_in = h_in + (w + p.pad_w) * p.input_stride_w + p.pad_c;
                                              const size_t w_out = h_out + w;
                                              const int8_t* row = output + w_in;
                                              size_t c = 0;
                                              for (; c + 3 < p.C; c += 4) {
                                                  rp[w_out + c * p.output_stride_c]       = (row[c]   - p.zero_point_f) * p.scale;
                                                  rp[w_out + (c+1) * p.output_stride_c]   = (row[c+1] - p.zero_point_f) * p.scale;
                                                  rp[w_out + (c+2) * p.output_stride_c]   = (row[c+2] - p.zero_point_f) * p.scale;
                                                  rp[w_out + (c+3) * p.output_stride_c]   = (row[c+3] - p.zero_point_f) * p.scale;
                                              }
                                              for (; c < p.C; ++c)
                                                  rp[w_out + c * p.output_stride_c] = (row[c] - p.zero_point_f) * p.scale;
                                          }
                                      }
                                  }
                                  outs[i] = std::move(result);
                              }
                          });
        return outs;
    }

    void add_to_track(int track_id, std::tuple<int, cv::Vec4f, int, float> entry)
    {
        auto& dq = detection_tracks_[track_id];
        dq.push_back(std::move(entry));
        while (static_cast<int>(dq.size()) > stable_frames_required_)
            dq.pop_front();
    }

    float compute_iou(const cv::Vec4f& b1, const cv::Vec4f& b2)
    {
        float x1 = b1[0] - b1[2] / 2.f, y1 = b1[1] - b1[3] / 2.f;
        float x2 = b1[0] + b1[2] / 2.f, y2 = b1[1] + b1[3] / 2.f;
        float x3 = b2[0] - b2[2] / 2.f, y3 = b2[1] - b2[3] / 2.f;
        float x4 = b2[0] + b2[2] / 2.f, y4 = b2[1] + b2[3] / 2.f;
        float ix = std::max(0.f, std::min(x2, x4) - std::max(x1, x3));
        float iy = std::max(0.f, std::min(y2, y4) - std::max(y1, y3));
        float inter = ix * iy;
        float uni = b1[2] * b1[3] + b2[2] * b2[3] - inter;
        return uni <= 0.f ? 0.f : inter / uni;
    }

    std::optional<int> match_detection_to_track(const cv::Vec4f& bbox, int class_id)
    {
        float best_iou = 0.f;
        std::optional<int> best_id;
        for (auto& [tid, history] : detection_tracks_) {
            if (history.empty()) continue;
            auto& [lf, lb, lc, lconf] = history.back();
            if (lc != class_id) continue;
            float iou = compute_iou(bbox, lb);
            if (iou > best_iou && iou >= iou_threshold_tracking_) {
                best_iou = iou;
                best_id = tid;
            }
        }
        return best_id;
    }

    float compute_bbox_variance(int track_id)
    {
        auto& hist = detection_tracks_[track_id];
        if (hist.size() < 2) return 0.f;
        float sx = 0.f, sy = 0.f;
        float n = static_cast<float>(hist.size());
        for (auto& [f, b, c, conf] : hist) { sx += b[0]; sy += b[1]; }
        float mx = sx / n, my = sy / n;
        float vx = 0.f, vy = 0.f;
        for (auto& [f, b, c, conf] : hist) {
            vx += (b[0] - mx) * (b[0] - mx);
            vy += (b[1] - my) * (b[1] - my);
        }
        return (std::sqrt(vx / n) + std::sqrt(vy / n)) / 2.f;
    }

    bool is_track_stable(int track_id)
    {
        auto& hist = detection_tracks_[track_id];
        if (static_cast<int>(hist.size()) < stable_frames_required_) return false;
        return compute_bbox_variance(track_id) < variance_threshold_;
    }

    bool has_object_destabilized(int track_id, const cv::Vec4f& cur)
    {
        auto it = last_stable_bbox_.find(track_id);
        if (it == last_stable_bbox_.end()) return false;
        const auto& last = it->second;
        float dist = std::sqrt(std::pow(cur[0] - last[0], 2.f) + std::pow(cur[1] - last[1], 2.f));
        float size_change = std::abs(cur[2] - last[2]) + std::abs(cur[3] - last[3]);
        return dist >= destabilize_threshold_ || size_change >= destabilize_threshold_;
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) 
    {
        try {
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            process_image(frame);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void process_image(const cv::Mat& frame) 
    {
        // ============================================================
        // Frame skipping
        // ============================================================
        frame_count_++;
        if (frame_count_ % every_n_frames_ != 0)
            return;

        // ============================================================
        // Crop image by crop_ratio on each side (keep centre region)
        // ============================================================
        int crop_x = static_cast<int>(frame.cols * crop_ratio_);
        int crop_y = static_cast<int>(frame.rows * crop_ratio_);
        cv::Mat cropped_frame = frame(
            cv::Rect(crop_x, crop_y,
                     frame.cols - 2 * crop_x,
                     frame.rows - 2 * crop_y)).clone();

        // Preprocess + run inference using preallocated buffers
        auto [processed_input, scale, x_offset, y_offset] = preprocess_frame(
            cropped_frame, input_infos_[0], mean_, stddev_, padded_buffer_, quantized_buffer_);

        // Copy processed input to buffer
        std::memcpy(input_args_[0].ptr, processed_input.data(), processed_input.size());

        // Run model inference
        if (axr_run_model_instance(instance_, input_args_.data(), input_args_.size(),
                                   output_args_.data(), output_args_.size()) != AXR_SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to run model instance");
            return;
        }

        // Process outputs using member's precomputed params (fast path)
        auto dequantized_outputs = process_outputs(output_data_);

        // Run ONNX postprocess on the persistent session
        auto detections = postprocess_model_output(
            *onnx_session_, *allocator_, input_names_, output_names_,
            dequantized_outputs, confidence_threshold_, nms_threshold_,
            num_classes_, has_objectness_,
            this->get_logger());

        RCLCPP_INFO(this->get_logger(), "Number of detections: %zu", detections.size());

        // Keep only the single highest-confidence detection
        if (detections.size() > 1) {
            auto best = std::max_element(detections.begin(), detections.end(),
                [](const Detection& a, const Detection& b) {
                    return a.confidence < b.confidence;
                });
            detections = {*best};
        }

        // Visualization and publishing (on cropped frame - boxes are in cropped coords)
        cv::Mat annotated_frame = plot_detections(
            cropped_frame, detections, box_type_, labels_, model_name_, scale, x_offset, y_offset);

        // Publish annotated image
        if (publish_annotated_image_) {
            auto msg_out = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", annotated_frame).toImageMsg();
            msg_out->header.stamp = this->now();
            msg_out->header.frame_id = "camera_frame";
            image_pub_->publish(*msg_out);
        }

        // Publish detection string
        if (publish_detection_strings_ && !detections.empty()) {
            auto detection_msg = create_detection_message(detections, labels_);
            detection_pub_->publish(detection_msg);
        }

        // ============================================================
        // Convert detections to cropped-frame coords (xc, yc, w, h)
        // and filter by max_bbox_ratio
        // ============================================================
        float cropped_area = static_cast<float>(cropped_frame.cols * cropped_frame.rows);
        float max_bbox_area = cropped_area * max_bbox_ratio_;

        // pair: (xywh-centre bbox in cropped-frame coords, class_id)
        std::vector<std::pair<cv::Vec4f, int>> filtered_detections;
        for (const auto& det : detections) {
            float x1_c = (det.box.x - x_offset) / scale;
            float y1_c = (det.box.y - y_offset) / scale;
            float x2_c = ((det.box.x + det.box.width)  - x_offset) / scale;
            float y2_c = ((det.box.y + det.box.height) - y_offset) / scale;
            float w_c  = x2_c - x1_c;
            float h_c  = y2_c - y1_c;
            if (w_c <= 0.f || h_c <= 0.f) continue;
            float bbox_area = w_c * h_c;
            if (bbox_area > max_bbox_area) {
                RCLCPP_WARN(this->get_logger(),
                    "Filtered large bbox: %.1f%% of image (cls=%d)",
                    bbox_area / cropped_area * 100.f, det.class_id);
                continue;
            }
            float xc_c = x1_c + w_c / 2.f;
            float yc_c = y1_c + h_c / 2.f;
            filtered_detections.push_back({{xc_c, yc_c, w_c, h_c}, det.class_id});
        }

        // ============================================================
        // Keep only the largest detection (if enabled)
        // ============================================================
        if (keep_largest_only_ && filtered_detections.size() > 1) {
            auto largest = std::max_element(
                filtered_detections.begin(), filtered_detections.end(),
                [](const auto& a, const auto& b) {
                    return (a.first[2] * a.first[3]) < (b.first[2] * b.first[3]);
                });
            filtered_detections = {*largest};
        }

        if (filtered_detections.empty())
            return;

        // ============================================================
        // Match detections to existing tracks
        // ============================================================
        std::set<int> matched_tracks;
        for (const auto& [bbox, cls_id] : filtered_detections) {
            auto track_id_opt = match_detection_to_track(bbox, cls_id);
            int track_id = track_id_opt.has_value() ? track_id_opt.value() : next_object_id_++;
            add_to_track(track_id, {frame_count_, bbox, cls_id, 0.f});
            matched_tracks.insert(track_id);
        }

        // Remove stale tracks (not seen in last 60 processed frames)
        std::vector<int> tracks_to_remove;
        for (auto& [tid, hist] : detection_tracks_) {
            if (!hist.empty() && frame_count_ - std::get<0>(hist.back()) > 60 * every_n_frames_)
                tracks_to_remove.push_back(tid);
        }
        for (int tid : tracks_to_remove) {
            detection_tracks_.erase(tid);
            object_states_.erase(tid);
            last_stable_bbox_.erase(tid);
        }

        // ============================================================
        // Placement detection state machine
        // DETECTING -> STABLE -> DESTABILIZED -> DETECTING
        // ============================================================
        std::vector<int> tracks_to_publish;
        for (int track_id : matched_tracks) {
            auto& hist = detection_tracks_[track_id];
            auto& [fn, bbox, cls_id, conf] = hist.back();

            if (!object_states_.count(track_id))
                object_states_[track_id] = "DETECTING";

            auto& state = object_states_[track_id];

            if (state == "DETECTING") {
                if (is_track_stable(track_id)) {
                    state = "STABLE";
                    last_stable_bbox_[track_id] = bbox;
                    tracks_to_publish.push_back(track_id);
                }
            } else if (state == "STABLE") {
                if (has_object_destabilized(track_id, bbox)) {
                    state = "DESTABILIZED";
                    detection_tracks_[track_id].clear();
                    auto it = last_published_objects_.find(track_id);
                    if (it != last_published_objects_.end()) {
                        int pub_cls = it->second.first;
                        std::string cname = class_id_map_.count(pub_cls) ?
                            class_id_map_.at(pub_cls) : "unknown";
                        last_published_objects_.erase(it);
                        RCLCPP_INFO(this->get_logger(),
                            "[STOP PUBLISH] Int32: %d (%s) - Track #%d (object moved)",
                            pub_cls, cname.c_str(), track_id);
                    }
                }
            } else if (state == "DESTABILIZED") {
                state = "DETECTING";
            }
        }

        // ============================================================
        // Publish newly stable objects
        // ============================================================
        auto now = std::chrono::steady_clock::now();
        for (int track_id : tracks_to_publish) {
            auto& [fn, bbox, cls_id, conf] = detection_tracks_[track_id].back();
            std::string cname = class_id_map_.count(cls_id) ?
                class_id_map_.at(cls_id) : "unknown";
            last_published_objects_[track_id] = {cls_id, now};

            if (publish_class_id_) {
                std_msgs::msg::Int32 class_msg;
                class_msg.data = cls_id;
                class_pub_->publish(class_msg);
            }
            RCLCPP_INFO(this->get_logger(), "[PUBLISH] Int32: %d (%s) - Track #%d",
                cls_id, cname.c_str(), track_id);

            if (save_results_) {
                std::filesystem::create_directories(save_dir_);
                std::string out_path = save_dir_ + "/frame_" +
                    std::to_string(frame_count_) + ".jpg";
                cv::imwrite(out_path, annotated_frame);
                RCLCPP_INFO(this->get_logger(), "Saved annotated image: %s", out_path.c_str());
            }
        }

        // ============================================================
        // Re-publish already stable objects at interval
        // ============================================================
        std::vector<int> tracks_to_republish;
        for (auto& [tid, pub_info] : last_published_objects_) {
            if (object_states_.count(tid) && object_states_[tid] == "STABLE") {
                double elapsed = std::chrono::duration<double>(now - pub_info.second).count();
                if (elapsed >= publish_interval_)
                    tracks_to_republish.push_back(tid);
            }
        }
        for (int track_id : tracks_to_republish) {
            int cls_id = last_published_objects_[track_id].first;
            last_published_objects_[track_id].second = now;
            if (publish_class_id_) {
                std_msgs::msg::Int32 class_msg;
                class_msg.data = cls_id;
                class_pub_->publish(class_msg);
            }
            std::string cname = class_id_map_.count(cls_id) ?
                class_id_map_.at(cls_id) : "unknown";
            RCLCPP_INFO(this->get_logger(), "[RE-PUBLISH] Int32: %d (%s) - Track #%d",
                cls_id, cname.c_str(), track_id);
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AxeleraYoloInference>());
    rclcpp::shutdown();
    return 0;
}

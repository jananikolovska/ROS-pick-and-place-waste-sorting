#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <span>
#include <sstream>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cv_bridge/cv_bridge.h>

#include "axruntime/axruntime.hpp"

struct Detection {
    int class_id;
    float confidence;
    cv::Rect2f box;
};

constexpr auto DEFAULT_LABELS = "ax_datasets/labels/coco.names";

using namespace std::string_literals;

std::vector<size_t> compute_unpadded_shape(
    const size_t* dims,
    const size_t (*padding)[2],
    size_t ndim)
{
    std::vector<size_t> unpadded;
    unpadded.reserve(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        const size_t pad_left = padding[i][0];
        const size_t pad_right = padding[i][1];
        assert(dims[i] >= (pad_left + pad_right));
        unpadded.push_back(dims[i] - pad_left - pad_right);
    }

    return unpadded;
}

size_t get_flat_index_NHWC(
    size_t n, size_t h, size_t w, size_t c,
    size_t H, size_t W, size_t C)
{
    return ((n * H + h) * W + w) * C + c;
}

size_t get_flat_index_NCHW(
    size_t n, size_t c, size_t h, size_t w,
    size_t C, size_t H, size_t W)
{
    return ((n * C + c) * H + h) * W + w;
}

std::vector<float> transpose_NHWC_to_NCHW(
    const std::vector<float>& input,
    size_t N, size_t H, size_t W, size_t C)
{
    assert(input.size() == N * H * W * C && "Input size mismatch");

    std::vector<float> output(N * C * H * W);
    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t c = 0; c < C; ++c) {
                    const size_t in_idx = get_flat_index_NHWC(n, h, w, c, H, W, C);
                    const size_t out_idx = get_flat_index_NCHW(n, c, h, w, C, H, W);
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
    return output;
}

static constexpr size_t kTransposeTile = 16;

std::vector<std::vector<float>> process_outputs(
    const std::vector<std::unique_ptr<std::int8_t[]>>& outputs,
    const std::vector<axrTensorInfo>& output_infos,
    rclcpp::Logger /*logger*/)
{
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
    const std::vector<std::vector<float>>& inputs_list)
{
    std::vector<Ort::Value> input_tensors;
    const size_t num_model_inputs = session.GetInputCount();

    if (inputs_list.size() != num_model_inputs) {
        throw std::invalid_argument("Number of inputs does not match model input count.");
    }

    for (size_t i = 0; i < num_model_inputs; ++i) {
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();

        const size_t actual_size = inputs_list[i].size();
        for (auto& dim : input_shape) {
            if (dim == -1) {
                dim = static_cast<int64_t>(actual_size);
            }
        }

        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            allocator.GetInfo(),
            const_cast<float*>(inputs_list[i].data()),
            actual_size,
            input_shape.data(),
            input_shape.size());

        input_tensors.push_back(std::move(tensor));
    }

    try {
        return session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size());
    } catch (const Ort::Exception& e) {
        std::cout << "Error code: " << e.GetOrtErrorCode() << std::endl;
        throw;
    }
}

std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>>
extract_bounding_boxes(
    const std::vector<std::vector<float>>& predictions,
    bool has_objectness,
    float confidence_threshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (const auto& pred : predictions) {
        float conf;
        int cls_id;

        if (has_objectness) {
            const float obj_conf = pred[4];
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

        if (conf < confidence_threshold) {
            continue;
        }

        const float x_center = pred[0];
        const float y_center = pred[1];
        const float width = pred[2];
        const float height = pred[3];
        const float x1 = x_center - width / 2.0f;
        const float y1 = y_center - height / 2.0f;
        const float x2 = x_center + width / 2.0f;
        const float y2 = y_center + height / 2.0f;

        cv::Rect box{cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2))};
        boxes.push_back(box);
        confidences.push_back(conf);
        class_ids.push_back(cls_id);
    }

    return {boxes, confidences, class_ids};
}

std::vector<Detection> postprocess_model_output(
    Ort::Session& onnx_session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names,
    const std::vector<std::vector<float>>& inputs_list,
    float confidence_threshold,
    float nms_threshold,
    int num_classes,
    bool has_objectness,
    rclcpp::Logger /*logger*/)
{
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
            if (shape.size() != 3 || shape[2] != stride) {
                throw std::runtime_error("Unexpected result shape for YOLOv5 (has_objectness=true)");
            }
            N = shape[1];
        } else {
            if (shape.size() != 3 || shape[1] != stride) {
                throw std::runtime_error("Unexpected result shape for YOLOv8 (has_objectness=false)");
            }
            N = shape[2];
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        boxes.reserve(static_cast<size_t>(N));
        confidences.reserve(static_cast<size_t>(N));
        class_ids.reserve(static_cast<size_t>(N));

        auto parse_range = [&](int start, int end,
                               std::vector<cv::Rect>& out_boxes,
                               std::vector<float>& out_confs,
                               std::vector<int>& out_classes) {
            out_boxes.reserve(static_cast<size_t>(end - start));
            out_confs.reserve(static_cast<size_t>(end - start));
            out_classes.reserve(static_cast<size_t>(end - start));

            for (int i = start; i < end; ++i) {
                float x, y, w, h;

                if (has_objectness) {
                    const float* p = data + static_cast<int64_t>(i) * stride;
                    x = p[0];
                    y = p[1];
                    w = p[2];
                    h = p[3];
                    const float obj = p[4];

                    int best_class = 0;
                    float best_score = 0.0f;
                    for (int c = 0; c < num_classes; ++c) {
                        const float conf = obj * p[5 + c];
                        if (conf > best_score) {
                            best_score = conf;
                            best_class = c;
                        }
                    }

                    if (best_score >= confidence_threshold) {
                        out_boxes.emplace_back(
                            int(x - w / 2.0f),
                            int(y - h / 2.0f),
                            int(w),
                            int(h));
                        out_confs.push_back(best_score);
                        out_classes.push_back(best_class);
                    }
                } else {
                    x = data[0 * N + i];
                    y = data[1 * N + i];
                    w = data[2 * N + i];
                    h = data[3 * N + i];

                    int best_class = 0;
                    float best_score = 0.0f;
                    for (int c = 0; c < num_classes; ++c) {
                        const float cc = data[(4 + c) * N + i];
                        if (cc > best_score) {
                            best_score = cc;
                            best_class = c;
                        }
                    }

                    if (best_score >= confidence_threshold) {
                        out_boxes.emplace_back(
                            int(x - w / 2.0f),
                            int(y - h / 2.0f),
                            int(w),
                            int(h));
                        out_confs.push_back(best_score);
                        out_classes.push_back(best_class);
                    }
                }
            }
        };

        constexpr int kParallelThreshold = 1024;

        if (N < kParallelThreshold) {
            parse_range(0, static_cast<int>(N), boxes, confidences, class_ids);
        } else {
            const int num_threads = std::max(1u, std::thread::hardware_concurrency());
            const int chunk_count = std::min<int>(num_threads, static_cast<int>(N));
            std::vector<std::vector<cv::Rect>> boxes_chunks(static_cast<size_t>(chunk_count));
            std::vector<std::vector<float>> conf_chunks(static_cast<size_t>(chunk_count));
            std::vector<std::vector<int>> class_chunks(static_cast<size_t>(chunk_count));

            cv::parallel_for_(cv::Range(0, chunk_count), [&](const cv::Range& range) {
                for (int chunk = range.start; chunk < range.end; ++chunk) {
                    const int start = static_cast<int>((chunk * N) / chunk_count);
                    const int end = static_cast<int>(((chunk + 1) * N) / chunk_count);
                    parse_range(start, end,
                                boxes_chunks[static_cast<size_t>(chunk)],
                                conf_chunks[static_cast<size_t>(chunk)],
                                class_chunks[static_cast<size_t>(chunk)]);
                }
            });

            size_t total_kept = 0;
            for (int chunk = 0; chunk < chunk_count; ++chunk) {
                total_kept += boxes_chunks[static_cast<size_t>(chunk)].size();
            }

            boxes.reserve(total_kept);
            confidences.reserve(total_kept);
            class_ids.reserve(total_kept);

            for (int chunk = 0; chunk < chunk_count; ++chunk) {
                auto& b = boxes_chunks[static_cast<size_t>(chunk)];
                auto& c = conf_chunks[static_cast<size_t>(chunk)];
                auto& ids = class_chunks[static_cast<size_t>(chunk)];

                boxes.insert(boxes.end(), b.begin(), b.end());
                confidences.insert(confidences.end(), c.begin(), c.end());
                class_ids.insert(class_ids.end(), ids.begin(), ids.end());
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices);

        final_detections.reserve(final_detections.size() + indices.size());
        for (int idx : indices) {
            Detection det;
            det.class_id = class_ids[static_cast<size_t>(idx)];
            det.confidence = confidences[static_cast<size_t>(idx)];
            det.box = boxes[static_cast<size_t>(idx)];
            final_detections.push_back(det);
        }
    }

    return final_detections;
}

std::tuple<float, int, int> preprocess_frame(
    const cv::Mat& frame,
    const axrTensorInfo& info,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& stddev,
    cv::Mat& padded_buffer,
    int8_t* quantized_ptr)
{
    const int height = static_cast<int>(info.dims[1]);
    const int width = static_cast<int>(info.dims[2]);
    const int channels = static_cast<int>(info.dims[3]);

    const auto [y_pad_left, y_pad_right] = info.padding[1];
    const auto [x_pad_left, x_pad_right] = info.padding[2];

    const int unpadded_height = height - static_cast<int>(y_pad_left) - static_cast<int>(y_pad_right);
    const int unpadded_width = width - static_cast<int>(x_pad_left) - static_cast<int>(x_pad_right);

    const float scale = std::min(
        static_cast<float>(unpadded_width) / static_cast<float>(frame.cols),
        static_cast<float>(unpadded_height) / static_cast<float>(frame.rows));

    const int resized_width = static_cast<int>(frame.cols * scale);
    const int resized_height = static_cast<int>(frame.rows * scale);

    const int x_offset = (unpadded_width - resized_width) / 2;
    const int y_offset = (unpadded_height - resized_height) / 2;

    cv::Mat roi = padded_buffer(cv::Rect(x_offset, y_offset, resized_width, resized_height));
    cv::resize(frame, roi, roi.size(), 0, 0, cv::INTER_LINEAR);

    if (y_offset > 0) {
        padded_buffer.rowRange(0, y_offset).setTo(cv::Scalar(0, 0, 0));
    }
    if (y_offset + resized_height < height) {
        padded_buffer.rowRange(y_offset + resized_height, height).setTo(cv::Scalar(0, 0, 0));
    }
    if (x_offset > 0) {
        padded_buffer.colRange(0, x_offset).setTo(cv::Scalar(0, 0, 0));
    }
    if (x_offset + resized_width < width) {
        padded_buffer.colRange(x_offset + resized_width, width).setTo(cv::Scalar(0, 0, 0));
    }

    const float inv255 = 1.0f / 255.0f;
    const float inv_scale = 1.0f / info.scale;
    float mul[3], add[3];
    for (int c = 0; c < 3; ++c) {
        const float norm_mul = 1.0f / stddev[c];
        const float norm_add = -mean[c] / stddev[c];
        mul[c] = norm_mul * inv_scale * inv255;
        add[c] = norm_add * inv_scale + info.zero_point;
    }

    const int HW = height * width;
    const uint8_t* src = padded_buffer.ptr<uint8_t>();
    for (int i = 0; i < HW; ++i) {
        const int src_idx = i * 3;
        const int dst_idx = i * channels;

        const float r = src[src_idx + 2];
        const float g = src[src_idx + 1];
        const float b = src[src_idx + 0];

        const float q0 = std::clamp(r * mul[0] + add[0], -128.f, 127.f);
        const float q1 = std::clamp(g * mul[1] + add[1], -128.f, 127.f);
        const float q2 = std::clamp(b * mul[2] + add[2], -128.f, 127.f);

        quantized_ptr[dst_idx + 0] = static_cast<int8_t>(std::lrintf(q0));
        quantized_ptr[dst_idx + 1] = static_cast<int8_t>(std::lrintf(q1));
        quantized_ptr[dst_idx + 2] = static_cast<int8_t>(std::lrintf(q2));
    }

    return {scale, x_offset, y_offset};
}

cv::Mat plot_detections(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const std::string& /*box_type*/,
    const std::vector<std::string>& labels,
    const std::string& model_name,
    float scale,
    int x_offset,
    int y_offset)
{
    cv::Mat annotated_frame = frame.clone();

    for (const auto& detection : detections) {
        const int class_id = detection.class_id;
        const float confidence = detection.confidence;
        const cv::Rect2f box = detection.box;

        const std::string label =
            (class_id >= 0 && class_id < static_cast<int>(labels.size()))
                ? labels[static_cast<size_t>(class_id)]
                : "Unknown";

        const float x1 = (box.x - x_offset) / scale;
        const float y1 = (box.y - y_offset) / scale;
        const float x2 = ((box.x + box.width) - x_offset) / scale;
        const float y2 = ((box.y + box.height) - y_offset) / scale;

        const int x1_int = std::max(0, std::min(static_cast<int>(std::round(x1)), frame.cols - 1));
        const int y1_int = std::max(0, std::min(static_cast<int>(std::round(y1)), frame.rows - 1));
        const int x2_int = std::max(0, std::min(static_cast<int>(std::round(x2)), frame.cols - 1));
        const int y2_int = std::max(0, std::min(static_cast<int>(std::round(y2)), frame.rows - 1));

        const cv::Scalar color =
            (confidence >= 0.5f) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(
            annotated_frame,
            cv::Point(x1_int, y1_int),
            cv::Point(x2_int, y2_int),
            color,
            2);

        std::ostringstream text;
        text << label << " " << static_cast<int>(confidence * 100.0f) << "%";
        cv::putText(
            annotated_frame,
            text.str(),
            cv::Point(x1_int, std::max(0, y1_int - 10)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255, 255, 255),
            2);

        const int cx = (x1_int + x2_int) / 2;
        const int cy = (y1_int + y2_int) / 2;
        cv::circle(annotated_frame, cv::Point(cx, cy), 3, cv::Scalar(0, 0, 255), -1);
    }

    cv::putText(
        annotated_frame,
        "Model used: " + model_name,
        cv::Point(10, 25),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(0, 0, 0),
        2);

    return annotated_frame;
}

std_msgs::msg::String create_detection_message(
    const std::vector<Detection>& detections,
    const std::vector<std::string>& labels)
{
    std_msgs::msg::String msg;
    std::ostringstream ss;

    for (const auto& detection : detections) {
        const int class_id = detection.class_id;
        const float confidence = detection.confidence;
        const cv::Rect2f box = detection.box;

        const std::string label =
            (class_id >= 0 && class_id < static_cast<int>(labels.size()))
                ? labels[static_cast<size_t>(class_id)]
                : "Unknown";

        ss << "Detection: " << label << " (" << confidence * 100.0f << "%) at ("
           << box.x << ", " << box.y << ", " << box.width << ", " << box.height << ")\n";
    }

    msg.data = ss.str();
    return msg;
}

auto read_labels(const std::string& path) -> std::vector<std::string>
{
    std::vector<std::string> labels;

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + path);
    }

    nlohmann::json j;
    file >> j;

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

void logger(void* arg, axrLogLevel level, const char* msg)
{
    (void)arg;
    (void)level;
    (void)msg;
}

class AxeleraYoloInference : public rclcpp::Node
{
    // Track stabilization timing metrics
    std::unordered_map<int, std::chrono::steady_clock::time_point> track_first_detected_time_;
    std::deque<double> track_stabilization_times_ms_;
    double cum_sum_track_stabilization_ms_ = 0.0;
    size_t cum_count_track_stabilization_ = 0;
    static constexpr size_t TRACK_STABILIZATION_WINDOW = 50;
public:
    AxeleraYoloInference()
        : Node("axelera_yolo_inference"),
          env_(ORT_LOGGING_LEVEL_WARNING, "axelera_yolo_inference")
    {
        RCLCPP_INFO(this->get_logger(), "Starting Axelera YOLO Inference Node...");

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
        this->declare_parameter("crop_ratio", 0.2);
        this->declare_parameter("max_bbox_ratio", 0.9);
        this->declare_parameter("stable_frames_required", 10);
        this->declare_parameter("variance_threshold", 15.0);
        this->declare_parameter("destabilize_threshold", 50.0);
        this->declare_parameter("false_detection_frames", 10);
        this->declare_parameter("keep_largest_only", true);
        this->declare_parameter("save_results", true);
        this->declare_parameter("publish_interval", 0.25);
        this->declare_parameter("class_detections_topic", "/detections");
        this->declare_parameter("save_dir", "yolo_results");
        this->declare_parameter("debug", false);
        this->declare_parameter("compute_metrics", true);

        model_name_ = this->get_parameter("model_name").as_string();
        aipu_cores_ = this->get_parameter("aipu_cores").as_int();
        input_topic_ = this->get_parameter("input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        confidence_threshold_ = static_cast<float>(this->get_parameter("confidence_threshold").as_double());
        nms_threshold_ = static_cast<float>(this->get_parameter("nms_threshold").as_double());


        const auto mean_param = this->get_parameter("mean").as_double_array();
        const auto stddev_param = this->get_parameter("stddev").as_double_array();

        mean_.fill(0.0f);
        stddev_.fill(1.0f);
        for (size_t i = 0; i < 3 && i < mean_param.size(); ++i) {
            mean_[i] = static_cast<float>(mean_param[i]);
        }
        for (size_t i = 0; i < 3 && i < stddev_param.size(); ++i) {
            stddev_[i] = static_cast<float>(stddev_param[i]);
        }

        const std::string annotated_topic_param = this->get_parameter("annotated_topic").as_string();
        crop_ratio_ = static_cast<float>(this->get_parameter("crop_ratio").as_double());
        max_bbox_ratio_ = static_cast<float>(this->get_parameter("max_bbox_ratio").as_double());
        stable_frames_required_ = this->get_parameter("stable_frames_required").as_int();
        variance_threshold_ = static_cast<float>(this->get_parameter("variance_threshold").as_double());
        destabilize_threshold_ = static_cast<float>(this->get_parameter("destabilize_threshold").as_double());
        false_detection_frames_ = this->get_parameter("false_detection_frames").as_int();
        keep_largest_only_ = this->get_parameter("keep_largest_only").as_bool();
        save_results_ = this->get_parameter("save_results").as_bool();
        publish_interval_ = static_cast<float>(this->get_parameter("publish_interval").as_double());
        class_detections_topic_ = this->get_parameter("class_detections_topic").as_string();
        save_dir_ = this->get_parameter("save_dir").as_string();
        publish_annotated_image_ = this->get_parameter("publish_annotated_image").as_bool();
        publish_detection_strings_ = this->get_parameter("publish_detection_strings").as_bool();
        publish_class_id_ = this->get_parameter("publish_class_id").as_bool();
        num_classes_ = static_cast<int>(this->get_parameter("num_classes").as_int());
        has_objectness_ = this->get_parameter("has_objectness").as_bool();
        box_type_ = this->get_parameter("box_type").as_string();
        debug_ = this->get_parameter("debug").as_bool();
        compute_metrics_ = this->get_parameter("compute_metrics").as_bool();

        if (debug_) {
            RCLCPP_INFO(this->get_logger(), "Parameters loaded:");
            RCLCPP_INFO(this->get_logger(), "  Model name: %s", model_name_.c_str());
            RCLCPP_INFO(this->get_logger(), "  AIPU cores: %d", aipu_cores_);
            RCLCPP_INFO(this->get_logger(), "  Input topic: %s", input_topic_.c_str());
            RCLCPP_INFO(this->get_logger(), "  Output topic: %s", output_topic_.c_str());
            RCLCPP_INFO(this->get_logger(), "  Confidence threshold: %.2f", confidence_threshold_);
            RCLCPP_INFO(this->get_logger(), "  NMS threshold: %.2f", nms_threshold_);
        }

        const std::string model_info_path =
            "../build/" + model_name_ + "/" + model_name_ + "/model_info.json";
        labels_ = read_labels(model_info_path);
        num_classes_ = static_cast<int>(labels_.size());

        class_id_map_.clear();
        class_id_map_.reserve(static_cast<size_t>(num_classes_));
        for (int i = 0; i < num_classes_; ++i) {
            class_id_map_[i] = labels_[static_cast<size_t>(i)];
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

        const std::string annotated_topic =
            annotated_topic_param.empty() ? input_topic_ + "_annotated" : annotated_topic_param;

        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(annotated_topic, 10);
        detection_pub_ = this->create_publisher<std_msgs::msg::String>(output_topic_, 10);
        class_pub_ = this->create_publisher<std_msgs::msg::Int32>(class_detections_topic_, 10);
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            input_topic_,
            10,
            std::bind(&AxeleraYoloInference::image_callback, this, std::placeholders::_1));

        ctx_ = axr::to_ptr(axr_create_context());
        if (!ctx_) {
            throw std::runtime_error("Failed to create runtime context");
        }
        axr_set_logger(ctx_.get(), AXR_LOG_WARNING, logger, nullptr);

        const std::string model_json_path =
            "../build/" + model_name_ + "/" + model_name_ + "/1/model.json";
        model_ = axr_load_model(ctx_.get(), model_json_path.c_str());
        if (!model_) {
            throw std::runtime_error("Failed to load model from path: " + model_json_path);
        }

        const auto inputs = axr_num_model_inputs(model_);
        for (size_t n = 0; n < inputs; ++n) {
            input_infos_.push_back(axr_get_model_input(model_, n));
        }

        const auto outputs = axr_num_model_outputs(model_);
        for (size_t n = 0; n < outputs; ++n) {
            output_infos_.push_back(axr_get_model_output(model_, n));
        }

        const auto batch_size = input_infos_[0].dims[0];

        connection_ = axr_device_connect(ctx_.get(), nullptr, batch_size, nullptr);
        if (!connection_) {
            throw std::runtime_error("Failed to connect to device");
        }

        const auto props =
            "input_dmabuf=0;num_sub_devices=" + std::to_string(batch_size) +
            ";aipu_cores=" + std::to_string(aipu_cores_);
        auto properties = axr_create_properties(ctx_.get(), props.c_str());

        instance_ = axr_load_model_instance(connection_, model_, properties);
        if (!instance_) {
            throw std::runtime_error("Failed to create model instance");
        }

        input_args_.resize(inputs);
        output_args_.resize(outputs);
        input_data_.resize(inputs);
        output_data_.resize(outputs);

        for (int n = 0; n < static_cast<int>(inputs); ++n) {
            input_data_[static_cast<size_t>(n)] =
                std::make_unique<std::int8_t[]>(axr_tensor_size(&input_infos_[static_cast<size_t>(n)]));
            input_args_[static_cast<size_t>(n)].ptr = input_data_[static_cast<size_t>(n)].get();
            input_args_[static_cast<size_t>(n)].fd = 0;
            input_args_[static_cast<size_t>(n)].offset = 0;
        }

        for (int n = 0; n < static_cast<int>(outputs); ++n) {
            output_data_[static_cast<size_t>(n)] =
                std::make_unique<std::int8_t[]>(axr_tensor_size(&output_infos_[static_cast<size_t>(n)]));
            output_args_[static_cast<size_t>(n)].ptr = output_data_[static_cast<size_t>(n)].get();
            output_args_[static_cast<size_t>(n)].fd = 0;
            output_args_[static_cast<size_t>(n)].offset = 0;
        }

        onnx_model_path_ =
            "../build/" + model_name_ + "/" + model_name_ + "/1/postprocess_graph.onnx";

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        onnx_session_ =
            std::make_unique<Ort::Session>(env_, onnx_model_path_.c_str(), session_options);
        allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();

        input_names_.clear();
        output_names_.clear();
        owned_input_names_.clear();
        owned_output_names_.clear();

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

        normalization_mul_[0] = 1.0f / stddev_[0];
        normalization_mul_[1] = 1.0f / stddev_[1];
        normalization_mul_[2] = 1.0f / stddev_[2];
        normalization_add_[0] = -mean_[0] / stddev_[0];
        normalization_add_[1] = -mean_[1] / stddev_[1];
        normalization_add_[2] = -mean_[2] / stddev_[2];

        input_scale_ = input_info.scale;
        input_zero_point_ = input_info.zero_point;
        quantized_buffer_size_ = input_height_ * input_width_ * input_channels_;

        padded_buffer_ = cv::Mat(
            static_cast<int>(input_height_),
            static_cast<int>(input_width_),
            CV_8UC3);
        quantized_buffer_.resize(quantized_buffer_size_);

        const auto& output_info = output_infos_[0];
        output_unpadded_shape_ =
            compute_unpadded_shape(output_info.dims, output_info.padding, output_info.ndims);
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

        all_output_params_.clear();
        all_output_params_.reserve(output_infos_.size());
        for (const auto& oi : output_infos_) {
            OutputParams params;
            params.unpadded_shape = compute_unpadded_shape(oi.dims, oi.padding, oi.ndims);
            params.N = params.unpadded_shape[0];
            params.H = params.unpadded_shape[1];
            params.W = params.unpadded_shape[2];
            params.C = params.unpadded_shape[3];
            params.padded_H = oi.dims[1];
            params.padded_W = oi.dims[2];
            params.padded_C = oi.dims[3];
            params.input_stride_n = params.padded_H * params.padded_W * params.padded_C;
            params.input_stride_h = params.padded_W * params.padded_C;
            params.input_stride_w = params.padded_C;
            params.output_stride_n = params.C * params.H * params.W;
            params.output_stride_c = params.H * params.W;
            params.output_stride_h = params.W;
            params.pad_n = oi.padding[0][0];
            params.pad_h = oi.padding[1][0];
            params.pad_w = oi.padding[2][0];
            params.pad_c = oi.padding[3][0];
            params.scale = oi.scale;
            params.zero_point_f = static_cast<float>(oi.zero_point);
            params.result_size = params.N * params.C * params.H * params.W;
            all_output_params_.push_back(params);
        }

        dequant_buffers_.resize(all_output_params_.size());
        for (size_t i = 0; i < all_output_params_.size(); ++i) {
            dequant_buffers_[i].resize(all_output_params_[i].result_size);
        }

        RCLCPP_INFO(this->get_logger(), "Axelera YOLO Inference Node initialized successfully.");

        if (compute_metrics_) {
            metrics_timer_ = this->create_wall_timer(
                std::chrono::seconds(5),
                std::bind(&AxeleraYoloInference::print_metrics, this));
            RCLCPP_INFO(this->get_logger(), "[METRICS] Performance metrics enabled (printed every 5 s).");

            std::filesystem::create_directories(save_dir_);
            const auto now_sys = std::chrono::system_clock::now();
            const std::time_t now_t = std::chrono::system_clock::to_time_t(now_sys);
            char ts_buf[32];
            std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d_%H-%M-%S", std::localtime(&now_t));
            const std::string log_path = save_dir_ + "/metrics_inference_" + ts_buf + ".txt";
            metrics_log_.open(log_path, std::ios::out | std::ios::trunc);
            if (metrics_log_.is_open()) {
                RCLCPP_INFO(this->get_logger(), "[METRICS] Logging to file: %s", log_path.c_str());
            } else {
                RCLCPP_WARN(this->get_logger(), "[METRICS] Could not open log file: %s", log_path.c_str());
            }
        }
    }

    ~AxeleraYoloInference() override
    {
        if (compute_metrics_) {
            RCLCPP_INFO(this->get_logger(), "[METRICS] === Final Metrics on Shutdown ===");
            print_metrics();
            if (metrics_log_.is_open()) {
                metrics_log_.flush();
                metrics_log_.close();
            }
        }

        instance_ = nullptr;
        connection_ = nullptr;
        model_ = nullptr;
        ctx_ = nullptr;
    }

private:
    enum class TrackState {
        Detecting,
        Stable,
        Destabilized
    };

    using TrackEntry = std::tuple<int, cv::Vec4f, int, float>;
    using TrackHistory = std::deque<TrackEntry>;

    struct DetectionProj {
        Detection det;
        float x1_c, y1_c, x2_c, y2_c;
        float w_c, h_c, xc_c, yc_c;
    };

    struct OutputParams {
        std::vector<size_t> unpadded_shape;
        size_t N{}, H{}, W{}, C{};
        size_t padded_H{}, padded_W{}, padded_C{};
        size_t input_stride_n{}, input_stride_h{}, input_stride_w{};
        size_t output_stride_n{}, output_stride_c{}, output_stride_h{};
        size_t pad_n{}, pad_h{}, pad_w{}, pad_c{};
        float scale{}, zero_point_f{};
        size_t result_size{};
    };

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

    std::vector<std::string> labels_;
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
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr class_pub_;

    std::string model_name_;
    int aipu_cores_{};
    std::string input_topic_;
    std::string output_topic_;
    float confidence_threshold_{};
    float nms_threshold_{};
    std::array<float, 3> mean_{};
    std::array<float, 3> stddev_{};

    size_t input_height_ = 0;
    size_t input_width_ = 0;
    size_t input_channels_ = 0;
    size_t y_pad_left_ = 0, y_pad_right_ = 0;
    size_t x_pad_left_ = 0, x_pad_right_ = 0;
    size_t unpadded_height_ = 0, unpadded_width_ = 0;
    std::array<float, 3> normalization_mul_{};
    std::array<float, 3> normalization_add_{};
    float input_scale_ = 0.0f;
    float input_zero_point_ = 0.0f;
    size_t quantized_buffer_size_ = 0;

    cv::Mat padded_buffer_;
    std::vector<std::int8_t> quantized_buffer_;

    std::vector<size_t> output_unpadded_shape_;
    size_t output_N_ = 0, output_H_ = 0, output_W_ = 0, output_C_ = 0;
    size_t output_padded_H_ = 0, output_padded_W_ = 0, output_padded_C_ = 0;
    size_t output_input_stride_n_ = 0, output_input_stride_h_ = 0, output_input_stride_w_ = 0;
    size_t output_output_stride_n_ = 0, output_output_stride_c_ = 0, output_output_stride_h_ = 0;
    size_t output_pad_n_ = 0, output_pad_h_ = 0, output_pad_w_ = 0, output_pad_c_ = 0;
    float output_scale_ = 0.0f, output_zero_point_f_ = 0.0f;
    size_t output_result_size_ = 0;

    std::vector<OutputParams> all_output_params_;
    std::vector<std::vector<float>> dequant_buffers_;

    int num_classes_{};
    bool has_objectness_{};
    std::string box_type_;

    float crop_ratio_{};
    float max_bbox_ratio_{};
    int stable_frames_required_{};
    float variance_threshold_{};
    float destabilize_threshold_{};
    bool keep_largest_only_{};
    bool save_results_{};
    float publish_interval_{};
    std::string class_detections_topic_;
    std::string save_dir_;
    bool publish_annotated_image_{};
    bool publish_detection_strings_{};
    bool publish_class_id_{};
    bool debug_ = false;

    std::unordered_map<int, std::string> class_id_map_;

    int frame_count_ = 0;
    int next_object_id_ = 0;
    float iou_threshold_tracking_ = 0.5f;

    std::vector<Detection> detections_buf_;
    std::vector<DetectionProj> projected_detections_buf_;
    std::vector<std::pair<cv::Vec4f, int>> filtered_detections_buf_;
    std::unordered_set<int> matched_tracks_buf_;
    std::vector<int> tracks_to_remove_buf_;
    std::vector<int> tracks_to_publish_buf_;
    std::vector<int> tracks_to_republish_buf_;

    std::unordered_map<int, TrackHistory> detection_tracks_;
    std::unordered_map<int, TrackState> object_states_;
    std::unordered_map<int, int> track_absent_frames_;
    std::unordered_map<int, cv::Vec4f> last_stable_bbox_;
    std::unordered_map<int, int> track_streak_;
    std::unordered_map<int, cv::Vec2f> track_avg_center_;
    std::unordered_map<int, int> track_disruption_;
    std::unordered_map<int, std::pair<int, std::chrono::steady_clock::time_point>> last_published_objects_;

    int false_detection_frames_{};

    bool compute_metrics_ = false;

    uint64_t metric_frames_received_ = 0;
    uint64_t metric_frames_processed_ = 0;
    uint64_t metric_detections_pub_ = 0;
    uint64_t metric_republishes_ = 0;
    uint64_t metric_tracks_created_ = 0;
    uint64_t metric_tracks_stabilized_ = 0;

    static constexpr size_t METRIC_WINDOW = 100;
    std::deque<double> lat_callback_queue_;
    std::deque<double> lat_preprocess_;
    std::deque<double> lat_aipu_;
    std::deque<double> lat_dequant_;
    std::deque<double> lat_onnx_;
    std::deque<double> lat_tracking_;
    std::deque<double> lat_end_to_end_;
    std::deque<double> lat_pub_interval_;

    // Cumulative stats for metrics
    double cum_sum_callback_queue_ = 0.0;
    double cum_sum_preprocess_ = 0.0;
    double cum_sum_aipu_ = 0.0;
    double cum_sum_dequant_ = 0.0;
    double cum_sum_onnx_ = 0.0;
    double cum_sum_tracking_ = 0.0;
    double cum_sum_end_to_end_ = 0.0;
    double cum_sum_pub_interval_ = 0.0;
    size_t cum_count_ = 0;

    static constexpr size_t METRIC_FPS_WINDOW = 60;
    std::deque<std::chrono::steady_clock::time_point> fps_window_;

    std::chrono::steady_clock::time_point metric_callback_t0_;
    rclcpp::TimerBase::SharedPtr metrics_timer_;
    std::ofstream metrics_log_;

    std::vector<std::vector<float>>& process_outputs(
        const std::vector<std::unique_ptr<std::int8_t[]>>& outputs)
    {
        for (size_t i = 0; i < outputs.size(); ++i) {
            const int8_t* __restrict__ raw = outputs[i].get();
            const auto& p = all_output_params_[i];
            float* __restrict__ result_ptr = dequant_buffers_[i].data();

            const float scale = p.scale;
            const float zp = p.zero_point_f;
            const size_t HW = p.H * p.W;
            const size_t C = p.C;

            static constexpr size_t kTileW = 16;

            cv::parallel_for_(
                cv::Range(0, static_cast<int>(p.N * p.H)),
                [&](const cv::Range& range) {
                    alignas(64) float tile[kTileW * 256];

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                    const float32x4_t v_scale = vdupq_n_f32(scale);
                    const float32x4_t v_zp = vdupq_n_f32(zp);
#endif

                    for (int nh = range.start; nh < range.end; ++nh) {
                        const size_t n = static_cast<size_t>(nh) / p.H;
                        const size_t h = static_cast<size_t>(nh) % p.H;

                        const size_t in_row =
                            (n + p.pad_n) * p.input_stride_n +
                            (h + p.pad_h) * p.input_stride_h;
                        const size_t out_n = n * p.output_stride_n;
                        const size_t out_hw = h * p.W;

                        for (size_t w0 = 0; w0 < p.W; w0 += kTileW) {
                            const size_t tw = std::min(kTileW, p.W - w0);

                            for (size_t dw = 0; dw < tw; ++dw) {
                                const int8_t* __restrict__ src =
                                    raw + in_row + (w0 + dw + p.pad_w) * p.input_stride_w + p.pad_c;
                                float* __restrict__ td = tile + dw * C;

                                size_t c = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                                for (; c + 16 <= C; c += 16) {
                                    int8x16_t vi = vld1q_s8(src + c);
                                    int16x8_t lo16 = vmovl_s8(vget_low_s8(vi));
                                    int16x8_t hi16 = vmovl_s8(vget_high_s8(vi));
                                    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                                    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                                    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                                    float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                                    vst1q_f32(td + c, vmulq_f32(vsubq_f32(f0, v_zp), v_scale));
                                    vst1q_f32(td + c + 4, vmulq_f32(vsubq_f32(f1, v_zp), v_scale));
                                    vst1q_f32(td + c + 8, vmulq_f32(vsubq_f32(f2, v_zp), v_scale));
                                    vst1q_f32(td + c + 12, vmulq_f32(vsubq_f32(f3, v_zp), v_scale));
                                }
#endif
                                for (; c < C; ++c) {
                                    td[c] = (static_cast<float>(src[c]) - zp) * scale;
                                }
                            }

                            for (size_t c = 0; c < C; ++c) {
                                float* __restrict__ dst = result_ptr + out_n + c * HW + out_hw + w0;
                                for (size_t dw = 0; dw < tw; ++dw) {
                                    dst[dw] = tile[dw * C + c];
                                }
                            }
                        }
                    }
                });
        }

        return dequant_buffers_;
    }

    void add_to_track(int track_id, TrackEntry entry)
    {
        auto& dq = detection_tracks_[track_id];
        dq.push_back(std::move(entry));
        while (static_cast<int>(dq.size()) > stable_frames_required_) {
            dq.pop_front();
        }
    }

    float compute_iou(const cv::Vec4f& b1, const cv::Vec4f& b2)
    {
        const float x1 = b1[0] - b1[2] / 2.0f;
        const float y1 = b1[1] - b1[3] / 2.0f;
        const float x2 = b1[0] + b1[2] / 2.0f;
        const float y2 = b1[1] + b1[3] / 2.0f;

        const float x3 = b2[0] - b2[2] / 2.0f;
        const float y3 = b2[1] - b2[3] / 2.0f;
        const float x4 = b2[0] + b2[2] / 2.0f;
        const float y4 = b2[1] + b2[3] / 2.0f;

        const float ix = std::max(0.0f, std::min(x2, x4) - std::max(x1, x3));
        const float iy = std::max(0.0f, std::min(y2, y4) - std::max(y1, y3));
        const float inter = ix * iy;
        const float uni = b1[2] * b1[3] + b2[2] * b2[3] - inter;
        return uni <= 0.0f ? 0.0f : inter / uni;
    }

    std::optional<int> match_detection_to_track(const cv::Vec4f& bbox, int class_id)
    {
        float best_iou = 0.0f;
        std::optional<int> best_id;

        for (auto& [tid, history] : detection_tracks_) {
            if (history.empty()) {
                continue;
            }

            auto& [lf, lb, lc, lconf] = history.back();
            (void)lf;
            (void)lconf;

            if (lc != class_id) {
                continue;
            }

            const float iou = compute_iou(bbox, lb);
            if (iou > best_iou && iou >= iou_threshold_tracking_) {
                best_iou = iou;
                best_id = tid;
            }
        }

        return best_id;
    }

    bool update_track_streak(int track_id, const cv::Vec4f& bbox)
    {
        const cv::Vec2f cur = {bbox[0], bbox[1]};
        auto it = track_avg_center_.find(track_id);

        if (it == track_avg_center_.end()) {
            track_avg_center_[track_id] = cur;
            track_streak_[track_id] = 1;
            track_disruption_[track_id] = 0;
            // Record first detection time for this track
            track_first_detected_time_[track_id] = std::chrono::steady_clock::now();
        } else {
            auto& avg = it->second;
            const float dx = cur[0] - avg[0];
            const float dy = cur[1] - avg[1];
            const float dist = std::sqrt(dx * dx + dy * dy);

            if (dist <= variance_threshold_) {
                track_disruption_[track_id] = 0;
                track_streak_[track_id]++;
                avg[0] = avg[0] * 0.8f + cur[0] * 0.2f;
                avg[1] = avg[1] * 0.8f + cur[1] * 0.2f;
            } else {
                track_disruption_[track_id]++;
                if (track_disruption_[track_id] >= false_detection_frames_) {
                    track_streak_[track_id] = 1;
                    track_disruption_[track_id] = 0;
                    track_avg_center_[track_id] = cur;
                    // Reset first detection time for this track
                    track_first_detected_time_[track_id] = std::chrono::steady_clock::now();
                }
            }
        }

        bool stabilized = track_streak_[track_id] >= stable_frames_required_;
        if (stabilized && track_first_detected_time_.count(track_id)) {
            auto now = std::chrono::steady_clock::now();
            auto first_detected = track_first_detected_time_[track_id];
            double ms = std::chrono::duration<double, std::milli>(now - first_detected).count();
            track_stabilization_times_ms_.push_back(ms);
            cum_sum_track_stabilization_ms_ += ms;
            cum_count_track_stabilization_++;
            if (track_stabilization_times_ms_.size() > TRACK_STABILIZATION_WINDOW) {
                track_stabilization_times_ms_.pop_front();
            }
            track_first_detected_time_.erase(track_id);
        }
        return stabilized;
    }

    bool has_object_destabilized(int track_id, const cv::Vec4f& cur)
    {
        auto it = last_stable_bbox_.find(track_id);
        if (it == last_stable_bbox_.end()) {
            return false;
        }

        const auto& last = it->second;
        const float dx = cur[0] - last[0];
        const float dy = cur[1] - last[1];
        const float dist = std::sqrt(dx * dx + dy * dy);
        const float size_change = std::abs(cur[2] - last[2]) + std::abs(cur[3] - last[3]);

        return dist >= destabilize_threshold_ || size_change >= destabilize_threshold_;
    }

    static inline double ms_between(
        const std::chrono::steady_clock::time_point& a,
        const std::chrono::steady_clock::time_point& b)
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

    void metric_push(std::deque<double>& dq, double val)
    {
        dq.push_back(val);
        if (dq.size() > METRIC_WINDOW) {
            dq.pop_front();
        }
    }

    static std::pair<double, double> metric_mean_std(const std::deque<double>& dq)
    {
        if (dq.empty()) {
            return {0.0, 0.0};
        }

        double sum = 0.0;
        for (double v : dq) {
            sum += v;
        }
        const double mean = sum / static_cast<double>(dq.size());

        double var = 0.0;
        for (double v : dq) {
            var += (v - mean) * (v - mean);
        }

        return {mean, std::sqrt(var / static_cast<double>(dq.size()))};
    }

    void print_metrics()
    {
        if (!compute_metrics_) {
            return;
        }

        const auto [cq_m, cq_s] = metric_mean_std(lat_callback_queue_);
        const auto [pre_m, pre_s] = metric_mean_std(lat_preprocess_);
        const auto [ai_m, ai_s] = metric_mean_std(lat_aipu_);
        const auto [dq_m, dq_s] = metric_mean_std(lat_dequant_);
        const auto [on_m, on_s] = metric_mean_std(lat_onnx_);
        const auto [tr_m, tr_s] = metric_mean_std(lat_tracking_);
        const auto [e2_m, e2_s] = metric_mean_std(lat_end_to_end_);
        const auto [pi_m, pi_s] = metric_mean_std(lat_pub_interval_);

        // Cumulative averages
        double cum_cq = cum_count_ ? cum_sum_callback_queue_ / cum_count_ : 0.0;
        double cum_pre = cum_count_ ? cum_sum_preprocess_ / cum_count_ : 0.0;
        double cum_ai = cum_count_ ? cum_sum_aipu_ / cum_count_ : 0.0;
        double cum_dq = cum_count_ ? cum_sum_dequant_ / cum_count_ : 0.0;
        double cum_on = cum_count_ ? cum_sum_onnx_ / cum_count_ : 0.0;
        double cum_tr = cum_count_ ? cum_sum_tracking_ / cum_count_ : 0.0;
        double cum_e2 = cum_count_ ? cum_sum_end_to_end_ / cum_count_ : 0.0;
        double cum_pi = cum_count_ ? cum_sum_pub_interval_ / cum_count_ : 0.0;

        double fps = 0.0;
        if (fps_window_.size() >= 2) {
            const double w = ms_between(fps_window_.front(), fps_window_.back()) / 1000.0;
            if (w > 0.0) {
                fps = static_cast<double>(fps_window_.size() - 1) / w;
            }
        }

        double stability_rate = 0.0;
        if (metric_tracks_created_ > 0) {
            stability_rate =
                100.0 * static_cast<double>(metric_tracks_stabilized_) /
                static_cast<double>(metric_tracks_created_);
        }

        // Track stabilization metrics
        double mean_stabilization = 0.0;
        double std_stabilization = 0.0;
        if (!track_stabilization_times_ms_.empty()) {
            double sum = 0.0;
            for (double ms : track_stabilization_times_ms_) sum += ms;
            mean_stabilization = sum / track_stabilization_times_ms_.size();
            double sq_sum = 0.0;
            for (double ms : track_stabilization_times_ms_) sq_sum += (ms - mean_stabilization) * (ms - mean_stabilization);
            std_stabilization = std::sqrt(sq_sum / track_stabilization_times_ms_.size());
        }
        double cum_mean_stabilization = cum_count_track_stabilization_ > 0 ? cum_sum_track_stabilization_ms_ / cum_count_track_stabilization_ : 0.0;

        char buf[4096];
        std::snprintf(
            buf,
            sizeof(buf),
            "\n========== Inference Node Metrics =========="
            "\n  Frames received         : %lu"
            "\n  Frames processed        : %lu"
            "\n  Inference FPS           : %.2f fps  (last %zu-frame window)"
            "\n  --- Stage Latencies (mean +/- std ms, last %zu frames) ---"
            "\n  Callback queue delay    : %7.2f +/- %.2f ms"
            "\n  Preprocessing           : %7.2f +/- %.2f ms"
            "\n  AIPU inference          : %7.2f +/- %.2f ms"
            "\n  Dequantisation          : %7.2f +/- %.2f ms"
            "\n  ONNX postprocess        : %7.2f +/- %.2f ms"
            "\n  Tracking state machine  : %7.2f +/- %.2f ms"
            "\n  End-to-end (callback)   : %7.2f +/- %.2f ms"
            "\n  --- Cumulative Averages (all frames) ---"
            "\n  Callback queue delay    : %7.2f ms"
            "\n  Preprocessing           : %7.2f ms"
            "\n  AIPU inference          : %7.2f ms"
            "\n  Dequantisation          : %7.2f ms"
            "\n  ONNX postprocess        : %7.2f ms"
            "\n  Tracking state machine  : %7.2f ms"
            "\n  End-to-end (callback)   : %7.2f ms"
            "\n  --- Publish Stats ---"
            "\n  Int32 detections pub'd  : %lu"
            "\n  Int32 re-publishes      : %lu"
            "\n  Re-pub interval jitter  : %7.2f +/- %.2f ms  (target %.0f ms)"
            "\n  --- Track Stats ---"
            "\n  Tracks created          : %lu"
            "\n  Tracks stabilised       : %lu"
            "\n  Stability rate          : %.1f%%"
            "\n  Track stabilization time (ms): %7.2f +/- %.2f (rolling last %zu)"
            "\n  Track stabilization time (ms): %7.2f (cumulative)"
            "\n============================================",
            metric_frames_received_,
            metric_frames_processed_,
            fps,
            fps_window_.size(),
            METRIC_WINDOW,
            cq_m, cq_s,
            pre_m, pre_s,
            ai_m, ai_s,
            dq_m, dq_s,
            on_m, on_s,
            tr_m, tr_s,
            e2_m, e2_s,
            cum_cq,
            cum_pre,
            cum_ai,
            cum_dq,
            cum_on,
            cum_tr,
            cum_e2,
            metric_detections_pub_,
            metric_republishes_,
            pi_m, pi_s, publish_interval_ * 1000.0f,
            metric_tracks_created_,
            metric_tracks_stabilized_,
            stability_rate,
            mean_stabilization, std_stabilization, TRACK_STABILIZATION_WINDOW,
            cum_mean_stabilization);

        RCLCPP_INFO(this->get_logger(), "%s", buf);

        if (metrics_log_.is_open()) {
            const std::time_t now_t =
                std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            char ts[32];
            std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&now_t));
            metrics_log_ << "[" << ts << "]\n" << buf << "\n\n";
            metrics_log_.flush();
        }
    }

    std::tuple<float, int, int> preprocess_frame(
        const cv::Mat& frame,
        const axrTensorInfo& info,
        const std::array<float, 3>& mean,
        const std::array<float, 3>& stddev,
        cv::Mat& padded_buffer,
        int8_t* quantized_ptr)
    {
        return ::preprocess_frame(frame, info, mean, stddev, padded_buffer, quantized_ptr);
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (compute_metrics_) {
            metric_frames_received_++;
            rclcpp::Time stamp(msg->header.stamp);
            if (stamp.nanoseconds() > 0) {
                const double qd = (this->now() - stamp).seconds() * 1000.0;
                metric_push(lat_callback_queue_, qd);
            }
            metric_callback_t0_ = std::chrono::steady_clock::now();
        }

        try {
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            process_image(frame);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void process_image(const cv::Mat& frame)
    {
        frame_count_++;

        if (compute_metrics_) {
            metric_frames_processed_++;
            const auto now_fp = std::chrono::steady_clock::now();
            fps_window_.push_back(now_fp);
            if (fps_window_.size() > METRIC_FPS_WINDOW) {
                fps_window_.pop_front();
            }
            // Increment cumulative count
            cum_count_++;
        }

        const auto t_proc_start = std::chrono::steady_clock::now();

        const int crop_x = static_cast<int>(frame.cols * crop_ratio_);
        const int crop_y = static_cast<int>(frame.rows * crop_ratio_);
        const cv::Rect crop_roi(
            crop_x,
            crop_y,
            frame.cols - 2 * crop_x,
            frame.rows - 2 * crop_y);
        const cv::Mat cropped_frame = frame(crop_roi);

        const auto t0 = std::chrono::steady_clock::now();

        auto [scale, x_offset, y_offset] = preprocess_frame(
            cropped_frame,
            input_infos_[0],
            mean_,
            stddev_,
            padded_buffer_,
            reinterpret_cast<int8_t*>(input_args_[0].ptr));

        const auto t1 = std::chrono::steady_clock::now();

        if (axr_run_model_instance(
                instance_,
                input_args_.data(), input_args_.size(),
                output_args_.data(), output_args_.size()) != AXR_SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Failed to run model instance");
            return;
        }

        const auto t2 = std::chrono::steady_clock::now();
        auto& dequantized_outputs = process_outputs(output_data_);

        const auto t3 = std::chrono::steady_clock::now();

        detections_buf_ = postprocess_model_output(
            *onnx_session_,
            *allocator_,
            input_names_,
            output_names_,
            dequantized_outputs,
            confidence_threshold_,
            nms_threshold_,
            num_classes_,
            has_objectness_,
            this->get_logger());

        if (debug_) {
            RCLCPP_INFO(this->get_logger(), "Number of detections: %zu", detections_buf_.size());
        }

        const auto t4 = std::chrono::steady_clock::now();

        projected_detections_buf_.clear();
        projected_detections_buf_.reserve(detections_buf_.size());

        const float cropped_w = static_cast<float>(cropped_frame.cols);
        const float cropped_h = static_cast<float>(cropped_frame.rows);
        const float cropped_area = cropped_w * cropped_h;
        const float max_area = cropped_area * max_bbox_ratio_;

        for (const auto& det : detections_buf_) {
            const float x1_c = (det.box.x - x_offset) / scale;
            const float y1_c = (det.box.y - y_offset) / scale;
            const float x2_c = ((det.box.x + det.box.width) - x_offset) / scale;
            const float y2_c = ((det.box.y + det.box.height) - y_offset) / scale;

            const float w_c = x2_c - x1_c;
            const float h_c = y2_c - y1_c;
            if (w_c <= 0.0f || h_c <= 0.0f) {
                continue;
            }

            const float area = w_c * h_c;
            if (area > max_area) {
                if (debug_) {
                    RCLCPP_WARN(
                        this->get_logger(),
                        "Filtered large bbox: %.1f%% of cropped image (cls=%d)",
                        area / cropped_area * 100.0f,
                        det.class_id);
                }
                continue;
            }

            const float xc_c = x1_c + w_c * 0.5f;
            const float yc_c = y1_c + h_c * 0.5f;

            projected_detections_buf_.push_back(
                DetectionProj{det, x1_c, y1_c, x2_c, y2_c, w_c, h_c, xc_c, yc_c});
        }

        detections_buf_.clear();
        detections_buf_.reserve(projected_detections_buf_.size());
        for (const auto& p : projected_detections_buf_) {
            detections_buf_.push_back(p.det);
        }

        if (detections_buf_.size() > 1) {
            auto best = std::max_element(
                detections_buf_.begin(),
                detections_buf_.end(),
                [](const Detection& a, const Detection& b) {
                    return a.confidence < b.confidence;
                });

            const Detection best_det = *best;
            detections_buf_.assign(1, best_det);

            auto best_proj = std::find_if(
                projected_detections_buf_.begin(),
                projected_detections_buf_.end(),
                [&](const DetectionProj& p) {
                    return p.det.class_id == best_det.class_id &&
                           p.det.confidence == best_det.confidence &&
                           p.det.box == best_det.box;
                });

            if (best_proj != projected_detections_buf_.end()) {
                const DetectionProj keep = *best_proj;
                projected_detections_buf_.assign(1, keep);
            } else {
                projected_detections_buf_.clear();
            }
        }

        TrackState display_state = TrackState::Detecting;
        for (const auto& [tid, state] : object_states_) {
            if (state == TrackState::Stable) {
                auto it_abs = track_absent_frames_.find(tid);
                const bool absent =
                    (it_abs != track_absent_frames_.end() && it_abs->second > 0);
                display_state = absent ? TrackState::Destabilized : TrackState::Stable;
                break;
            }
        }

        static const std::vector<Detection> empty_detections;
        const std::vector<Detection>& display_detections =
            (display_state == TrackState::Stable) ? detections_buf_ : empty_detections;

        const bool need_annotated_frame = publish_annotated_image_ || save_results_;
        cv::Mat annotated_frame;

        if (need_annotated_frame) {
            annotated_frame = plot_detections(
                cropped_frame.clone(),
                display_detections,
                box_type_,
                labels_,
                model_name_,
                scale,
                x_offset,
                y_offset);

            const char* status_text = "DETECTING";
            cv::Scalar status_color(255, 255, 255);

            if (display_state == TrackState::Stable) {
                status_text = "STABLE";
                status_color = cv::Scalar(0, 255, 0);
            } else if (display_state == TrackState::Destabilized) {
                status_text = "DESTABILIZING";
                status_color = cv::Scalar(0, 140, 255);
            }

            int baseline = 0;
            const cv::Size text_size = cv::getTextSize(
                status_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
            const int tx = annotated_frame.cols - text_size.width - 10;

            cv::putText(
                annotated_frame,
                status_text,
                cv::Point(tx + 1, 31),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 0, 0),
                3);
            cv::putText(
                annotated_frame,
                status_text,
                cv::Point(tx, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                2);

            if (display_state == TrackState::Stable && !display_detections.empty()) {
                const int cls = display_detections[0].class_id;
                auto it_cls = class_id_map_.find(cls);
                const std::string cls_name =
                    (it_cls != class_id_map_.end()) ? it_cls->second : "unknown";

                const std::string det_text = "Detected: " + cls_name;
                const int y_base = annotated_frame.rows - 15;
                cv::putText(
                    annotated_frame,
                    det_text,
                    cv::Point(10, y_base),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 255),
                    1);
            }
        }

        if (publish_annotated_image_ && !annotated_frame.empty()) {
            auto msg_out =
                cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", annotated_frame).toImageMsg();
            msg_out->header.stamp = this->now();
            msg_out->header.frame_id = "camera_frame";
            image_pub_->publish(*msg_out);
        }

        if (publish_detection_strings_ && !detections_buf_.empty()) {
            auto detection_msg = create_detection_message(detections_buf_, labels_);
            detection_pub_->publish(detection_msg);
        }

        filtered_detections_buf_.clear();
        filtered_detections_buf_.reserve(projected_detections_buf_.size());

        for (const auto& p : projected_detections_buf_) {
            filtered_detections_buf_.push_back({
                {p.xc_c, p.yc_c, p.w_c, p.h_c},
                p.det.class_id
            });
        }

        if (keep_largest_only_ && filtered_detections_buf_.size() > 1) {
            auto largest = std::max_element(
                filtered_detections_buf_.begin(),
                filtered_detections_buf_.end(),
                [](const auto& a, const auto& b) {
                    return (a.first[2] * a.first[3]) < (b.first[2] * b.first[3]);
                });
            const auto keep = *largest;
            filtered_detections_buf_.assign(1, keep);
        }

        matched_tracks_buf_.clear();

        for (const auto& [bbox, cls_id] : filtered_detections_buf_) {
            const auto track_id_opt = match_detection_to_track(bbox, cls_id);
            const bool is_new_track = !track_id_opt.has_value();
            const int track_id = is_new_track ? next_object_id_++ : track_id_opt.value();

            if (compute_metrics_ && is_new_track) {
                metric_tracks_created_++;
            }

            add_to_track(track_id, TrackEntry{frame_count_, bbox, cls_id, 0.0f});
            matched_tracks_buf_.insert(track_id);
        }

        tracks_to_remove_buf_.clear();
        tracks_to_remove_buf_.reserve(detection_tracks_.size());

        for (auto& [tid, hist] : detection_tracks_) {
            if (!hist.empty() &&
                frame_count_ - std::get<0>(hist.back()) > 60) {
                tracks_to_remove_buf_.push_back(tid);
            }
        }

        for (int tid : tracks_to_remove_buf_) {
            detection_tracks_.erase(tid);
            object_states_.erase(tid);
            last_stable_bbox_.erase(tid);
            track_streak_.erase(tid);
            track_avg_center_.erase(tid);
            track_disruption_.erase(tid);
            track_absent_frames_.erase(tid);
            last_published_objects_.erase(tid);
        }

        for (auto& [tid, state] : object_states_) {
            const bool matched = (matched_tracks_buf_.find(tid) != matched_tracks_buf_.end());

            if (state == TrackState::Stable && !matched) {
                int& absent_frames = track_absent_frames_[tid];
                absent_frames++;

                if (absent_frames < false_detection_frames_) {
                    if (debug_) {
                        RCLCPP_INFO(
                            this->get_logger(),
                            "[ABSENT] Track #%d missing for %d/%d frames - holding STABLE",
                            tid, absent_frames, false_detection_frames_);
                    }
                    continue;
                }

                absent_frames = 0;
                state = TrackState::Destabilized;
                detection_tracks_[tid].clear();
                track_streak_[tid] = 0;
                track_disruption_[tid] = 0;
                track_avg_center_.erase(tid);

                if (debug_) {
                    RCLCPP_INFO(this->get_logger(), "[DESTABILIZED] Track #%d disappeared", tid);
                }

                auto it_pub = last_published_objects_.find(tid);
                if (it_pub != last_published_objects_.end()) {
                    const int pub_cls = it_pub->second.first;
                    auto it_cls = class_id_map_.find(pub_cls);
                    const std::string cname =
                        (it_cls != class_id_map_.end()) ? it_cls->second : "unknown";

                    last_published_objects_.erase(it_pub);

                    if (debug_) {
                        RCLCPP_INFO(
                            this->get_logger(),
                            "[STOP PUBLISH] Int32: %d (%s) - Track #%d (disappeared)",
                            pub_cls, cname.c_str(), tid);
                    }
                }
            } else if (matched) {
                track_absent_frames_[tid] = 0;
            }
        }

        tracks_to_publish_buf_.clear();
        tracks_to_publish_buf_.reserve(matched_tracks_buf_.size());

        for (int track_id : matched_tracks_buf_) {
            auto& hist = detection_tracks_[track_id];
            auto& [fn, bbox, cls_id, conf] = hist.back();
            (void)fn;
            (void)conf;

            auto [it_state, inserted] =
                object_states_.try_emplace(track_id, TrackState::Detecting);
            (void)inserted;
            TrackState& state = it_state->second;

            if (state == TrackState::Detecting) {
                if (update_track_streak(track_id, bbox)) {
                    state = TrackState::Stable;
                    if (compute_metrics_) {
                        metric_tracks_stabilized_++;
                    }
                    last_stable_bbox_[track_id] = bbox;
                    tracks_to_publish_buf_.push_back(track_id);
                }
            } else if (state == TrackState::Stable) {
                if (has_object_destabilized(track_id, bbox)) {
                    state = TrackState::Destabilized;
                    detection_tracks_[track_id].clear();
                    track_streak_[track_id] = 0;
                    track_disruption_[track_id] = 0;
                    track_avg_center_.erase(track_id);

                    if (debug_) {
                        RCLCPP_INFO(
                            this->get_logger(),
                            "[DESTABILIZED] Track #%d object moved",
                            track_id);
                    }

                    auto it_pub = last_published_objects_.find(track_id);
                    if (it_pub != last_published_objects_.end()) {
                        const int pub_cls = it_pub->second.first;
                        auto it_cls = class_id_map_.find(pub_cls);
                        const std::string cname =
                            (it_cls != class_id_map_.end()) ? it_cls->second : "unknown";

                        last_published_objects_.erase(it_pub);

                        if (debug_) {
                            RCLCPP_INFO(
                                this->get_logger(),
                                "[STOP PUBLISH] Int32: %d (%s) - Track #%d (object moved)",
                                pub_cls, cname.c_str(), track_id);
                        }
                    }
                }
            } else if (state == TrackState::Destabilized) {
                state = TrackState::Detecting;
            }
        }

        const auto now = std::chrono::steady_clock::now();

        for (int track_id : tracks_to_publish_buf_) {
            auto& [fn, bbox, cls_id, conf] = detection_tracks_[track_id].back();
            (void)fn;
            (void)bbox;
            (void)conf;

            int publish_cls_id = cls_id;

            auto it_cls = class_id_map_.find(publish_cls_id);
            const std::string cname =
                (it_cls != class_id_map_.end()) ? it_cls->second : "unknown";

            last_published_objects_.clear();
            last_published_objects_[track_id] = {publish_cls_id, now};

            if (publish_class_id_) {
                std_msgs::msg::Int32 class_msg;
                class_msg.data = publish_cls_id;
                class_pub_->publish(class_msg);
                if (compute_metrics_) {
                    metric_detections_pub_++;
                }
            }

            if (debug_) {
                RCLCPP_INFO(
                    this->get_logger(),
                    "[PUBLISH] Int32: %d (%s) - Track #%d",
                    publish_cls_id, cname.c_str(), track_id);
            }

            if (save_results_ && !annotated_frame.empty()) {
                std::filesystem::create_directories(save_dir_);
                const std::string out_path =
                    save_dir_ + "/frame_" + std::to_string(frame_count_) + ".jpg";
                cv::imwrite(out_path, annotated_frame);
                if (debug_) {
                    RCLCPP_INFO(
                        this->get_logger(),
                        "Saved annotated image: %s",
                        out_path.c_str());
                }
            }
        }

        tracks_to_republish_buf_.clear();
        tracks_to_republish_buf_.reserve(last_published_objects_.size());

        for (auto& [tid, pub_info] : last_published_objects_) {
            auto it_state = object_states_.find(tid);
            if (it_state != object_states_.end() &&
                it_state->second == TrackState::Stable) {
                const double elapsed =
                    std::chrono::duration<double>(now - pub_info.second).count();
                if (elapsed >= publish_interval_) {
                    tracks_to_republish_buf_.push_back(tid);
                }
            }
        }

        for (int track_id : tracks_to_republish_buf_) {
            if (compute_metrics_) {
                const double elapsed = std::chrono::duration<double, std::milli>(
                    now - last_published_objects_[track_id].second).count();
                metric_push(lat_pub_interval_, elapsed);
                metric_republishes_++;
            }

            const int cls_id = last_published_objects_[track_id].first;
            last_published_objects_[track_id].second = now;

            if (publish_class_id_) {
                std_msgs::msg::Int32 class_msg;
                class_msg.data = cls_id;
                class_pub_->publish(class_msg);
            }

            auto it_cls = class_id_map_.find(cls_id);
            const std::string cname =
                (it_cls != class_id_map_.end()) ? it_cls->second : "unknown";

            if (debug_) {
                RCLCPP_INFO(
                    this->get_logger(),
                    "[RE-PUBLISH] Int32: %d (%s) - Track #%d",
                    cls_id, cname.c_str(), track_id);
            }
        }

        if (compute_metrics_) {
            const auto t5 = std::chrono::steady_clock::now();
            metric_push(lat_preprocess_, ms_between(t0, t1));
            metric_push(lat_aipu_, ms_between(t1, t2));
            metric_push(lat_dequant_, ms_between(t2, t3));
            metric_push(lat_onnx_, ms_between(t3, t4));
            metric_push(lat_tracking_, ms_between(t4, t5));
            metric_push(lat_end_to_end_, ms_between(t_proc_start, t5));

            // Update cumulative sums
            cum_sum_callback_queue_ += lat_callback_queue_.empty() ? 0.0 : lat_callback_queue_.back();
            cum_sum_preprocess_ += ms_between(t0, t1);
            cum_sum_aipu_ += ms_between(t1, t2);
            cum_sum_dequant_ += ms_between(t2, t3);
            cum_sum_onnx_ += ms_between(t3, t4);
            cum_sum_tracking_ += ms_between(t4, t5);
            cum_sum_end_to_end_ += ms_between(t_proc_start, t5);
            cum_sum_pub_interval_ += lat_pub_interval_.empty() ? 0.0 : lat_pub_interval_.back();
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AxeleraYoloInference>());
    rclcpp::shutdown();
    return 0;
}
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <std_srvs/srv/set_bool.hpp>
#include <gazebo_msgs/srv/delete_entity.hpp>
#include <gazebo_msgs/srv/spawn_entity.hpp>
#include <std_msgs/msg/int32.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <fstream>
#include <sstream>
#include <atomic>
#include <thread>
#include <deque>
#include <cmath>
#include <ctime>
#include <filesystem>

// ---------------------------------------------------------------
// Pick-and-place performance metrics
// ---------------------------------------------------------------
struct PnpMetrics {
    // Class names (indices 0-3)
    static constexpr const char* CLASS_NAMES[4] = {"glass", "metal", "paper", "plastic"};
    // Bin grouping: 0+1 = green, 2 = red, 3 = yellow
    static constexpr const char* BIN_NAMES[4]   = {"green", "green", "red", "yellow"};

    std::atomic<uint64_t> detections_received{0};
    std::atomic<uint64_t> detections_dropped{0};
    std::atomic<uint64_t> cycles_succeeded{0};
    std::atomic<uint64_t> cycles_failed{0};

    // Per-class counters  [0]=glass [1]=metal [2]=paper [3]=plastic
    std::atomic<uint64_t> class_count[4]     = {};
    std::atomic<uint64_t> class_succeeded[4] = {};
    std::atomic<uint64_t> class_failed[4]    = {};

    std::mutex         mtx;
    std::deque<double> detection_to_start_ms;   // receive -> thread start
    std::deque<double> cycle_duration_s;         // receive -> cycle complete (all classes)
    std::deque<double> class_duration_s[4];      // per-class rolling window
    static constexpr size_t WINDOW = 50;

    // Cumulative running mean per class (Welford-lite: sum + count)
    double  class_dur_sum[4] = {};
    uint64_t class_dur_cnt[4] = {};

    void push(std::deque<double>& dq, double val) {
        std::lock_guard<std::mutex> lk(mtx);
        dq.push_back(val);
        if (dq.size() > WINDOW) dq.pop_front();
    }

    void push_class(int cls, double dur_s) {
        std::lock_guard<std::mutex> lk(mtx);
        if (cls >= 0 && cls < 4) {
            class_duration_s[cls].push_back(dur_s);
            if (class_duration_s[cls].size() > WINDOW) class_duration_s[cls].pop_front();
            class_dur_sum[cls] += dur_s;
            class_dur_cnt[cls]++;
        }
    }

    std::pair<double,double> mean_std_locked(const std::deque<double>& dq) {
        std::lock_guard<std::mutex> lk(mtx);
        return mean_std_unlocked(dq);
    }

    // Call only while holding mtx
    std::pair<double,double> mean_std_unlocked(const std::deque<double>& dq) {
        if (dq.empty()) return {0.0, 0.0};
        double sum = 0.0;
        for (double v : dq) sum += v;
        double mean = sum / static_cast<double>(dq.size());
        double var  = 0.0;
        for (double v : dq) var += (v - mean) * (v - mean);
        return {mean, std::sqrt(var / static_cast<double>(dq.size()))};
    }

    // Cumulative mean for a class (safe to call without lock since doubles
    // are read-only from the print thread and written from the worker thread;
    // minor tearing is acceptable for a thesis metric printout)
    double class_cumulative_mean(int cls) const {
        if (cls < 0 || cls >= 4 || class_dur_cnt[cls] == 0) return 0.0;
        return class_dur_sum[cls] / static_cast<double>(class_dur_cnt[cls]);
    }
};

// Pick and Place Node with YOLO Detector Integration
// ====================================================
// This node subscribes to Int32 messages from the YOLO detector containing
// detected object class IDs and performs corresponding pick-and-place operations.
//
// Topic: /detections (configurable via launch parameter)
// Message Type: std_msgs::msg::Int32
//
// Class ID Mapping (from YOLO detector):
//   0 = glass   -> Place in green bin (60°)
//   1 = metal   -> Place in green bin (60°)
//   2 = paper   -> Place in red bin (120°)
//   3 = plastic -> Place in yellow bin (-60°)

using namespace std::chrono_literals;

bool perform_pick_and_place_cycle(
    std::shared_ptr<rclcpp::Node> node,
    moveit::planning_interface::MoveGroupInterface& move_group_interface,
    int cycle_number,
    int message_value,
    bool debug)
{
  auto const logger = node->get_logger();
  
  // Determine color and position based on YOLO detection class ID
  // Class mapping (matches YOLO detector):
  //   0 = glass   -> green bin at 60°
  //   1 = metal   -> green bin at 60°
  //   2 = paper   -> red bin at 120°
  //   3 = plastic -> yellow bin at -60°
  std::string color;
  std::string model_color;
  double place_joint_angle;
  
  if (message_value == 0 || message_value == 1) {
    color = "Green";
    model_color = "green";
    place_joint_angle = 1.047;  // 60 degrees
  } else if (message_value == 2) {
    color = "Red";
    model_color = "red";
    place_joint_angle = 2.094;  // 120 degrees
  } else { // message_value == 3
    color = "Yellow";
    model_color = "yellow";
    place_joint_angle = -1.047;  // -60 degrees
  }
  
  if (debug) RCLCPP_INFO(logger, "Starting pick and place cycle %d - Color: %s, Number: %d", cycle_number, color.c_str(), message_value);
  
  // Spawn the box
  rclcpp::Client<gazebo_msgs::srv::SpawnEntity>::SharedPtr spawn_client =
    node->create_client<gazebo_msgs::srv::SpawnEntity>("/spawn_entity");
  
  // Read the model SDF file
  std::string model_xml;
  std::string package_path = ament_index_cpp::get_package_share_directory("pick_and_place_simulation");
  std::string model_path = package_path + "/models/box_" + model_color + "/model.sdf";
  std::ifstream model_file(model_path);
  if (model_file.is_open())
  {
    std::stringstream buffer;
    buffer << model_file.rdbuf();
    model_xml = buffer.str();
    model_file.close();
  }
  else
  {
    RCLCPP_ERROR(logger, "Failed to open box model file at: %s", model_path.c_str());
    return false;
  }
  
  auto spawn_request = std::make_shared<gazebo_msgs::srv::SpawnEntity::Request>();
  spawn_request->name = "box_" + std::to_string(cycle_number);
  spawn_request->xml = model_xml;
  spawn_request->initial_pose.position.x = -0.03;
  spawn_request->initial_pose.position.y = 0.54;
  spawn_request->initial_pose.position.z = 0.5;
  spawn_request->initial_pose.orientation.w = 1.0;
  
  if (!spawn_client->wait_for_service(5s)) {
    RCLCPP_ERROR(logger, "Spawn service not available");
    return false;
  }
  
  auto spawn_future = spawn_client->async_send_request(spawn_request);
  if (spawn_future.wait_for(10s) == std::future_status::timeout) {
    RCLCPP_ERROR(logger, "Spawn service call timed out");
    return false;
  }
  
  auto spawn_result = spawn_future.get();
  if (!spawn_result->success) {
    RCLCPP_ERROR(logger, "Failed to spawn box: %s", spawn_result->status_message.c_str());
    return false;
  }
  
  // Wait for box to settle
  rclcpp::sleep_for(2s);
  
  if (debug) RCLCPP_INFO(logger, "Moving to pick position...");
  
  // Pick
  std::vector<double> joint_group_positions_pick = {0.0, 0.802, 0.942, 1.361};
  move_group_interface.setJointValueTarget(joint_group_positions_pick);

  auto const [pick_success, pick_plan] = [&move_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(move_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  if (pick_success)
  {
    move_group_interface.execute(pick_plan);
    if (debug) RCLCPP_INFO(logger, "Pick motion executed");
  }
  else
  {
    RCLCPP_ERROR(logger, "Pick planning failed!");
    return false;
  }

  // Activate vacuum gripper
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr vacuum_client =
    node->create_client<std_srvs::srv::SetBool>("switch");

  auto pick_request = std::make_shared<std_srvs::srv::SetBool::Request>();
  pick_request->data = true;

  if (!vacuum_client->wait_for_service(5s)) {
    RCLCPP_ERROR(logger, "Vacuum service not available");
    return false;
  }

  auto pick_future = vacuum_client->async_send_request(pick_request);
  if (pick_future.wait_for(5s) == std::future_status::timeout) {
    RCLCPP_ERROR(logger, "Vacuum service call timed out");
    return false;
  }
  if (debug) RCLCPP_INFO(logger, "Vacuum activated");

  if (debug) RCLCPP_INFO(logger, "Moving to place position...");

  // Place position based on color: green at 60°, red at 120°, yellow at -60°
  std::vector<double> joint_group_positions_place = {place_joint_angle, 0.802, 0.942, 1.361};
  
  move_group_interface.setJointValueTarget(joint_group_positions_place);

  auto const [place_success, place_plan] = [&move_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(move_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  if (place_success)
  {
    move_group_interface.execute(place_plan);
    if (debug) RCLCPP_INFO(logger, "Place motion executed");
  }
  else
  {
    RCLCPP_ERROR(logger, "Place planning failed!");
    return false;
  }

  // Deactivate vacuum gripper
  auto place_request = std::make_shared<std_srvs::srv::SetBool::Request>();
  place_request->data = false;

  auto place_future = vacuum_client->async_send_request(place_request);
  if (place_future.wait_for(5s) == std::future_status::timeout) {
    RCLCPP_ERROR(logger, "Vacuum deactivate service call timed out");
    return false;
  }
  if (debug) RCLCPP_INFO(logger, "Vacuum deactivated");

  // Delete the box entity from Gazebo
  rclcpp::Client<gazebo_msgs::srv::DeleteEntity>::SharedPtr delete_client =
    node->create_client<gazebo_msgs::srv::DeleteEntity>("/delete_entity");

  auto delete_request = std::make_shared<gazebo_msgs::srv::DeleteEntity::Request>();
  delete_request->name = "box_" + std::to_string(cycle_number);

  if (!delete_client->wait_for_service(5s)) {
    RCLCPP_ERROR(logger, "Delete service not available");
    return false;
  }

  auto delete_future = delete_client->async_send_request(delete_request);
  if (delete_future.wait_for(5s) == std::future_status::timeout) {
    RCLCPP_ERROR(logger, "Delete service call timed out");
    return false;
  }
  
  auto delete_result = delete_future.get();
  if (delete_result->success)
  {
    if (debug) RCLCPP_INFO(logger, "Box deleted successfully for cycle %d", cycle_number);
  }
  else
  {
    RCLCPP_WARN(logger, "Failed to delete box: %s", delete_result->status_message.c_str());
  }
  
  if (debug) RCLCPP_INFO(logger, "Pick and place cycle %d completed successfully!", cycle_number);
  return true;
}

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto const node = std::make_shared<rclcpp::Node>(
      "test_send_joint_space_goal_node", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto const logger = node->get_logger();

  // Declare and get the detection topic parameter
  // If provided via launch parameter overrides, it is already declared.
  if (!node->has_parameter("detections_topic")) {
    node->declare_parameter("detections_topic", "/detections");
  }
  std::string detections_topic = node->get_parameter("detections_topic").as_string();
  
  bool debug = false;
  if (!node->has_parameter("debug")) {
    node->declare_parameter("debug", false);
  }
  debug = node->get_parameter("debug").as_bool();

  bool compute_metrics = true;
  if (!node->has_parameter("compute_metrics")) {
    node->declare_parameter("compute_metrics", true);
  }
  compute_metrics = node->get_parameter("compute_metrics").as_bool();

  auto metrics = std::make_shared<PnpMetrics>();
  auto pnp_log  = std::make_shared<std::ofstream>();
  rclcpp::TimerBase::SharedPtr metrics_timer;
  if (compute_metrics) {
    std::string metrics_dir = "pnp_metrics";
    std::filesystem::create_directories(metrics_dir);
    std::time_t now_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d_%H-%M-%S", std::localtime(&now_t));
    std::string log_path = metrics_dir + "/metrics_pnp_" + std::string(ts_buf) + ".txt";
    pnp_log->open(log_path, std::ios::out | std::ios::trunc);
    if (pnp_log->is_open())
      RCLCPP_INFO(logger, "[METRICS] Logging to file: %s", log_path.c_str());
    else
      RCLCPP_WARN(logger, "[METRICS] Could not open log file: %s", log_path.c_str());

    metrics_timer = node->create_wall_timer(
        std::chrono::seconds(5),
        [&metrics, &logger, pnp_log]() {
            auto [d2s_m, d2s_s] = metrics->mean_std_locked(metrics->detection_to_start_ms);
            auto [cyc_m, cyc_s] = metrics->mean_std_locked(metrics->cycle_duration_s);

            // Per-class rolling means (grab lock once)
            double cls_roll[4], cls_cum[4];
            {
                std::lock_guard<std::mutex> lk(metrics->mtx);
                for (int c = 0; c < 4; ++c) {
                    cls_roll[c] = metrics->mean_std_unlocked(metrics->class_duration_s[c]).first;
                    cls_cum[c]  = metrics->class_cumulative_mean(c);
                }
            }
            uint64_t green_cnt = metrics->class_count[0] + metrics->class_count[1];
            double   green_cum = (metrics->class_dur_cnt[0] + metrics->class_dur_cnt[1]) > 0
                ? (metrics->class_dur_sum[0] + metrics->class_dur_sum[1])
                  / static_cast<double>(metrics->class_dur_cnt[0] + metrics->class_dur_cnt[1])
                : 0.0;

            char buf[4096];
            std::snprintf(buf, sizeof(buf),
                "\n========== Pick&Place Node Metrics ==========\n"
                "  Detections received   : %lu\n"
                "  Detections dropped    : %lu  (node busy)\n"
                "  Cycles succeeded      : %lu\n"
                "  Cycles failed         : %lu\n"
                "  Detection->start lag  : %.2f +/- %.2f ms\n"
                "  Cycle duration (all)  : %.2f +/- %.2f s  (rolling last %zu)\n"
                "  --- Per-class cycle counts & avg duration (rolling | cumulative) ---\n"
                "  [0] glass   : %lu cycles  ok=%lu  fail=%lu  | roll=%.2fs  cum=%.2fs\n"
                "  [1] metal   : %lu cycles  ok=%lu  fail=%lu  | roll=%.2fs  cum=%.2fs\n"
                "  [2] paper   : %lu cycles  ok=%lu  fail=%lu  | roll=%.2fs  cum=%.2fs\n"
                "  [3] plastic : %lu cycles  ok=%lu  fail=%lu  | roll=%.2fs  cum=%.2fs\n"
                "  --- Bin totals ---\n"
                "  green  (glass+metal) : %lu cycles  cum avg=%.2fs\n"
                "  red    (paper)       : %lu cycles  cum avg=%.2fs\n"
                "  yellow (plastic)     : %lu cycles  cum avg=%.2fs\n"
                "=============================================",
                metrics->detections_received.load(),
                metrics->detections_dropped.load(),
                metrics->cycles_succeeded.load(),
                metrics->cycles_failed.load(),
                d2s_m, d2s_s,
                cyc_m, cyc_s, PnpMetrics::WINDOW,
                metrics->class_count[0].load(), metrics->class_succeeded[0].load(), metrics->class_failed[0].load(), cls_roll[0], cls_cum[0],
                metrics->class_count[1].load(), metrics->class_succeeded[1].load(), metrics->class_failed[1].load(), cls_roll[1], cls_cum[1],
                metrics->class_count[2].load(), metrics->class_succeeded[2].load(), metrics->class_failed[2].load(), cls_roll[2], cls_cum[2],
                metrics->class_count[3].load(), metrics->class_succeeded[3].load(), metrics->class_failed[3].load(), cls_roll[3], cls_cum[3],
                green_cnt, green_cum,
                metrics->class_count[2].load(), cls_cum[2],
                metrics->class_count[3].load(), cls_cum[3]);

            RCLCPP_INFO(logger, "%s", buf);

            if (pnp_log && pnp_log->is_open()) {
                std::time_t nt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                char ts[32]; std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&nt));
                (*pnp_log) << "[" << ts << "]\n" << buf << "\n\n";
                pnp_log->flush();
            }
        });
    RCLCPP_INFO(logger, "[METRICS] Pick&place metrics enabled (printed every 5 s).");
  }
  
  if (debug) RCLCPP_INFO(logger, "Pick and place node started. Waiting for detections on topic: %s", detections_topic.c_str());

  using moveit::planning_interface::MoveGroupInterface;
  auto move_group_interface = MoveGroupInterface(node, "arm_group");

  // Flag to track if currently processing
  std::atomic<bool> is_processing(false);
  std::atomic<int> cycle_counter(0);

  // Subscribe to detections topic (Int32 messages from YOLO detector)
  auto subscription = node->create_subscription<std_msgs::msg::Int32>(
    detections_topic,
    10,
    [&](const std_msgs::msg::Int32::SharedPtr msg) {
      if (compute_metrics) metrics->detections_received++;

      // Check if already processing
      if (is_processing.load()) {
        RCLCPP_WARN(logger, "Already processing a cycle. Ignoring message: %d", msg->data);
        if (compute_metrics) metrics->detections_dropped++;
        return;
      }
      
      is_processing.store(true);
      cycle_counter++;
      int current_cycle = cycle_counter.load();

      const auto t_receive = std::chrono::steady_clock::now();
      
      if (debug) RCLCPP_INFO(logger, "Received detection class ID: %d - Starting pick and place cycle %d", msg->data, current_cycle);
      
      // Spawn a separate thread to handle the pick and place cycle
      std::thread pick_place_thread([&, current_cycle, msg, t_receive]() {
        if (compute_metrics) {
            auto t_thread_start = std::chrono::steady_clock::now();
            double lag_ms = std::chrono::duration<double, std::milli>(
                t_thread_start - t_receive).count();
            metrics->push(metrics->detection_to_start_ms, lag_ms);
        }

        // Execute the pick and place cycle
        bool ok = perform_pick_and_place_cycle(node, move_group_interface, current_cycle, msg->data, debug);

        if (compute_metrics) {
            auto t_done = std::chrono::steady_clock::now();
            double dur_s = std::chrono::duration<double>(t_done - t_receive).count();
            metrics->push(metrics->cycle_duration_s, dur_s);
            int cls = msg->data;
            metrics->push_class(cls, dur_s);
            metrics->class_count[cls < 4 ? cls : 0]++;
            if (ok) { metrics->cycles_succeeded++; metrics->class_succeeded[cls < 4 ? cls : 0]++; }
            else    { metrics->cycles_failed++;     metrics->class_failed[cls < 4 ? cls : 0]++;    }
        }

        if (ok)
        {
          if (debug) RCLCPP_INFO(logger, "Cycle %d completed successfully", current_cycle);
        }
        else
        {
          RCLCPP_ERROR(logger, "Cycle %d failed!", current_cycle);
        }
        
        is_processing.store(false);
        if (debug) RCLCPP_INFO(logger, "Ready for next message...");
      });
      
      // Detach the thread so it runs independently
      pick_place_thread.detach();
    });

  rclcpp::spin(node);
  
  if (compute_metrics) {
    RCLCPP_INFO(logger, "[METRICS] === Final Pick&Place Metrics on Shutdown ===");
    auto [d2s_m, d2s_s] = metrics->mean_std_locked(metrics->detection_to_start_ms);
    auto [cyc_m, cyc_s] = metrics->mean_std_locked(metrics->cycle_duration_s);
    double cls_roll[4], cls_cum[4];
    {
        std::lock_guard<std::mutex> lk(metrics->mtx);
        for (int c = 0; c < 4; ++c) {
            cls_roll[c] = metrics->mean_std_unlocked(metrics->class_duration_s[c]).first;
            cls_cum[c]  = metrics->class_cumulative_mean(c);
        }
    }
    uint64_t green_cnt = metrics->class_count[0] + metrics->class_count[1];
    double   green_cum = (metrics->class_dur_cnt[0] + metrics->class_dur_cnt[1]) > 0
        ? (metrics->class_dur_sum[0] + metrics->class_dur_sum[1])
          / static_cast<double>(metrics->class_dur_cnt[0] + metrics->class_dur_cnt[1])
        : 0.0;

    char buf[4096];
    std::snprintf(buf, sizeof(buf),
        "\n========== Pick&Place Node Metrics ==========\n"
        "  Detections received   : %lu\n"
        "  Detections dropped    : %lu  (node busy)\n"
        "  Cycles succeeded      : %lu\n"
        "  Cycles failed         : %lu\n"
        "  Detection->start lag  : %.2f +/- %.2f ms\n"
        "  Cycle duration (all)  : %.2f +/- %.2f s  (rolling last %zu)\n"
        "  --- Per-class cycle counts & avg duration (rolling | cumulative) ---\n"
        "  [0] glass   : %lu cycles  ok=%lu  fail=%lu  | roll=%.2fs  cum=%.2fs\n"
        "  [1] metal   : %lu cycles  ok=%lu  fail=%lu  | roll=%.2fs  cum=%.2fs\n"
        "  [2] paper   : %lu cycles  ok=%lu  fail=%lu  | roll=%.2fs  cum=%.2fs\n"
        "  [3] plastic : %lu cycles  ok=%lu  fail=%lu  | roll=%.2fs  cum=%.2fs\n"
        "  --- Bin totals ---\n"
        "  green  (glass+metal) : %lu cycles  cum avg=%.2fs\n"
        "  red    (paper)       : %lu cycles  cum avg=%.2fs\n"
        "  yellow (plastic)     : %lu cycles  cum avg=%.2fs\n"
        "=============================================",
        metrics->detections_received.load(),
        metrics->detections_dropped.load(),
        metrics->cycles_succeeded.load(),
        metrics->cycles_failed.load(),
        d2s_m, d2s_s,
        cyc_m, cyc_s, PnpMetrics::WINDOW,
        metrics->class_count[0].load(), metrics->class_succeeded[0].load(), metrics->class_failed[0].load(), cls_roll[0], cls_cum[0],
        metrics->class_count[1].load(), metrics->class_succeeded[1].load(), metrics->class_failed[1].load(), cls_roll[1], cls_cum[1],
        metrics->class_count[2].load(), metrics->class_succeeded[2].load(), metrics->class_failed[2].load(), cls_roll[2], cls_cum[2],
        metrics->class_count[3].load(), metrics->class_succeeded[3].load(), metrics->class_failed[3].load(), cls_roll[3], cls_cum[3],
        green_cnt, green_cum,
        metrics->class_count[2].load(), cls_cum[2],
        metrics->class_count[3].load(), cls_cum[3]);

    RCLCPP_INFO(logger, "%s", buf);

    if (pnp_log && pnp_log->is_open()) {
        std::time_t nt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char ts[32]; std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&nt));
        (*pnp_log) << "[" << ts << "] FINAL\n" << buf << "\n\n";
        pnp_log->flush();
        pnp_log->close();
    }
  }

  if (debug) RCLCPP_INFO(logger, "Shutting down...");
  rclcpp::shutdown();
  return 0;
}
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <std_srvs/srv/set_bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <gazebo_msgs/srv/delete_entity.hpp>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <queue>
#include <sstream>
#include <set>
#include <mutex>

using namespace std::chrono_literals;

struct ObjectTask {
  std::string id;
  std::string color;
  double place_angle;
};

class VisionBasedPickAndPlace : public rclcpp::Node {
public:
  VisionBasedPickAndPlace() : Node("vision_based_pick_and_place_node") {
    // Subscribe to camera detection
    detection_sub_ = this->create_subscription<std_msgs::msg::String>(
      "detected_objects", 10,
      std::bind(&VisionBasedPickAndPlace::detectionCallback, this, std::placeholders::_1));

    // Publisher to notify when task is completed
    completion_pub_ = this->create_publisher<std_msgs::msg::String>("task_completed", 10);

    // Clients and interfaces
    suction_client_ = this->create_client<std_srvs::srv::SetBool>("switch");
    delete_client_ = this->create_client<gazebo_msgs::srv::DeleteEntity>("/delete_entity");

    // Wait for services
    while (!suction_client_->wait_for_service(1s)) {
      RCLCPP_INFO(this->get_logger(), "Waiting for vacuum suction service...");
    }

    while (!delete_client_->wait_for_service(1s)) {
      RCLCPP_INFO(this->get_logger(), "Waiting for delete service...");
    }

    RCLCPP_INFO(this->get_logger(), "✓ Vision-based Pick and Place Node Ready!");
    RCLCPP_INFO(this->get_logger(), "Listening for camera detections...");
  }

  void init() {
    // Initialize MoveIt
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      this->shared_from_this(), "arm_group");

    // Wait for MoveIt to be ready
    std::this_thread::sleep_for(2s);

    // Main thread to process detected objects
    processing_thread_ = std::thread(&VisionBasedPickAndPlace::processingLoop, this);
  }

private:
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr detection_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr completion_pub_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr suction_client_;
  rclcpp::Client<gazebo_msgs::srv::DeleteEntity>::SharedPtr delete_client_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  std::queue<ObjectTask> task_queue_;
  std::set<std::string> processing_objects_;  // Track objects being processed
  std::mutex queue_mutex_;  // Protect queue and set access
  std::thread processing_thread_;
  bool should_exit_ = false;

  void detectionCallback(const std_msgs::msg::String::SharedPtr msg) {
    // Parse message: "OBJECT_DETECTED|obj_1|GREEN|0.0|0.115565"
    std::istringstream iss(msg->data);
    std::string token;
    std::vector<std::string> parts;

    while (std::getline(iss, token, '|')) {
      parts.push_back(token);
    }

    if (parts.size() >= 3 && parts[0] == "OBJECT_DETECTED") {
      std::string obj_id = parts[1];
      std::string color = parts[2];

      // Only add if not already being processed
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (processing_objects_.find(obj_id) == processing_objects_.end()) {
        ObjectTask task;
        task.id = obj_id;
        task.color = color;
        task.place_angle = (color == "GREEN") ? 0.7 : -0.7;

        task_queue_.push(task);
        processing_objects_.insert(obj_id);
        RCLCPP_INFO(this->get_logger(), 
                    "📸 Received detection: %s | Color: %s | Destination: %s",
                    obj_id.c_str(), color.c_str(), 
                    (color == "GREEN" ? "RIGHT" : "LEFT"));
      }
    }
  }

  void processingLoop() {
    while (rclcpp::ok() && !should_exit_) {
      ObjectTask task;
      bool has_task = false;
      
      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!task_queue_.empty()) {
          task = task_queue_.front();
          task_queue_.pop();
          has_task = true;
        }
      }
      
      if (has_task) {

        RCLCPP_INFO(this->get_logger(), 
                    "========================================");
        RCLCPP_INFO(this->get_logger(), 
                    "🤖 EXECUTING PICK AND PLACE FOR: %s (Color: %s)",
                    task.id.c_str(), task.color.c_str());
        RCLCPP_INFO(this->get_logger(), 
                    "========================================");

        executePickAndPlace(task);
      }
      std::this_thread::sleep_for(100ms);
    }
  }

  void executePickAndPlace(const ObjectTask& task) {
    // Pick from center (12 o'clock)
    RCLCPP_INFO(this->get_logger(), "→ Picking object from 12 o'clock...");
    std::vector<double> pick_center = {0.0, 0.802, 0.942, 1.361};
    move_group_->setJointValueTarget(pick_center);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(move_group_->plan(plan));

    if (success) {
      move_group_->execute(plan);
      RCLCPP_INFO(this->get_logger(), "✓ Reached pick position");
    } else {
      RCLCPP_ERROR(this->get_logger(), "✗ Planning failed");
      return;
    }

    // Activate suction
    auto pick_request = std::make_shared<std_srvs::srv::SetBool::Request>();
    pick_request->data = true;
    auto pick_result = suction_client_->async_send_request(pick_request);
    pick_result.wait_for(5s);
    RCLCPP_INFO(this->get_logger(), "✓ Suction activated");

    // Place at appropriate location based on color
    std::string direction = (task.color == "GREEN") ? "RIGHT (3 o'clock)" : "LEFT (9 o'clock)";
    RCLCPP_INFO(this->get_logger(), "→ Placing object to %s...", direction.c_str());
    
    std::vector<double> place_position = {task.place_angle, 0.802, 0.942, 1.361};
    move_group_->setJointValueTarget(place_position);

    plan = moveit::planning_interface::MoveGroupInterface::Plan();
    success = static_cast<bool>(move_group_->plan(plan));

    if (success) {
      move_group_->execute(plan);
      RCLCPP_INFO(this->get_logger(), "✓ Reached place position");
    } else {
      RCLCPP_ERROR(this->get_logger(), "✗ Planning failed");
      return;
    }

    // Deactivate suction
    auto place_request = std::make_shared<std_srvs::srv::SetBool::Request>();
    place_request->data = false;
    auto place_result = suction_client_->async_send_request(place_request);
    place_result.wait_for(5s);
    RCLCPP_INFO(this->get_logger(), "✓ Object released at %s", direction.c_str());

    // Wait for object to settle
    std::this_thread::sleep_for(300ms);

    // Delete object
    RCLCPP_INFO(this->get_logger(), "→ Deleting object from simulation...");
    auto delete_request = std::make_shared<gazebo_msgs::srv::DeleteEntity::Request>();
    delete_request->name = task.id;

    auto delete_result = delete_client_->async_send_request(delete_request);
    
    // Wait for the delete service to complete (executor is already spinning)
    if (delete_result.wait_for(5s) == std::future_status::ready) {
      auto result = delete_result.get();  // Get once and store
      if (result->success) {
        RCLCPP_INFO(this->get_logger(), "✓ Object %s deleted from Gazebo", task.id.c_str());
      } else {
        RCLCPP_ERROR(this->get_logger(), "✗ Failed to delete object %s: %s", 
                     task.id.c_str(), result->status_message.c_str());
      }
    } else {
      RCLCPP_ERROR(this->get_logger(), "✗ Delete service call timeout for object %s", task.id.c_str());
    }

    // Wait for Gazebo to process deletion
    std::this_thread::sleep_for(800ms);

    // Return to home
    RCLCPP_INFO(this->get_logger(), "→ Returning to home (12 o'clock)...");
    std::vector<double> home = {0.0, 0.802, 0.942, 1.361};
    move_group_->setJointValueTarget(home);

    plan = moveit::planning_interface::MoveGroupInterface::Plan();
    success = static_cast<bool>(move_group_->plan(plan));

    if (success) {
      move_group_->execute(plan);
      RCLCPP_INFO(this->get_logger(), "✓ Returned to home position");
    }

    std::this_thread::sleep_for(500ms);
    RCLCPP_INFO(this->get_logger(), " ");
    
    // Notify camera that object processing is complete
    auto completion_msg = std::make_unique<std_msgs::msg::String>();
    completion_msg->data = task.id;
    completion_pub_->publish(std::move(completion_msg));
    RCLCPP_INFO(this->get_logger(), "✓ Notified camera: %s processing complete", task.id.c_str());
    
    // Remove from processing set
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      processing_objects_.erase(task.id);
    }
  }

public:
  ~VisionBasedPickAndPlace() {
    should_exit_ = true;
    if (processing_thread_.joinable()) {
      processing_thread_.join();
    }
  }
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VisionBasedPickAndPlace>();
  
  // Initialize MoveIt after node is fully constructed
  node->init();
  
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  
  std::thread executor_thread([&executor]() {
    executor.spin();
  });
  
  executor_thread.join();
  rclcpp::shutdown();
  return 0;
}

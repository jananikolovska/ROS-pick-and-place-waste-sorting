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
    int message_value)
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
  
  RCLCPP_INFO(logger, "Starting pick and place cycle %d - Color: %s, Number: %d", cycle_number, color.c_str(), message_value);
  
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
  
  RCLCPP_INFO(logger, "Moving to pick position...");
  
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
    RCLCPP_INFO(logger, "Pick motion executed");
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
  RCLCPP_INFO(logger, "Vacuum activated");

  RCLCPP_INFO(logger, "Moving to place position...");

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
    RCLCPP_INFO(logger, "Place motion executed");
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
  RCLCPP_INFO(logger, "Vacuum deactivated");

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
    RCLCPP_INFO(logger, "Box deleted successfully for cycle %d", cycle_number);
  }
  else
  {
    RCLCPP_WARN(logger, "Failed to delete box: %s", delete_result->status_message.c_str());
  }
  
  RCLCPP_INFO(logger, "Pick and place cycle %d completed successfully!", cycle_number);
  return true;
}

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto const node = std::make_shared<rclcpp::Node>(
      "test_send_joint_space_goal_node", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto const logger = node->get_logger();

  // Declare and get the detection topic parameter
  node->declare_parameter("detections_topic", "/detections");
  std::string detections_topic = node->get_parameter("detections_topic").as_string();
  
  RCLCPP_INFO(logger, "Pick and place node started. Waiting for detections on topic: %s", detections_topic.c_str());

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
      // Check if already processing
      if (is_processing.load()) {
        RCLCPP_WARN(logger, "Already processing a cycle. Ignoring message: %d", msg->data);
        return;
      }
      
      is_processing.store(true);
      cycle_counter++;
      int current_cycle = cycle_counter.load();
      
      RCLCPP_INFO(logger, "Received detection class ID: %d - Starting pick and place cycle %d", msg->data, current_cycle);
      
      // Spawn a separate thread to handle the pick and place cycle
      std::thread pick_place_thread([&, current_cycle, msg]() {
        // Execute the pick and place cycle
        if (perform_pick_and_place_cycle(node, move_group_interface, current_cycle, msg->data))
        {
          RCLCPP_INFO(logger, "Cycle %d completed successfully", current_cycle);
        }
        else
        {
          RCLCPP_ERROR(logger, "Cycle %d failed!", current_cycle);
        }
        
        is_processing.store(false);
        RCLCPP_INFO(logger, "Ready for next message...");
      });
      
      // Detach the thread so it runs independently
      pick_place_thread.detach();
    });

  rclcpp::spin(node);
  
  RCLCPP_INFO(logger, "Shutting down...");
  rclcpp::shutdown();
  return 0;
}
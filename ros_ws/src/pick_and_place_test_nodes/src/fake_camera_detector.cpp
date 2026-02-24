#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <gazebo_msgs/srv/spawn_entity.hpp>
#include <gazebo_msgs/srv/delete_entity.hpp>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

using namespace std::chrono_literals;

// Custom message structure for detected objects
struct DetectedObject {
  std::string id;
  std::string color;  // "GREEN" or "PURPLE"
  double x;
  double y;
  bool is_available;
};

class FakeCameraDetector : public rclcpp::Node {
public:
  FakeCameraDetector() : Node("fake_camera_detector_node"), task_completed_(false) {
    // Publisher for detected objects
    object_pub_ = this->create_publisher<std_msgs::msg::String>("detected_objects", 10);
    
    // Subscribe to task completion notifications
    completion_sub_ = this->create_subscription<std_msgs::msg::String>(
      "task_completed", 10,
      std::bind(&FakeCameraDetector::completionCallback, this, std::placeholders::_1));
    
    // Clients for Gazebo
    spawn_client_ = this->create_client<gazebo_msgs::srv::SpawnEntity>("/spawn_entity");
    delete_client_ = this->create_client<gazebo_msgs::srv::DeleteEntity>("/delete_entity");
    
    // Wait for services
    while (!spawn_client_->wait_for_service(1s)) {
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for spawn service.");
        return;
      }
      RCLCPP_INFO(this->get_logger(), "Spawn service not available, waiting...");
    }
    
    while (!delete_client_->wait_for_service(1s)) {
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for delete service.");
        return;
      }
      RCLCPP_INFO(this->get_logger(), "Delete service not available, waiting...");
    }

    RCLCPP_INFO(this->get_logger(), "All services available. Starting object detection simulation...");
    
    // Start detection thread (not using timer to avoid executor conflicts)
    detection_thread_ = std::thread(&FakeCameraDetector::detectionLoop, this);
  }

  void detectionLoop() {
    int object_counter = 0;
    const int NUM_OBJECTS = 5;
    
    while (rclcpp::ok() && object_counter < NUM_OBJECTS) {
      object_counter++;
      std::string object_id = "obj_" + std::to_string(object_counter);
      std::string color = getRandomColor();
      
      RCLCPP_INFO(this->get_logger(), " ");
      RCLCPP_INFO(this->get_logger(), "========================================");
      RCLCPP_INFO(this->get_logger(), "🎥 OBJECT %d/%d - Color: %s", 
                  object_counter, NUM_OBJECTS, color.c_str());
      RCLCPP_INFO(this->get_logger(), "========================================");
      
      // Spawn object
      spawnObject(object_id, color);
      
      // Let it settle
      std::this_thread::sleep_for(1s);
      
      // Reset completion flag
      {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        task_completed_ = false;
      }
      
      // Continuously publish detection until robot processes the object
      RCLCPP_INFO(this->get_logger(), "🎥 Camera continuously detecting %s...", object_id.c_str());
      int detection_count = 0;
      while (rclcpp::ok()) {
        // Check if task is completed
        {
          std::lock_guard<std::mutex> lock(completion_mutex_);
          if (task_completed_) {
            break;
          }
        }
        
        // Publish detection periodically
        detection_count++;
        publishDetection(object_id, color, detection_count);
        
        // Wait 1.5 seconds before next detection
        std::this_thread::sleep_for(1500ms);
      }
      
      if (rclcpp::ok()) {
        RCLCPP_INFO(this->get_logger(), "✓ Object %s processed and removed by robot (sent %d detections)", 
                    object_id.c_str(), detection_count);
        std::this_thread::sleep_for(500ms);  // Small delay before next
      }
    }
    
    RCLCPP_INFO(this->get_logger(), " ");
    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "🎥 Camera: All %d objects processed!", NUM_OBJECTS);
    RCLCPP_INFO(this->get_logger(), "========================================");
  }

  ~FakeCameraDetector() {
    if (detection_thread_.joinable()) {
      detection_thread_.join();
    }
  }

private:
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr object_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr completion_sub_;
  rclcpp::Client<gazebo_msgs::srv::SpawnEntity>::SharedPtr spawn_client_;
  rclcpp::Client<gazebo_msgs::srv::DeleteEntity>::SharedPtr delete_client_;
  
  std::thread detection_thread_;
  std::mutex completion_mutex_;
  std::condition_variable completion_cv_;
  bool task_completed_;

  void completionCallback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "✓ Received completion notification for: %s", msg->data.c_str());
    std::lock_guard<std::mutex> lock(completion_mutex_);
    task_completed_ = true;
    completion_cv_.notify_one();
  }

  std::string getRandomColor() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 1);
    return dis(gen) == 0 ? "GREEN" : "PURPLE";
  }

  void spawnObject(const std::string& object_id, const std::string& color) {
    auto spawn_request = std::make_shared<gazebo_msgs::srv::SpawnEntity::Request>();
    spawn_request->name = object_id;
    
    // Create colored box
    std::string color_name, rgb_ambient, rgb_diffuse;
    if (color == "GREEN") {
      color_name = "Gazebo/Green";
      rgb_ambient = "0 1 0 1";
      rgb_diffuse = "0 1 0 1";
    } else {
      color_name = "Gazebo/Purple";
      rgb_ambient = "1 0 1 1";
      rgb_diffuse = "1 0 1 1";
    }

    std::string spawn_xml = R"(<?xml version="1.0"?>
<sdf version='1.7'>
  <model name=')" + object_id + R"('>
    <link name='link'>
      <inertial>
        <mass>0.095</mass>
        <inertia>
          <ixx>0.00016352</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.00027373</iyy>
          <iyz>0</iyz>
          <izz>0.00032967</izz>
        </inertia>
      </inertial>
      <visual name='visual'>
        <geometry><box><size>0.229751 0.162283 0.113616</size></box></geometry>
        <material>
          <script><name>)" + color_name + R"(</name><uri>file://media/materials/scripts/gazebo.material</uri></script>
          <ambient>)" + rgb_ambient + R"(</ambient>
          <diffuse>)" + rgb_diffuse + R"(</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
      <collision name='collision'>
        <geometry><box><size>0.229751 0.162283 0.113616</size></box></geometry>
        <surface>
          <friction><ode><mu>1</mu><mu2>1</mu2></ode></friction>
          <contact><ode><kp>1e+13</kp><kd>1</kd></ode></contact>
        </surface>
      </collision>
    </link>
    <static>0</static>
  </model>
</sdf>)";

    spawn_request->xml = spawn_xml;
    spawn_request->robot_namespace = "";
    
    geometry_msgs::msg::Pose pose;
    pose.position.x = 0.0;
    pose.position.y = 0.115565;
    pose.position.z = 0.056808;
    pose.orientation.w = 1.0;
    spawn_request->initial_pose = pose;

    auto result = spawn_client_->async_send_request(spawn_request);
    
    // Wait for result without spinning the executor
    if (result.wait_for(5s) == std::future_status::ready && result.get()->success) {
      RCLCPP_INFO(this->get_logger(), "✓ %s object spawned at 12 o'clock!", color.c_str());
    } else {
      RCLCPP_ERROR(this->get_logger(), "✗ Failed to spawn object");
    }
  }

  void publishDetection(const std::string& object_id, const std::string& color, int count = 0) {
    auto msg = std::make_unique<std_msgs::msg::String>();
    msg->data = "OBJECT_DETECTED|" + object_id + "|" + color + "|0.0|0.115565";
    object_pub_->publish(std::move(msg));
    RCLCPP_INFO(this->get_logger(), "🎥 Detection #%d: {ID: %s, Color: %s}", 
                count, object_id.c_str(), color.c_str());
  }
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FakeCameraDetector>();
  
  // Use multi-threaded executor to handle both detection thread and node operations
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  
  std::thread executor_thread([&executor]() {
    executor.spin();
  });
  
  // Detection loop runs in the detection thread
  executor_thread.join();
  rclcpp::shutdown();
  return 0;
}

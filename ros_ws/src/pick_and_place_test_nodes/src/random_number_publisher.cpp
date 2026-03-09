#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>

#include <chrono>
#include <random>

using namespace std::chrono_literals;

class RandomNumberPublisher : public rclcpp::Node
{
public:
  RandomNumberPublisher()
  : Node("random_number_publisher"), publish_count_(0)
  {
    publisher_ = this->create_publisher<std_msgs::msg::Int32>("/detections", 10);
    
    // Setup random number generator
    std::random_device rd;
    rng_ = std::mt19937(rd());
    dist_ = std::uniform_int_distribution<int>(0, 3);
    
    RCLCPP_INFO(this->get_logger(), "Random Number Publisher started!");
    
    // Create timer to publish every 20 seconds
    timer_ = this->create_wall_timer(
      20s,
      std::bind(&RandomNumberPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    // Generate random number between 0 and 3
    int random_number = dist_(rng_);
    
    auto message = std_msgs::msg::Int32();
    message.data = random_number;
    
    publisher_->publish(message);
    publish_count_++;
    
    RCLCPP_INFO(this->get_logger(), "Published: %d (total messages: %d)", random_number, publish_count_);
  }

  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  int publish_count_;
  std::mt19937 rng_;
  std::uniform_int_distribution<int> dist_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RandomNumberPublisher>());
  rclcpp::shutdown();
  return 0;
}

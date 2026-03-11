# ROS-pick-and-place-waste-sorting

Jana Nikolovska  
University of Bologna  
Master Thesis in Artificial Intelligence

**Purpose:** This project demonstrates the integration of the Axelera Metis M.2 AI accelerator with ROS 2 for real-time perception, using a validation scenario focused on intelligent waste sorting. The system is designed to showcase a complete perception-to-action pipeline, with a focus on robust, hardware-accelerated object detection and its use in a simulated pick-and-place task.

---

## System Overview

- **Platform:** Ubuntu 22.04, ROS 2 Humble, Gazebo Classic (x86 architecture only), Axelera Voyager SDK 1.4
- **Perception:** Object detection is performed using YOLO models, fine-tuned in the `yolo_training` directory. The models are compiled and deployed for the Axelera Metis M.2 accelerator.
- **Simulation & Integration:** The `ros_ws` workspace contains:
  - The full simulation environment (robot, world, and manipulation logic)
  - Inference nodes: two generic nodes for YOLO models (`ax_inference_node_cpp` and `ax_inference_node_python`, which can be extended for custom models)
  - The `recycle_inference_node_cpp`, an advanced node with custom YOLO network, preprocessing, and postprocessing tailored for the pick-and-place sorting task
- **Flow:** Camera frames are published, processed by the inference node (running on the Axelera accelerator), and detection results are used to trigger the robotic arm in simulation to pick and sort objects based on class.
- **Results:** The `results` directory contains experimental results and data included in the thesis.

For a detailed description of the ROS-based flow, node parameters, and launch instructions, see the README inside `ros_ws`.

---

### yolo_training
Contains scripts and notebooks for dataset preparation, model training, and fine-tuning of YOLO object detection models for the waste sorting task.

### ros_ws
Contains the ROS 2 workspace with:
- Simulation environment and robot manipulation logic
- Inference nodes for generic and custom YOLO models
- All launch files, helper nodes, and configuration

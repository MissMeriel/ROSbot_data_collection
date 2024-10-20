#!/bin/bash

# Source ROS environment
source /opt/ros/humble/setup.bash
source /home/husarion/ros2_ws/install/setup.bash

# test rosbotxl subsystems one at a time
echo "attempting to start lidar"
ros2 launch rplidar_ros rplidar_a3_launch.py &
echo "attempting to start camera"
ros2 run usb_cam usb_cam_node_exe --ros-args --params-file ~/ros2_ws/src/usb_cam/config/params_1.yaml &
echo "attempting to drive"
python3 /home/husarion/ros2_ws/src/final/final/steering_NN.py

## TO QUIT: spam Ctrl+C in terminal when want to kill self driving mode
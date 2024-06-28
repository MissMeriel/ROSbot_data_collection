import os


print("attempting to start lidar")
os.system("ros2 launch rplidar_ros rplidar_a3_launch.py &")
print("attempting to start camera")
os.system("ros2 run usb_cam usb_cam_node_exe --ros-args --params-file ~/ros2_ws/src/usb_cam/config/params_1.yaml &")
print("attempting to drive")
os.system("python3 /home/husarion/ros2_ws/src/final/final/steering_NN.py")

## TO QUIT: spam Ctrl+C in terminal when want to kill self driving mode
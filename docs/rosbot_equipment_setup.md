# ROSbot Equipments Setup

This document contains instructions to set up the [ROS Equipments](https://husarion.com/tutorials/ros-equipment/) on your ROSbot.

The ROS Equipments included in this document:
- [Zed Camera](https://husarion.com/tutorials/ros-equipment/zed/)
- [SLAMTEC RPlidar](https://husarion.com/tutorials/ros-equipment/rplidar/)
- Wireless Controller

## Zed Camera
[Stereolabs ZED Official Documentation](https://www.stereolabs.com/docs/)

[ZED SDK v4.1](https://www.stereolabs.com/developers/release)
[ZED ROS 2 Repository](https://github.com/stereolabs/zed-ros2-wrapper)
[ZED Docker Repository](https://github.com/husarion/zed-docker)

If encounter the Error `[zed.zed_node]: Error opening camera: NO GPU DETECTED`:
1. Use the usb_cam package to parse the input of the ZED2 camera.
[usb_cam](https://github.com/ros-drivers/usb_cam)
2. Install the package and run the `/usb_cam` node following the instruction within the package repository.
3. Check the `default_cam.yaml` to match the `ros_parameters` with your Zed Camera. 
   - For example, set `video_device` to the correct port that your camera is plugged to. Usually it is `/dev/video0` or `dev/video1`.
   - Set `camera_name` to your camera model name. In our project, we use `ZED2`.
4. Run `ros2 run usb_cam usb_cam_node_exe --ros-args --params-file /home/husarion/.ros/camera_info/default_cam.yaml` to start the camera.

## SLAMTEC RPlidar


## Wireless Controller



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
   - For example, set `video_device` to the correct port that your camera is plugged to. Usually it is `/dev/video0` or `/dev/video1`.
   - Set `camera_name` to your camera model name. In our project, we use `ZED2`.
4. Run `ros2 run usb_cam usb_cam_node_exe --ros-args --params-file /home/husarion/.ros/camera_info/default_cam.yaml` to start the camera.

## SLAMTEC RPlidar
[Slamtec RPlidar Official Documentation](https://www.slamtec.com/en/) 

[RPlidar ROS 2 Repository](https://github.com/Slamtec/rplidar_ros/tree/ros2/)
[RPlidar Docker Repository](https://github.com/husarion/rplidar-docker)

If encounter the Error `[rplidar_node]: Error, operation time out. SL_RESULT_OPERATION_TIMEOUT`:
1. Set the parameter `serial_port` to match the USB port of your lidar. Usually the port the lidar uses is `ttyUSB1` or `ttyUSB0`.
2. Make sure to give port permission to the USB port of the lidar through `sudo chmod 777 /dev/ttyUSB1` or `sudo chmod 777 /dev/ttyUSB0` depending on the USB port your lidar is using.
3. Set the appropriate baudrate for your lidar. For RPlidar A3 that we use in the project, baudrate is `256000`.
4. Connect the lidar to the USB serial converter that comes in the box with lidar. Plug the USB serial converter cable in the USB port inside the robot (serial port on the auxiliary board). Check below two images for reference. The yellow box on image shows where the USB serial converter cable plugged in.
   - [RPLidar USB Serial Converter Plugged](image/RPLidar%20USB%20Serial%20Converter%20Plugged.png)
   - [RPLidar USB Serial Converter Plugged CloseUp](image/RPLidar%20USB%20Serial%20Converter%20Plugged%20CloseUp.png)

## Wireless Controller


# ROSbot Equipments Setup

This document contains instructions to set up the [ROS Equipments](https://husarion.com/tutorials/ros-equipment/) on your ROSbot.

The ROS Equipments included in this document:
- [Zed Camera](https://husarion.com/tutorials/ros-equipment/zed/)
- [SLAMTEC RPlidar](https://husarion.com/tutorials/ros-equipment/rplidar/)
- Wireless Controller

## Equipment Mounting
1. Run the cable of the equipment through your ROSbot's cover
2. The Zed camera stand is mounted on ROSbot's cover with one screw. The RPLidar is mounted on ROSbot's cover with two screws.
   - Check this [Husarion tutorial video](https://youtu.be/FKI4aFbu7lo) for demonstration of mounting external components on ROSbot.
3. For the mounting location, please check the following images for reference:
   - [ROSbot's Cover Top-Side View](image/ROSbot's%20Cover%20Top.jpg)
   - [ROSbot's Cover Top-Side View Labeled](image/ROSbot's%20Cover%20Top%20Labeled.jpg)
   - [ROSbot's Cover Bottom-Side View](image/ROSbot's%20Cover%20Bottom.jpg)
   - [ROSbot's Cover Bottom-Side View Labeled](image/ROSbot's%20Cover%20Bottom%20Labeled.jpg)

## Zed Camera
[Stereolabs ZED Official Documentation](https://www.stereolabs.com/docs/)

Download [ZED SDK v4.1](https://www.stereolabs.com/developers/release) for Ubuntu 22. Install the ZED SDK following the instruction [here](https://www.stereolabs.com/docs/installation/linux#download-and-install-the-zed-sdk).\
Install the ROS 2 package `zed_ros2_wrapper` for using Stereolabs ZED Camera cameras: [ZED ROS 2 Repository](https://github.com/stereolabs/zed-ros2-wrapper).

Use Cheese to test your camera:
1. In your terminal, install cheese through `sudo apt-get install cheese`.
2. Run `cheese`. Cheese should automatically detect your webcam and display live video stream.
3. To record video or take a snapshot, click either on photo or video and select Take a Photo or Start recording.
   - If Cheese doesn't display the output from your webcam, test that the problem is not with Cheese before trying anything else by using one of the programs in the next section.
   - If Cheese displays a garbled picture or one with poor colour/brightness/contrast, go to the Ubuntu wiki [Webcam Troubleshooting](https://help.ubuntu.com/community/Webcam/Troubleshooting) page for steps to resolve this.
   - If Cheese does not run and throws an error where it cannot access something in a .config folder, check if the folder is owned by root, if so you will need to change ownership of it from 'root' to 'husarion'.

If encounter the Error `[zed.zed_node]: Error opening camera: NO GPU DETECTED`:
1. Use the usb_cam package to parse the input of the ZED2 camera.
[usb_cam](https://github.com/ros-drivers/usb_cam)
2. Install the package and run the `/usb_cam` node following the instruction within the package repository.
3. Check the `default_cam.yaml` from the path `$HOME/.ros/camera_info/defeault_cam.yaml` to match the `ros_parameters` with your Zed Camera. 
   - For example, set `video_device` to the correct port that your camera is plugged to. Usually it is `/dev/video0` or `/dev/video1`.
     The default `video_device` is `/dev/video0`. Check if `/video0` is shown in your `/dev` directory by running `ls /dev`. If `/video0` is absent, change `video_device` to the video device displayed after running `ls -l /dev/video*`.
   - Set `camera_name` to your camera model name. In our project, we use `ZED2`.
4. Run `ros2 run usb_cam usb_cam_node_exe --ros-args --params-file /home/husarion/.ros/camera_info/default_cam.yaml` to start the camera.
   - If the above command does not work, try running `ros2 run usb_cam usb_cam_node_exe --ros-args --params-file ~/ros2_ws/src/usb_cam/config/params_1.yaml`. 
   - You can replace `params_1.yaml` with your designated configuration file.
   - If you encounter the error to the effect of "specified format `mjpeg2rgb` is unsupported by the selected device":
         - Open the .yaml file you are using for params, this should be found under `~/ros2_ws/src/usb_cam/config`.
         - In your params.yaml file set `pixel_format` to `yuyv2rgb`. 
6. Run ROS visualization through `rviz2`. Inside RViz, click `Add` button and create visualization `By topic`. The image data is published to the `/image_raw` topic.
   - Check the camera screen displayed and make sure it is clear. 
   - If it looks something like this (but cheese is clear):
     
   ![Photo Jun 11 2024, 15 33 43](https://github.com/jamission/ROSbot_data_collection/assets/171875030/20639f93-e673-4c9c-99cc-b8280abebf26)

   - You have the wrong resolution in your configuration file.
   - Open the .yaml file you are using for params, this should be found under `~/ros2_ws/src/usb_cam/config`.
   - In your params.yaml file set `image_width` and `image_height` to the same resolution as output by your camera. For the ZED camera in this project, it is 1344 x 376 respectively. 
   - If the camera is different and you do not know the resolution, run `cheese` and take a photo, and open the photo in the photoviewer, you will see the resolution.
     
Other resources to view:
[ZED Docker Repository](https://github.com/husarion/zed-docker)
[Ubuntu Documentation on how to choose, setup and use a webcam](https://help.ubuntu.com/community/Webcam)

## SLAMTEC RPlidar
[Slamtec RPlidar Official Documentation](https://www.slamtec.com/en/) 

Install the SLAMTEC LIDAR ROS2 Package: [RPlidar ROS 2 Repository](https://github.com/Slamtec/rplidar_ros/tree/ros2/)

If encounter the Error `[rplidar_node]: Error, operation time out. SL_RESULT_OPERATION_TIMEOUT`:
1. Set the parameter `serial_port` to match the USB port of your lidar. Usually the port the lidar uses is `ttyUSB1` or `ttyUSB0`. You can determine the device number by observing the changes for `ttyUSB` devices detected in your `/dev` directory by running `ls /dev` when you plug and unplug the lidar. 
2. Make sure to give port permission to the USB port of the lidar through `sudo chmod 777 /dev/ttyUSB1` or `sudo chmod 777 /dev/ttyUSB0` depending on the USB port your lidar is using.
3. Set the appropriate baudrate for your lidar. For RPlidar A3 that we use in the project, baudrate is `256000`.
4. Run `sudo apt upgrade` to upgrade the installed packages on your system to their latest versions and run `sudo apt update` to refresh your system's package information to the latest available. 
5. Connect the lidar to the USB serial converter that comes in the box with lidar. Plug the USB serial converter cable in the USB port inside the ROSbot (serial port on the auxiliary board). Check below two images for reference. The yellow box on image shows where the USB serial converter cable plugged in.
   - [RPLidar USB Serial Converter Plugged](image/RPLidar%20USB%20Serial%20Converter%20Plugged.jpg)
   - [RPLidar USB Serial Converter Plugged CloseUp](image/RPLidar%20USB%20Serial%20Converter%20Plugged%20CloseUp.png)

If RViz is not automatically prompted open, run ROS visualization through `rviz2`. Inside RViz, click `Add` button and create visualization `By topic`. The Lidar scan is published to the `/scan` topic.

Other resources to view:
[RPlidar Docker Repository](https://github.com/husarion/rplidar-docker)

## Wireless Controller

For Logitech Gamepad F710, follow the instruction in [Husarion Tutorial](https://husarion.com/tutorials/ros-equipment/gamepad-f710/).

For **Xbox** controller that we use in the project:
1. Connect your controller to the ROSbot through bluetooth. For instructions on connecting as the first time, refer to the [rosbot_installation First time connecting your controller to the ROSbot](rosbot_installation.md#first-time-connecting-your-controller-to-the-rosbot) section.
   - If you encounter bluetooth connection problem, refer to the [bluetooth-controller-connection-fix](rosbot_installation.md#bluetooth-controller-connection-fix) section.
3. Install [`joystick_drivers`](https://github.com/ros-drivers/joystick_drivers/tree/ros2?tab=readme-ov-file) and [`teleop_twist_joy`](https://github.com/ros2/teleop_twist_joy/tree/humble) packages.
3. Run `ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox'` to start the `/teleop_twist_joy_node`.

If your controller cannot operate your ROSbot to move, but your controller is successfully connected to ROSbot and both `/joy_node` and `/teleop_twist_joy_node` show in your `ros2 node list` AND `/joy` and `/cmd_vel` show in your `ros2 topic list`:
1. Double check that you have run source after build.  
2. If the LED2 light is also flashing in red, run `docker compose up` or `docker compose up rosbot-xl microros` to (re)create and start your docker container `rosbot-xl` and `microros`. 
Confirm the docker containers running status by the command `docker ps`.
3. Run `ros2 topic echo /joy`. Press buttons and move the joystick on your controller to check whether the `axes` and `buttons` values change accordingly.
4. Run `ros2 topic echo /cmd_vel`. Press the button displayed as your `Teleop enable button` in `[INFO] [TeleopTwistJoy]` message, and observe whether `linear` and `angular` values change accordingly.
   - To change the `Teleop enable button`, modify the `enable_button` value in `xbox.config.yaml`.
   - You can use `ros2 topic echo /joy` to inspect each button's corresponding `enable_button` value.
   

#!/bin/bash
source /opt/ros/humble/setup.bash
source /home/husarion/ros2_ws/install/setup.bash

#sudo rfkill block wifi
export ROSBOT_VER=ROSBOT_2.0
export ROS_MASTER_URI=http://localhost:11311 
export ROS_IP=127.0.0.1
export ROS_IPV6=off
sudo service bluetooth start
timeout 15s bluetoothctl scan on
# change MAC to match your Xbox controller
bluetoothctl connect 68:6C:E6:72:4A:56
devblk=$(ls /dev | grep -m 1 sd)
sudo umount /dev/"$devblk"1
echo /dev/"$devblk"1
mkdir -p ~/media/usb
sudo mount /dev/"$devblk"1 ~/media/usb -o umask=000

ros2 launch final data_collection.launch.py

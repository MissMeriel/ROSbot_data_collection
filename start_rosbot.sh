. /opt/ros/melodic/setup.bash
. ~/husarion_ws/devel/setup.sh

export ROSBOT_VER=ROSBOT_2.0
#export ROS_MASTER_URI=http://master:11311
export ROS_MASTER_URI=http://localhost:11311 
export ROS_IP=127.0.0.1
export ROS_IPV6=off
#sudo mount /dev/sda1 /media/usb
sudo service bluetooth start
timeout 15s bluetoothctl scan on
sudo umount /media/usb
sudo mount /dev/sdb1 /home/husarion/media/usb -o umask=000
source ~/husarion_ws/devel/setup.sh
#roslaunch husarion_ros rosbot_drivers.launch
roslaunch datacoll data_collector.launch dest:=/home/husarion/media/usb/rosbot_dataset

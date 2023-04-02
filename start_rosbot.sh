. /opt/ros/melodic/setup.bash
. ~/husarion_ws/devel/setup.sh

sudo rfkill block wifi
export ROSBOT_VER=ROSBOT_2.0
export ROS_MASTER_URI=http://localhost:11311 
export ROS_IP=127.0.0.1
export ROS_IPV6=off
# uncomment the line below if you want to mount from a known static mount point
# sudo mount /dev/sda1 /media/usb 
sudo service bluetooth start
timeout 15s bluetoothctl scan on
devblk=$(ls /dev | grep -m 1 sd)
sudo umount /media/usb
echo /dev/"$devblk"1
sudo mount /dev/"$devblk"1 /home/husarion/media/usb -o umask=000
source ~/husarion_ws/devel/setup.sh
# uncomment the line below and comment the datacoll roslaunch line if you only want to launch drivers
# roslaunch husarion_ros rosbot_drivers.launch 
roslaunch datacoll data_collector.launch dest:=/home/husarion/media/usb/rosbot_dataset

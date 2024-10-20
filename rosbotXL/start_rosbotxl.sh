#!/bin/bash

# Source ROS environment
source /opt/ros/humble/setup.bash
source /home/husarion/ros2_ws/install/setup.bash

# Ensure script is run by the user 'husarion'
CURRENT_USER=$(whoami)
if [ "$CURRENT_USER" != "husarion" ]; then
  echo "This script can only be run by the user 'husarion'."
  exit 1
fi

# Define color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Start Bluetooth service and connect to Xbox controller
sudo service bluetooth start
timeout 15s bluetoothctl scan on
# Change MAC to match your Xbox controller
bluetoothctl connect A8:8C:3E:54:75:DE

# Mount USB drive
devblk=$(ls /dev | grep -m 1 sd)
sudo umount /dev/"$devblk"1
echo /dev/"$devblk"1
mkdir -p ~/media/usb
sudo mount /dev/"$devblk"1 ~/media/usb -o umask=000

# Check if the required Docker images are pulled
echo -e "${GREEN}[1/2]\r\nChecking if the required Docker Images are pulled ...${NC}"
COMPOSE_FILE="/home/husarion/compose.yaml"

# Temporary solution for shared memory communication between host and docker container
export DOCKER_UID=$(id -u husarion)
export DOCKER_GID=$(id -g husarion)

SERVICE_IMAGES=$(docker compose -f $COMPOSE_FILE config | grep 'image:' | awk '{print $2}')
IMAGE_NOT_FOUND=0

for IMAGE in $SERVICE_IMAGES; do
    if [ -z "$(docker images -q $IMAGE)" ]; then
        echo -e "${YELLOW}Image ${BOLD}$IMAGE${NC}${YELLOW} is not pulled.${NC}"
        IMAGE_NOT_FOUND=1
    else
        echo -e "${GREEN}Image ${BOLD}$IMAGE${NC}${GREEN} is pulled.${NC}"
    fi
done

if [ $IMAGE_NOT_FOUND -eq 1 ]; then
    echo -e "${GREEN}Pulling missing images...${NC}"
    docker compose -f $COMPOSE_FILE pull
    echo -e "${GREEN}done${NC}"
fi

# Launch ROS 2 Driver
echo -e "\r\n${GREEN}[2/2]\r\nLaunching ROS 2 Driver${NC}"
prefix="file://"
if [[ $CYCLONEDDS_URI == "$prefix"* ]]; then
    export CYCLONEDDS_PATH=${CYCLONEDDS_URI#file://}
else
    export CYCLONEDDS_PATH=""
fi

mkdir -p ~/.ros
docker compose -f $COMPOSE_FILE up -d

sleep 3

ros2 daemon stop

# Start ROS 2 nodes
#echo -e "\r\n${GREEN}Starting teleop_twist_joy node...${NC}"
#ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox' &

#echo -e "\r\n${GREEN}Starting rplidar a3...${NC}"
#ros2 launch rplidar_ros view_rplidar_a3_launch.py &

#echo -e "\r\n${GREEN}Starting zed2 camera...${NC}"
#ros2 run usb_cam usb_cam_node_exe --ros-args --params-file /home/husarion/.ros/camera_info/default_cam.yaml &

echo -e "${GREEN}done. Type ${BOLD}ros2 topic list${NC}${GREEN} to see available ROS 2 topics ${NC}"

# Launch data collection
ros2 launch final data_collection.launch.py

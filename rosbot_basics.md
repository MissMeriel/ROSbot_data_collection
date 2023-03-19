# ROSbot Dataset Collection

This is a ROS workspace for collecting a dataset by driving around using the Husarion ROSbot 2.0.

## Environment Setup

Run:
```
python3 -m venv venv-dcoll
. venv-dcoll/bin/activate
pip install -f requirements.txt
```



## Connecting to the ROSbot

If you haven't yet connected your ROSbot to your network, follow instructions [here](https://husarion.com/manuals/rosbot/#system-reinstallation).


First, run `ifconfig` to determine your ip address and the subnet mask. You should see output similar to this:
```
$ ifconfig
    enp3s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
    inet 192.168.0.16  netmask 255.255.255.0  broadcast 192.168.0.255
```

Then, find your ROSBOT_IP using nmap on your gateway IP address (`inet`) and the subnet mask (`netmask`) you just found. It should be listed as device name `husarion`, or possibly `Unknown`.
SSH into your ROSbot and connect it to the same network as your desktop machine:
```
$ sudo nmap -sn 192.168.0.0/24
$ ssh husarion@<ROSBOT_IP>
```
SSH password is "husarion".
You will likely have to change some default values in the ROSbot `~/.bashrc`, such as `ROS_IPV6` which needs to be set to `off`. I recommend adding the following to the end of your `~/.bashrc` file:
```
export ROS_MASTER_URI=http://<DESKTOP_IP>:11311
export ROS_IP=<rosbot_ip>
export ROS_IPV6=off
```

To launch rosserial communication and ROSbot firmware run:
```
source husarion_ws/devel/setup.bash
export ROS_MASTER_URI=http://<DESKTOP_IP>:11311
export ROS_IP=<rosbot_ip>
export ROS_IPV6=off
roslaunch husarion_ros rosbot_drivers.launch
```
To only run sensors, run `roslaunch astra_launch astra.launch`
To only run actuation, run `roslaunch rosbot_ekf all.launch`


On your desktop machine:
```
export ROS_MASTER_URI=http://<DESKTOP_IP>:11311
export ROS_HOSTNAME=<DESKTOP_IP>
export ROS_IP=<ROSBOT_IP>
export ROS_IPV6=off
rostopic list
```

This should return all the topics running on the ROSbot.
Now we are ready to send a motion command:
```
rostopic pub -r 100 /mavros/setpoint_velocity/cmd_vel geometry_msgs/TwistStamped"{header: auto,  twist.linear: {x: 1, y: 2, z: 3}, twist.angular: {x: 1,y: 1,z: 1}"
```

# Basic ROSbot control
In your first terminal on your desktop, run roscore:
```
cd <path/to/ROSbot_data_collection>
. venv-dcoll/bin/activate
source devel/setup.bash
roscore
```

In your second terminal, connect to the ROSbot and start the drivers:
```
ssh husarion@<ROSBOT_IP>
source husarion_ws/devel/setup.bash
export ROS_MASTER_URI=http://<DESKTOP_IP>:11311
export ROS_IP=<ROSBOT_IP>
export ROS_IPV6=off
roslaunch husarion_ros rosbot_drivers.launch
```

Open a second terminal and run:
```
cd <path/to/ROSbot_data_collection>
. venv-dcoll/bin/activate
source devel/setup.bash
roslaunch data_collection data_collector.launch dest:=$(pwd)/dataset

```

Open a third terminal and run:
```
cd <path/to/ROSbot_data_collection>
. venv-dcoll/bin/activate
source devel/setup.bash
sudo apt-get install ros-<distro>-teleop-twist-keyboard
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```

# ROSbot data

## Pre-installed ROSbot packages
```
astra_launch    rosbot_description  route_admin_panel
astra_camera    husarion_ros    rosbot_ekf  rplidar_ros
```

To view what topics ROSbot has available: [ROS API Software](https://husarion.com/manuals/rosbot/#ros-api)

# ROSbot Troubleshooting

## Husarion References

* [Husarion ROS1 Tutorials](https://husarion.com/tutorials/ros-tutorials/1-ros-introduction/)
* [ROSbot simple kinematics](https://husarion.com/tutorials/ros-tutorials/3-simple-kinematics-for-mobile-robot/)
# ROSbot Data Collection Quickstart

This document contains instructions to start your ROSbot so you can collect a dataset. It assumes that the necessary apt packages and ROS scripts have already been installed on your ROSbot.

Hardware required:
* ROSbot
* Wireless Xbox controller
* small form-factor external USB drive (like [this one](https://a.co/d/4fmXYWw))
* Bluetooth 4.0 dongle (like [this one](https://a.co/d/e7X3SpB))

## ROSbot Setup

To charge your ROSbot, follow the Husarion [charging instructions](https://husarion.com/manuals/rosbot/#charging-rosbot).

When you are ready to drive the ROSbot around, follow these steps. 
These directions assume that you are starting with your ROSbot turned off, charged, and nothing plugged in:
1. First, find the two USB ports on the rear panel of the ROSbot. Plug the Bluetooth dongle into the right USB port (labelled XBOX) and the external data storage device into the left USB port (labelled EXT). DO NOT USE the dongle that came with the XBox controller; it only works for Windows systems.
2. Turn your XBox controller into pairing mode. This is accomplished by pressing and holding together the XBox button and the button on the back of the controller with the ")))" symbol next to it. See figure below showing which buttons to hold. When it is in pairing mode, the XBox button will show a fast steady blink. If you don't hold them down long enough, it will show a heartbeat blink.
3. Turn on your ROSbot. It will take ~10 seconds to wake up. The lidar turret will begin to spin and the fast, steady blinking XBox button on the controller will turn to constantly lit to indicate it is paired.
4. You're ready to drive :)
5. When you're ready to stop, simply toggle the on/off switch on the ROSbot. See figure below to locate the on/off switch.


## First time connecting your controller to the ROSbot

You should have your Xbox controller's MAC address before you begin. The easiest way to find it out is to connect it to a bluetooth-enabled laptop and inspect the device using your bluetooth settings.

1. You should also have `bluez` already installed on your ROSbot. Find out by running `bluetoothctl`. If not, run `sudo apt install bluetoothctl`.
2. Run `sudo service bluetooth restart; bluetoothctl`. This will take you into the bluetoothctl prompt.
3. Within the bluetoothctl prompt, run `remove all`
4. Within the bluetoothctl prompt, run `bluetoothctl scan on`
4. Within the bluetoothctl prompt, run `connect <your-controller-MAC>`
5. You should see output similar to ``. If not, or if it is connecting and disconnecting, refer to Troubleshooting.

## Troubleshooting

### WiFi and Bluetooth
If you are using a wireless network and intend to operate the ROSbot outside that network, test your ROSbot startup routine when not connected to that network. 
You may need to disable the WiFi on your ROSbot to successfully complete the startup routine. 
You may also want to change the `ROS_IP` set in `start_rosbot.sh`.

Fix for controller bluetooth disconnect-reconnect: [Bluetooth Problem Ubuntu 18.04 LTS](https://askubuntu.com/questions/1040497/bluetooth-problem-ubuntu-18-04-lts)

### Bluetooth controller connection fix

1. Ensure bluetooth service is started "sudo service bluetooth start"
2. start bluetoothctl via "bluetoothctl" in the command line
3. systemctl restart bluetooth
4. forget [Xbox Wireless Controller] (via the Xbox Controller's mac address)
5. remove [Xbox Wireless Controller]
6. set Xbox controller to pairable mode
7. set pairable on
8. start scan via "scan on"
9. connect [Xbox Wireless Controller]

### Camera calibration
Basics of ROS camera calibration: [ros wiki](http://wiki.ros.org/camera_calibration)

## Husarion References

* [Husarion ROS1 Tutorials](https://husarion.com/tutorials/ros-tutorials/1-ros-introduction/)
* [ROSbot simple kinematics](https://husarion.com/tutorials/ros-tutorials/3-simple-kinematics-for-mobile-robot/)
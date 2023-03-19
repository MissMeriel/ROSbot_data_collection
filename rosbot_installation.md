# ROSbot Setup

This document contains instructions to set up the data collection scripts on your ROSbot so you can run these scripts natively and collect a dataset.

Hardware required:
* ROSbot
* Wireless Xbox controller
* small form-factor external USB drive (like [this one](https://a.co/d/4fmXYWw))
* Bluetooth 4.0 dongle (like [this one](https://a.co/d/e7X3SpB))


## Setup Instructions

1. Copy `start_rosbot.sh` file and `src` directory to your USB drive. Plug the USB drive into your ROSbot.
2. Connect to your ROSbot (ssh, plug in a monitor and mouse/keyboard, etc.).
3. Copy `start_rosbot.sh` to `~` (aka `/home/husarion`). 
4. Give `start_rosbot.sh` executable permissions by running `chmod +x start_rosbot.sh`.
5. Copy the `datacoll` directory inside `src` to `~/husarion_ws/src`.
6. Navigate to `~/husarion_ws/src/datacoll/src` and add executable permissions to all `.py` scripts.
7. `cd ~/husarion_ws/src; source devel/setup.bash`
8. Try running the startup script: `./start_rosbot.sh`. If you experience errors, refer to troubleshooting.
9. If step #8 runs smoothly, add `./start_rosbot.sh` to your startup routine. For more detailed screenshots, see section below on adding scripts to your startup routine.
    1. Go to Dash and type "Start" into the search bar. On HusarionOS, this is the magnifying glass icon at the bottom of the screen.
    2. Double click to open "Session and Startup".
    3. Hit the "Add" icon to add a new routine to startup.
    4. Write a name and description. In the "Command" field, type `/bin/bash -c "sleep 10 & /home/husarion/start_rosbot.sh"`.
10. Test if your script starts on startup. Turn off your ROSbot and turn it back on again. The LiDAR turret should spin within 20-30 seconds. Let it run for a few seconds and then check the external USB drive to see if the dataset wrote to disk.

## Adding scripts to your startup routine on HusarionOS
1. Go to Dash and type "Start" into the search bar. On HusarionOS, this is the magnifying glass icon at the bottom of the screen.
![husarion OS home screen](figures/husarionOS-homescreen.png)
2. Double click to open "Session and Startup".
![](figures/session-and-startup.png)
3. Hit the "Add" icon to add a new routine to startup.
![](figures/session-and-startup-add.png)
4. Write a name and description. In the "Command" field, type `/bin/bash -c "sleep 10 & /home/husarion/start_rosbot.sh"`.
![](figures/session-and-startup-add-complete.png)

## Troubleshooting

If you are using a wireless network, test your ROSbot startup routine when not connected to that network. You may need to disable the WiFi on your ROSbot to successfully complete the startup routine.

Here's a [reference](https://net2.com/how-to-run-applications-at-startup-on-ubuntu-18-04/) for adding scripts to an Ubuntu 18 LSB system.
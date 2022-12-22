# ROSbot Dataset Collection

This repository contains instructions and source code for using the ROSbot to collect a dataset by driving around using the Husarion ROSbot 2.0.
It contains 

## ROSbot Setup

To charge your ROSbot: [link](https://husarion.com/manuals/rosbot/#charging-rosbot)

When you are ready to drive the ROSbot around, follow these steps. 
These directions assume that you are starting with your ROSbot turned off, charged, and nothing plugged in:
1. First, find the two USB ports on the rear panel of the ROSbot. Plug the Bluetooth dongle into the right USB port (labelled XBOX) and the external data storage device into the left USB port (labelled EXT). DO NOT USE the dongle that came with the XBox controller; it only works for Windows systems.
2. Turn your XBox controller into pairing mode. This is accomplished by pressing and holding together the XBox button and the button on the back of the controller with the ")))" symbol next to it. See figure below showing which buttons to hold. When it is in pairing mode, the XBox button will show a fast steady blink. If you don't hold them down long enough, it will show a heartbeat blink.
3. Turn on your ROSbot. It will take ~10 seconds to wake up. The lidar turret will begin to spin and the fast, steady blinking XBox button on the controller will turn to constantly lit to indicate it is paired.
4. You're ready to drive :)
5. When you're ready to stop, simply toggle the on/off switch on the ROSbot. See figure below to locate the on/off switch.

![rosbot rear panel](figures/rosbot-rear-panel.png)
ROSbot rear panel guide.

![xbox pairing buttons](figures/xbox-pairing-buttons.jpeg)
XBox controller pairing buttons.

## Driving your ROSbot
![controller-mapping](figures/xbox-controller-mapping.png)
Upon startup, the ROSbot immediately begins collecting datapoints at a rate of 10Hz.
Each image is 480x640 RGB and ~110KB. It collects 1GB of data approximately every 15 minutes.
Data collection can be paused by pressing A and resumed by pressing B.
The maximum turning speed can be adjusted up or down by pressing right or left on the directional pad, respectively.
The maximum longitudinal (forward/back) speed can be adjusted up or down by pressing up or down on the directional pad, respectively.
Each adjustment changes the max speed by ±0.2 on a scale of (-1, 1).

# ROSbot data
Husarion puts out an annotated list of ROSbot topics via the [ROSbot API](https://husarion.com/manuals/rosbot/#ros-api).
For a full list of available ROSbot data, see [list of published topics](ROSbot-pub-topics.txt).

# ROSbot Troubleshooting

Guide to the LED output on the back: [LEDs and buttons](https://husarion.com/manuals/core2/#leds-and-buttons)

## Husarion References

* [Husarion ROS1 Tutorials](https://husarion.com/tutorials/ros-tutorials/1-ros-introduction/)
* [ROSbot simple kinematics](https://husarion.com/tutorials/ros-tutorials/3-simple-kinematics-for-mobile-robot/)

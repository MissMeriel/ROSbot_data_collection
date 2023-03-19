#!/usr/bin/env python
import rospy
import os
import random
import string
import time
import numpy as np
import cv2
from cv_bridge import CvBridge
import threading

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, CompressedImage, Joy, Range, BatteryState, LaserScan
from std_msgs.msg import String

# globals
MAX_SPEED_LONG = 1
MAX_SPEED_LAT = 1
speed_cmd = 0.0
turn_cmd = 0.0
collection_paused = False

def hook():
    cv2.destroyAllWindows()
    exit(0)

def joy_callback(data):
    global MAX_SPEED_LONG, MAX_SPEED_LAT, speed_cmd, turn_cmd, collection_paused
    # adjust max longitudinal speed
    if data.axes[7] > 0:
        MAX_SPEED_LONG = max(MAX_SPEED_LONG + 0.2, 1.0)
    elif data.axes[7] < 0:
        MAX_SPEED_LONG = max(MAX_SPEED_LONG - 0.2, 0.01)
    # adjust max lateral/turning speed
    if data.axes[6] < 0:
        MAX_SPEED_LAT = max(MAX_SPEED_LAT + 0.2, 1.0)
    elif data.axes[6] > 0:
        MAX_SPEED_LAT = max(MAX_SPEED_LAT - 0.2, 0.01)
    # attenuate speed and turn commands
    speed_cmd = data.axes[1] * MAX_SPEED_LONG
    turn_cmd = data.axes[0] * MAX_SPEED_LAT
    # check if user wants to pause data collection
    if data.buttons[0] > 0 and collection_paused:
        collection_paused = False
    elif data.buttons[1] > 0 and not collection_paused:
        collection_paused = True

def main_loop():
    global speed_cmd, turn_cmd, collection_paused, MAX_SPEED_LAT, MAX_SPEED_LONG
    rospy.init_node('teleop_joy_concurrent', anonymous=False)

    rospy.Subscriber("/joy", Joy, joy_callback, queue_size=1)
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    
    rate = rospy.Rate(5)
    img_count = 0
    # wait for subscribed topics
    rospy.sleep(5)

    while not rospy.is_shutdown():

        cmd_msg = Twist()
        cmd_msg.linear.x = speed_cmd
        cmd_msg.angular.z = turn_cmd
        cmd_vel_pub.publish(cmd_msg)
            
        rate.sleep()

    rospy.spin()
    

if __name__ == '__main__':
    rospy.on_shutdown(hook)
    main_loop()

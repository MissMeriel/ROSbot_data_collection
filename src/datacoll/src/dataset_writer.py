#!/usr/bin/env python
from datetime import datetime
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
image = None
speed_cmd = 0.0
turn_cmd = 0.0
collecting = False
batt_state = vel_state = lidar_state = None
range_fl = range_rl = range_fr = range_rr = None

class ImWriteThread(threading.Thread):
    def __init__(self, dataset_subdir):
        super(ImWriteThread, self).__init__()
        self.im = None
        self.im_timestamp = None
        self.img_filename = ""
        self.speed_cmd = None
        self.turn_cmd = None
        self.batt_state = None
        self.vel_lin_x = None
        # adding other lin and ang values here
        self.vel_lin_y = None
        self.vel_lin_z = None
        self.vel_ang_x = None
        self.vel_ang_y = None
        
        self.vel_ang_z = None
        self.lidar_angle_min = None
        self.lidar_angle_max = None
        self.lidar_angle_increment = None
        self.lidar_range_min = None
        self.lidar_range_max = None
        self.lidar_ranges = None
        self.lidar_intensities = None
        self.range_fl = None
        self.range_fr = None
        self.range_rl = None
        self.range_rr = None
        self.dataset_subdir = dataset_subdir
        self.img_count = 0
        self.timeout = None
        self.done = False
        self.condition = threading.Condition()
        self.start()

    def run(self):
        while not self.done:
            self.condition.acquire()
            self.condition.wait(self.timeout)
            if self.im is not None:
                self.img_filename = "astra-{:05d}.jpg".format(self.img_count)
                cv2.imwrite("{}/{}".format(self.dataset_subdir, self.img_filename), self.im)

                with open(self.dataset_subdir + "/data.csv", 'a') as f:
                    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(self.img_filename, self.im_timestamp, datetime.now(), self.speed_cmd, self.turn_cmd, self.batt_state, self.vel_lin_x, 
                    # adding other lin and ang values here
                    self.vel_lin_y,
                    self.vel_lin_z,
                    self.vel_ang_x,
                    self.vel_ang_y,         
                    self.vel_ang_z, self.lidar_angle_min, self.lidar_angle_max, self.lidar_angle_increment, self.lidar_range_min, self.lidar_range_max, self.lidar_ranges, self.lidar_intensities, self.range_fl, self.range_fr, self.range_rl, self.range_rr))
                self.img_count += 1
                # with open("{}/{}".format(dataset_subdir, self.img_filename), 'wb') as imfile:
                    # np.save(imfile, bridge_img)
            self.condition.release()

    def update(self, im, im_timestamp, speed_cmd, turn_cmd, batt_state, 
               vel_lin_x, vel_lin_y, vel_lin_z, vel_ang_x, vel_ang_y, vel_ang_z,
                    lidar_ranges, lidar_intensities,
                    range_fl, range_fr, range_rl, range_rr):
        self.condition.acquire()
        self.im = im
        self.im_timestamp = im_timestamp
        self.speed_cmd = speed_cmd
        self.turn_cmd = turn_cmd
        self.batt_state = batt_state
        self.vel_lin_x = vel_lin_x
        # adding other lin and ang values here
        self.vel_lin_y = vel_lin_y
        self.vel_lin_z = vel_lin_z
        self.vel_ang_x = vel_ang_x
        self.vel_ang_y = vel_ang_y
        self.vel_ang_z = vel_ang_z
        self.lidar_ranges = lidar_ranges
        self.lidar_intensities = lidar_intensities
        self.range_fl = range_fl
        self.range_fr = range_fr
        self.range_rl = range_rl
        self.range_rr = range_rr
        # self.img_filename = img_filename
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(0, 0, 0, 0, 0, 0)
        self.join()

def hook():
    cv2.destroyAllWindows()
    exit(0)

'''
ROSbot produces 480x640x3 images
'''
def img_callback(data):
    global image
    # rospy.loginfo(rospy.get_caller_id() + " image received %s,%s", data.height, data.width)
    image = data

# Only voltage reported
def battery_callback(data):
    global batt_state
    batt_state = data.voltage

# Only linear.x and angular.z reported
def velocity_callback(data):
    global vel_state
    vel_state = data

# ranges & intensities
def lidar_callback(data):
    global lidar_state
    lidar_state = data

# All ranges have:
#   fov: 0.26
#   min range: 0.03
#   max range: 0.9
# Only changing field is range
def range_fl_callback(data):
    global range_fl
    range_fl = data.range

def range_fr_callback(data):
    global range_fr
    range_fr = data.range

def range_rl_callback(data):
    global range_rl
    range_rl = data.range

def range_rr_callback(data):
    global range_rr
    range_rr = data.range

def cmd_vel_callback(data):
    global speed_cmd, turn_cmd
    speed_cmd = data.linear.x 
    turn_cmd = data.angular.z

def main_loop():
    global image, speed_cmd, turn_cmd, collecting
    global batt_state, vel_state, lidar_state
    global range_fl, range_rl, range_fr, range_rr

    rospy.init_node('ROSbot_dataset_writer_node', anonymous=False)
    thread_lock = threading.Lock()
    dataset_dir = rospy.get_param(rospy.get_name()+"/dest", ".")
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    # timestr = str(localtime.tm_mon) + "_" + str(localtime.tm_mday) + "-" + str(localtime.tm_hour) + "_" + str(localtime.tm_min)
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    dataset_subdir = dataset_dir+"/rosbot-"+timestr+"-"+randstr
    if not os.path.exists(dataset_subdir):
        og_umask = os.umask(0)
        os.makedirs(dataset_subdir, mode=0o777)
        rospy.loginfo("Writing dataset to "+dataset_subdir)
    os.umask(og_umask)

    rospy.Subscriber("/camera/rgb/image_rect_color", Image, img_callback, queue_size=1)
    #rospy.Subscriber("/joy", Joy, joy_callback, queue_size=1)
    rospy.Subscriber("/battery", BatteryState, battery_callback, queue_size=1)
    rospy.Subscriber("/velocity", Twist, velocity_callback, queue_size=1)
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)
    rospy.Subscriber("/range/fl", Range, range_fl_callback, queue_size=1)
    rospy.Subscriber("/range/fr", Range, range_fr_callback, queue_size=1)
    rospy.Subscriber("/range/rl", Range, range_rl_callback, queue_size=1)
    rospy.Subscriber("/range/rr", Range, range_rr_callback, queue_size=1)
    rospy.Subscriber('cmd_vel', Twist, cmd_vel_callback, queue_size = 1)

    # cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    imwrite_thread = ImWriteThread(dataset_subdir)
    bridge = CvBridge()
    rate = rospy.Rate(5)
    try:
        with open(dataset_subdir+"/data.csv", "w") as f:
            
            # write dataframe file header
            # self.lidar_angle_min, self.lidar_angle_max, self.lidar_angle_increment, self.lidar_range_min, self.lidar_range_max, 
            f.write("IMAGE,IMAGE_HEADER_TIME,ROS_TIME,CMD_VEL_LONG,CMD_VEL_LAT,BATT_VOLTAGE,VELOCITY_LIN_X,VELOCITY_LIN_Y,VELOCITY_LIN_Z,VELOCITY_ANG_X,VELOCITY_ANG_Y,VELOCITY_ANG_Z,LIDAR_ANGLE_MIN,LIDAR_ANGLE_MAX,LIDAR_ANGLE_INCREMENT,LIDAR_RANGE_MIN,LIDAR_RANGE_MAX,LIDAR_RANGE,LIDAR_INTENSITY,RANGE_FL,RANGE_FR,RANGE_RL,RANGE_RR\n")
    except Exception as e:
        print(e)
        exit(0)

    try:
        # wait for subscribed topics
        rospy.sleep(5)

        while not rospy.is_shutdown():
            collecting = rospy.get_param("collecting", True)
            # print("img_count", imwrite_thread.img_count)
            if image is not None and collecting:
                bridge_img = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')[...,::-1]
                imwrite_thread.update(bridge_img, str(image.header.stamp.secs) + ":" + str(image.header.stamp.nsecs), speed_cmd, turn_cmd, batt_state, 
                                        vel_state.linear.x, 
                                        # adding other vel values here
                                        vel_state.linear.y,
                                        vel_state.linear.z,
                                        vel_state.angular.x,
                                        vel_state.angular.y,
                                        vel_state.angular.z, 
                                        str(lidar_state.ranges).replace(",", " "), 
                                        str(lidar_state.intensities).replace(",", " "), 
                                        range_fl, range_fr, range_rl, range_rr)

                if imwrite_thread.img_count % 100 == 0:
                    rospy.loginfo("Dataset size="+str(imwrite_thread.img_count))
            rate.sleep()
    except Exception as e:
        print(e)

    finally:
        imwrite_thread.stop()
        restoreTerminalSettings(settings)
        rospy.spin()
    

if __name__ == '__main__':
    rospy.on_shutdown(hook)
    main_loop()

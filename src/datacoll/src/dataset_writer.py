#!/usr/bin/env python
import rospy
import os
import random
import string
import time
import numpy as np
import cv2
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image, CompressedImage, Joy
from std_msgs.msg import String

# globals
image = None
MAX_SPEED_LONG = 1
MAX_SPEED_LAT = 1
speed_cmd = 0.0
turn_cmd = 0.0
collection_paused = False

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

def img_preprocess(img, grayscale=True, crop_size=(200,200), target_size=(224,224)):
    if grayscale:
        img = np.clip(0.07 * img[...,0]  + 0.72 * img[...,1] + 0.21 * img[...,2], 0, 255)

    if target_size is not None:
        if (img.shape[0], img.shape[1]) != target_size:
            img = cv2.resize(img, target_size)

    if crop_size is not None:
        # img = central_image_crop(img, crop_size[0], crop_size[1])
        half_the_width = int(img.shape[1] / 2)
        img = img[img.shape[0] - crop_size[1]: img.shape[0],
              half_the_width - int(crop_size[0] / 2):
              half_the_width + int(crop_size[0] / 2)]

    return np.asarray(img, dtype=np.float32)


def cmd_vel_callback(data):
    # TODO
    pass

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
    global image, speed_cmd, turn_cmd, collection_paused, MAX_SPEED_LAT, MAX_SPEED_LONG
    rospy.init_node('ROSbot_dataset_writer_node', anonymous=False)
    print("In main_loop")
    dataset_dir = rospy.get_param(rospy.get_name()+"/dest", ".")
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    dataset_subdir = dataset_dir+"/rosbot-"+timestr+"-"+randstr
    if not os.path.exists(dataset_subdir):
        og_umask = os.umask(0)
        os.makedirs(dataset_subdir, mode=0o777)
        rospy.loginfo("Writing dataset to "+dataset_subdir)
    os.umask(og_umask)

    rospy.Subscriber("/camera/rgb/image_rect_color", Image, img_callback)
    # rospy.Subscriber("/camera/rgb/image_rect_color/compressed", CompressedImage, img_callback)
    # rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)
    rospy.Subscriber("/joy", Joy, joy_callback)

    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    
    bridge = CvBridge()
    rate = rospy.Rate(5)
    img_count = 0
    with open(dataset_subdir+"/data.csv", "w") as f:
        # write dataframe file header
        f.write("IMAGE,CMD_VEL_LAT,CMD_VEL_LONG,")
        while not rospy.is_shutdown():
            if image is not None:
                bridge_img = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')[...,::-1]
                if img_count % 100 == 0:
                    rospy.loginfo("Dataset size="+str(img_count))

                if not collection_paused:
                    cv2.imshow("/camera/rgb/image_raw", bridge_img)
                    img_filename = "astra-{:05d}.jpg".format(img_count)
                    f.write("{},{},{}\n".format(img_filename, speed_cmd, turn_cmd))
                    cv2.imwrite("{}/{}".format(dataset_subdir, img_filename), bridge_img)
                    img_count += 1
                # else:
                #     cv2.imshow("collection paused", np.zeros((480,640,3)))
                cv2.waitKey(1)
                cmd_msg = Twist()
                cmd_msg.linear.x = speed_cmd
                cmd_msg.angular.z = turn_cmd
                cmd_vel_pub.publish(cmd_msg)
                # speed_cmd = turn_cmd = 0 
        rate.sleep()
    # rospy.spin()
    

if __name__ == '__main__':
    # rospy.on_shutdown(hook)
    main_loop()

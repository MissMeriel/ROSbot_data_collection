#!/usr/bin/env python
import rospy
import os
import random
import string
import time
import numpy as np
import cv2
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String

# globals
image = None

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


def main_loop():
    global image
    rospy.init_node('ROSbot_dataset_writer_node', anonymous=False)
    print("In main_loop")
    dataset_dir = rospy.get_param(rospy.get_name()+"/dest", ".")
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    dataset_subdir = f"{dataset_dir}/rosbot-{timestr}-{randstr}"
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.exists(dataset_subdir):
        os.mkdir(dataset_subdir)
        rospy.loginfo(f"Writing dataset to {dataset_subdir}")

    rospy.Subscriber("/camera/rgb/image_rect_color", Image, img_callback)
    # rospy.Subscriber("/camera/rgb/image_rect_color/compressed", CompressedImage, img_callback)
    rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)

    # pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    bridge = CvBridge()
    rate = rospy.Rate(5)
    img_count = 0
    with open(f"{dataset_subdir}/data.csv", "w") as f:
        # write dataframe file header
        f.write("IMAGE,CMD_VEL,")
        while not rospy.is_shutdown():
            if image is not None:
                bridge_img = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
                if img_count % 100 == 0:
                    rospy.loginfo(f"Dataset size={img_count}")
                # cv2.imshow("/camera/rgb/image_raw",cv_img[...,::-1])
                cv2.imshow("/camera/rgb/image_raw", bridge_img[...,::-1])
                cv2.waitKey(1)
                cv2.imwrite(f"{dataset_subdir}/astra-{img_count:05d}.jpg", bridge_img)
                img_count += 1
            rate.sleep()
    # rospy.spin()
    

if __name__ == '__main__':
    # rospy.on_shutdown(hook)
    main_loop()
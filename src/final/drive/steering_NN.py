# This file imports the model weights & biases from a .pt file
# Uses the imported NN to steer the robot
# MAKE SURE TO SPLIT THE IMAGE, get 2 predictions and then average the steering outputs

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sensor_msgs.msg
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd
import sys
from torchvision.transforms import Compose, ToTensor
# sys.path.append("../models")
from .DAVE2pytorch import DAVE2v3
import torch


class Steering_NN(Node):
    def __init__(self):

        super().__init__('steering_NN')

        # Load model weights & bias
        self.input_shape = (2560, 720) # Change this value to match your input shape. Example: (width x height) for image input
        self.model=torch.load('/home/husarion/ros2_ws/src/final/models/Trained_Models/Dave2-Keras-off-4-plus-corner/model-DAVE2v3-2560x720-lr0.0001-100epoch-64batch-lossMSE-11Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur-epoch56.pt', map_location=torch.device('cpu')) # Change the path to match where you saved the model
        # self.model = DAVE2v3(input_shape=self.input_shape)
        # self.model.load_state_dict(torch.load('/home/husarion/ros2_ws/src/final/models/Trained_Models/Dave2-Keras-off-4-plus-corner/model-DAVE2v3-2560x720-lr0.0001-100epoch-64batch-lossMSE-11Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur-epoch56.pt', map_location=torch.device('cpu')))
        # self.model = torch.load('/home/husarion/ros2_ws/src/final/final/model-DAVE2v3-135x240-lr0.001-50epoch-64batch-lossMSE-7Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur-best.pt')
        self.model.eval()

        self.publisher_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.image_subscription = self.create_subscription(sensor_msgs.msg.Image, '/image_raw', self.image_callback, 10)
        print("lidar was subscribed to")
        self.lidar_subscription = self.create_subscription(sensor_msgs.msg.LaserScan, '/scan', self.lidar_callback, 10)

        self.min_pause_distance = 0.35
        self.obstacle_closeby = False
        self.postObstacle_counterTurn = 0
        self.sign = 1

        self.bridge = CvBridge()
        self.bridged_image = None
        self.unsplit_image = None

        self.left_image = None
        self.right_image = None

        self.vel = Twist()
        self.max_speed = 0.3
        self.vel.linear.x = self.max_speed
        self.vel.angular.z = 0.0

        # Timer callback to publish the velocities at that moment
        self.timer = self.create_timer(0.2, self.timer_callback)


    def timer_callback(self):
        
        #UNSPLIT IMAGE
        if self.unsplit_image == None or self.obstacle_closeby or self.postObstacle_counterTurn > 0:
            return
        transformed_image = Compose([ToTensor()])(self.unsplit_image)
        input_image = transformed_image.unsqueeze(0)
        self.vel.angular.z = self.model(input_image).item()
        print('NEURAL NETWORK TURN')
        # make neural network turns more drastic
        if self.vel.angular.z < 0:
            self.vel.angular.z = self.vel.angular.z 
        if self.vel.angular.z > 0:
            self.vel.angular.z = self.vel.angular.z * 3
        self.publisher_vel.publish(self.vel)
        print(self.vel.angular.z)



        # Model inference to output angular velocity prediction
        #SPLIT IMAGE CODE
        '''
        if self.left_image == None or self.right_image == None or self.obstacle_closeby:
            return

        left_transformed_image = Compose([ToTensor()])(self.left_image)
        right_transformed_image = Compose([ToTensor()])(self.right_image)

        left_input_image = left_transformed_image.unsqueeze(0)
        right_input_image = right_transformed_image.unsqueeze(0)

        left_input_image = torch.autograd.Variable(left_input_image)
        right_input_image = torch.autograd.Variable(right_input_image)

        left_output = self.model(left_input_image)
        right_output = self.model(right_input_image)

        predicted_angular_velocity = (left_output.item()+right_output.item())/2
        self.vel.angular.z = predicted_angular_velocity

        print(self.vel.angular.z)

        self.publisher_vel.publish(self.vel)
        '''
    
    def image_callback(self, msg):
        #might need to do some reversing, not too sure yet
        self.bridged_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
        # cv2.imshow('', self.bridged_image)
        # print("Should get image here")
        img = Image.fromarray(self.bridged_image)
        self.unsplit_image = img.resize((640,360))

        #SPLIT IMAGE CODE
        '''
        img = Image.fromarray(self.bridged_image)
        width, height = img.size

        self.left_image = img.crop((0, 0, width // 2, height))
        self.right_image = img.crop((width // 2, 0, width, height))
        self.left_image = self.left_image.resize((320, 360))
        self.right_image = self.right_image.resize((320, 360))
        '''

        # print("Left image size:", self.left_image.size)
        # print("Right Image size:", self.right_image.size)
    
    def lidar_callback(self, msg):
        #the lidar scans right infront of it as index 0
        lidar_ranges = msg.ranges
        self.vel.linear.x = self.max_speed

        """
        STEERING CORRECTION BUBBLE
        Values go from 0 to 1800 (CCW)
        """
        left_distances = lidar_ranges[0:600]
        right_distances = lidar_ranges[1200:1800]


        left_total = 0
        right_total = 0
        left_close_count = 0
        right_close_count = 0
        bubble_radius = .4
        threshold = len(left_distances) * bubble_radius

        for dist in left_distances:
            if dist == float('inf'):
                left_total += 0.15
                left_close_count += 1
            elif dist > bubble_radius:
                left_total += bubble_radius
            else:
                left_close_count += 1
                left_total += dist
        
        for dist in right_distances:
            if dist == float('inf'):
                right_total += 0.15
                right_close_count += 1
            elif dist > bubble_radius:
                right_total += bubble_radius
            else:
                right_close_count += 1
                right_total += dist

        if self.postObstacle_counterTurn > 0:
            self.vel.angular.z = self.sign * -0.4
            self.postObstacle_counterTurn -= 1

        #if left_total < threshold and left_total < right_total:
        if left_close_count > len(left_distances) // 3:
            self.unsplit_image = None
            self.obstacle_closeby = True
            self.vel.angular.z = (left_total - threshold) * 0.005

        #elif right_total < threshold:
        elif right_close_count > len(right_distances) // 3:
            self.unsplit_image = None
            self.obstacle_closeby = True
            self.vel.angular.z = (threshold - right_total) * 0.005
        else:
            if self.obstacle_closeby == True:
                self.postObstacle_counterTurn = 2
                self.sign = (self.vel.angular.z / abs(self.vel.angular.z))
                self.vel.angular.z = self.sign * -0.4
            print("bubble is not in range")
            self.obstacle_closeby = False

        

        if self.obstacle_closeby or self.postObstacle_counterTurn > 0:
            if not self.obstacle_closeby:
                print('COUNTER TURN')
            else:
                print('BUBBLE TURN')
            self.publisher_vel.publish(self.vel)
            print(self.vel.angular.z)


        """
        STOPPING IN FRONT OF OBSTACLES
        This should also turn away from the direction that had the points closest to it
        """
        
        front_indicies = len(lidar_ranges) // 6 #~60 degrees of points

        # might be really slow
        front_ranges = lidar_ranges[-1 - front_indicies: -1] #the left 60 degrees from the middle 
        front_ranges.extend(lidar_ranges[0:front_indicies]) # the right 60 degrees from the middle
        
        close_counter = 0
        for curr_range in front_ranges:
            if curr_range < self.min_pause_distance or curr_range == float('inf'):
                close_counter += 1
        if close_counter > len(front_ranges) // 3:
            self.obstacle_closeby = True
            #if this becomes false, we want the nn to predict on a new image, not some old stored one
            self.unsplit_image = None
            self.vel.linear.x = -1 * self.max_speed
            self.publisher_vel.publish(self.vel)


            print("TOO CLOSE!!!")
        else:
            self.obstacle_closeby = False



def main(args=None):

    print("hello STEERING")
    rclpy.init(args=args)
    node = Steering_NN()
    print("bonjour")

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
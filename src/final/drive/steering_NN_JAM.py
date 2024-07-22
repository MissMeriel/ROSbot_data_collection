# This file imports the model weights & biases from a .pt file
# Uses the imported NN to steer the robot
# MAKE SURE TO SPLIT THE IMAGE, get 2 predictions and then average the steering outputs

# I mainly rewrote lidar callback so it doesnt do donuts 
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
sys.path.append("home/husarion/ros2_ws/src/final/final")
from DAVE2pytorch import DAVE2v3
import torch


class Steering_NN(Node):
    def __init__(self):

        super().__init__('steering_NN')

        # Load model weights & bias

        # ORIGINAL #
        #self.input_shape = (1344, 376) # Change this value to match your input shape. Example: (width x height) for image input
        
        # ORIGINAL / 2 #
        self.input_shape = (672, 188) # Change this value to match your input shape. Example: (width x height) for image input

        # ORIGINAL / 3 #
        #self.input_shape = (448, 125) # Change this value to match your input shape. Example: (width x height) for image input

        
        
        # models
        #model_path = '/home/husarion/ros2_ws/src/final/models/model-DAVE2v3-1344x376-lr0.0001-100epoch-64batch-lossMSE-5Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur-best-math.pt' # Change the path to match where you saved the model
        #model_path = '/home/husarion/ros2_ws/src/final/models/model-DAVE2v3-1344x376-lr0.0001-35epoch-64batch-lossMSE-7Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur.pt'
        #model_path='/home/husarion/ros2_ws/src/final/models/johann-loops6.pt'
        #model_path='/home/husarion/ros2_ws/src/final/models/johann-loops37.pt'
        #model_path='/home/husarion/ros2_ws/src/final/models/johann_first.pt'
        #model_path='/home/husarion/ros2_ws/src/final/models/MathModels/model-DAVE2v3-1344x376-lr0.0001-100epoch-64batch-lossMSE-5Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur-best-math.pt'
        #model_path = '/home/husarion/ros2_ws/src/final/models/math_first.pt'
        #model_path = '/home/husarion/ros2_ws/src/final/models/model-DAVE2v3-1344x376-lr0.0001-35epoch-64batch-lossMSE-7Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur.pt'
        #model_path = '/home/husarion/ros2_ws/src/final/models/MathModels/model-DAVE2v3-1344x376-lr0.0001-100epoch-64batch-lossMSE-32Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur-best-Math-week-6.pt'
        #model_path='/home/husarion/ros2_ws/src/final/models/johann_third.pt'
        model_path = '/home/husarion/ros2_ws/src/final/models/20240718/model-DAVE2v3-448x125-lr0.0001-100epoch-64batch-lossMSE-7Ksamples-robustificationFalse-LOOPSTRAIGHT45-LOWERIMG.pt'
        #model_path = '/home/husarion/ros2_ws/src/final/models/20240718/model-DAVE2v3-448x125-lr0.0001-100epoch-64batch-lossMSE-7Ksamples-robustificationFalse-LOOPSTRAIGHT45-LOWERIMG-epoch94.pt'
        #model_path = '/home/husarion/ros2_ws/src/final/models/20240718/model-DAVE2v3-1344x376-lr0.0001-100epoch-64batch-lossMSE-7Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur.pt'
        # model_path = '/home/husarion/ros2_ws/src/final/models/model-DAVE2v3-448x125-lr0.0001-45epoch-64batch-lossMSE-8Ksamples-robustificationFalse-LOOPSTRAIGHT45-LOWERIMG.pt'       
        #model_path = '/home/husarion/ros2_ws/src/final/models/libraryinitial/model-DAVE2v3-448x125-lr0.0001-100epoch-64batch-lossMSE-8Ksamples-robustificationFalse-LOOPSTRAIGHT45-LOWERIMG.pt'
        #model_path = '/home/husarion/ros2_ws/src/final/models/libraryinitialaugment/model-DAVE2v3-448x125-lr0.0001-100epoch-64batch-lossMSE-8Ksamples-robustificationFalse-LOOPSTRAIGHT45-LOWERIMG.pt'
        #model_path = '/home/husarion/ros2_ws/src/final/models/libraryinitialaugment/model-DAVE2v3-448x125-lr0.0001-100epoch-64batch-lossMSE-8Ksamples-robustificationFalse-LOOPSTRAIGHT45-LOWERIMG.pt'
        model_path = '/home/husarion/ros2_ws/src/final/models/model-DAVE2v3-672x188-lr0.0001-100epoch-64batch-lossMSE-7Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur-epoch99.pt'

        try:
            self.model = DAVE2v3(input_shape=self.input_shape)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        except TypeError as e:
            try:
                self.model = torch.load(model_path, map_location=torch.device('cpu'))
            except TypeError as e:
                print(e)

        self.publisher_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        
        self.lidar_subscription = self.create_subscription(sensor_msgs.msg.LaserScan, '/scan', self.lidar_callback, 10)
        #print("lidar was subscribed to")
        self.image_subscription = self.create_subscription(sensor_msgs.msg.Image, '/image_raw', self.image_callback, 3)
        
        

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
        print("Inside timer_callback")
        #UNSPLIT IMAGE
        print(f"{self.unsplit_image=}, {self.obstacle_closeby=}, {self.postObstacle_counterTurn=}")
        if self.unsplit_image == None or self.obstacle_closeby or self.postObstacle_counterTurn > 0:
            return
        transformed_image = Compose([ToTensor()])(self.unsplit_image)
        input_image = transformed_image.unsqueeze(0)
        self.vel.angular.z = self.model(input_image).item()
        
        # make neural network turns more drastic
        if self.vel.angular.z < 0:
            self.vel.angular.z = self.vel.angular.z 
        if self.vel.angular.z > 0:
            self.vel.angular.z = self.vel.angular.z * 3
        self.publisher_vel.publish(self.vel)
        print(f"NEURAL NETWORK TURN: {self.vel.angular.z:.3f}")



        # Model inference to output angular velocity prediction
        #SPLIT IMAGE CODE
        """
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
        """
    
    def image_callback(self, msg):
        #might need to do some reversing, not too sure yet
        self.bridged_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
        # cv2.imshow('', self.bridged_image)
        print("Inside image_callback")
        #self.unsplit_image = Image.fromarray(self.bridged_image)
        img = Image.fromarray(self.bridged_image)
        #self.unsplit_image = img.resize((448,125))
        self.unsplit_image = img.resize(self.input_shape)



        #SPLIT IMAGE CODE
        """
        img = Image.fromarray(self.bridged_image)
        width, height = img.size

        self.left_image = img.crop((0, 0, width // 2, height))
        self.right_image = img.crop((width // 2, 0, width, height))
        self.left_image = self.left_image.resize((672, 376))
        self.right_image = self.right_image.resize((672, 376))
        
        """
        # print("Left image size:", self.left_image.size)
        # print("Right Image size:", self.right_image.size)
    
    
    def lidar_callback(self,msg):
        # Does not make donuts
        print("inside lidar callback")
        lidar_ranges = msg.ranges
        left_distances = lidar_ranges[0:600]
        right_distances = lidar_ranges[1200:1800]
        #self.twist = Twist()
        self.safedist = .3 
        #self.velsign = min(msg.ranges)/(abs(min(msg.ranges)))

        
        if abs(min(left_distances)) < self.safedist:
            print("obstacle detected LEFT")
            print("yellow")
            self.vel.linear.x = 0.0
            
            # backup and correct
        # self.vel.linear.x = -0.4
        # self.vel.angular.z = -0.4
        elif abs(min(right_distances)) < self.safedist:
            print("obstacle detected RIGHT")
            print("blue")
            self.vel.linear.x = 0.0
            
            # backup and correct
        # self.vel.linear.x = -0.4
        # self.vel.angular.z = 0.4
        else: 
            print("red")
            self.vel.linear.x = 0.3
            #self.vel.angular.z = 0.0

        print('green')
        
        #self.publisher_vel.publish(self.vel)


def main(args=None):

    print("hello STEERING")
    rclpy.init(args=args)
    node = Steering_NN()
    print("bonjour")

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
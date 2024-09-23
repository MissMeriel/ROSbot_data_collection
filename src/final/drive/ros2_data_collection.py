import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
import cv2
import numpy as np
import os
from rclpy.parameter import Parameter
import os
import time

class DataCollectionNode(Node):
    def __init__(self):

        super().__init__('ros2_data_collection')
        self.subscription_twist = self.create_subscription(Twist, '/cmd_vel', self.vel_cmd_callback, 10)
        self.subscription_image = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.subscription_joystick = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.lidar_subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        
        self.declare_parameters(namespace='', parameters=[("dest", rclpy.Parameter.Type.STRING)])

        # self.subscription_image.add_on_set_parameters_callback(self.on_parameters_set).add_on_set_parameters_callback(self.on_parameters_set).add_on_change_callback(self.on_parameter_change)
                
        # self.subscription = self.create_subscription(Image, '/zed_camera/rgb/right_image', self.right_image_callback, 10)

        self.bridge = CvBridge()
        self.bridged_image = None
        self.lidar_ranges = None
        self.image_num = 1
        
        self.linear_speed_x = 0.0
        self.angular_speed_z = 0.0
        self.is_turning = False
        self.is_manually_off_course = False

        self.parent_path = self.get_parameter("dest").value
        self.get_logger().info(f"{self.parent_path=}")
        self.dataset_subdir_id = "rosbotxl_data/"
        os.makedirs(f"{self.parent_path}/{self.dataset_subdir_id}/", exist_ok=True)
        self.collection_iter = len(os.listdir(f"{self.parent_path}/{self.dataset_subdir_id}"))
        self.dataset_subsubdir_id = f"collection{self.collection_iter:03d}"
        os.makedirs(f"{self.parent_path}/{self.dataset_subdir_id}/{self.dataset_subsubdir_id}", exist_ok=True)

        self.get_logger().info(f"Logging to {self.parent_path}/{self.dataset_subdir_id}/{self.dataset_subsubdir_id}")
        self.dataset_path = f"{self.parent_path}/{self.dataset_subdir_id}/{self.dataset_subsubdir_id}/data.csv"

        # write csv header
        with open(self.dataset_path, 'a') as f:
            f.write("{},{},{},{},{},{},{}\n".format("timestamp, image_name", "linear_speed_x", "angular_speed_z", "is_turning", "is_manually_off_course", "lidar_ranges"))

        #timer callback to save the image and publish the velocities at that moment
        self.timer = self.create_timer(0.2, self.timer_callback)


    def timer_callback(self):
        if np.array_equal(self.bridged_image, None):
            return
        image_filename = "rivian-{:05d}.jpg".format(self.image_num) #:05d means 5 places
        cv2.imwrite(f"{self.parent_path}/{self.dataset_subdir_id}/{self.dataset_subsubdir_id}/{image_filename}", self.bridged_image[:,:,::-1])

        with open(self.dataset_path, 'a') as f:
            if self.lidar_ranges is not None:
                stringified_lidar_ranges = [str(i) for i in self.lidar_ranges]
                stringified_lidar_ranges = " ".join(stringified_lidar_ranges)
                #self.get_logger().info(stringified_lidar_ranges)
            else:
                stringified_lidar_ranges = None
                
            f.write("{},{},{},{},{},{},{}\n".format(time.time(), image_filename, self.linear_speed_x, self.angular_speed_z, self.is_turning, self.is_manually_off_course, stringified_lidar_ranges))

        self.image_num += 1 #will need to diff image numbers if we are doing left and right cameras

    def vel_cmd_callback(self, msg):
        self.linear_speed_x = msg.linear.x
        self.angular_speed_z = msg.angular.z
        
        #Test through logger
        #self.get_logger().info(f"Velocity command received: linear_speed_x={self.linear_speed_x}, angular_speed_z={self.angular_speed_z}, is_turning={self.is_turning}, is_manually_off_course={self.is_manually_off_course}")

    
    def image_callback(self, msg):
        self.bridged_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
    
    def joy_callback(self, msg):
        if msg.axes[2] != -0.0 or msg.axes[3] != -0.0:
            self.is_turning = True
        else:
            self.is_turning = False
         #Test through logger   
        #self.get_logger().info(f"Joystick command received: axes2={msg.axes[2]}, axes3={msg.axes[3]}")
        #self.get_logger().info(f"Joystick command received: is_turning={self.is_turning}")
        if msg.buttons[0] == 1: #Xbox Controller button 'A'
        	self.is_manually_off_course = True
        else:
        	self.is_manually_off_course = False
        
    def lidar_callback(self, msg):
        #the lidar scans right infront of it as index 0
        
        self.lidar_ranges = np.array(msg.ranges, dtype=np.float32)

    def on_parameters_set(self, parameters):
        self.get_logger().info("Parameters set: {}".format(parameters))

    def on_parameter_change(self, parameters):
        self.get_logger().info("Parameter change: {}".format(parameters))


def main(args=None):
    rclpy.init(args=args)
    print("args", args)
    node = DataCollectionNode()

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

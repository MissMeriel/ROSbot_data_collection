import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class DriveNode(Node):
    def __init__(self):
        super().__init__('drive')
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 1)

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.max_speed = 0.3

        self.vel = Twist()
        self.vel.linear.x = 0.0
        self.vel.angular.z = 0.0

    def joy_callback(self, msg):
        if abs(msg.axes[1]) < 0.1:
            self.vel.linear.x = 0.0
        else:
            self.vel.linear.x = self.max_speed if msg.axes[1] > 0 else -self.max_speed
        
        # previously: 
        if abs(msg.axes[2]) < 0.1: # Protects angular motion against small, un-intended movements on right joystick
            self.vel.angular.z = 0.0
        else:
            self.vel.angular.z = msg.axes[2]
        # if abs(msg.axes[2]) < 0.1:
        #     self.vel.angular.z = 0.0
        # elif abs(msg.axes[2]) < 0.75:
        #     self.vel.angular.z = 0.3
        # else:
        #     self.vel.angular.z = 0.8
        # self.vel.angular.z = self.vel.angular.z if msg.axes[2] > 0 else -self.vel.angular.z
        self.publisher.publish(self.vel)


    def timer_callback(self):
        # print("msg", self.vel)
        #self.publisher.publish(self.vel)
        pass
    

def main(args=None):
    rclpy.init(args=args)
    node = DriveNode()

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
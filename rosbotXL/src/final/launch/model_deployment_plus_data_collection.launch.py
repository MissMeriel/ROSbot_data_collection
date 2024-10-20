from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.substitutions import TextSubstitution
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    dest_launch_arg = DeclareLaunchArgument("dest", default_value=TextSubstitution(text="/home/husarion/media/usb3"))
	
    return LaunchDescription([
    	dest_launch_arg,
    	IncludeLaunchDescription(
    	    PythonLaunchDescriptionSource([
    	        PathJoinSubstitution([
    	            FindPackageShare('rplidar_ros'),
    	            'launch',
    	            'view_rplidar_a3_launch.py'
    	        ])
    	    ]),
    	    launch_arguments={}.items()
    	),
        Node(
            package='joy',
            namespace='',
            executable='joy_node',
            name='joy'
        ),
        Node(
            package='final',
            executable='steering_NN',
            name='sim',
            parameters=[
            	{"dest": LaunchConfiguration("dest")},
            ],
            output='screen',
            emulate_tty=True
        ),
        Node(
            package='final',
            executable='ros2_data_collection',
            name='sim',
            parameters=[
            	{"dest": LaunchConfiguration("dest")},
            ],
            output='screen',
            emulate_tty=True
        ),
        Node(
            package='usb_cam',
            namespace ='',
            executable='usb_cam_node_exe',
            arguments=['--ros-args', '--params-file', '/home/husarion/.ros/camera_info/default_cam.yaml']
            #parameters=[{'params-file': '/home/husarion/.ros/camera_info/default_cam.yaml'}]
        )
    ])

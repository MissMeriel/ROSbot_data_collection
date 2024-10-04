from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'final'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	(os.path.join('share', package_name, 'launch'), glob('launch/*launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='husarion',
    maintainer_email='husarion@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drive = final.drive:main', 
            'ros2_data_collection = final.ros2_data_collection:main', 
            'steering_NN = final.steering_NN:main',
            'onnx_steering_NN_transformer = final.onnx_steering_NN_transformer:main'
        ],
    },
)

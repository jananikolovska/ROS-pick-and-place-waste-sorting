from setuptools import setup
from glob import glob
import os

package_name = 'ax_inference_node_python'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('ax_inference_node_python/launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Ax Camera YOLO ROS2 package',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'inference_python = ax_inference_node_python.inference_node:main',
        ],
    },
)

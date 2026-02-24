from setuptools import setup
from glob import glob
import os

package_name = 'helper_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('helper_nodes/launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Helper ROS2 package',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'camera_node = helper_nodes.camera_node:main',
            'display_node = helper_nodes.display_node:main',
            'video_stream_node = helper_nodes.video_stream_node:main',
            'save_annotated_node = helper_nodes.save_annotated_node:main',
        ],
    },
)

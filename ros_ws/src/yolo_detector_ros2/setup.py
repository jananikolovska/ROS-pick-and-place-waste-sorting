from setuptools import setup

package_name = "yolo_detector_ros2"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="your_name",
    maintainer_email="you@example.com",
    description="YOLO detector: subscribes to /camera_frame, saves annotated frames, publishes /detections.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_detector_node = yolo_detector_ros2.yolo_detector_node:main",
        ],
    },
)
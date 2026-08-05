"""
Copyright MIT
MIT License

BWSI Autonomous RACECAR Course
Racecar Neo LTS

File Name: camera_real.py
File Description: Contains the Camera module of the racecar_core library
"""

from camera import Camera

# General
import cv2 as cv
import numpy as np
class NDArray:  # inline stub; nptyping is deliberately not a dependency
    def __class_getitem__(cls, item): return cls

# ROS2
import rclpy as ros2
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSProfile,
)
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


_NO_DEPTH = (
    "The NeoRacer has no depth camera. Its forward camera is a monocular USB "
    "webcam, not the RealSense D435i the racecar_neo platform carries, so "
    "there is no depth stream to read. Use rc.lidar for range data."
)


class CameraReal(Camera):
    # The NeoRacer camera node publishes the forward USB camera here, under
    # the same name realsense.launch.py remaps the D435i color stream onto.
    # There is no /camera/depth: this car has no depth sensor.
    __COLOR_TOPIC = "/camera/color"

    def __init__(self):
        self.__bridge = CvBridge()
        self.node = ros2.create_node("image_sub", start_parameter_services=False)

        qos_profile = QoSProfile(depth=10)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE

        self.__color_image_sub = self.node.create_subscription(
            Image, self.__COLOR_TOPIC, self.__color_callback, qos_profile
        )
        self.__color_image = None
        self.__color_image_new = None

    def __color_callback(self, data):
        # The NeoRacer camera node republishes the USB camera's native MJPG
        # buffer with encoding 'jpeg', so the frame is decoded here rather than
        # handed to cv_bridge.
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            self.__color_image_new = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        except CvBridgeError as e:
            print(f"camera_real: failed to decode color frame: {e}")

    def __update(self):
        self.__color_image = self.__color_image_new

    def get_color_image_no_copy(self) -> NDArray[(480, 640, 3), np.uint8]:
        return self.__color_image

    def get_color_image_async(self) -> NDArray[(480, 640, 3), np.uint8]:
        return self.__color_image_new

    def get_depth_image(self) -> NDArray[(480, 640), np.float32]:
        raise NotImplementedError(_NO_DEPTH)

    def get_depth_image_async(self) -> NDArray[(480, 640), np.float32]:
        raise NotImplementedError(_NO_DEPTH)

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


class CameraReal(Camera):
    # RACECAR topics fed by the RealSense D435i; realsense.launch.py remaps the
    # RealSense color/image_raw and depth/image_rect_raw onto these names. On
    # NeoRacer the color stream comes from the USB camera node instead and
    # there is no depth sensor; see __color_callback and get_depth_image.
    __COLOR_TOPIC = "/camera/color"
    __DEPTH_TOPIC = "/camera/depth"

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

        self.__depth_image_sub = self.node.create_subscription(
            Image, self.__DEPTH_TOPIC, self.__depth_callback, qos_profile
        )
        self.__depth_image = None
        self.__depth_image_new = None

    def __color_callback(self, data):
        # The NeoRacer camera node republishes the USB camera's native MJPG
        # buffer with encoding 'jpeg', so the frame is decoded here rather than
        # handed to cv_bridge.
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            self.__color_image_new = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        except CvBridgeError as e:
            print(f"camera_real: failed to decode color frame: {e}")

    def __depth_callback(self, data):
        # RealSense depth is 16UC1 (uint16) in millimeters; the API returns cm.
        try:
            depth_mm = self.__bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.__depth_image_new = depth_mm.astype(np.float32) / 10.0
        except CvBridgeError as e:
            print(f"camera_real: failed to decode depth frame: {e}")

    def __update(self):
        self.__color_image = self.__color_image_new
        self.__depth_image = self.__depth_image_new

    def get_color_image_no_copy(self) -> NDArray[(480, 640, 3), np.uint8]:
        return self.__color_image

    def get_color_image_async(self) -> NDArray[(480, 640, 3), np.uint8]:
        return self.__color_image_new

    def get_depth_image(self) -> NDArray[(480, 640), np.float32]:
        return self.__depth_image

    def get_depth_image_async(self) -> NDArray[(480, 640), np.float32]:
        return self.__depth_image_new

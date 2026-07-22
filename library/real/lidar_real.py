"""
Copyright MIT
MIT License

BWSI Autonomous RACECAR Course
Racecar Neo LTS

File Name: lidar_real.py
File Description: Contains the Lidar module of the racecar_core library
"""

from lidar import Lidar

# General
import numpy as np
class NDArray:  # inline stub; nptyping is deliberately not a dependency
    def __class_getitem__(cls, item): return cls

# ROS2
import rclpy as ros2
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class LidarReal(Lidar):
    # The ROS topic from which we get Lidar data
    __SCAN_TOPIC = "/scan"

    # The NeoRacer's LakiBeam1 publishes a full-circle scan of ~1440 bins
    # (0.25 deg per bin). The sensor's live window is 270 deg (45-315 in its
    # own frame): the rear ~90 deg wedge never carries a return and reads 0.
    _NUM_SAMPLES: int = 1440

    def __init__(self):
        # ROS node
        self.node = ros2.create_node("scan_sub", start_parameter_services=False)

        # subscribe to the scan topic, which will call
        # __scan_callback every time the lidar sends data
        self.__scan_sub = self.node.create_subscription(
            LaserScan, self.__SCAN_TOPIC, self.__scan_callback, qos_profile_sensor_data
        )

        self.__samples = np.empty(0)
        self.__samples_new = np.empty(0)

    # Meters -> cm; flip to clockwise order to match the sim convention; roll a
    # half turn so index 0 is the front of the car. No-return samples (inf) are
    # zeroed to match the sim, so the blind rear wedge reads 0.
    def __scan_callback(self, data):
        scan_data = np.asarray(data.ranges, dtype=np.float32) * 100.0
        scan_data = np.roll(np.flip(scan_data), len(scan_data) // 2)
        scan_data[~np.isfinite(scan_data)] = 0.0
        self.__samples_new = scan_data

    def __update(self):
        self.__samples = self.__samples_new

    def get_samples(self) -> NDArray[1440, np.float32]:
        return self.__samples

    def get_samples_async(self) -> NDArray[1440, np.float32]:
        return self.__samples_new

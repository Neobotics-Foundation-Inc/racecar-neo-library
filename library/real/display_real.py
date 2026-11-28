"""
Copyright MIT
GNU General Public License v3.0

BWSI Autonomous RACECAR Course
Racecar Neo LTS

File Name: display_real.py
File Description: Contains the Display module of the racecar_core library
"""
import cv2 as cv
import numpy as np
import os
class NDArray:  # stub - no runtime dependency on nptyping
    def __class_getitem__(cls, _): return cls

import rclpy as ros2
from std_msgs.msg import String

from display import Display


_NO_PIXELS = (
    "The NeoRacer dot matrix takes text only. Its 8x8 panel is driven by a "
    "serial firmware that renders ASCII, with no raw-pixel or intensity "
    "command, unlike the MAX7219 chain on the racecar_neo platform. Use "
    "rc.display.show_text() instead."
)


class DisplayReal(Display):
    __WINDOW_NAME: str = "RACECAR display window"
    __DISPLAY: str = ":0"
    __TEXT_TOPIC: str = "/dotmatrix/text"
    # Mirrors idle_text in neoracer_ros2_driver/config/led_matrix.yaml, which
    # led_matrix_node writes on startup and shutdown.
    __IDLE_TEXT: str = "N"

    def __init__(self, isHeadless):
        Display.__init__(self, isHeadless)
        self.__display_found = \
            self.__DISPLAY in os.popen('cd /tmp/.X11-unix && for x in X*; do echo ":${x#X}"; done ').read()

        if self.__display_found:
            os.environ["DISPLAY"] = self.__DISPLAY
        else:
            print(f"Display {self.__DISPLAY} not found.")

        # The panel is an 8x8 module behind a serial firmware that accepts
        # text only (see neoracer_ros2_driver/led_matrix_node.py), so only
        # show_text() reaches hardware. There is no /dotmatrix/pixels here.
        self.node = ros2.create_node("display_node", start_parameter_services=False)
        self.__text_pub = self.node.create_publisher(
            String, self.__TEXT_TOPIC, qos_profile=1
        )

    def create_window(self) -> None:
        if not self._Display__isHeadless and self.__display_found:
            cv.namedWindow(self.__WINDOW_NAME)

    def show_color_image(self, image: NDArray) -> None:
        if not self._Display__isHeadless and self.__display_found:
            cv.imshow(self.__WINDOW_NAME, image)
            cv.waitKey(1)

    def set_matrix(self, matrix: NDArray[(8, 24), np.uint8]) -> None:
        raise NotImplementedError(_NO_PIXELS)

    def get_matrix(self) -> NDArray[(8, 24), np.uint8]:
        raise NotImplementedError(_NO_PIXELS)

    def show_text(self, text: str, scroll_speed: float = 2.0) -> None:
        # The panel firmware does not use scroll_speed. It scrolls text that is
        # wider than the 8-pixel panel by itself.
        msg = String()
        msg.data = text
        self.__text_pub.publish(msg)

    def clear(self) -> None:
        # The panel holds the last frame written, so returning it to the idle
        # frame is the clear. show_text(" ") blanks it instead.
        self.show_text(self.__IDLE_TEXT)

    def set_matrix_intensity(self, intensity: float) -> None:
        raise NotImplementedError(_NO_PIXELS)

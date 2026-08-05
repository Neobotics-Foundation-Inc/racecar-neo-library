"""
Copyright MIT
MIT License

BWSI Autonomous RACECAR Course
Racecar Neo LTS

File Name: led_real.py
File Description: Contains the Led module of the racecar_core library.
The NeoRacer carries no addressable LED strip, so every method raises rather
than publishing to /led/pixels. The API is kept so code written against the
racecar_neo platform imports and type-checks unchanged; it fails where it
would have driven hardware that is not there.
"""

from typing import List, Tuple

from led import Led

import rclpy as ros2

Color = Tuple[int, int, int]

_NO_STRIP = (
    "The NeoRacer has no addressable LED strip. rc.led drives the WS2812B "
    "strip on the racecar_neo NEO-PIT board, which this car does not carry. "
    "Use rc.display for the 8x8 dot matrix instead."
)


class LedReal(Led):
    def __init__(self):
        # The node exists only so racecar_core_real can register it alongside
        # the other modules; it advertises nothing.
        self.node = ros2.create_node("led_pub", start_parameter_services=False)

    def get_num_pixels(self) -> int:
        raise NotImplementedError(_NO_STRIP)

    def set_pixel(self, index: int, color: Color) -> None:
        raise NotImplementedError(_NO_STRIP)

    def set_color(self, color: Color) -> None:
        raise NotImplementedError(_NO_STRIP)

    def set_pixels(self, colors: List[Color]) -> None:
        raise NotImplementedError(_NO_STRIP)

    def clear(self) -> None:
        raise NotImplementedError(_NO_STRIP)

    def get_pixels(self) -> List[Color]:
        raise NotImplementedError(_NO_STRIP)

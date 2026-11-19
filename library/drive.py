"""
Copyright MIT
GNU General Public License v3.0

BWSI Autonomous RACECAR Course
Racecar Neo LTS

File Name: drive.py
File Description: Defines the interface of the Drive module of the racecar_core library.
"""

import abc


class Drive(abc.ABC):
    """
    Controls the car's speed and steering.
    """

    @abc.abstractmethod
    def set_speed_angle(self, speed: float, angle: float) -> None:
        """
        Sets the car's target speed and the angle of the front wheels.

        Args:
            speed: The target speed from -1.0 (full backward) to 1.0 (full forward),
                scaled by the maximum speed (see set_max_speed).
            angle: The amount to turn the front wheels from -1.0 (full left) to 1.0
                (full right).

        Note:
            The speed and angle arguments are unitless ratios. The car holds the
            requested speed with a closed-loop controller on its drive encoder.

        Example::

            if counter < 1:
                # Drive straight forward at full speed
                rc.drive.set_speed_angle(1, 0)
            elif counter < 2:
                # Drive reverse at full speed with the wheels pointing fully to the left
                rc.drive.set_speed_angle(-1, -1)
            else:
                # Drive 70% to the right at half speed forward
                rc.drive.set_speed_angle(0.5, 0.7)
        """
        pass

    def stop(self) -> None:
        """
        Brings the car to a stop and points the front wheels forward.

        Note:
            stop is equivalent to rc.drive.set_speed_angle(0, 0)

        Example::

            # Stops the car if the counter is greater than 5
            if counter > 5:
                rc.drive.stop()
        """
        self.set_speed_angle(0, 0)

    @abc.abstractmethod
    def set_max_speed(self, max_speed: float = 1.0) -> None:
        """
        Sets the maximum speed in the forward and backward direction.

        Args:
            max_speed: The scale factor applied to speed inputs, ranging from
                0.0 to 1.0. Defaults to 1.0 on the NeoRacer and 0.25 in
                RacecarSim.

        Note:
            The NeoRacer starts at a max speed of 1.0 (full scale). A speed of
            1.0 then commands the driver's top speed, which is 6 m/s in the
            driver's default settings. RacecarSim starts at a max speed of 0.25.
            Lower the max speed to slow the whole program down in one place.

        Warning:
            The RACECAR contains expensive and fragile equipment.  Please only run
            full speed in a safe environment without the potential for hard
            collisions.

        Example::

            # Update the max speed to 0.5
            rc.drive.set_max_speed(0.5)
        """
        pass

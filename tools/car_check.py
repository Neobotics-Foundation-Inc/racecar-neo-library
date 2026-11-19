"""
Neobotics Foundation Inc.
GPLv3 License
racecar-neo-library/tools/

File Name: car_check.py

Title: Library Car Check

Author: Neobotics

Version: 0.1.0

Purpose: Call every rc.* getter of the student library once on the NeoRacer and
print what each one returns.

Methodology:
- Load the library from this clone, ahead of any copy installed on the car.
- Start the real racecar_core. On the NeoRacer, go() enters the program at once.
- Show CHECK on the dot matrix, then wait three seconds so the battery reading arrives.
- Call each getter once and print its value, or the shape and range of an array.
- Expect NotImplementedError with its message from the depth, LED, dot matrix pixel
  and battery current calls. The NeoRacer has no depth camera, no LED strip and no
  current sensor, and its dot matrix takes text only.
- Count a call as failed when it raises anything else, when the camera or the lidar
  returns no data, when the camera image is not 640 x 480 with three color channels,
  or when the lidar scan is not 1440 samples long.
- Clear the dot matrix, print a summary, and exit with status 1 when a call failed.

Expected Outcome: The car does not move. The dot matrix shows CHECK for about
three seconds and then returns to its idle letter. The terminal prints one line
per call and a summary line, and the program exits on its own.
- A healthy car: every line starts with OK, RAISES or WARN, the summary reports 0
  failed, and the exit status is 0.
- The camera or the lidar is off: its lines start with FAIL and the exit status is 1.
- The battery reading has not arrived yet: the voltage line starts with WARN.
- SLAM is not running: rc.slam reports mode "off", no pose and no map. That is not
  a failure.
- A call hangs: the terminal stops after the last line it printed. Press Ctrl+C.
- Ctrl+C before the summary: the program exits with status 1 and prints no summary.
  The dot matrix keeps showing CHECK until another program writes to it.

Run it on the car from this folder: python3 car_check.py
"""

import os
import sys
from pathlib import Path

import numpy as np

# import the racecar library.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "library"))
import racecar_core
import racecar_utils as rc_utils

# >> Constants <<

# WAIT_S - seconds of frames before the calls, so the battery reading (every two seconds) arrives
# IMAGE_SHAPE - the rows, columns and color channels of a NeoRacer camera image
# LIDAR_SAMPLES - the length of a NeoRacer lidar scan
# NO_DEPTH - the start of the depth camera NotImplementedError message
# NO_STRIP - the start of the LED strip NotImplementedError message
# NO_PIXELS - the start of the dot matrix pixel NotImplementedError message
# NO_CURRENT - the start of the battery current NotImplementedError message
WAIT_S = 3.0
IMAGE_SHAPE = (480, 640, 3)
LIDAR_SAMPLES = 1440
NO_DEPTH = "The NeoRacer has no depth camera"
NO_STRIP = "The NeoRacer has no addressable LED strip"
NO_PIXELS = "The NeoRacer dot matrix takes text only"
NO_CURRENT = "The NeoRacer cannot measure battery current"

# >> Variables <<

rc = racecar_core.create_racecar(False)
elapsed = 0.0
reported = False
counts = {"OK": 0, "RAISES": 0, "WARN": 0, "FAIL": 0}


"""
Identifier: describe
Usage: Turns a getter's return value into one short line of text.
Parameters:
- value: the value a getter returned
Outputs:
- the text: shape, type and range for an array, the length for a list, else the value itself
"""
def describe(value):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return f"empty array {value.shape}"
        return (
            f"array {value.shape} {value.dtype}, "
            f"min {float(np.min(value)):.4g}, max {float(np.max(value)):.4g}"
        )
    if isinstance(value, list):
        return f"list of {len(value)}: {value[:3]!r}{' ...' if len(value) > 3 else ''}"
    return repr(value)


"""
Identifier: report
Usage: Prints one result line and counts it.
Parameters:
- status: OK, RAISES, WARN or FAIL
- name: the call, as a student would write it
- detail: what the call returned or raised
Outputs:
- none
"""
def report(status, name, detail):
    counts[status] += 1
    print(f"{status:<6} {name}: {detail}", flush=True)


"""
Identifier: check
Usage: Calls a getter once and reports its value. A call that raises fails.
Parameters:
- name: the call, as a student would write it
- call: a function that makes the call
- need_data: True when None or an empty result counts as a failure
Outputs:
- the value the call returned, or None when it raised
"""
def check(name, call, need_data=False):
    try:
        value = call()
        if need_data and (value is None or len(value) == 0):
            report("FAIL", name, f"returned no data ({describe(value)})")
        else:
            report("OK", name, describe(value))
    except Exception as error:
        report("FAIL", name, f"raised {type(error).__name__}: {error}")
        return None
    return value


"""
Identifier: expect_raise
Usage: Calls a function for hardware the NeoRacer lacks and checks that it raises
NotImplementedError with the expected message.
Parameters:
- name: the call, as a student would write it
- call: a function that makes the call
- message: the start of the expected message
Outputs:
- none
"""
def expect_raise(name, call, message):
    try:
        value = call()
    except NotImplementedError as error:
        if str(error).startswith(message):
            report("RAISES", name, f"NotImplementedError: {error}")
        else:
            report("FAIL", name, f"NotImplementedError with another message: {error}")
        return
    except Exception as error:
        report("FAIL", name, f"raised {type(error).__name__} instead: {error}")
        return
    report("FAIL", name, f"returned {describe(value)} instead of raising")


"""
Identifier: run_checks
Usage: Calls every rc.* getter once, plus the calls that must raise on the NeoRacer.
Parameters:
- none
Outputs:
- one printed line per call
"""
def run_checks():
    check("rc.get_delta_time()", rc.get_delta_time)

    # camera
    image = check("rc.camera.get_color_image()", rc.camera.get_color_image, need_data=True)
    check("rc.camera.get_color_image_no_copy()", rc.camera.get_color_image_no_copy, need_data=True)
    check("rc.camera.get_color_image_async()", rc.camera.get_color_image_async, need_data=True)
    width = check("rc.camera.get_width()", rc.camera.get_width)
    height = check("rc.camera.get_height()", rc.camera.get_height)
    check("rc.camera.get_max_range()", rc.camera.get_max_range)
    if image is not None and len(image) > 0:
        if np.shape(image) == IMAGE_SHAPE and (height, width) == IMAGE_SHAPE[:2]:
            report("OK", "camera image size", f"{width} x {height}, {IMAGE_SHAPE[2]} color channels")
        else:
            report(
                "FAIL",
                "camera image size",
                f"image {np.shape(image)}, get_width() {width}, get_height() {height}",
            )
    expect_raise("rc.camera.get_depth_image()", rc.camera.get_depth_image, NO_DEPTH)
    expect_raise("rc.camera.get_depth_image_async()", rc.camera.get_depth_image_async, NO_DEPTH)

    # controller
    button, trigger, joystick = rc.controller.Button, rc.controller.Trigger, rc.controller.Joystick
    check("rc.controller.is_down(Button.A)", lambda: rc.controller.is_down(button.A))
    check("rc.controller.was_pressed(Button.A)", lambda: rc.controller.was_pressed(button.A))
    check("rc.controller.was_released(Button.A)", lambda: rc.controller.was_released(button.A))
    check("rc.controller.get_trigger(Trigger.LEFT)", lambda: rc.controller.get_trigger(trigger.LEFT))
    check("rc.controller.get_trigger(Trigger.RIGHT)", lambda: rc.controller.get_trigger(trigger.RIGHT))
    check("rc.controller.get_joystick(Joystick.LEFT)", lambda: rc.controller.get_joystick(joystick.LEFT))
    check("rc.controller.get_joystick(Joystick.RIGHT)", lambda: rc.controller.get_joystick(joystick.RIGHT))

    # display
    check("rc.display.new_matrix()", rc.display.new_matrix)
    expect_raise("rc.display.get_matrix()", rc.display.get_matrix, NO_PIXELS)
    expect_raise(
        "rc.display.set_matrix(matrix)",
        lambda: rc.display.set_matrix(rc.display.new_matrix()),
        NO_PIXELS,
    )
    expect_raise(
        "rc.display.set_matrix_intensity(0.5)",
        lambda: rc.display.set_matrix_intensity(0.5),
        NO_PIXELS,
    )

    # led
    expect_raise("rc.led.get_num_pixels()", rc.led.get_num_pixels, NO_STRIP)
    expect_raise("rc.led.get_pixels()", rc.led.get_pixels, NO_STRIP)
    expect_raise("rc.led.set_pixel(0, red)", lambda: rc.led.set_pixel(0, (255, 0, 0)), NO_STRIP)
    expect_raise("rc.led.set_color(red)", lambda: rc.led.set_color((255, 0, 0)), NO_STRIP)
    expect_raise("rc.led.set_pixels([red])", lambda: rc.led.set_pixels([(255, 0, 0)]), NO_STRIP)
    expect_raise("rc.led.clear()", rc.led.clear, NO_STRIP)

    # lidar
    scan = check("rc.lidar.get_samples()", rc.lidar.get_samples, need_data=True)
    check("rc.lidar.get_samples_async()", rc.lidar.get_samples_async, need_data=True)
    count = check("rc.lidar.get_num_samples()", rc.lidar.get_num_samples)
    if scan is not None and len(scan) > 0:
        if len(scan) == LIDAR_SAMPLES and count == LIDAR_SAMPLES:
            report("OK", "lidar scan length", f"{len(scan)} samples, front reads {scan[0]:.1f} cm")
        else:
            report("FAIL", "lidar scan length", f"{len(scan)} samples, get_num_samples() {count}")

    # physics
    check("rc.physics.get_linear_acceleration()", rc.physics.get_linear_acceleration)
    check("rc.physics.get_angular_velocity()", rc.physics.get_angular_velocity)
    check("rc.physics.get_magnetic_field()", rc.physics.get_magnetic_field)
    check("rc.physics.get_encoder_speed()", rc.physics.get_encoder_speed)
    try:
        voltage = rc.physics.get_battery_voltage()
    except Exception as error:
        report("FAIL", "rc.physics.get_battery_voltage()", f"raised {type(error).__name__}: {error}")
    else:
        if voltage > 0:
            report("OK", "rc.physics.get_battery_voltage()", describe(voltage))
        else:
            report("WARN", "rc.physics.get_battery_voltage()", f"{voltage!r}: no /battery/state reading yet")
    expect_raise("rc.physics.get_battery_current()", rc.physics.get_battery_current, NO_CURRENT)
    check("rc.physics.get_rc_channels()", rc.physics.get_rc_channels)

    # slam
    check("rc.slam.get_mode()", rc.slam.get_mode)
    check("rc.slam.get_map_name()", rc.slam.get_map_name)
    check("rc.slam.is_localized()", rc.slam.is_localized)
    check("rc.slam.get_pose()", rc.slam.get_pose)
    check("rc.slam.get_pose_async()", rc.slam.get_pose_async)
    check("rc.slam.get_map()", rc.slam.get_map)
    check("rc.slam.get_save_status()", rc.slam.get_save_status)

    # telemetry and vision
    check("rc.telemetry.get_diagnostics()", lambda: sorted(rc.telemetry.get_diagnostics()))
    check("rc.vision.get_detections()", rc.vision.get_detections)
    check("rc.vision.get_detections_async()", rc.vision.get_detections_async)

    # utils
    check("rc_utils.get_data_dir()", rc_utils.get_data_dir)


"""
Identifier: start
Usage: Holds the car still and shows CHECK on the dot matrix while the sensors start.
Parameters:
- none
Outputs:
- none
"""
def start():
    rc.drive.stop()
    check('rc.display.show_text("CHECK")', lambda: rc.display.show_text("CHECK"))
    print(f">> Waiting {WAIT_S:.0f} seconds for sensor data...", flush=True)


"""
Identifier: update
Usage: Counts the frames' time, then runs every check once, prints the summary,
and stops the program.
Parameters:
- none
Outputs:
- none
"""
def update():
    global elapsed, reported
    if reported:
        return
    elapsed += rc.get_delta_time()
    if elapsed < WAIT_S:
        return
    reported = True
    try:
        run_checks()
    except Exception as error:
        report("FAIL", "remaining checks", f"skipped after {type(error).__name__}: {error}")
    check("rc.display.clear()", rc.display.clear)
    print(
        f">> {counts['OK']} OK, {counts['RAISES']} raised as expected, "
        f"{counts['WARN']} warnings, {counts['FAIL']} failed",
        flush=True,
    )
    # Exit at once with the result. A Ctrl+C sent from inside update() lets the
    # run thread publish once more after ROS shuts down, which prints a traceback.
    os._exit(1 if counts["FAIL"] else 0)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
    # go() returns only when the program stops before update() exits with the
    # result, for example on Ctrl+C, so the check did not finish.
    sys.exit(1)

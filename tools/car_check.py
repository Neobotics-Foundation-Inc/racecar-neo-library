"""
Neobotics Foundation Inc.
GPLv3 License
racecar-neo-library/tools/

File Name: car_check.py

Title: Library Car Check

Author: Neobotics

Version: 0.2.0

Purpose: Call every rc.* getter of the student library once on the NeoRacer and
report the results the way a test suite does.

Methodology:
- Load the library from this clone, ahead of any copy installed on the car.
- Hold everything the library prints, its start-up lines included. The held lines
  are shown at the end only when a check fails or the run stops early.
- Start the real racecar_core. On the NeoRacer, go() enters the program at once.
- Show CHECK on the dot matrix, then wait three seconds so the battery reading arrives.
- Run each check once. A check passes when its getter returns, or when a call for
  hardware the NeoRacer lacks raises NotImplementedError with its message. The
  NeoRacer has no depth camera, no LED strip and no current sensor, and its dot
  matrix takes text only.
- A check fails when a call raises anything else, when the camera or the lidar
  returns no data, when the camera image is not 640 x 480 with three color channels,
  or when the lidar scan is not 1440 samples long. A check that needs an earlier
  result that is missing is skipped.
- Print one row of marks per module, then the failures, the warnings and a summary
  line. With -v, print one line per check with its value instead of the marks.
- Clear the dot matrix and exit with status 1 when a check failed.

Expected Outcome: The car does not move. The dot matrix shows CHECK for about
three seconds and then returns to its idle letter. The terminal prints a header,
one row of marks per module and a summary line, and the program exits on its own.
- A healthy car: every mark is a dot, the summary line reads "51 passed", and the
  exit status is 0.
- The camera or the lidar is off: its marks are F, the FAILURES section names each
  failed check with what it returned, the library's held lines follow, and the exit
  status is 1.
- The battery reading has not arrived yet: its mark is W, and the warnings section
  says so. A warning does not change the exit status.
- SLAM is not running: rc.slam reports mode "off", no pose and no map. That is not
  a failure.
- A call hangs: the terminal stops partway through a row. Press Ctrl+C. The program
  prints the checks it finished, the check that was still running, the library's
  held lines and a summary line, and exits with status 1. The dot matrix keeps
  showing CHECK until another program writes to it.

Run it on the car from this folder: python3 car_check.py, or python3 car_check.py -v
"""

import io
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# import the racecar library.
LIBRARY = Path(__file__).resolve().parent.parent / "library"
sys.path.insert(0, str(LIBRARY))
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
# VALUE, DATA, RAISES, CUSTOM - how a check judges its call (see outcome)
# MARKS, WORDS - how a result prints in a row of marks and with -v
# WIDTH - the width of the rows and the banners, from the terminal
WAIT_S = 3.0
IMAGE_SHAPE = (480, 640, 3)
LIDAR_SAMPLES = 1440
NO_DEPTH = "The NeoRacer has no depth camera"
NO_STRIP = "The NeoRacer has no addressable LED strip"
NO_PIXELS = "The NeoRacer dot matrix takes text only"
NO_CURRENT = "The NeoRacer cannot measure battery current"
VALUE, DATA, RAISES, CUSTOM = "value", "data", "raises", "custom"
MARKS = {"PASS": ".", "FAIL": "F", "WARN": "W", "SKIP": "s"}
WORDS = {"PASS": "PASSED", "FAIL": "FAILED", "WARN": "WARNING", "SKIP": "SKIPPED"}
WIDTH = min(max(shutil.get_terminal_size((80, 24)).columns, 60), 100)

# >> Variables <<

# OUT - the terminal. sys.stdout holds the library's own lines in HELD until the end.
OUT = sys.stdout
HELD = io.StringIO()
sys.stdout = HELD
COLOR = OUT.isatty()
VERBOSE = "-v" in sys.argv[1:] or "--verbose" in sys.argv[1:]
began = time.monotonic()
rc = None
elapsed = 0.0
reported = False
results = []
values = {}
row = {"section": None, "length": 0}
running = None


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
Identifier: paint
Usage: Colors text for the terminal, as pytest does. Text sent to a file stays plain.
Parameters:
- text: the text to color
- status: PASS (green), FAIL (red), WARN or SKIP (yellow)
Outputs:
- the text, with color codes when the terminal shows them
"""
def paint(text, status):
    if not COLOR:
        return text
    code = {"PASS": "32", "FAIL": "31", "WARN": "33", "SKIP": "33"}[status]
    return f"\033[{code}m{text}\033[0m"


"""
Identifier: banner
Usage: Centers a title in a line of = signs, as pytest prints its headers.
Parameters:
- title: the text in the middle
Outputs:
- the line
"""
def banner(title):
    return f" {title} ".center(WIDTH, "=")


"""
Identifier: end_row
Usage: Finishes the current row of marks with the share of checks done so far.
Parameters:
- none
Outputs:
- none
"""
def end_row():
    if row["section"] is None:
        return
    done = f"[{len(results) * 100 // TOTAL:3d}%]"
    OUT.write(" " * max(1, WIDTH - row["length"] - len(done)) + done + "\n")
    OUT.flush()
    row["section"] = None


"""
Identifier: report
Usage: Records one result and prints its mark, or its line with -v.
Parameters:
- section: the module the check belongs to, which names its row
- name: the call, as a student would write it
- status: PASS, FAIL, WARN or SKIP
- detail: what the call returned or raised
Outputs:
- none
"""
def report(section, name, status, detail):
    if VERBOSE:
        results.append((section, name, status, detail))
        OUT.write(f"{paint(f'{WORDS[status]:<7}', status)} {name}: {detail}\n")
        OUT.flush()
        return
    if row["section"] != section:
        end_row()
        row["section"], row["length"] = section, len(section) + 1
        OUT.write(section + " ")
    results.append((section, name, status, detail))
    OUT.write(paint(MARKS[status], status))
    OUT.flush()
    row["length"] += 1


"""
Identifier: outcome
Usage: Makes one call and judges what it returned.
Parameters:
- name: the call, as a student would write it
- kind: VALUE passes on any return; DATA also needs a non-empty result; RAISES needs
  NotImplementedError with the message; CUSTOM returns its own status and detail
- call: a function that makes the call
- message: for RAISES, the start of the expected message
Outputs:
- the status and the detail. An exception from the call goes to the caller.
"""
def outcome(name, kind, call, message):
    if kind == RAISES:
        try:
            value = call()
        except NotImplementedError as error:
            if str(error).startswith(message):
                return "PASS", f"raises NotImplementedError: {message}"
            return "FAIL", f"NotImplementedError with another message: {error}"
        return "FAIL", f"returned {describe(value)} instead of raising"
    value = call()
    if kind == CUSTOM:
        return value
    values[name] = value
    if kind == DATA and (value is None or len(value) == 0):
        return "FAIL", f"returned no data ({describe(value)})"
    return "PASS", describe(value)


"""
Identifier: run
Usage: Runs one check and reports it. A call that raises fails.
Parameters:
- check: (section, name, kind, call) or (section, name, RAISES, call, message)
Outputs:
- none
"""
def run(check):
    global running
    section, name, kind, call = check[:4]
    message = check[4] if len(check) > 4 else None
    running = f"{section}: {name}"
    try:
        status, detail = outcome(name, kind, call, message)
    except Exception as error:
        status, detail = "FAIL", f"raised {type(error).__name__}: {error}"
    running = None
    report(section, name, status, detail)


"""
Identifier: image_size
Usage: Compares the camera image read earlier with get_width() and get_height().
Parameters:
- none
Outputs:
- the status and the detail; skipped when there was no image
"""
def image_size():
    image = values.get("rc.camera.get_color_image()")
    width, height = values.get("rc.camera.get_width()"), values.get("rc.camera.get_height()")
    if image is None or len(image) == 0:
        return "SKIP", "no camera image to measure"
    if np.shape(image) == IMAGE_SHAPE and (height, width) == IMAGE_SHAPE[:2]:
        return "PASS", f"{width} x {height}, {IMAGE_SHAPE[2]} color channels"
    return "FAIL", f"image {np.shape(image)}, get_width() {width}, get_height() {height}"


"""
Identifier: scan_length
Usage: Compares the lidar scan read earlier with 1440 and get_num_samples().
Parameters:
- none
Outputs:
- the status and the detail; skipped when there was no scan
"""
def scan_length():
    scan = values.get("rc.lidar.get_samples()")
    count = values.get("rc.lidar.get_num_samples()")
    if scan is None or len(scan) == 0:
        return "SKIP", "no lidar scan to measure"
    if len(scan) == LIDAR_SAMPLES and count == LIDAR_SAMPLES:
        return "PASS", f"{len(scan)} samples, front reads {scan[0]:.1f} cm"
    return "FAIL", f"{len(scan)} samples, get_num_samples() {count}"


"""
Identifier: battery_voltage
Usage: Reads the battery voltage. A reading of 0 means none has arrived yet.
Parameters:
- none
Outputs:
- the status and the detail
"""
def battery_voltage():
    voltage = rc.physics.get_battery_voltage()
    if voltage > 0:
        return "PASS", describe(voltage)
    return "WARN", f"{voltage!r}: no /battery/state reading yet"


# >> Checks <<

# Each check is (section, name, kind, call), or (section, name, RAISES, call, message).
# The calls read rc when they run, after main() has made it.
START_CHECKS = [
    ("start", 'rc.display.show_text("CHECK")', VALUE, lambda: rc.display.show_text("CHECK")),
]
CHECKS = [
    ("core", "rc.get_delta_time()", VALUE, lambda: rc.get_delta_time()),
    ("camera", "rc.camera.get_color_image()", DATA, lambda: rc.camera.get_color_image()),
    ("camera", "rc.camera.get_color_image_no_copy()", DATA, lambda: rc.camera.get_color_image_no_copy()),
    ("camera", "rc.camera.get_color_image_async()", DATA, lambda: rc.camera.get_color_image_async()),
    ("camera", "rc.camera.get_width()", VALUE, lambda: rc.camera.get_width()),
    ("camera", "rc.camera.get_height()", VALUE, lambda: rc.camera.get_height()),
    ("camera", "rc.camera.get_max_range()", VALUE, lambda: rc.camera.get_max_range()),
    ("camera", "camera image size", CUSTOM, image_size),
    ("camera", "rc.camera.get_depth_image()", RAISES, lambda: rc.camera.get_depth_image(), NO_DEPTH),
    ("camera", "rc.camera.get_depth_image_async()", RAISES,
     lambda: rc.camera.get_depth_image_async(), NO_DEPTH),
    ("controller", "rc.controller.is_down(Button.A)", VALUE,
     lambda: rc.controller.is_down(rc.controller.Button.A)),
    ("controller", "rc.controller.was_pressed(Button.A)", VALUE,
     lambda: rc.controller.was_pressed(rc.controller.Button.A)),
    ("controller", "rc.controller.was_released(Button.A)", VALUE,
     lambda: rc.controller.was_released(rc.controller.Button.A)),
    ("controller", "rc.controller.get_trigger(Trigger.LEFT)", VALUE,
     lambda: rc.controller.get_trigger(rc.controller.Trigger.LEFT)),
    ("controller", "rc.controller.get_trigger(Trigger.RIGHT)", VALUE,
     lambda: rc.controller.get_trigger(rc.controller.Trigger.RIGHT)),
    ("controller", "rc.controller.get_joystick(Joystick.LEFT)", VALUE,
     lambda: rc.controller.get_joystick(rc.controller.Joystick.LEFT)),
    ("controller", "rc.controller.get_joystick(Joystick.RIGHT)", VALUE,
     lambda: rc.controller.get_joystick(rc.controller.Joystick.RIGHT)),
    ("display", "rc.display.new_matrix()", VALUE, lambda: rc.display.new_matrix()),
    ("display", "rc.display.get_matrix()", RAISES, lambda: rc.display.get_matrix(), NO_PIXELS),
    ("display", "rc.display.set_matrix(matrix)", RAISES,
     lambda: rc.display.set_matrix(rc.display.new_matrix()), NO_PIXELS),
    ("display", "rc.display.set_matrix_intensity(0.5)", RAISES,
     lambda: rc.display.set_matrix_intensity(0.5), NO_PIXELS),
    ("led", "rc.led.get_num_pixels()", RAISES, lambda: rc.led.get_num_pixels(), NO_STRIP),
    ("led", "rc.led.get_pixels()", RAISES, lambda: rc.led.get_pixels(), NO_STRIP),
    ("led", "rc.led.set_pixel(0, red)", RAISES, lambda: rc.led.set_pixel(0, (255, 0, 0)), NO_STRIP),
    ("led", "rc.led.set_color(red)", RAISES, lambda: rc.led.set_color((255, 0, 0)), NO_STRIP),
    ("led", "rc.led.set_pixels([red])", RAISES, lambda: rc.led.set_pixels([(255, 0, 0)]), NO_STRIP),
    ("led", "rc.led.clear()", RAISES, lambda: rc.led.clear(), NO_STRIP),
    ("lidar", "rc.lidar.get_samples()", DATA, lambda: rc.lidar.get_samples()),
    ("lidar", "rc.lidar.get_samples_async()", DATA, lambda: rc.lidar.get_samples_async()),
    ("lidar", "rc.lidar.get_num_samples()", VALUE, lambda: rc.lidar.get_num_samples()),
    ("lidar", "lidar scan length", CUSTOM, scan_length),
    ("physics", "rc.physics.get_linear_acceleration()", VALUE, lambda: rc.physics.get_linear_acceleration()),
    ("physics", "rc.physics.get_angular_velocity()", VALUE, lambda: rc.physics.get_angular_velocity()),
    ("physics", "rc.physics.get_magnetic_field()", VALUE, lambda: rc.physics.get_magnetic_field()),
    ("physics", "rc.physics.get_encoder_speed()", VALUE, lambda: rc.physics.get_encoder_speed()),
    ("physics", "rc.physics.get_battery_voltage()", CUSTOM, battery_voltage),
    ("physics", "rc.physics.get_battery_current()", RAISES,
     lambda: rc.physics.get_battery_current(), NO_CURRENT),
    ("physics", "rc.physics.get_rc_channels()", VALUE, lambda: rc.physics.get_rc_channels()),
    ("slam", "rc.slam.get_mode()", VALUE, lambda: rc.slam.get_mode()),
    ("slam", "rc.slam.get_map_name()", VALUE, lambda: rc.slam.get_map_name()),
    ("slam", "rc.slam.is_localized()", VALUE, lambda: rc.slam.is_localized()),
    ("slam", "rc.slam.get_pose()", VALUE, lambda: rc.slam.get_pose()),
    ("slam", "rc.slam.get_pose_async()", VALUE, lambda: rc.slam.get_pose_async()),
    ("slam", "rc.slam.get_map()", VALUE, lambda: rc.slam.get_map()),
    ("slam", "rc.slam.get_save_status()", VALUE, lambda: rc.slam.get_save_status()),
    ("telemetry", "rc.telemetry.get_diagnostics()", VALUE, lambda: sorted(rc.telemetry.get_diagnostics())),
    ("vision", "rc.vision.get_detections()", VALUE, lambda: rc.vision.get_detections()),
    ("vision", "rc.vision.get_detections_async()", VALUE, lambda: rc.vision.get_detections_async()),
    ("utils", "rc_utils.get_data_dir()", VALUE, lambda: rc_utils.get_data_dir()),
]
FINISH_CHECKS = [
    ("finish", "rc.display.clear()", VALUE, lambda: rc.display.clear()),
]
TOTAL = len(START_CHECKS) + len(CHECKS) + len(FINISH_CHECKS)


"""
Identifier: finish
Usage: Prints the failures, the warnings, the library's held lines when needed, and
the summary line, then exits.
Parameters:
- stopped: why the run ended early, or None when every check ran
Outputs:
- none; the program exits with status 1 when a check failed or the run stopped early
"""
def finish(stopped=None):
    end_row()
    failed = [r for r in results if r[2] == "FAIL"]
    warned = [r for r in results if r[2] == "WARN"]
    if stopped:
        OUT.write("\n" + banner(f"stopped: {stopped}") + "\n")
        if running:
            OUT.write(f"still running: {running}\n")
    for title, picked in (("FAILURES", failed), ("warnings summary", warned)):
        if picked:
            OUT.write("\n" + banner(title) + "\n")
            for section, name, _, detail in picked:
                OUT.write(f"{section}: {name}\n    {detail}\n")
    held = HELD.getvalue().strip()
    if (failed or stopped) and held:
        OUT.write("\n" + banner("library output") + "\n" + held + "\n")
    counts = [
        (len(failed), "failed"),
        (sum(r[2] == "PASS" for r in results), "passed"),
        (len(warned), "warning" if len(warned) == 1 else "warnings"),
        (sum(r[2] == "SKIP" for r in results), "skipped"),
        (TOTAL - len(results) if stopped else 0, "not run"),
    ]
    summary = ", ".join(f"{n} {word}" for n, word in counts if n)
    status = "FAIL" if failed or stopped else "WARN" if warned else "PASS"
    OUT.write("\n" + paint(banner(f"{summary} in {time.monotonic() - began:.1f}s"), status) + "\n")
    OUT.flush()
    # Exit at once with the result. A Ctrl+C sent from inside update() lets the
    # run thread publish once more after ROS shuts down, which prints a traceback.
    os._exit(1 if failed or stopped else 0)


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
    OUT.write(f"collected {TOTAL} checks\n\n")
    for check in START_CHECKS:
        run(check)


"""
Identifier: update
Usage: Counts the frames' time, then runs every check once, clears the dot matrix
and finishes.
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
        for check in CHECKS:
            run(check)
    except Exception as error:
        report("finish", "remaining checks", "FAIL", f"skipped after {type(error).__name__}: {error}")
    for check in FINISH_CHECKS:
        run(check)
    finish()


"""
Identifier: main
Usage: Prints the header, starts the library and runs the checks from start() and update().
Parameters:
- none
Outputs:
- none; finish() ends the program
"""
def main():
    global rc
    OUT.write(banner("NeoRacer car check") + "\n")
    OUT.write(f"library: {LIBRARY}\n")
    OUT.write("starting the library and waiting for sensor data ...\n")
    OUT.flush()
    try:
        rc = racecar_core.create_racecar(False)
        rc.set_start_update(start, update)
        rc.go()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        finish(f"the library did not start ({type(error).__name__}: {error})")
    # go() returns only when the program stops before update() finishes, for
    # example on Ctrl+C, so the check did not finish.
    finish("Ctrl+C before the check finished")


if __name__ == "__main__":
    main()

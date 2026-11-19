"""The NeoRacer's changes to the real (ROS 2) modules: drive, display, LED, lidar, go(), nodes."""

import math
import threading
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")


@pytest.fixture
def ros(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "95")  # away from a running car
    rclpy.init()
    from rclpy.executors import SingleThreadedExecutor

    executor = SingleThreadedExecutor()
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield executor
    executor.shutdown()
    rclpy.try_shutdown()


def wait(done, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        time.sleep(0.02)
    return done()


def listen(executor, msg_type, topic):
    """Subscribe a test node to a topic and return the list its messages land in."""
    node = rclpy.create_node("test_listener")
    received = []
    node.create_subscription(msg_type, topic, received.append, 10)
    executor.add_node(node)
    return received


# -- drive --


def test_drive_starts_at_full_speed(ros):
    from drive_real import DriveReal

    drive = DriveReal()
    message = drive._DriveReal__message
    drive.set_speed_angle(1.0, 0.0)
    assert message.drive.speed == pytest.approx(1.0)
    drive.set_max_speed(0.5)
    drive.set_speed_angle(1.0, 0.0)
    assert message.drive.speed == pytest.approx(0.5)
    drive.set_max_speed()
    drive.set_speed_angle(-1.0, 0.0)
    assert message.drive.speed == pytest.approx(-1.0)
    # Integer inputs still reach the message as floats.
    drive.set_max_speed(1)
    drive.set_speed_angle(1, 0)
    assert isinstance(message.drive.speed, float)


def test_drive_publishes_on_update_without_a_timer(ros):
    from ackermann_msgs.msg import AckermannDriveStamped
    from drive_real import DriveReal

    drive = DriveReal()
    assert list(drive._DriveReal__node.timers) == []
    received = listen(ros, AckermannDriveStamped, "/drive")
    drive.set_speed_angle(0.5, 0.25)
    time.sleep(0.2)
    assert received == []  # nothing publishes until the run loop calls __update
    assert wait(lambda: (drive._DriveReal__update(), len(received))[1] > 0)
    msg = received[0]
    assert msg.drive.speed == pytest.approx(0.5)
    assert msg.drive.steering_angle == pytest.approx(-0.25)  # positive angle is right
    assert msg.header.frame_id == "base_link"


# -- display and LED --


def test_display_clear_shows_the_idle_text(ros):
    from display_real import DisplayReal
    from std_msgs.msg import String

    display = DisplayReal(True)
    received = listen(ros, String, "/dotmatrix/text")
    assert wait(lambda: (display.clear(), len(received))[1] > 0)
    assert received[-1].data == "N"


def test_display_pixel_calls_raise(ros):
    from display_real import DisplayReal

    display = DisplayReal(True)
    for call in (
        lambda: display.set_matrix(display.new_matrix()),
        display.get_matrix,
        lambda: display.set_matrix_intensity(0.5),
    ):
        with pytest.raises(NotImplementedError, match="The NeoRacer dot matrix takes text only"):
            call()


def test_led_calls_raise(ros):
    from led_real import LedReal

    led = LedReal()
    for call in (
        led.get_num_pixels,
        lambda: led.set_pixel(0, (255, 0, 0)),
        lambda: led.set_color((255, 0, 0)),
        lambda: led.set_pixels([(255, 0, 0)]),
        led.clear,
        led.get_pixels,
    ):
        with pytest.raises(NotImplementedError, match="The NeoRacer has no addressable LED strip"):
            call()


# -- lidar --


def test_lidar_keeps_the_neoracer_geometry(ros):
    from lidar_real import LidarReal
    from sensor_msgs.msg import LaserScan

    lidar = LidarReal()
    ros.add_node(lidar.node)
    pub_node = rclpy.create_node("fake_lidar")
    pub = pub_node.create_publisher(LaserScan, "/scan", 10)
    ranges = [1.0 + i * 0.001 for i in range(1440)]
    ranges[5] = math.inf  # no return
    msg = LaserScan()
    msg.ranges = ranges
    assert wait(lambda: (pub.publish(msg), len(lidar.get_samples_async()))[1] > 0)
    lidar._LidarReal__update()
    scan = lidar.get_samples()

    assert lidar.get_num_samples() == 1440 and len(scan) == 1440
    # Meters become cm. Index 0 is the front of the car, half a turn from the
    # sensor's index 0, and the samples run clockwise.
    assert scan[0] == pytest.approx(ranges[719] * 100, rel=1e-6)
    assert scan[1] == pytest.approx(ranges[718] * 100, rel=1e-6)
    assert int((scan == 0).sum()) == 1  # the inf reads 0


# -- nodes --


def test_library_nodes_skip_parameter_services(ros, monkeypatch):
    created = []
    real_create_node = rclpy.create_node

    def create_node(name, **kwargs):
        created.append((name, kwargs.get("start_parameter_services", True)))
        return real_create_node(name, **kwargs)

    monkeypatch.setattr(rclpy, "create_node", create_node)
    from camera_real import CameraReal
    from controller_real import ControllerReal
    from display_real import DisplayReal
    from drive_real import DriveReal
    from led_real import LedReal
    from lidar_real import LidarReal
    from physics_real import PhysicsReal
    from slam_real import SlamReal
    from telemetry_real import TelemetryReal
    from vision_real import VisionReal

    for make in (
        CameraReal,
        lambda: ControllerReal(None),
        lambda: DisplayReal(True),
        DriveReal,
        LedReal,
        LidarReal,
        PhysicsReal,
        SlamReal,
        TelemetryReal,
        VisionReal,
    ):
        make()
    assert len(created) == 10
    assert [name for name, services in created if services] == []


# -- go() --


class FakeExecutor:
    """Spins go()'s loop; ends it after a set number of spins so a regression fails, not hangs."""

    def __init__(self, error=None, max_spins=50):
        self.error = error
        self.max_spins = max_spins
        self.spins = 0
        self.racecar = None

    def spin_once(self, timeout_sec=None):
        if self.error is not None:
            raise self.error
        self.spins += 1
        if self.spins >= self.max_spins:
            self.racecar._RacecarReal__running = False


class FakeLidar:
    def get_samples(self):
        return np.ones(1440, dtype=np.float32)


def bare_racecar(executor):
    """A RacecarReal with only what go() reads, so no node or thread starts."""
    from racecar_core_real import RacecarReal

    rc = RacecarReal.__new__(RacecarReal)
    rc._RacecarReal__executor = executor
    rc._RacecarReal__user_start = None
    rc._RacecarReal__user_update = None
    rc._RacecarReal__user_update_slow = None
    rc.lidar = FakeLidar()
    executor.racecar = rc
    return rc


def test_go_enters_the_user_program_without_start(monkeypatch):
    import racecar_core_real

    monkeypatch.setattr(racecar_core_real.ros2, "ok", lambda *a, **k: False)
    rc = bare_racecar(FakeExecutor())
    calls = []

    def start():
        calls.append("start")
        rc._RacecarReal__running = False  # end go() after one pass

    def update():
        calls.append("update")

    rc.set_start_update(start, update)
    rc.go()
    # The FlySky handset has no START button: go() runs start() itself.
    assert calls == ["start"]
    assert rc._RacecarReal__cur_update is update


@pytest.mark.parametrize("context_up, shutdowns", [(False, 0), (True, 1)])
def test_go_returns_on_ctrl_c(monkeypatch, context_up, shutdowns):
    import racecar_core_real

    done = []
    monkeypatch.setattr(racecar_core_real.ros2, "ok", lambda *a, **k: context_up)
    monkeypatch.setattr(racecar_core_real.ros2, "shutdown", lambda *a, **k: done.append(1))
    rc = bare_racecar(FakeExecutor(KeyboardInterrupt()))
    rc.set_start_update(lambda: None, lambda: None)
    rc.go()  # returns instead of raising
    # rclpy's Ctrl-C handler may have shut the context down already.
    assert len(done) == shutdowns

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
    """Spins go()'s loop. It ends the loop after a set number of spins, so a regression fails and does not hang."""

    def __init__(self, error=None, max_spins=50):
        self.error = error
        self.max_spins = max_spins
        self.spins = 0
        self.racecar = None

    def spin_once(self, timeout_sec=None):
        if self.error is not None:
            raise self.error
        self.racecar.lidar.receive()
        self.spins += 1
        if self.spins >= self.max_spins:
            self.racecar._RacecarReal__running = False


class FakeLidar:
    """A lidar whose scan reaches get_samples() on the frame after it arrives."""

    def __init__(self, ready=True):
        self.incoming = np.ones(1440, dtype=np.float32) if ready else np.empty(0)
        self.frame = self.incoming

    def receive(self):
        self.incoming = np.ones(1440, dtype=np.float32)

    def get_samples(self):
        return self.frame

    def get_samples_async(self):
        return self.incoming

    def _LidarReal__update(self):
        self.frame = self.incoming


class FakeModule:
    """Stands in for a library module and records each per-frame update the run loop makes."""

    def __init__(self, name, updated):
        self.name = name
        self.updated = updated

    def __getattr__(self, attr):
        if attr.endswith("__update"):
            return lambda: self.updated.append(self.name)
        raise AttributeError(attr)


def bare_racecar(executor, scan_ready=True):
    """A RacecarReal with only what go() reads, so no node or thread starts."""
    from racecar_core_real import RacecarReal

    rc = RacecarReal.__new__(RacecarReal)
    rc._RacecarReal__executor = executor
    rc._RacecarReal__user_start = None
    rc._RacecarReal__user_update = None
    rc._RacecarReal__user_update_slow = None
    rc.updated = []
    for name in ("drive", "controller", "camera", "physics", "vision", "slam", "telemetry"):
        setattr(rc, name, FakeModule(name, rc.updated))
    rc.lidar = FakeLidar(scan_ready)
    executor.racecar = rc
    return rc


def test_run_loop_publishes_drive():
    # The NeoRacer drive module has no timer, so the run loop's per-frame update publishes /drive.
    rc = bare_racecar(FakeExecutor())
    rc._RacecarReal__update_modules()
    assert "drive" in rc.updated


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


def test_go_waits_for_the_first_scan(monkeypatch):
    import racecar_core_real

    monkeypatch.setattr(racecar_core_real.ros2, "ok", lambda *a, **k: False)
    rc = bare_racecar(FakeExecutor(), scan_ready=False)
    seen = []

    def start():
        seen.append(len(rc.lidar.get_samples()))
        rc._RacecarReal__running = False  # end go() after one pass

    rc.set_start_update(start, lambda: None)
    rc.go()
    # start() already reads a full scan.
    assert seen == [1440]


def test_go_async_enters_the_user_program_without_start():
    rc = bare_racecar(FakeExecutor())
    calls = []
    rc.set_start_update(lambda: calls.append("start"), lambda: None)
    rc.go_async()
    rc._RacecarReal__async_thread.join(timeout=5)
    # As under go(), the program starts without a START button.
    assert calls == ["start"]


@pytest.mark.parametrize("context_up, shutdowns", [(False, 0), (True, 1)])
@pytest.mark.parametrize("stop", ["ctrl_c", "external_shutdown"])
def test_go_returns_on_ctrl_c(monkeypatch, context_up, shutdowns, stop):
    import racecar_core_real

    done = []
    monkeypatch.setattr(racecar_core_real.ros2, "ok", lambda *a, **k: context_up)
    monkeypatch.setattr(racecar_core_real.ros2, "shutdown", lambda *a, **k: done.append(1))
    if stop == "ctrl_c":
        error = KeyboardInterrupt()
    else:
        error = racecar_core_real.ros2.executors.ExternalShutdownException()
    rc = bare_racecar(FakeExecutor(error))
    rc.set_start_update(lambda: None, lambda: None)
    try:
        rc.go()  # returns instead of raising
    except KeyboardInterrupt:
        pytest.fail("go() let Ctrl+C through instead of returning")
    # rclpy's Ctrl+C handler may have shut the context down already.
    assert len(done) == shutdowns

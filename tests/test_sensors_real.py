"""CameraReal subscribes on request and decodes on read; PhysicsReal reads the battery state."""

import math
import threading
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")


@pytest.fixture
def ros(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "94")  # away from a running car
    rclpy.init()
    from rclpy.executors import SingleThreadedExecutor

    executor = SingleThreadedExecutor()
    thread = threading.Thread(target=executor.spin, daemon=True)
    yield executor, thread
    executor.shutdown()
    rclpy.try_shutdown()


def wait(done, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        time.sleep(0.02)
    return done()


def image_msg():
    from sensor_msgs.msg import Image

    msg = Image(height=2, width=3, encoding="bgr8", step=9)
    msg.data = bytes(range(18))
    return msg


# A flat color survives JPEG compression within a few counts.
JPEG_BGR = (40, 120, 200)


def jpeg_msg():
    """A frame as the NeoRacer camera node sends it: the camera's JPEG bytes, encoding 'jpeg'."""
    import cv2 as cv
    from sensor_msgs.msg import Image

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :] = JPEG_BGR
    ok, jpeg = cv.imencode(".jpg", frame)
    assert ok
    msg = Image(height=48, width=64, encoding="jpeg", step=0)
    msg.data = jpeg.tobytes()
    return msg


def publish_until_stopped(pub, make):
    stop = threading.Event()

    def publish():
        while not stop.is_set():
            pub.publish(make())
            time.sleep(0.02)

    threading.Thread(target=publish, daemon=True).start()
    return stop


def camera_with_publisher(ros):
    executor, thread = ros
    from camera_real import CameraReal
    from sensor_msgs.msg import Image

    cam = CameraReal()
    executor.add_node(cam.node)
    pub_node = rclpy.create_node("fake_camera")
    executor.add_node(pub_node)
    pub = pub_node.create_publisher(Image, "/camera/color", 10)
    thread.start()
    return cam, pub


def test_camera_subscribes_on_the_first_read(ros):
    cam, pub = camera_with_publisher(ros)
    assert cam.node.get_subscriptions_info_by_topic("/camera/color") == []

    stop = publish_until_stopped(pub, image_msg)
    try:
        image = cam.get_color_image_async()
        assert image is not None and image.shape == (2, 3, 3)
        assert image[1, 2].tolist() == [15, 16, 17]
        assert cam.get_color_image_async() is not None
    finally:
        stop.set()


def test_camera_decodes_the_neoracer_jpeg_frames(ros):
    cam, pub = camera_with_publisher(ros)

    stop = publish_until_stopped(pub, jpeg_msg)
    try:
        image = cam.get_color_image_async()
        assert image is not None and image.shape == (48, 64, 3)
        assert image.dtype == np.uint8
        assert np.abs(image.astype(int) - JPEG_BGR).max() <= 4
    finally:
        stop.set()


def test_camera_returns_none_for_a_bad_jpeg_frame(ros, capsys):
    from sensor_msgs.msg import Image

    cam, pub = camera_with_publisher(ros)

    def bad():
        return Image(height=48, width=64, encoding="jpeg", step=0, data=b"\xff\xd8 not a jpeg")

    def empty():
        return Image(height=48, width=64, encoding="jpeg", step=0, data=b"")

    for make in (bad, empty):
        stop = publish_until_stopped(pub, make)
        try:
            assert wait(lambda: cam.get_color_image_async() is None)
            time.sleep(0.1)
            assert cam.get_color_image_async() is None
        finally:
            stop.set()
    assert "failed to decode color frame" in capsys.readouterr().out


def test_camera_without_frames_returns_none(ros):
    executor, thread = ros
    from camera_real import CameraReal

    cam = CameraReal()
    executor.add_node(cam.node)
    thread.start()
    start = time.monotonic()
    assert cam.get_color_image_async() is None
    assert time.monotonic() - start < 2.0


def test_depth_raises_on_the_neoracer(ros):
    executor, thread = ros
    from camera_real import CameraReal

    cam = CameraReal()
    executor.add_node(cam.node)
    thread.start()
    for read in (cam.get_depth_image, cam.get_depth_image_async):
        with pytest.raises(NotImplementedError, match="The NeoRacer has no depth camera"):
            read()
    assert cam.node.get_subscriptions_info_by_topic("/camera/depth") == []


def test_battery_voltage_comes_from_the_state(ros):
    executor, thread = ros
    from physics_real import PhysicsReal
    from sensor_msgs.msg import BatteryState

    physics = PhysicsReal()
    executor.add_node(physics.node)
    pub_node = rclpy.create_node("fake_controller")
    executor.add_node(pub_node)
    pub = pub_node.create_publisher(BatteryState, "/battery/state", 10)
    thread.start()
    assert wait(lambda: pub.get_subscription_count() > 0)
    # The NeoRacer controller node sends the pack voltage and NaN for current.
    pub.publish(BatteryState(voltage=11.4, current=math.nan))
    assert wait(lambda: physics.get_battery_voltage() > 0)
    assert physics.get_battery_voltage() == pytest.approx(11.4)
    with pytest.raises(NotImplementedError, match="cannot measure battery current"):
        physics.get_battery_current()


def test_get_data_dir(tmp_path, monkeypatch):
    import racecar_utils as rc_utils

    monkeypatch.setenv("RACECAR_DATA_DIR", str(tmp_path / "root"))
    assert rc_utils.get_data_dir() == str(tmp_path / "root")
    images = rc_utils.get_data_dir("images")
    assert images == str(tmp_path / "root" / "images")
    assert (tmp_path / "root" / "images").is_dir()

"""The NeoRacer's changes that need no ROS 2: defaults, sample counts, nodes, and the sim display."""

import ast
import inspect
from pathlib import Path

import numpy as np

REAL = Path(__file__).resolve().parent.parent / "library" / "real"


def calls(path, name):
    """Every call to a function or method called `name` in a source file."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if called == name:
                out.append(node)
    return out


def test_every_real_node_skips_parameter_services():
    found = 0
    for path in sorted(REAL.glob("*.py")):
        for call in calls(path, "create_node"):
            found += 1
            keywords = {k.arg: k.value for k in call.keywords}
            flag = keywords.get("start_parameter_services")
            assert isinstance(flag, ast.Constant) and flag.value is False, (
                f"{path.name}:{call.lineno} creates a node with parameter services"
            )
    assert found >= 11  # ten modules and racecar_core_real's rate node


def test_drive_defaults_to_full_speed():
    import drive

    default = inspect.signature(drive.Drive.set_max_speed).parameters["max_speed"].default
    assert default == 1.0


def test_drive_real_has_no_publish_timer():
    assert calls(REAL / "drive_real.py", "create_timer") == []


def test_lidar_sample_counts():
    import lidar

    assert lidar.Lidar._NUM_SAMPLES == 1080  # RacecarSim
    source = (REAL / "lidar_real.py").read_text()
    assert "_NUM_SAMPLES: int = 1440" in source  # the NeoRacer


class FakeSim:
    """The part of RacecarSim that DisplaySim sends through."""

    class Header:
        display_set_matrix = 33
        display_show_text = 34

    def __init__(self):
        self._RacecarSim__in_call = True
        self.sent = []

    def _RacecarSim__send_data(self, data, is_async):
        self.sent.append(data)


def test_display_sim_clear_blanks_the_matrix():
    from display_sim import DisplaySim

    sim = FakeSim()
    display = DisplaySim(sim, True)
    display.set_matrix(np.ones((8, 24), dtype=np.uint8))
    display.clear()
    assert not display.get_matrix().any()
    # Header byte, then 8 x 24 bits of zeros
    assert sim.sent[-1] == bytes([33]) + bytes(24)

"""Drone App: connection checks, verdict, and starting the website."""
import importlib.machinery
import importlib.util
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "windows" / "drone_app"
sys.path.insert(0, str(APP))

from drone_flight.web.drone_checks import build_checks, last_landing_reason, verdict  # noqa: E402

USB = {"com_ports": ["COM4", "COM5"]}
REC = {"on": True, "seconds": 75, "bytes": 3_000_000, "error": ""}


def checkup(**changes):
    data = {
        "service": {"connected": True, "port": "COM4", "fc_mode": "LOITER", "mode": "DISARMED", "flight_ready": True,
                    "flight_blocker": "", "last_fault": "", "last_command": "HEAD 1",
                    "telemetry": {"laser_ok": True, "height_mm": 40, "flow_quality": 12, "armed": False}},
        "page_heard_s": 0.2,
        "radio": {"port": "COM7", "note": ""},
        "battery": {"volts": 24.6, "percent": None},
        "head": {"found": True, "port": "COM5", "reading": True, "steering_on": True},
        "eyes": [{"label": "EYE 1", "fps": 9.0, "error": "", "running": True},
                 {"label": "EYE 2", "fps": 8.5, "error": "", "running": True}],
        "eeg": {"enabled": True, "state": "streaming", "model_ready": True, "prediction": {"decision": "STOP"},
                "brain_control": {"enabled": False}},
        "mavlink_log": None,
    }
    for path, value in changes.items():
        target = data
        keys = path.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return data


def by_key(checks):
    return {c.key: c for c in checks}


class ChecklistTests(unittest.TestCase):
    def test_all_connected_is_ready_to_fly(self):
        checks = build_checks("up", checkup(), USB, REC)
        self.assertEqual(verdict(checks, checkup()).title, "READY TO FLY")
        self.assertEqual(by_key(checks)["battery"].detail, "24.6 V")
        self.assertEqual(by_key(checks)["recording"].detail, "Screen + data · 1:15 · 3 MB")

    def test_website_down_says_reopen_the_app(self):
        checks = build_checks("down", None, USB, REC)
        result = verdict(checks, None)
        self.assertEqual(result.title, "NOT READY")
        self.assertIn("Flying website - Reopen the Drone App.", result.note)
        self.assertEqual(by_key(checks)["drone"].level, "wait")
        self.assertEqual(by_key(checks)["radio"].level, "wait")

    def test_radio_found_on_whatever_port_the_drone_answers(self):
        checks = by_key(build_checks("up", checkup(**{"radio.port": "COM9"}), USB, REC))
        self.assertEqual((checks["radio"].level, checks["radio"].detail), ("ok", "Drone link on COM9"))

    def test_drone_without_battery(self):
        data = checkup(**{"service.connected": False, "service.telemetry": None, "battery": None})
        checks = by_key(build_checks("up", data, USB, REC))
        self.assertEqual(checks["drone"].level, "bad")
        self.assertIn("battery", checks["drone"].fix)
        self.assertEqual(checks["laser"].level, "wait")

    def test_radio_unplugged(self):
        data = checkup(**{"service.connected": False, "service.telemetry": None,
                          "radio": {"port": "", "note": "no USB radio plugged in"}})
        checks = build_checks("up", data, {"com_ports": ["COM5"]}, REC)
        self.assertEqual(by_key(checks)["radio"].level, "bad")
        self.assertEqual(by_key(checks)["drone"].fix, "Plug in the radio first.")
        self.assertIn("Radio (USB)", verdict(checks, data).note)

    def test_radio_plugged_in_but_drone_off_keeps_looking(self):
        data = checkup(**{"service.connected": False, "service.telemetry": None,
                          "radio": {"port": "", "note": "drone not heard on COM7"}})
        checks = by_key(build_checks("up", data, USB, REC))
        self.assertEqual((checks["radio"].level, checks["radio"].detail), ("warn", "drone not heard on COM7"))
        self.assertEqual(checks["drone"].level, "bad")
        self.assertIn("battery", checks["drone"].fix)

    def test_battery_levels(self):
        def level(volts):
            return by_key(build_checks("up", checkup(battery={"volts": volts, "percent": None}), USB, REC))["battery"]
        self.assertEqual(level(24.0).level, "ok")
        self.assertEqual(level(22.6).level, "warn")
        self.assertEqual(level(21.9).level, "bad")
        checks = build_checks("up", checkup(battery={"volts": 21.9, "percent": None}), USB, REC)
        self.assertEqual(verdict(checks, checkup()).title, "NOT READY", "it would refuse to take off")

    def test_blind_floor_camera_on_the_ground_does_not_block(self):
        checks = build_checks("up", checkup(), USB, REC)
        self.assertEqual(by_key(checks)["floor"].level, "warn")
        self.assertEqual(verdict(checks, checkup()).title, "READY TO FLY")

    def test_page_closed_or_hidden(self):
        closed = build_checks("up", checkup(page_heard_s=None), USB, REC)
        self.assertEqual(by_key(closed)["page"].fix, "Press OPEN FLYING PAGE.")
        self.assertEqual(verdict(closed, checkup()).title, "NOT READY")
        hidden = build_checks("up", checkup(page_heard_s=20.0), USB, REC)
        self.assertIn("Click on the flying page", verdict(hidden, checkup()).note)

    def test_optional_devices_never_block(self):
        data = checkup(head={"found": False}, eyes=[], eeg={"enabled": False})
        checks = by_key(build_checks("up", data, USB, REC))
        self.assertEqual([checks[k].level for k in ("head", "eyes", "eeg")], ["off", "off", "off"])
        self.assertIn("CONNECT EEG", checks["eeg"].fix)
        self.assertEqual(verdict(list(checks.values()), data).title, "READY TO FLY")

    def test_head_sensor_with_bad_signal_says_check_wiring(self):
        data = checkup(head={"found": False, "note": "COM7 sends unreadable data"})
        head = by_key(build_checks("up", data, USB, REC))["head"]
        self.assertEqual(head.level, "warn")
        self.assertIn("UART-RVC", head.fix)
        missing = by_key(build_checks("up", checkup(head={"found": False, "note": "no head sensor adapter plugged in"}),
                                      USB, REC))["head"]
        self.assertEqual(missing.level, "off")

    def test_one_eye_camera_broken(self):
        data = checkup(eyes=[{"label": "EYE 1", "fps": 9.0, "error": "", "running": True},
                             {"label": "EYE 2", "fps": 0.0, "error": "no picture from this camera", "running": True}])
        self.assertEqual(by_key(build_checks("up", data, USB, REC))["eyes"].detail,
                         "1 of 2 working (EYE 2: no picture from this camera)")

    def test_flying_and_landing_reason(self):
        data = checkup(**{"service.mode": "HOVERING", "service.telemetry.armed": True,
                          "service.last_fault": "WRONG_DIRECTION", "service.fault_help": "It did not go where asked."})
        checks = build_checks("up", data, USB, REC)
        self.assertEqual(verdict(checks, data).title, "FLYING")
        self.assertEqual(by_key(checks)["floor"].level, "bad", "blind camera in the air")
        self.assertIn("WRONG_DIRECTION", last_landing_reason(data))

    def test_starting(self):
        self.assertEqual(verdict(build_checks("starting", None, USB, REC), None).title, "STARTING")


def load_app():
    loader = importlib.machinery.SourceFileLoader("drone_app_main", str(APP / "drone_app.pyw"))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("drone_app_main", loader))
    loader.exec_module(module)
    module.POLL_S = 0.0
    return module


class AppFileTests(unittest.TestCase):
    """The Drone App only starts the website and opens the flying page: no window or recording of its own."""

    def test_app_imports_without_windows(self):
        app = load_app()
        self.assertIn("--disable-backgrounding-occluded-windows", app.BROWSER_FLAGS,
                      "the page keeps sending keys when covered")
        self.assertFalse((APP / "drone_recording.py").exists())

    def run_main(self, app, health):
        calls = []
        app.first_instance = lambda: True
        app.get_json = lambda path, timeout=1.0: (calls.append(path), health)[1]
        app.close_page_windows = lambda: calls.append("close pages")
        app.start_website = lambda: calls.append("restart") or ""
        app.open_page = lambda: calls.append("open page") or ""
        app.tell = lambda text: calls.append(("told", text))
        return app.main(), calls

    def test_flying_drone_is_left_alone(self):
        app = load_app()
        result, calls = self.run_main(app, (200, {"mode": "HOVERING"}))
        self.assertEqual(result, 1)
        self.assertEqual(calls, ["/launcher-health", ("told", app.BUSY)],
                         "no restart and no new page: opening the page would land the drone")

    def test_disarmed_closes_old_pages_then_restarts_and_opens_one(self):
        app = load_app()
        for health in ((200, {"mode": "DISARMED"}), (0, None)):
            result, calls = self.run_main(app, health)
            self.assertEqual(result, 0)
            self.assertEqual(calls, ["/launcher-health", "close pages", "restart", "open page"],
                             "an old page's token would keep the drone on the ground")

    def test_restart_waits_for_the_new_website(self):
        app = load_app()
        answers = iter([(200, None), (0, None), (0, None), (200, {"mode": "DISARMED"})])
        asked = []
        app.get_json = lambda path, timeout=1.0: (asked.append(path), next(answers))[1]
        app.run_start_task = lambda: ""
        self.assertEqual(app.start_website(), "")
        self.assertEqual(len(asked), 4, "waits until the old website went away and the new one answers")
        self.assertNotIn("/", asked, "never loads the page itself")

    def test_reports_a_website_that_never_starts(self):
        app = load_app()
        app.START_WAIT_S = 0.05
        app.get_json = lambda path, timeout=1.0: (0, None)
        app.run_start_task = lambda: ""
        self.assertIn("did not start", app.start_website())
        app.run_start_task = lambda: "NeuralFlightAppStart failed: not found"
        self.assertEqual(app.start_website(), "NeuralFlightAppStart failed: not found")


if __name__ == "__main__":
    unittest.main()

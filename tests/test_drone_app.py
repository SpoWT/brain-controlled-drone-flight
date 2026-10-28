"""Drone App: connection checks, verdict and session recording."""
import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "drone_app"
sys.path.insert(0, str(APP))  # drone_checks.py lives at the top, shared with the website

from drone_checks import build_checks, last_landing_reason, verdict  # noqa: E402
from drone_recording import DataLog, MavlinkSlice, ScreenRecorder, Session  # noqa: E402
from window_capture import PAGE_TITLE, Window, choose_window  # noqa: E402

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


class FakeWriter:
    def __init__(self):
        self.frames, self.released = 0, False

    def write(self, frame):
        self.frames += 1

    def release(self):
        self.released = True


class RecordingTests(unittest.TestCase):
    def test_screen_keeps_real_time_and_splits_files(self):
        import cv2  # noqa: F401  (imported first: the first import takes a moment)
        import numpy as np
        writers = []

        def writer(path, fps, size):
            writers.append((path.name, size, FakeWriter()))
            return writers[-1][2]

        folder = Path(tempfile.mkdtemp())
        recorder = ScreenRecorder(folder, grab=lambda: np.zeros((1600, 2560, 3), np.uint8), writer=writer,
                                  fps=10, segment_s=0.3)
        recorder.start()
        time.sleep(1.0)
        recorder.stop()
        self.assertEqual(writers[0][1], (1920, 1200), "scaled down to 1920 wide")
        self.assertGreaterEqual(len(writers), 2, "a new file every segment")
        self.assertTrue(all(w.released for _, _, w in writers))
        self.assertGreaterEqual(recorder.frames, 6, "about 10 frames per second of real time")
        self.assertEqual(recorder.error, "")

    def test_closed_page_keeps_the_timeline_going(self):
        import numpy as np
        pictures = [None, None, np.zeros((100, 200, 3), np.uint8)] + [None] * 1000
        writers = []

        def writer(path, fps, size):
            writers.append(FakeWriter())
            return writers[-1]

        recorder = ScreenRecorder(Path(tempfile.mkdtemp()), grab=lambda: pictures.pop(0), writer=writer, fps=10)
        recorder.start()
        time.sleep(0.9)
        recorder.stop()
        self.assertEqual(len(writers), 1, "starts with the first picture")
        self.assertGreaterEqual(writers[0].frames, 3, "keeps writing the last picture while the page is closed")
        self.assertEqual(recorder.error, "")

    def test_screen_error_is_reported_not_raised(self):
        def broken():
            raise OSError("screen locked")
        recorder = ScreenRecorder(Path(tempfile.mkdtemp()), grab=broken, writer=lambda *a: FakeWriter())
        recorder.start()
        recorder.stop()
        self.assertIn("screen locked", recorder.error)

    def test_events_tell_the_story(self):
        folder = Path(tempfile.mkdtemp())
        log = DataLog(folder)
        ground = checkup()
        flying = checkup(**{"service.mode": "HOVERING", "service.last_command": "TAKEOFF 300",
                            "service.telemetry.armed": True})
        landed = checkup(**{"service.last_fault": "OVER_HEIGHT", "service.fault_help": "Too high."})
        for data in (ground, flying, flying, landed):
            checks = build_checks("up", data, USB, REC)
            log.record("up", data, checks, verdict(checks, data))
        log.close()
        events = (folder / "events.txt").read_text()
        self.assertIn("READY TO FLY", events)
        self.assertIn("Command sent: TAKEOFF 300", events)
        self.assertIn("Drone mode: HOVERING", events)
        self.assertIn("AUTOMATIC LANDING: OVER_HEIGHT - Too high.", events)
        self.assertEqual(events.count("Drone mode: HOVERING"), 1, "only changes are written")
        lines = (folder / "data.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertEqual(json.loads(lines[1])["checkup"]["service"]["mode"], "HOVERING")

    def test_session_copies_the_drone_messages_heard_while_open(self):
        root = Path(tempfile.mkdtemp())
        mav = root / "mav_old.jsonl"
        mav.write_text("before\n")
        session = Session(root / "rec", screen=ScreenRecorder(root, grab=lambda: None, writer=None))
        session.screen._stop.set()
        data = checkup(mavlink_log={"path": str(mav), "error": ""})
        session.mavlink._start = time.time() + 60  # the log file is older than this session
        checks = build_checks("up", data, USB, REC)
        session.record("up", data, checks, verdict(checks, data))
        with open(mav, "a") as handle:
            handle.write("during\n")
        session.close()
        self.assertEqual((session.folder / "drone_mavlink.jsonl").read_text(), "during\n")
        summary = json.loads((session.folder / "session.json").read_text())
        self.assertEqual(summary["mavlink_sources"], [str(mav)])
        self.assertIn("Drone App closed", (session.folder / "events.txt").read_text())

    def test_mavlink_slice_new_file_from_the_start(self):
        root = Path(tempfile.mkdtemp())
        mav = root / "mav_new.jsonl"
        mav.write_text("first\n")
        piece = MavlinkSlice(time.time())
        piece.see(str(mav))
        self.assertIsNone(piece.copy_to(root / "out.jsonl"))
        self.assertEqual((root / "out.jsonl").read_text(), "first\n")


class AppFileTests(unittest.TestCase):
    def test_app_imports_without_windows(self):
        loader = importlib.machinery.SourceFileLoader("drone_app_main", str(APP / "drone_app.pyw"))
        module = importlib.util.module_from_spec(importlib.util.spec_from_loader("drone_app_main", loader))
        loader.exec_module(module)
        self.assertFalse(module.flying({"website": "up", "checkup": checkup()}))
        up = {"website": "up", "checkup": checkup(**{"service.mode": "HOVERING", "service.telemetry.armed": True})}
        self.assertTrue(module.flying(up), "asks twice before stopping the recording in flight")
        self.assertIn("--disable-backgrounding-occluded-windows", module.BROWSER_FLAGS,
                      "the page keeps sending keys when covered")

    def test_finds_our_flying_page_window(self):
        windows = [Window(1, 50, "Optical Hover Control - Google Chrome", "Chrome_WidgetWin_1", 900, True),
                   Window(2, 77, PAGE_TITLE, "Chrome_WidgetWin_1", 4000, True),
                   Window(3, 77, "", "Chrome_WidgetWin_1", 10, True),
                   Window(4, 77, PAGE_TITLE, "Chrome_WidgetWin_1", 9000, False)]
        self.assertEqual(choose_window(windows, {77}), 2, "our browser's biggest visible window")
        self.assertEqual(choose_window(windows, {99}), 2, "else the app-mode window titled like the page")
        self.assertIsNone(choose_window(windows[:1], set()), "never a normal browser tab")


if __name__ == "__main__":
    unittest.main()

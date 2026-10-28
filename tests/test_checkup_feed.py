"""GET /api/checkup: read-only status and connection lights."""
import unittest
from types import SimpleNamespace as NS

from flask import Flask

from checkup_feed import UsbPorts, battery, create_checkup_blueprint


class FakeLink:
    device = "COM7"  # wherever the drone answered

    def __init__(self, sys_status=None):
        self.sys_status = sys_status

    def latest(self, kind, max_age):
        return self.sys_status if kind == "SYS_STATUS" else None


class FakeHead:
    port, error = "COM5", ""

    def latest(self, max_age):
        return 12.5


class FakeService:
    def __init__(self):
        self.commands = []

    def status(self):
        return {"mode": "DISARMED", "phase": "GROUND", "connected": True, "port": "COM4", "calibration": {"big": 1},
                "telemetry": {"laser_ok": True}}

    def page_heard_age(self):
        return 0.4

    def eeg_go_active(self):
        return False

    def __getattr__(self, name):  # any command (arm, land, authorize ...) would be recorded here
        self.commands.append(name)
        return lambda *a, **k: None


class FakeEEG:
    def status(self):
        return {"enabled": True, "state": "streaming", "eeg": [[1.0] * 500] * 4, "prediction": {"decision": "GO"},
                "model_ready": True}


class FakeCamera:
    def __init__(self):
        self.touched = 0

    def touch(self):
        self.touched += 1


class FakeEyes:
    def __init__(self):
        self.cameras = [FakeCamera(), FakeCamera()]

    def status(self):
        return [{"label": "EYE 1", "fps": 9.8, "error": "", "running": True},
                {"label": "EYE 2", "fps": 0.0, "error": "no picture", "running": True}]


class CheckupTests(unittest.TestCase):
    def make(self, sys_status=None):
        service, eyes = FakeService(), FakeEyes()
        transport = NS(link=FakeLink(sys_status), head=FakeHead(), _head_on=True)
        recorder = NS(path="outputs/mavlink_logs/mav_1.jsonl", error="")
        app = Flask(__name__)
        app.extensions["eeg_flight_control"] = NS(status=lambda: {"enabled": False, "message": "off"})
        usb = UsbPorts(lister=lambda: ["COM4", "COM5"])
        app.register_blueprint(create_checkup_blueprint(service, transport, FakeEEG(), eyes, recorder, usb))
        return app.test_client(), service, eyes

    def test_everything_in_one_read_only_answer(self):
        client, service, eyes = self.make(NS(voltage_battery=24310, battery_remaining=-1))
        data = client.get("/api/checkup").get_json()  # no flight token needed
        self.assertEqual(data["battery"], {"volts": 24.31, "percent": None})
        self.assertEqual(data["page_heard_s"], 0.4)
        self.assertEqual(data["head"], {"found": True, "port": "COM5", "reading": True, "error": "", "steering_on": True})
        self.assertEqual(data["eyes"][1]["error"], "no picture")
        self.assertEqual(data["eeg"]["prediction"], {"decision": "GO"})
        self.assertEqual(data["eeg"]["brain_control"], {"enabled": False, "message": "off"})
        self.assertNotIn("eeg", data["eeg"], "the raw EEG trace is not sent")
        self.assertNotIn("calibration", data["service"])
        self.assertEqual(data["mavlink_log"]["path"], "outputs/mavlink_logs/mav_1.jsonl")
        self.assertEqual(data["radio"], {"port": "COM7", "note": ""})
        self.assertEqual(service.commands, [], "never commands the drone or touches the control session")
        self.assertEqual([c.touched for c in eyes.cameras], [1, 1], "eye cameras run while the app watches")

    def test_connection_lights_and_rec_for_the_page(self):
        client, _, _ = self.make(NS(voltage_battery=24310, battery_remaining=-1))
        data = client.get("/api/checkup").get_json()
        lights = {c["key"]: c for c in data["checks"]}
        self.assertEqual(lights["radio"]["level"], "ok")
        self.assertEqual(lights["radio"]["detail"], "Drone link on COM7")
        self.assertEqual(lights["battery"]["detail"], "24.3 V")
        self.assertEqual(lights["eyes"]["level"], "warn", "EYE 2 has no picture")
        self.assertIsNone(data["app_recording"], "no Drone App: no REC light")
        data = client.get("/api/checkup?recording=125").get_json()
        self.assertEqual(data["app_recording"], {"seconds": 125})
        self.assertEqual(client.get("/api/checkup?recording=-4").status_code, 200)

    def test_battery_unknown_until_the_fc_reports_it(self):
        self.assertIsNone(battery(FakeLink(None)))
        self.assertIsNone(battery(FakeLink(NS(voltage_battery=0xFFFF, battery_remaining=-1))))
        self.assertEqual(battery(FakeLink(NS(voltage_battery=22000, battery_remaining=40))), {"volts": 22.0, "percent": 40})

    def test_no_head_sensor(self):
        from checkup_feed import head
        self.assertEqual(head(NS(head=None)), {"found": False, "note": ""})
        self.assertEqual(head(NS(head=None), NS(note="COM7 sends unreadable data"))["note"], "COM7 sends unreadable data")


if __name__ == "__main__":
    unittest.main()

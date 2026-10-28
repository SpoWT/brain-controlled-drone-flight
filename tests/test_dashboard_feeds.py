"""Read-only feeds for the one-screen dashboard: steering (/api/direction) and the goggle eye cameras."""
import time
import unittest
from types import SimpleNamespace as NS

from flask import Flask

from dashboard_feeds import create_dashboard_blueprint
from eye_cameras import EyeCamera, EyeCameras, EyeDevice, create_eye_blueprint, find_eye_cameras, measure

HUB = r"PCIROOT(0)#PCI(0201)#USBROOT(0)#USB(3)"  # the goggles' hub


def device(port, device_id="", path=""):
    """An eye camera on port `port` of the goggles' hub."""
    device_id = device_id or f"ID{port}"
    return EyeDevice(path or f"cam-{device_id}", device_id, f"{HUB}#USB({port})")


class FakeTransport:
    def direction_status(self):
        return {"head_sensor": True, "head_on": True, "head_offset": 32.0, "drone_offset": 28.0,
                "turn_rate": 6, "arrow_rate": 0}


class FakeService:
    def __init__(self):
        self.recentred = []

    def eeg_go_active(self):
        return True

    def recenter_head(self, token):
        if token != "tok":
            raise PermissionError("This browser is not the active flight page")
        self.recentred.append(token)


class DirectionFeedTests(unittest.TestCase):
    def test_direction_feed_combines_head_drone_and_brain(self):
        app = Flask(__name__)
        app.register_blueprint(create_dashboard_blueprint(FakeTransport(), FakeService()))
        data = app.test_client().get("/api/direction").get_json()
        self.assertEqual(data["head_offset"], 32.0)
        self.assertEqual(data["turn_rate"], 6)
        self.assertTrue(data["brain_go"])

    def test_direction_feed_says_why_there_is_no_head_sensor(self):
        app = Flask(__name__)
        watch = NS(note="COM7 sends unreadable data (250 bytes/s)")
        app.register_blueprint(create_dashboard_blueprint(FakeTransport(), FakeService(), watch))
        self.assertEqual(app.test_client().get("/api/direction").get_json()["head_note"], watch.note)

    def test_recenter_needs_the_flight_page_token(self):
        service = FakeService()
        app = Flask(__name__)
        app.register_blueprint(create_dashboard_blueprint(FakeTransport(), service))
        client = app.test_client()
        self.assertEqual(client.post("/api/head/recenter", headers={"X-Flight-Token": "tok"}).status_code, 200)
        self.assertEqual(service.recentred, ["tok"])
        self.assertEqual(client.post("/api/head/recenter", headers={"X-Flight-Token": "other"}).status_code, 403)


class EyeCameraTests(unittest.TestCase):
    def test_finds_only_the_goggle_cameras(self):
        names = ["ASUS FHD webcam", "Innomaker-U20CAM-1080p-S1", "Innomaker-U20CAM-1080p-S1"]
        self.assertEqual(find_eye_cameras(names), [1, 2])

    def test_measure_reports_pupil_size_and_position(self):
        result = NS(locked=True, contour=None, radius=12.0, x=0.5, y=0.4, confidence=0.8, brightness=0.5, source="model")
        m = measure(result, 640)
        self.assertEqual(m["diameter_px"], 24.0)
        self.assertEqual(m["source"], "MediaPipe iris")
        self.assertFalse(measure(NS(locked=False, confidence=0.1, brightness=0.5), 640)["detected"])

    def test_feeds_serve_frames_and_numbers_then_release_the_camera(self):
        class FakeCapture:
            released = False

            def read(self):
                return True, NS(shape=(480, 640, 3))

            def release(self):
                FakeCapture.released = True

        result = NS(locked=True, contour=None, radius=10.0, x=0.5, y=0.5, confidence=0.9, brightness=0.4, source="classic")
        camera = EyeCamera(0, "EYE 1", device=device(1), resolver=lambda path: 1, opener=lambda index: FakeCapture(),
                           detector=lambda frame, previous: (result, b"JPEGDATA"))
        app = Flask(__name__)
        app.register_blueprint(create_eye_blueprint(EyeCameras([camera])))
        client = app.test_client()
        client.get("/api/eyes")
        for _ in range(50):
            if camera.latest_jpeg():
                break
            time.sleep(0.02)
        eyes = client.get("/api/eyes").get_json()["eyes"]
        self.assertTrue(eyes[0]["detected"])
        self.assertEqual(eyes[0]["diameter_px"], 20.0)
        stream = client.get("/eyes/0.mjpg")
        chunk = next(stream.response)
        self.assertIn(b"--frame", chunk)
        self.assertIn(b"JPEGDATA", chunk)
        stream.close()
        self.assertEqual(client.get("/eyes/5.mjpg").status_code, 404)


class EyeDeviceTests(unittest.TestCase):
    """The two cameras are the same model: told apart by device path, ordered by the USB port they sit on."""

    PATH_A = r"\\?\usb#vid_0c45&pid_6366&mi_00#a&183af011&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
    PATH_B = r"\\?\usb#vid_0c45&pid_6366&mi_00#a&f787245&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"

    def test_path_parts(self):
        from eye_cameras import instance_id, short_id, usb_port
        self.assertEqual(short_id(self.PATH_B), "F787245")
        self.assertEqual(instance_id(self.PATH_A), r"USB\VID_0C45&PID_6366&MI_00\A&183AF011&0&0000")
        self.assertEqual(usb_port(HUB + "#USB(1)"), (HUB, 1))
        self.assertEqual(usb_port(""), ("", 0))

    def test_eye_cameras_are_ordered_by_usb_port_not_plug_order(self):
        from eye_cameras import find_eye_devices
        devices = [("ASUS FHD webcam", r"\\?\usb#vid_3277&pid_0010&mi_00#7&df5034d&0&0000#{x}\global"),
                   ("Innomaker-U20CAM-1080p-S1", self.PATH_B), ("Innomaker-U20CAM-1080p-S1", self.PATH_A)]
        ports = {self.PATH_A: HUB + "#USB(1)", self.PATH_B: HUB + "#USB(3)"}
        found = find_eye_devices(devices, locate=ports.get)
        self.assertEqual([d.device_id for d in found], ["183AF011", "F787245"])
        # re-plugged: Windows swapped the IDs, but the camera on port 1 is still EYE 1
        swapped = {self.PATH_A: HUB + "#USB(3)", self.PATH_B: HUB + "#USB(1)"}
        self.assertEqual([d.device_id for d in find_eye_devices(devices, locate=swapped.get)], ["F787245", "183AF011"])

    def test_cameras_cannot_be_listed_keeps_what_is_there(self):
        eyes = EyeCameras([EyeCamera(0, "EYE 1", device=device(1)), EyeCamera(1, "EYE 2", device=device(3))],
                          finder=lambda: None)
        eyes.refresh()
        self.assertEqual([c.device_id for c in eyes.cameras], ["ID1", "ID3"])

    def test_rotation_follows_the_cameras_port_in_the_goggles(self):
        import tempfile
        from pathlib import Path
        config = Path(tempfile.mkdtemp()) / "eyes.json"
        eyes = EyeCameras([EyeCamera(0, "EYE 1", device=device(1, "183AF011")),
                           EyeCamera(1, "EYE 2", device=device(3, "F787245"))], config_path=config)
        eyes.rotate(1, 90)
        # re-plugged into another laptop port: new device IDs, same ports inside the goggles
        again = EyeCameras([EyeCamera(0, "EYE 1", device=device(1, "9A9A9A")),
                            EyeCamera(1, "EYE 2", device=device(3, "7B7B7B"))], config_path=config)
        self.assertEqual([c.rotation for c in again.cameras], [0, 90])
        self.assertEqual([c.settings_key for c in again.cameras], ["usb port 1", "usb port 3"])

    def test_old_settings_by_device_id_are_kept(self):
        import json
        import tempfile
        from pathlib import Path
        config = Path(tempfile.mkdtemp()) / "eyes.json"
        config.write_text(json.dumps({"183AF011": {"rotation": 180}, "usb port 3": "broken"}))
        eyes = EyeCameras([EyeCamera(0, "EYE 1", device=device(1, "183AF011")),
                           EyeCamera(1, "EYE 2", device=device(3))], config_path=config)
        self.assertEqual([c.rotation for c in eyes.cameras], [180, 0], "a broken entry is ignored, not a crash")
        self.assertEqual(json.loads(config.read_text())["usb port 1"], {"rotation": 180, "slot": 0},
                         "saved under the port")

    def test_slots_stay_put_while_a_camera_is_missing(self):
        import tempfile
        from pathlib import Path
        from eye_cameras import assign_slots
        one, three = device(1), device(3)
        self.assertEqual(assign_slots([three], [one, three], lambda d: None), [None, three], "EYE 2 stays EYE 2")
        self.assertEqual(assign_slots([three], [None, None], lambda d: 1), [None, three], "its slot from last time")
        self.assertEqual(assign_slots([three, one], [None, None], lambda d: None), [three, one], "else in order given")
        config = Path(tempfile.mkdtemp()) / "eyes.json"
        found = [[one, three]]
        EyeCameras.discover(config_path=config, finder=lambda: found[0])
        found[0] = [three]  # next start: the port 1 camera is unplugged
        eyes = EyeCameras.discover(config_path=config, finder=lambda: found[0])
        self.assertEqual([c.device_id for c in eyes.cameras], ["", "ID3"])

    def test_refresh_gives_slots_the_cameras_by_port_and_reopens_switched_ones(self):
        found = [[device(1)], [device(1), device(3)]]
        eyes = EyeCameras.discover(config_path=None, finder=lambda: found[0])
        self.assertEqual([c.device_id for c in eyes.cameras], ["ID1", ""])
        self.assertTrue(eyes.cameras[1].missing())
        reopened = []
        eyes.cameras[1].request_reopen = lambda reason: reopened.append(reason)
        found.pop(0)
        eyes.refresh()
        self.assertEqual([c.device_id for c in eyes.cameras], ["ID1", "ID3"])
        self.assertEqual(reopened, ["switching camera"], "only the slot that changed")


class ReplugTests(unittest.TestCase):
    """The two cameras can trade device paths when re-plugged; the USB port decides which eye is which."""

    def test_swapped_paths_follow_the_port(self):
        from eye_cameras import assign_slots
        before = [EyeDevice("PX", "X", f"{HUB}#USB(1)"), EyeDevice("PY", "Y", f"{HUB}#USB(3)")]
        after = [EyeDevice("PY", "Y", f"{HUB}#USB(1)"), EyeDevice("PX", "X", f"{HUB}#USB(3)")]
        slots = {f"{HUB}#USB(1)": 0, f"{HUB}#USB(3)": 1}
        chosen = assign_slots(after, before, lambda d: slots[d.location])
        self.assertEqual([d.location for d in chosen], [f"{HUB}#USB(1)", f"{HUB}#USB(3)"])

    def test_one_failed_port_lookup_changes_nothing_else(self):
        from eye_cameras import find_eye_devices, settings_key
        a = r"\\?\usb#vid_0c45&pid_6366&mi_00#a&1&0&0000#{x}\global"
        b = r"\\?\usb#vid_0c45&pid_6366&mi_00#a&2&0&0000#{x}\global"
        listing = [("Innomaker-U20CAM-1080p-S1", b), ("Innomaker-U20CAM-1080p-S1", a)]
        known = {}
        first = find_eye_devices(listing, locate={a: f"{HUB}#USB(1)", b: f"{HUB}#USB(3)"}.get, known=known)
        self.assertEqual([settings_key(d, first) for d in first], ["usb port 1", "usb port 3"])
        failed = find_eye_devices(listing, locate={a: f"{HUB}#USB(1)"}.get, known=known)
        self.assertEqual([settings_key(d, failed) for d in failed], ["usb port 1", "usb port 3"], "last good port reused")
        fresh = find_eye_devices(listing, locate={a: f"{HUB}#USB(1)"}.get, known={})
        self.assertEqual([d.location for d in fresh], [f"{HUB}#USB(1)", ""], "unknown port sorts last")
        self.assertEqual(settings_key(fresh[0], fresh), "usb port 1", "the other camera keeps its key")

    def test_a_cached_port_now_held_by_the_other_camera_is_not_reused(self):
        from eye_cameras import find_eye_devices
        listing = [("Innomaker-U20CAM-1080p-S1", "PX"), ("Innomaker-U20CAM-1080p-S1", "PY")]
        known = {}
        find_eye_devices(listing, locate={"PX": f"{HUB}#USB(1)", "PY": f"{HUB}#USB(3)"}.get, known=known)
        # re-plug swapped the paths; PX's lookup fails, and its cached port 1 now belongs to PY
        swapped = find_eye_devices(listing, locate={"PY": f"{HUB}#USB(1)"}.get, known=known)
        self.assertEqual({d.path: d.location for d in swapped}, {"PY": f"{HUB}#USB(1)", "PX": ""})

    def test_two_cameras_on_one_port_are_looked_at_again(self):
        looks = [[device(1, "X", "PX"), device(1, "Y", "PY")], [device(1, "Y", "PY"), device(3, "X", "PX")]]
        eyes = EyeCameras([EyeCamera(0, "EYE 1"), EyeCamera(1, "EYE 2")], finder=lambda: looks.pop(0))
        eyes.refresh()
        self.assertEqual([c.device_id for c in eyes.cameras], ["", ""], "impossible listing: not placed")
        eyes.refresh()
        self.assertEqual([c.device_id for c in eyes.cameras], ["Y", "X"], "port 1 is EYE 1")

    def test_path_now_on_another_port_is_not_opened(self):
        from eye_cameras import MOVED
        opened = []
        camera = EyeCamera(0, "EYE 1", device=device(1), resolver=lambda path: 1, locate=lambda path: f"{HUB}#USB(3)",
                           opener=lambda index: opened.append(index))
        camera.touch()
        for _ in range(100):
            if camera.status()["error"]:
                break
            time.sleep(0.02)
        self.assertEqual((camera.status()["error"], opened), (MOVED, []))
        self.assertTrue(camera.missing(), "so the cameras are looked for again")
        camera._watched_at = -1e9

    def test_list_changing_while_opening_opens_again(self):
        import eye_cameras

        class Capture:
            def __init__(self):
                self.released = False

            def read(self):
                return True, NS(shape=(480, 640, 3))

            def release(self):
                self.released = True

        indexes = iter([1, 2] + [1] * 1000)  # the second listing (right after opening) differs once
        captures = []
        result = NS(locked=False, confidence=0.0, brightness=0.3)

        def opener(index):
            captures.append(Capture())
            return captures[-1]
        camera = EyeCamera(0, "EYE 1", device=device(1), resolver=lambda path: next(indexes), opener=opener,
                           detector=lambda frame, previous: (result, b"J"))
        old = eye_cameras.MISSING_RETRY_S
        eye_cameras.MISSING_RETRY_S = 0.05
        try:
            camera.touch()
            for _ in range(200):
                if camera.latest_jpeg():
                    break
                time.sleep(0.02)
        finally:
            eye_cameras.MISSING_RETRY_S = old
        self.assertTrue(captures[0].released, "the first open was dropped")
        self.assertEqual(camera.latest_jpeg(), b"J")
        camera._watched_at = -1e9


class SharedIndexTests(unittest.TestCase):
    def test_an_index_held_by_the_other_slot_is_not_reused(self):
        from eye_cameras import LIST_CHANGED, DeviceClaims
        claims = DeviceClaims()
        one, two = EyeCamera(0, "EYE 1", claims=claims), EyeCamera(1, "EYE 2", claims=claims)
        asked = []
        two.request_reopen = lambda reason: asked.append(reason)
        self.assertEqual(claims.claim("PB", 0, slot=1), "", "EYE 2 opened camera B at index 0")
        # re-plug: camera A now lists first, so A's path resolves to index 0, which EYE 2 still holds
        self.assertEqual(claims.claim("PA", 0, slot=0), LIST_CHANGED)
        self.assertEqual(asked, [LIST_CHANGED], "EYE 2 is told to find its own camera again")
        claims.release(slot=1)
        self.assertEqual(claims.claim("PA", 0, slot=0), "")
        self.assertIsNotNone(one)

    def test_software_cameras_without_a_device_path_do_not_hide_the_eyes(self):
        from eye_cameras import _read, index_of

        class Bag:
            def Read(self, key, pErrorLog=None):
                if key == "DevicePath":
                    raise OSError("no such property")
                return "OBS Virtual Camera"
        self.assertEqual((_read(Bag(), "FriendlyName"), _read(Bag(), "DevicePath")), ("OBS Virtual Camera", ""))
        listing = [("OBS Virtual Camera", ""), ("Innomaker-U20CAM-1080p-S1", "cam-A")]
        self.assertEqual(index_of("cam-A", listing), 1)
        self.assertIsNone(index_of("", listing))

    def test_a_port_that_cannot_be_read_is_looked_at_again_before_guessing(self):
        from eye_cameras import PARTIAL_LOOKS
        looks = [[device(1), EyeDevice("cam-B", "B", "")]] * PARTIAL_LOOKS + [[device(1), device(3, "B", "cam-B")]]
        eyes = EyeCameras([EyeCamera(0, "EYE 1"), EyeCamera(1, "EYE 2")], finder=lambda: looks.pop(0))
        for _ in range(PARTIAL_LOOKS):
            eyes.refresh()
            self.assertEqual([c.device_id for c in eyes.cameras], ["", ""], "not placed while a port is unknown")
        eyes.refresh()
        self.assertEqual([c.device_id for c in eyes.cameras], ["ID1", "B"])


class OneCameraOneSlotTests(unittest.TestCase):
    """One camera is never open in both slots."""

    def test_second_slot_refuses_the_camera_the_first_has_open(self):
        from eye_cameras import DeviceClaims

        class Capture:
            def read(self):
                return True, NS(shape=(480, 640, 3))

            def release(self):
                pass

        opened, claims = [], DeviceClaims()
        result = NS(locked=False, confidence=0.0, brightness=0.3)
        same = device(1)

        def opener(index):
            opened.append(index)
            return Capture()
        one = EyeCamera(0, "EYE 1", device=same, claims=claims, resolver=lambda path: 1, opener=opener,
                        detector=lambda frame, previous: (result, b"J"))
        two = EyeCamera(1, "EYE 2", device=same, claims=claims, resolver=lambda path: 1, opener=opener,
                        detector=lambda frame, previous: (result, b"J"))
        one.touch()
        for _ in range(100):
            if one.latest_jpeg():
                break
            time.sleep(0.02)
        two.touch()
        for _ in range(100):
            if two.status()["error"]:
                break
            time.sleep(0.02)
        self.assertEqual(opened, [1], "opened once, by EYE 1")
        self.assertEqual(two.status()["error"], "already open in the other slot")

    def test_unplugged_camera_says_so_and_is_looked_for_again(self):
        from eye_cameras import NOT_PLUGGED, REFRESH_EVERY_S
        now = [0.0]
        calls = []
        camera = EyeCamera(0, "EYE 1", device=device(1), resolver=lambda path: None, clock=lambda: now[0])
        camera.touch()
        for _ in range(100):
            if camera.status()["error"]:
                break
            time.sleep(0.02)
        self.assertEqual(camera.status()["error"], NOT_PLUGGED)
        eyes = EyeCameras([camera], finder=lambda: calls.append(1) or [], clock=lambda: now[0])
        eyes.status()
        self.assertEqual(calls, [], "not before REFRESH_EVERY_S")
        now[0] += REFRESH_EVERY_S
        eyes.status()
        for _ in range(100):
            if calls:
                break
            time.sleep(0.02)
        self.assertEqual(calls, [1])
        now[0] += 1000  # let the camera thread stop


class EyeRotationTests(unittest.TestCase):
    """The goggle cameras are mounted sideways/upside down; each view is turned upright and remembered."""

    def make(self, tmp):
        import numpy as np
        seen = []

        class Capture:
            def read(self):
                frame = np.zeros((4, 6, 3), dtype=np.uint8)
                frame[0, 0] = 255  # mark the top-left corner
                return True, frame

            def release(self):
                pass

        def detector(frame, previous):
            seen.append(frame)
            return NS(locked=False, confidence=0.0, brightness=0.0), b"J"
        camera = EyeCamera(0, "EYE 1", device=device(1), resolver=lambda path: 1, opener=lambda index: Capture(),
                           detector=detector)
        eyes = EyeCameras([camera], config_path=tmp / "eyes.json")
        return eyes, camera, seen

    def test_rotate_turns_the_picture_and_is_remembered(self):
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp())
        eyes, camera, seen = self.make(tmp)
        app = Flask(__name__)
        app.register_blueprint(create_eye_blueprint(eyes))
        client = app.test_client()
        self.assertEqual(client.post("/api/eyes/0/rotate", json={"by": 90}).get_json()["eyes"][0]["rotation"], 90)
        camera.touch()
        for _ in range(100):
            if seen and seen[-1].shape[:2] == (6, 4):
                break
            time.sleep(0.02)
        self.assertEqual(seen[-1].shape[:2], (6, 4), "90 deg: 6 wide x 4 tall becomes 4 wide x 6 tall")
        self.assertEqual(seen[-1][0, 3, 0], 255, "clockwise: the top-left corner moves to the top-right")
        again, _, _ = self.make(tmp)
        self.assertEqual(again.cameras[0].rotation, 90, "kept after a restart")
        self.assertEqual(client.post("/api/eyes/0/rotate", json={"by": 45}).status_code, 400)
        self.assertEqual(client.post("/api/eyes/7/rotate", json={"by": 90}).status_code, 404)


class NoPictureTests(unittest.TestCase):
    """A camera with no picture is reopened by itself."""

    def test_silent_camera_is_reopened_until_pictures_come_back(self):
        import eye_cameras
        opened = []

        class Capture:
            def __init__(self, works):
                self.works = works

            def read(self):
                return (True, NS(shape=(480, 640, 3))) if self.works else (False, None)

            def release(self):
                pass

        def opener(index):
            opened.append(index)
            return Capture(works=len(opened) >= 3)  # dead after the blip, back on the 3rd open

        result = NS(locked=False, confidence=0.0, brightness=0.3)
        camera = EyeCamera(1, "EYE 2", device=device(3), resolver=lambda path: 2, opener=opener,
                           detector=lambda frame, previous: (result, b"J"))
        old = (eye_cameras.NO_PICTURE_REOPEN_S, eye_cameras.FIRST_FRAME_S)
        eye_cameras.NO_PICTURE_REOPEN_S, eye_cameras.FIRST_FRAME_S = 0.3, 0.2
        try:
            camera.touch()
            for _ in range(200):
                if camera.latest_jpeg():
                    break
                time.sleep(0.02)
        finally:
            eye_cameras.NO_PICTURE_REOPEN_S, eye_cameras.FIRST_FRAME_S = old
        self.assertGreaterEqual(len(opened), 3, "reopened after each silent spell")
        self.assertEqual(camera.latest_jpeg(), b"J", "pictures are back")
        self.assertEqual(camera.status()["error"], "")


class SamePictureTests(unittest.TestCase):
    """Two views of one camera: the later one is reopened, then the page asks for a re-plug."""

    def test_second_view_of_the_same_camera_is_reopened_then_reported(self):
        import numpy as np
        from eye_cameras import GAVE_UP, MAX_REOPENS, SAME_CHECKS
        picture = np.tile(np.arange(32, dtype="float32") * 6, (24, 1))  # has detail
        one, two = EyeCamera(1, "EYE 1"), EyeCamera(2, "EYE 2")
        eyes = EyeCameras([one, two])
        reopened = []
        two.request_reopen = lambda reason: reopened.append(reason)
        for camera in (one, two):
            camera._thumb, camera._thumb_at = picture, time.monotonic()
        for _ in range(SAME_CHECKS * (MAX_REOPENS + 1)):
            two._thumb_at = one._thumb_at = time.monotonic()
            statuses = eyes.status()
        self.assertEqual(len(reopened), MAX_REOPENS)
        self.assertEqual(statuses[1]["error"], GAVE_UP)

    def test_two_different_eyes_or_dark_views_are_left_alone(self):
        import numpy as np
        from eye_cameras import SAME_CHECKS, same_picture
        left = np.tile(np.arange(32, dtype="float32") * 6, (24, 1))
        self.assertFalse(same_picture(left, left[:, ::-1].copy()), "different pictures")
        dark = np.full((24, 32), 3, dtype="float32")
        self.assertFalse(same_picture(dark, dark), "dark, empty views are not compared")
        self.assertTrue(same_picture(left, left + 1))
        one, two = EyeCamera(1, "EYE 1"), EyeCamera(2, "EYE 2")
        eyes = EyeCameras([one, two])
        reopened = []
        two.request_reopen = lambda reason: reopened.append(reason)
        for _ in range(SAME_CHECKS * 3):
            one._thumb, one._thumb_at = left, time.monotonic()
            two._thumb, two._thumb_at = left[:, ::-1].copy(), time.monotonic()
            eyes.status()
        self.assertEqual(reopened, [])


if __name__ == "__main__":
    unittest.main()

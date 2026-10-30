"""Eye steering and camera tilt on their own: where the eyes look, and how far the camera tilts."""
import struct
import time
import unittest
from types import SimpleNamespace as NS

from drone_flight.head.eye_cameras import EyeCamera, EyeCameras, EyeDevice
from drone_flight.head.eye_steering import EYE_TILT_MAX_DEG, EYE_TURN_MAX_DEG, TILT_MAX_DEG, EyeGaze, tilt_for
from drone_flight.head.head_follow import HeadYawReader


class EyeGazeTests(unittest.TestCase):
    def test_no_reading_or_no_zero_adds_nothing(self):
        gaze = EyeGaze()
        self.assertEqual(gaze.offset({0: (0.2, 0.5)}), (0.0, 0.0), "not zeroed yet")
        gaze.zero({0: (0.5, 0.5)})
        self.assertEqual(gaze.offset({}), (0.0, 0.0), "eyes closed or cameras off")
        self.assertEqual(gaze.offset({1: (0.2, 0.5)}), (0.0, 0.0), "an eye that had no zero is not used")

    def test_looking_right_and_up(self):
        # the camera faces the eye: looking right moves the pupil left in the picture, looking up moves it up
        gaze = EyeGaze()
        gaze.zero({0: (0.5, 0.5)})
        right, up = gaze.offset({0: (0.4, 0.5)})
        self.assertGreater(right, 0)
        self.assertEqual(up, 0.0)
        right, up = gaze.offset({0: (0.5, 0.4)})
        self.assertEqual(right, 0.0)
        self.assertGreater(up, 0)
        right, up = gaze.offset({0: (0.6, 0.6)})
        self.assertLess(right, 0, "looking left")
        self.assertLess(up, 0, "looking down")

    def test_documented_numbers(self):
        gaze = EyeGaze()
        gaze.zero({0: (0.5, 0.5)})
        self.assertAlmostEqual(gaze.offset({0: (0.3, 0.5)})[0], 15.0, msg="0.2 of the picture = 15 degrees")
        self.assertAlmostEqual(gaze.offset({0: (0.4, 0.5)})[0], 7.5)
        self.assertAlmostEqual(gaze.offset({0: (0.5, 0.4)})[1], 7.5)
        self.assertEqual(gaze.offset({0: (0.475, 0.525)}), (0.0, 0.0), "under 0.03: jitter")

    def test_small_movements_are_ignored_and_big_ones_capped(self):
        gaze = EyeGaze()
        gaze.zero({0: (0.5, 0.5)})
        self.assertEqual(gaze.offset({0: (0.49, 0.51)}), (0.0, 0.0), "jitter")
        right, up = gaze.offset({0: (0.0, 0.0)})
        self.assertEqual((right, up), (EYE_TURN_MAX_DEG, EYE_TILT_MAX_DEG))

    def test_both_eyes_are_averaged(self):
        gaze = EyeGaze()
        gaze.zero({0: (0.5, 0.5), 1: (0.4, 0.5)})
        self.assertAlmostEqual(gaze.offset({0: (0.4, 0.5), 1: (0.2, 0.5)})[0], 11.25, msg="0.1 and 0.2: average 0.15")
        self.assertAlmostEqual(gaze.offset({0: (0.4, 0.5)})[0], 7.5, msg="one eye seen: that eye alone")

    def test_clear(self):
        gaze = EyeGaze()
        gaze.zero({0: (0.5, 0.5)})
        gaze.clear()
        self.assertEqual(gaze.offset({0: (0.2, 0.2)}), (0.0, 0.0))


class TiltTests(unittest.TestCase):
    def test_head_and_eyes_add_up_within_the_limit(self):
        self.assertEqual(tilt_for(10.0, 5.0), 15.0)
        self.assertEqual(tilt_for(-10.0, 0.0), -10.0)
        self.assertEqual(tilt_for(80.0, 10.0), TILT_MAX_DEG)
        self.assertEqual(tilt_for(-80.0, -10.0), -TILT_MAX_DEG)


def rvc(yaw_deg, pitch_deg):
    body = bytes([1]) + struct.pack("<hhhhhh", int(yaw_deg * 100), int(pitch_deg * 100), 0, 0, 0, 1000) + b"\x00" * 3
    return b"\xaa\xaa" + body + bytes([sum(body) & 0xFF])


class HeadPitchTests(unittest.TestCase):
    def test_reader_keeps_yaw_and_pitch(self):
        class Port:
            def __init__(self):
                self.data = rvc(66.0, 12.5)

            def read(self, size):
                data, self.data = self.data, b""
                if not data:
                    time.sleep(0.01)
                return data

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        reader = HeadYawReader("COM9", opener=Port)
        try:
            for _ in range(100):
                if reader.latest(1.0) is not None:
                    break
                time.sleep(0.01)
            self.assertEqual(reader.latest(1.0), 66.0)
            self.assertEqual(reader.latest_pitch(1.0), 12.5)
        finally:
            reader.close()


class EyeCameraPupilTests(unittest.TestCase):
    def test_median_of_recent_pupils_ignores_a_blink_frame(self):
        clock = [100.0]
        camera = EyeCamera(0, "EYE 1", clock=lambda: clock[0])
        view = (0, "cam")
        for x in (0.40, 0.41, 0.60, 0.42):  # 0.60: the lid half over the pupil for one frame
            camera._pupils.append((x, 0.5, clock[0], view))
        self.assertEqual(camera.pupil(0.5), (view, (0.415, 0.5)))
        self.assertEqual(EyeCameras([camera]).pupils(0.5), {(0, 0, "cam"): (0.415, 0.5)})
        camera._pupils.append((0.3, 0.3, clock[0], (90, "cam")))  # picture turned on the page
        self.assertEqual(EyeCameras([camera]).pupils(0.5), {(0, 90, "cam"): (0.3, 0.3)},
                         "a new key, so the old straight-ahead point is not used for it")
        clock[0] += 0.6
        self.assertIsNone(camera.pupil(0.5), "all older than max_age")

    def test_latest_locked_pupil_and_nothing_once_stale(self):
        class Capture:
            def read(self):
                return True, NS(shape=(480, 640, 3))

            def release(self):
                pass

        clock = [100.0]
        result = NS(locked=True, contour=None, radius=10.0, x=0.42, y=0.55, confidence=0.9, brightness=0.4,
                    source="classic")
        camera = EyeCamera(0, "EYE 1", device=EyeDevice(r"\\?\usb#cam1", "CAM1"), resolver=lambda path: 1,
                           opener=lambda index: Capture(), detector=lambda frame, previous: (result, b"J"),
                           clock=lambda: clock[0])
        eyes = EyeCameras([camera])
        for _ in range(100):
            if eyes.pupils(0.5):
                break
            time.sleep(0.02)
        key = (0, 0, r"\\?\usb#cam1")  # eye, rotation, camera
        self.assertEqual(eyes.pupils(0.5), {key: (0.42, 0.55)}, "reading the pupils keeps the camera running")
        result.locked = False  # a blink: frames without a pupil never replace the last locked one
        result.x = result.y = None
        time.sleep(0.2)
        self.assertEqual(eyes.pupils(0.5), {key: (0.42, 0.55)})
        clock[0] += 1.0
        self.assertEqual(eyes.pupils(0.5), {}, "too old")


if __name__ == "__main__":
    unittest.main()

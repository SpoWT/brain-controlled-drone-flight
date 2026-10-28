"""Both eye cameras run pupil detection at the same time, on pictures of different sizes."""
import threading
import unittest

import cv2
import numpy as np

from goggle_pupil import detect_pupil


def eye(width, height):
    frame = np.full((height, width, 3), 170, np.uint8)
    cv2.circle(frame, (width // 2, height // 2), 40, (60, 60, 60), -1)
    cv2.circle(frame, (width // 2, height // 2), 15, (10, 10, 10), -1)
    return frame


class TwoCameraTests(unittest.TestCase):
    def test_two_threads_with_different_frame_sizes(self):
        errors = []

        def run(frame):
            try:
                for _ in range(600):
                    detect_pupil(frame, (0.5, 0.5), None)
            except Exception as exc:  # collected and asserted below
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(eye(640, 480),)), threading.Thread(target=run, args=(eye(480, 640),))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()

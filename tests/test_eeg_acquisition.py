"""Ganglion connection: dongle detection and a BrainFlow synthetic stream."""
import importlib.util
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from bci_game import acquisition


def port(device, description="", vid=None, pid=None):
    return SimpleNamespace(device=device, description=description, vid=vid, pid=pid)


class DongleTests(unittest.TestCase):
    def find(self, ports):
        with mock.patch("serial.tools.list_ports.comports", return_value=ports):
            return acquisition.find_bled112_port()

    def test_one_dongle_is_found_by_usb_id_or_name(self):
        self.assertEqual(self.find([port("COM6", vid=0x2458, pid=0x0001), port("COM4", "CP210x")]), "COM6")
        self.assertEqual(self.find([port("/dev/cu.usbmodem1", "Low Energy Dongle")]), "/dev/cu.usbmodem1")

    def test_no_guess_between_several_or_none(self):
        self.assertIsNone(self.find([port("COM6", "BLED112"), port("COM7", "BLED112")]))
        self.assertIsNone(self.find([port("COM4", "CP210x")]))


@unittest.skipUnless(importlib.util.find_spec("brainflow"), "brainflow not installed")
class SyntheticStreamTests(unittest.TestCase):
    def test_synthetic_board_streams_four_channels(self):
        handle = acquisition.connect(simulate=True)
        try:
            time.sleep(0.3)
            chunk = acquisition.read_chunk(handle)
            eeg = acquisition.to_eeg_matrix(handle, chunk)
            self.assertEqual(handle.transport, "synthetic")
            self.assertGreater(len(eeg), 0)
            self.assertEqual(eeg.shape[1], 4)
            self.assertEqual(len(chunk[handle.timestamp_row]), len(eeg))
            self.assertEqual(len(handle.accel_rows), 3)
        finally:
            acquisition.release(handle)
        acquisition.release(handle)  # releasing twice is harmless
        acquisition.release(None)


if __name__ == "__main__":
    unittest.main()

"""The laptop flight recorder keeps every message and never breaks the link."""
import json
import tempfile
import unittest
from pathlib import Path

from drone_flight.flight.ardupilot_link import STREAMS, ArduPilotLink
from drone_flight.flight.mavlink_recorder import MavlinkRecorder
from tests.fake_ardupilot import FakeArduPilot, Msg


class RecorderTests(unittest.TestCase):
    def make(self, recorder):
        fc = FakeArduPilot()
        return ArduPilotLink("sim", connection_factory=fc.connection, start_reader=False, recorder=recorder)

    def test_records_fc_and_radio_messages(self):
        rec = MavlinkRecorder(Path(tempfile.mkdtemp()), clock=lambda: 12.5)
        link = self.make(rec)
        link.handle(Msg("VIBRATION", vibration_x=12.0, vibration_y=9.0, vibration_z=31.0, clipping_0=4))
        link.handle(Msg("RADIO_STATUS", src=(51, 68), rssi=180, remrssi=150))
        rec.close()
        rows = [json.loads(line) for line in rec.path.read_text().splitlines()]
        self.assertEqual([r["msg"] for r in rows], ["VIBRATION", "RADIO_STATUS"])
        self.assertEqual(rows[0]["f"]["vibration_z"], 31.0)
        self.assertEqual(rows[1]["src"], [51, 68])
        self.assertEqual(rows[0]["t"], 12.5)

    def test_write_failure_turns_recorder_off_but_link_keeps_working(self):
        rec = MavlinkRecorder(Path(tempfile.mkdtemp()))
        rec._file.close()  # simulate a disk/file error
        link = self.make(rec)
        link.handle(Msg("HEARTBEAT", base_mode=128, custom_mode=22))
        self.assertTrue(rec.error)
        self.assertTrue(link.armed())

    def test_radio_streams_stay_lean(self):
        # a full SiK link delays the laser readings
        ids = [msg_id for msg_id, _ in STREAMS]
        self.assertNotIn(241, ids)
        self.assertIn(132, ids, "the laser still comes over the radio")
        self.assertLessEqual(sum(hz for _, hz in STREAMS), 25)


if __name__ == "__main__":
    unittest.main()

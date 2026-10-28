"""Head steering: the drone's heading follows the headset's yaw."""
import struct
import unittest

from types import SimpleNamespace as NS

from head_follow import NO_ADAPTER, HeadFollow, HeadSensorWatch, parse_rvc_packet, scan


def rvc(yaw_deg, pitch_deg=0.0, roll_deg=0.0, index=1):
    body = bytes([index]) + struct.pack("<hhhhhh", int(yaw_deg * 100), int(pitch_deg * 100), int(roll_deg * 100), 0, 0, 1000)
    body += b"\x00\x00\x00"  # MI, MR, reserved
    return b"\xaa\xaa" + body + bytes([sum(body) & 0xFF])


class RvcPacketTests(unittest.TestCase):
    def test_parses_yaw_pitch_roll(self):
        self.assertEqual(parse_rvc_packet(rvc(66.09, -1.05, 1.06)), (66.09, -1.05, 1.06))

    def test_rejects_bad_checksum_and_header(self):
        bad = bytearray(rvc(10)); bad[-1] ^= 0xFF
        self.assertIsNone(parse_rvc_packet(bytes(bad)))
        self.assertIsNone(parse_rvc_packet(b"\x00" * 19))


class HeadFollowTests(unittest.TestCase):
    def test_turn_rate_follows_the_head_and_stops_when_matched(self):
        f = HeadFollow()
        f.zero(head_yaw=66.0, drone_yaw=10.0)
        self.assertEqual(f.rate(head_yaw=66.0, drone_yaw=10.0), 0, "looking ahead: no turn")
        self.assertGreater(f.rate(head_yaw=96.0, drone_yaw=10.0), 0, "head 30 deg right: turn right")
        self.assertLess(f.rate(head_yaw=36.0, drone_yaw=10.0), 0, "head 30 deg left: turn left")
        self.assertEqual(f.rate(head_yaw=96.0, drone_yaw=40.0), 0, "drone caught up with the head: stop")

    def test_turn_rate_is_capped_and_wraps_around_180(self):
        f = HeadFollow()
        f.zero(head_yaw=170.0, drone_yaw=0.0)
        self.assertLessEqual(abs(f.rate(head_yaw=-100.0, drone_yaw=0.0)), 45, "capped at 45 deg/s")
        self.assertGreater(f.rate(head_yaw=-170.0, drone_yaw=0.0), 0, "170 -> -170 is 20 deg to the right")

    def test_small_head_wobble_is_ignored(self):
        f = HeadFollow()
        f.zero(head_yaw=0.0, drone_yaw=0.0)
        self.assertEqual(f.rate(head_yaw=3.0, drone_yaw=0.0), 0)

    def test_no_reading_means_no_turn(self):
        f = HeadFollow()
        self.assertEqual(f.rate(head_yaw=None, drone_yaw=0.0), 0, "never zeroed")
        f.zero(head_yaw=0.0, drone_yaw=0.0)
        self.assertEqual(f.rate(head_yaw=None, drone_yaw=0.0), 0, "head sensor quiet")
        self.assertEqual(f.rate(head_yaw=40.0, drone_yaw=None), 0, "drone heading unknown")


class HeadSensorSearchTests(unittest.TestCase):
    """The website keeps looking for the head sensor and says what it sees."""

    def test_scan_reports_what_each_port_sends(self):
        self.assertEqual(scan(ports=[]), (None, NO_ADAPTER))
        data = {"COM7": b"\xfe\x7e\x60\x00" * 50, "COM8": b"", "COM9": b"\x00" + rvc(12.0) + rvc(12.5)}
        self.assertEqual(scan(ports=["COM7", "COM8"], listen_s=1.0, listener=lambda port, s: data[port]),
                         (None, "COM7 sends unreadable data (200 bytes/s), COM8 sends nothing"))
        self.assertEqual(scan(ports=["COM7", "COM9"], listener=lambda port, s: data[port]), ("COM9", ""))

    def test_watch_attaches_when_the_sensor_appears_and_lets_go_when_it_goes_silent(self):
        now = [0.0]
        found = [(None, "COM7 sends unreadable data")]

        class Reader:
            def __init__(self, port):
                self.port, self.closed, self.value = port, False, 10.0

            def latest(self, max_age):
                return self.value

            def close(self):
                self.closed = True

        transport = NS(head=None)
        watch = HeadSensorWatch(transport, exclude={"COM4"}, scanner=lambda exclude: found[-1],
                                reader_factory=Reader, clock=lambda: now[0])
        watch.step()
        self.assertIsNone(transport.head)
        self.assertEqual(watch.note, "COM7 sends unreadable data")
        found.append(("COM7", ""))
        watch.step()
        self.assertEqual(transport.head.port, "COM7")
        self.assertEqual(watch.note, "")
        reader = transport.head
        reader.value = None
        now[0] = 20.0
        watch.step()
        self.assertIsNone(transport.head, "silent sensor is let go so it can be found again")
        self.assertTrue(reader.closed)


if __name__ == "__main__":
    unittest.main()

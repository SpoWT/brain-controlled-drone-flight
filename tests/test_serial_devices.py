"""The radio and the head sensor are found by what they send, on any port, again and again."""
import os
import unittest
from types import SimpleNamespace as NS

os.environ.setdefault("MAVLINK20", "1")

from ardupilot_link import ArduPilotLink  # noqa: E402
from head_follow import HeadSensorWatch  # noqa: E402
from serial_devices import NO_RADIO, RADIO_SILENT_S, RadioWatch, find_radio, has_fc_heartbeat  # noqa: E402


def heartbeat_bytes(system=1, component=1, mav_type=2, armed=False):
    from pymavlink.dialects.v20 import ardupilotmega as mavlink
    mav = mavlink.MAVLink(None, srcSystem=system, srcComponent=component)
    return mav.heartbeat_encode(mav_type, 3, 209 if armed else 81, 0, 3).pack(mav)


RVC_NOISE = bytes([0xAA, 0xAA, 1, 2, 3] * 40)


class HeartbeatTests(unittest.TestCase):
    def test_only_the_flight_controllers_heartbeat_counts(self):
        self.assertTrue(has_fc_heartbeat(RVC_NOISE + heartbeat_bytes() + RVC_NOISE))
        self.assertFalse(has_fc_heartbeat(heartbeat_bytes(system=255, component=190, mav_type=6)), "a GCS")
        self.assertFalse(has_fc_heartbeat(heartbeat_bytes(system=2)), "another vehicle")
        self.assertFalse(has_fc_heartbeat(RVC_NOISE), "the head sensor")
        self.assertFalse(has_fc_heartbeat(heartbeat_bytes()[:-1] + b"\x00"), "bad checksum")


class FindRadioTests(unittest.TestCase):
    def test_finds_the_port_the_drone_answers_on(self):
        answers = {"COM4": False, "COM7": True}
        self.assertEqual(find_radio(ports=["COM4", "COM7"], prober=answers.get), ("COM7", ""))

    def test_last_radio_port_is_tried_first(self):
        tried = []
        find_radio(ports=["COM4", "COM7"], prefer="com7", prober=lambda port: tried.append(port))
        self.assertEqual(tried, ["COM7", "COM4"])

    def test_says_what_it_saw(self):
        def prober(port):
            if port == "COM5":
                raise OSError("access denied")
            return False
        self.assertEqual(find_radio(ports=["COM4", "COM5"], prober=prober), (None, "drone not heard on COM4, COM5 busy"))
        self.assertEqual(find_radio(ports=[]), (None, NO_RADIO))


class FakeLink:
    def __init__(self, device=""):
        self.device, self.hb, self.quiet, self.armed_recently = device, None, 0.0, False
        self.events = []

    def heartbeat(self):
        return self.hb

    def quiet_for(self):
        return self.quiet

    def armed_within(self, seconds):
        return self.armed_recently

    def attach(self, port):
        self.events.append(("attach", port))
        self.device = port

    def detach(self):
        self.events.append(("detach", self.device))
        self.device = ""


class RadioWatchTests(unittest.TestCase):
    def make(self, link, found=("COM7", ""), flying=False, busy=frozenset()):
        calls = []

        def finder(exclude, prefer):
            calls.append((set(exclude), prefer))
            return found
        return RadioWatch(link, finder=finder, flying=lambda: flying, busy=lambda: set(busy)), calls

    def test_answering_drone_is_left_alone(self):
        link = FakeLink("COM7")
        link.hb = object()
        watch, calls = self.make(link)
        watch.step()
        self.assertEqual((calls, link.events), ([], []))

    def test_wrong_port_is_let_go_and_the_drone_is_found_elsewhere(self):
        link = FakeLink("COM4")  # e.g. the head sensor took COM4 after re-plugging
        link.quiet = RADIO_SILENT_S - 1
        watch, calls = self.make(link, busy={"COM9"})
        watch.step()
        self.assertEqual(link.events, [], "not silent long enough yet")
        link.quiet = RADIO_SILENT_S
        watch.step()
        self.assertEqual(link.events, [("detach", "COM4"), ("attach", "COM7")])
        self.assertEqual(calls, [({"COM9"}, "COM4")], "skips the head sensor's port")
        self.assertEqual((link.device, watch.note), ("COM7", ""))

    def test_keeps_looking_while_the_drone_is_off(self):
        link = FakeLink()
        watch, calls = self.make(link, found=(None, "drone not heard on COM7"))
        for _ in range(3):
            watch.step()
        self.assertEqual(len(calls), 3, "checked again every time, never decided once")
        self.assertEqual(watch.note, "drone not heard on COM7")

    def test_never_lets_go_while_it_may_be_flying(self):
        for flying, armed in ((True, False), (False, True)):
            link = FakeLink("COM7")
            link.quiet, link.armed_recently = RADIO_SILENT_S - 1, armed
            watch, calls = self.make(link, found=("COM4", ""), flying=flying)
            watch.step()
            self.assertEqual((calls, link.events), ([], []), "a short dropout: just reopen the same port")
            self.assertIn("keeping the port", watch.note)
            link.quiet = 30.0
            watch, calls = self.make(link, found=(None, "drone not heard on COM4"), flying=flying)
            for _ in range(3):
                watch.step()
            self.assertEqual(len(calls), 3, "keeps checking the other adapters")
            self.assertEqual(link.events, [], "never lets go without a port the drone answers on")
            self.assertIn("keeping the port", watch.note)

    def test_after_a_minute_of_silence_in_the_air_every_port_is_checked_again(self):
        from serial_devices import FLYING_HOLD_S
        link = FakeLink("COM7")
        link.quiet = FLYING_HOLD_S
        watch, calls = self.make(link, found=("COM7", ""), flying=True)
        watch.step()
        self.assertEqual(link.events, [("detach", "COM7"), ("attach", "COM7")], "the old name is tried again too")

    def test_radio_back_under_another_name_while_flying(self):
        # USB glitch: the head sensor came back as COM7 (the old radio name), the radio as COM9
        link = FakeLink("COM7")
        link.quiet = RADIO_SILENT_S
        watch, calls = self.make(link, found=("COM9", ""), flying=True, busy={"COM4"})
        watch.step()
        self.assertEqual(link.events, [("attach", "COM9")], "straight to the new port, no detach")
        self.assertEqual(calls, [({"COM4", "COM7"}, "COM7")])


class FakeConnection:
    def __init__(self, device):
        self.device, self.closed, self.sent = device, False, []
        self.mav = NS(heartbeat_send=lambda *a: self.sent.append(a))

    def recv_match(self, **kw):
        return None

    def close(self):
        self.closed = True


def fc_heartbeat(armed):
    return NS(get_type=lambda: "HEARTBEAT", get_srcSystem=lambda: 1, get_srcComponent=lambda: 1,
              base_mode=209 if armed else 81, custom_mode=0)


class LinkPortTests(unittest.TestCase):
    def test_link_can_start_without_a_port_and_move_between_ports(self):
        now = [100.0]
        made = []

        def factory(device, **kw):
            made.append(FakeConnection(device))
            return made[-1]
        link = ArduPilotLink("", connection_factory=factory, clock=lambda: now[0], start_reader=False)
        self.assertEqual(link.device, "")
        link.send_gcs_heartbeat()  # nowhere to send: dropped, no error
        link.attach("COM4")
        link.send_gcs_heartbeat()
        self.assertEqual(len(made[0].sent), 1)
        now[0] += 7
        self.assertEqual(link.quiet_for(), 7)
        link.attach("COM7")
        self.assertTrue(made[0].closed, "the old port is closed")
        self.assertEqual((link.device, link.quiet_for()), ("COM7", 0))
        link.detach()
        self.assertTrue(made[1].closed)
        self.assertEqual(link.device, "")

    def test_remembers_whether_the_drone_was_armed(self):
        now = [0.0]
        link = ArduPilotLink("", connection_factory=FakeConnection, clock=lambda: now[0], start_reader=False)
        link.handle(fc_heartbeat(armed=True))
        now[0] += 30
        self.assertTrue(link.armed_within(60))
        link.handle(fc_heartbeat(armed=False))
        now[0] += 31
        self.assertFalse(link.armed_within(60))
        self.assertEqual(link.quiet_for(), 31)


class HeadSteeringBeforeTheDroneTests(unittest.TestCase):
    def test_head_steering_can_be_switched_on_before_the_drone_is_heard(self):
        from ardupilot_flight_transport import ArduPilotFlightTransport
        link = ArduPilotLink("", connection_factory=FakeConnection, start_reader=False)
        transport = ArduPilotFlightTransport("", link=link, start_worker=False)
        transport.send("HEAD 1")
        self.assertTrue(transport._head_on)
        with self.assertRaises(RuntimeError):
            transport.send("TAKEOFF 500")  # anything for the drone still needs it


class CutDuringRadioOutageTests(unittest.TestCase):
    def test_motor_cut_keeps_going_until_the_drone_is_heard_disarmed(self):
        from ardupilot_flight_transport import CUT_REPEAT_S
        from tests.test_ardupilot_transport import Sim
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        self.assertTrue(sim.fc.armed)
        sim.feed_fc = False  # nothing heard from the drone (e.g. the radio is moving to another port)
        cuts = []
        disarm = sim.link.disarm
        sim.link.disarm = lambda force=False: (cuts.append(sim.now), disarm(force))
        start = sim.now
        sim.transport.send("LOCK")
        sim.advance(CUT_REPEAT_S + 4, keys=(0, 0))
        self.assertGreater(max(cuts) - start, CUT_REPEAT_S + 3, "still cutting: a stale heartbeat proves nothing")
        sim.feed_fc = True  # heard again, motors off
        sim.advance(CUT_REPEAT_S, keys=(0, 0))
        cuts.clear()
        sim.advance(2, keys=(0, 0))
        self.assertEqual(cuts, [], "stops once a fresh heartbeat says disarmed")
        self.assertFalse(sim.link.armed())


class SecondCutTests(unittest.TestCase):
    def test_pressing_the_cut_again_keeps_it_repeating(self):
        from ardupilot_flight_transport import CUT_REPEAT_S
        from tests.test_ardupilot_transport import Sim
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.feed_fc = False
        sim.transport.send("LOCK")
        sim.advance(1, keys=(0, 0))
        sim.transport.send("LOCK")  # pressed again (or a second tab): must not drop the cut
        self.assertEqual(sim.transport._phase, "FAULT")
        cuts = []
        disarm = sim.link.disarm
        sim.link.disarm = lambda force=False: (cuts.append(sim.now), disarm(force))
        start = sim.now
        sim.advance(CUT_REPEAT_S + 3, keys=(0, 0))
        self.assertGreater(max(cuts) - start, CUT_REPEAT_S + 2, "still cutting")

    def test_lock_on_the_ground_without_a_pending_cut_is_idle(self):
        from tests.test_ardupilot_transport import Sim
        sim = Sim(self)
        sim.transport.send("LOCK")
        self.assertEqual(sim.transport._phase, "IDLE")


class HeadWatchExcludeTests(unittest.TestCase):
    def test_head_search_skips_the_radios_current_port(self):
        transport = NS(head=None, link=NS(device="COM7"))
        seen = []
        watch = HeadSensorWatch(transport, exclude=lambda: {transport.link.device},
                                scanner=lambda exclude: seen.append(set(exclude)) or (None, "nothing"))
        watch.step()
        transport.link.device = ""  # radio let go while the drone is off
        watch.step()
        self.assertEqual(seen, [{"COM7"}, set()])


if __name__ == "__main__":
    unittest.main()

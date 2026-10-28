"""ArduPilotFlightTransport + the unchanged AssistedFlightService against a simulated ArduPilot."""
import json
import statistics
import tempfile
import unittest
from pathlib import Path

from ardupilot_flight_transport import ArduPilotFlightTransport, throttle_for_climb
from ardupilot_link import ArduPilotLink
from assisted_flight_service import AssistedFlightService
from tests.fake_ardupilot import FakeArduPilot


class Sim:
    DT = 0.05

    def __init__(self, test, **fc_kw):
        self.now = 0.0
        self.fc = FakeArduPilot()
        for k, v in fc_kw.items():
            setattr(self.fc, k, v)
        clock = lambda: self.now  # noqa: E731
        self.link = ArduPilotLink("sim", connection_factory=self.fc.connection, clock=clock, start_reader=False)
        self.transport = ArduPilotFlightTransport("sim", link=self.link, clock=clock, start_worker=False)
        tmp = Path(tempfile.mkdtemp())
        (tmp / "cal.json").write_text(json.dumps({"scale_um": 1786, "axis_map": 5}))
        self.service = AssistedFlightService(self.transport, calibration_path=tmp / "cal.json",
                                             flight_log_dir=None, clock=clock, sleep=self.advance,
                                             start_worker=False)
        self.token = self.service.begin_browser_session()
        self.seq = 0
        self.feed_fc = True
        self.advance(0.5)

    def advance(self, seconds, keys=None):
        steps = max(1, int(round(seconds / self.DT)))
        for i in range(steps):
            self.now += self.DT
            self.fc.step(self.DT, self.now)
            if self.feed_fc:
                for msg in self.fc.messages():
                    self.link.handle(msg)
            if i % 2 == 0:
                self.transport.tick()
                if keys is not None:
                    self.service.browser_heartbeat(self.token, forward=keys[0], right=keys[1])
                self.service.tick()

    def next_seq(self):
        self.seq += 1
        return self.seq

    def arm(self, target=1000):
        self.service.arm_and_takeoff(self.token, self.next_seq(), target)

    def phase(self):
        return self.service.status()["phase"]


class FullFlightTests(unittest.TestCase):
    def test_arm_climbs_to_target_holds_flies_and_lands(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        self.assertTrue(sim.fc.armed)
        self.assertEqual(sim.fc.mode, 5, "flies in LOITER: the old drone's lock-the-spot position hold")
        self.assertAlmostEqual(sim.fc.height_m, 1.0, delta=0.1)
        north0 = sim.fc.north
        sim.advance(2, keys=(300, 0))
        self.assertGreater(sim.fc.north - north0, 0.3, "W flies forward")
        self.assertGreater(sim.fc.sticks[0], 0)
        sim.advance(1, keys=(0, 0))
        self.assertEqual(sim.fc.sticks[0], 0, "releasing keys = sticks centred = hold")
        sim.service.land(sim.token, sim.next_seq())
        sim.advance(12, keys=(0, 0))
        self.assertFalse(sim.fc.armed)
        self.assertEqual(sim.phase(), "LANDED")
        self.assertEqual(sim.service.status()["mode"], "DISARMED")

    def test_arms_only_with_idle_throttle_in_loiter(self):
        sim = Sim(self)
        sim.arm(800)
        sim.advance(1.0)
        arms = [i for i, e in enumerate(sim.fc.log) if e[:3] == ("COMMAND", 400, 1)]
        self.assertTrue(arms)
        before = [e for e in sim.fc.log[:arms[0]] if e[0] == "MANUAL_CONTROL"]
        self.assertEqual(before[-1][3], 0, "throttle at idle when ARM is sent")
        self.assertIn(("SET_MODE", 5), sim.fc.log[:arms[0]], "takes off in LOITER: spot hold from the first moment")

    def test_measured_move_forward(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        start = sim.fc.north
        sim.service.go_forward(sim.token, sim.next_seq(), 2000)
        sim.advance(12, keys=(0, 0))
        self.assertAlmostEqual(sim.fc.north - start, 2.0, delta=0.15)
        self.assertEqual(sim.phase(), "FLY")
        self.assertTrue(sim.service.status()["telemetry"]["move_done"])

    def test_emergency_cut_forces_disarm_in_air(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.service.emergency_lock(sim.token)
        sim.advance(0.2)
        self.assertFalse(sim.fc.armed)
        self.assertEqual(sim.fc.forced_disarms, 1)


class SafetyTests(unittest.TestCase):
    def test_no_lift_stops_motors_on_the_floor(self):
        sim = Sim(self, can_lift=False)
        sim.arm(1000)
        sim.advance(11, keys=(0, 0))
        self.assertFalse(sim.fc.armed)
        self.assertEqual(sim.service.status()["fault"], "NO_LIFT")

    def test_lost_height_sensor_in_air_lands(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.fc.range_alive = False
        sim.advance(1.0, keys=(0, 0))
        self.assertNotEqual(sim.fc.mode, 9, "a 1 s radio gap is not a lost laser (the FC flies on its own laser)")
        sim.advance(1.5, keys=(0, 0))
        self.assertEqual(sim.fc.mode, 9)
        self.assertEqual(sim.service.status()["last_fault"], "LASER_LOST")

    def test_over_height_lands(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.fc.height_m = 1.6
        sim.advance(0.3, keys=(0, 0))
        self.assertEqual(sim.phase(), "LAND", "laser fine -> the gentle website landing")
        self.assertLess(sim.fc.sticks[2], 400, "descending")

    def test_fc_silent_holds_then_lands_blind(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.feed_fc = False
        sim.advance(3.5, keys=(300, 0))
        self.assertEqual(sim.fc.sticks[:2], (0, 0), "no leaning while the FC cannot be heard")
        self.assertEqual(sim.fc.mode, 9, "height unknown -> LAND, sticks centred")

    def test_browser_silence_lands(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.advance(3.0)  # no browser heartbeats
        self.assertIn(sim.phase(), {"LAND", "FINAL_DESCENT"})
        self.assertLess(sim.fc.sticks[2], 400, "descending")

    def test_touchdown_backup_cuts_motors_if_fc_stays_armed(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        original = sim.fc.step

        def never_auto_disarm(dt, now):
            original(dt, now)
            sim.fc.on_ground_since = None if sim.fc.armed else sim.fc.on_ground_since
        sim.fc.step = never_auto_disarm
        sim.service.land(sim.token, sim.next_seq())
        touched_at = None
        for _ in range(180):  # up to 9 s
            sim.advance(0.05, keys=(0, 0))
            if touched_at is None and sim.fc.height_m <= 0.02:
                touched_at = sim.now
            if not sim.fc.armed:
                break
        self.assertFalse(sim.fc.armed)
        self.assertGreaterEqual(sim.fc.forced_disarms, 1)
        self.assertLessEqual(sim.now - touched_at, 1.6, "motors stop about 1 s after touching the ground")

    def test_takeoff_refused_when_not_on_the_floor(self):
        sim = Sim(self)
        sim.fc.height_m = 0.5
        sim.advance(0.3)
        with self.assertRaises(ValueError):
            sim.arm(1000)
        self.assertFalse(sim.fc.armed)

    def test_takeoff_refused_without_standby(self):
        sim = Sim(self)
        sim.transport.send("TAKEOFF 1000")
        sim.advance(1.0)
        self.assertFalse(sim.fc.armed)
        self.assertEqual(sim.transport.telemetry()[0].fault, "PRECONDITION")

    def test_mode_change_refused_keeps_motors_off(self):
        sim = Sim(self, accept_modes=False)
        with self.assertRaises(ValueError):
            sim.arm(1000)
            sim.advance(3)
            raise ValueError(sim.service.status()["fault"])
        self.assertFalse(sim.fc.armed)

    def test_gcs_heartbeat_keeps_flowing(self):
        sim = Sim(self)
        sim.advance(5)
        self.assertGreaterEqual(sim.fc.gcs_heartbeats, 5)


class LandStickTests(unittest.TestCase):
    def test_land_centres_sticks_even_while_moving(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.advance(1, keys=(400, 300))
        self.assertNotEqual(sim.fc.sticks[:2], (0, 0))
        sim.service.land(sim.token, sim.next_seq())
        sim.advance(0.2, keys=(400, 300))
        self.assertEqual(sim.fc.sticks[:2], (0, 0))
        self.assertEqual(sim.phase(), "LAND")


class StuckLaserTests(unittest.TestCase):
    def test_laser_stuck_at_2cm_while_climbing_lands_slowly_never_cuts(self):
        sim = Sim(self, range_stuck_cm=2)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        self.assertEqual(sim.fc.mode, 9, "stuck laser while the FC climbs -> LAND (slow descent)")
        self.assertLess(sim.fc.height_m, 0.8, "stopped climbing early, not to the old runaway height")
        self.assertEqual(sim.fc.forced_disarms, 0, "never cut motors in the air")
        self.assertEqual(sim.service.status()["last_fault"], "LASER_LOST")
        sim.advance(5, keys=(0, 0))
        self.assertFalse(sim.fc.armed, "landed and disarmed by the FC")

    def test_true_no_lift_ends_disarmed_on_the_floor(self):
        sim = Sim(self, can_lift=False, range_stuck_cm=2)
        sim.arm(1000)
        sim.advance(11, keys=(0, 0))
        self.assertFalse(sim.fc.armed)
        self.assertEqual(sim.service.status()["fault"], "NO_LIFT")


class Flight20261009Tests(unittest.TestCase):
    def test_height_datum_reset_on_arming_is_not_a_lost_laser(self):
        sim = Sim(self, ekf_z_bias=0.4)  # baro drifted while sitting; arming resets it
        sim.arm(1000)
        sim.advance(14, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        self.assertEqual(sim.service.status()["last_fault"], "NONE")

    def test_emergency_cut_repeats_until_the_fc_disarms(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        sim.fc.drop_commands = 4  # radio drops out at full power
        sim.service.emergency_lock(sim.token)
        sim.advance(0.3, keys=(0, 0))
        self.assertEqual(sim.fc.sticks[2], 0, "throttle to minimum at once, even before the disarm lands")
        sim.advance(2.0, keys=(0, 0))
        self.assertFalse(sim.fc.armed, "cut retried until the motors stopped")

    def test_lift_asks_for_a_gentle_climb(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(14, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        climb = [z for kind, *rest in sim.fc.log if kind == "MANUAL_CONTROL" for z in [rest[2]]]
        self.assertLessEqual(max(climb), 760, "no full-rate climb request that a radio dropout could freeze")


class LandWatchdogTests(unittest.TestCase):
    def test_land_that_climbs_is_cut(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(14, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        sim.fc.land_climbs = True
        sim.service.land(sim.token, sim.next_seq())
        sim.advance(3, keys=(0, 0))
        self.assertFalse(sim.fc.armed, "LAND climbing 40 cm -> motors cut instead of a flyaway")
        self.assertGreaterEqual(sim.fc.forced_disarms, 1)
        self.assertEqual(sim.service.status()["last_fault"], "LAND_FAILED")
        self.assertLess(sim.fc.height_m, 1.6)

    def test_normal_land_is_never_cut(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(14, keys=(0, 0))
        sim.service.land(sim.token, sim.next_seq())
        sim.advance(12, keys=(0, 0))
        self.assertFalse(sim.fc.armed)
        self.assertNotEqual(sim.service.status()["last_fault"], "LAND_FAILED")


class SlowFlightTests(unittest.TestCase):
    def test_held_key_settles_at_capped_speed_and_release_stops(self):
        sim = Sim(self, loiter_ok=False)  # FLOWHOLD fallback: lean from EKF speed feedback
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.advance(6, keys=(600, 0))  # page max 600 mm/s requested
        self.assertLess(sim.fc.vn, 0.45, "speed capped near 0.4 m/s, not accelerating forever")
        self.assertGreater(sim.fc.vn, 0.25)
        self.assertLessEqual(abs(sim.fc.sticks[0]), 300, "tilt never above the gentle limit")
        sim.advance(3, keys=(0, 0))
        self.assertLess(abs(sim.fc.vn), 0.05, "keys released -> stops")


class OldDroneLogicTests(unittest.TestCase):
    """Take-off, hold and landing behaviour."""

    def test_switches_to_loiter_once_off_the_floor(self):
        sim = Sim(self)
        sim.arm(1000)
        for _ in range(240):
            sim.advance(0.05, keys=(0, 0))
            if sim.phase() == "CLIMB":
                break
        self.assertEqual(sim.phase(), "CLIMB")
        sim.advance(1.0, keys=(0, 0))
        self.assertEqual(sim.fc.mode, 5, "position hold from the climb on, like the old drone")

    def test_key_is_a_speed_and_release_holds_the_spot(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.advance(4, keys=(300, 0))
        self.assertAlmostEqual(sim.fc.vn, 0.3, delta=0.03, msg="W = the key speed")
        sim.advance(2, keys=(0, 0))
        self.assertLess(abs(sim.fc.vn), 0.02, "released -> brakes")
        spot = sim.fc.north
        sim.advance(3, keys=(0, 0))
        self.assertAlmostEqual(sim.fc.north, spot, delta=0.03, msg="holds the spot")
        sim.advance(3, keys=(0, -300))
        self.assertLess(sim.fc.east, -0.6, "A = left")

    def test_loiter_refused_falls_back_to_alt_hold_and_still_flies(self):
        sim = Sim(self, loiter_ok=False)
        sim.arm(1000)
        sim.advance(16, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        self.assertEqual(sim.fc.mode, 2, "ALT_HOLD, never FLOWHOLD (it floods the radio)")
        north0 = sim.fc.north
        sim.advance(2, keys=(300, 0))
        self.assertGreater(sim.fc.north - north0, 0.3)

    def test_gentle_landing_like_the_old_drone(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.service.land(sim.token, sim.next_seq())
        rates, touched_at = {}, None
        for _ in range(400):  # up to 20 s
            sim.advance(0.05, keys=(0, 0))
            rates.setdefault(sim.phase(), []).append(sim.fc.vz)
            self.assertNotEqual(sim.fc.mode, 9, "the website lands it, not the FC's LAND")
            if touched_at is None and sim.fc.height_m <= 0.02:
                touched_at = sim.now
            if not sim.fc.armed:
                break
        settled = {phase: statistics.median(v) for phase, v in rates.items()}
        self.assertAlmostEqual(settled["LAND"], -0.15, delta=0.02, msg="0.15 m/s down to 10 cm")
        self.assertAlmostEqual(settled["FINAL_DESCENT"], -0.10, delta=0.02, msg="then 0.1 m/s")
        self.assertFalse(sim.fc.armed)
        self.assertLessEqual(sim.now - touched_at, 0.8, "motors off right after touchdown")
        self.assertEqual(sim.phase(), "LANDED")

    def test_laser_lost_while_landing_hands_over_to_the_fc(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.service.land(sim.token, sim.next_seq())
        sim.advance(1, keys=(0, 0))
        sim.fc.range_alive = False
        sim.advance(2.5, keys=(0, 0))
        self.assertEqual(sim.fc.mode, 9)
        self.assertEqual(sim.phase(), "EMERGENCY_DESCENT")

    def test_fc_landing_by_itself_is_followed(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(12, keys=(0, 0))
        sim.fc.mode = 9  # e.g. the FC's own EKF failsafe
        sim.advance(0.5, keys=(300, 0))
        self.assertEqual(sim.phase(), "EMERGENCY_DESCENT")
        self.assertEqual(sim.fc.sticks[:2], (0, 0))


class RadioLagTests(unittest.TestCase):
    """Laser readings arrive late or not at all over a busy radio."""

    def test_never_uses_flowhold(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(16, keys=(0, 0))
        sim.service.land(sim.token, sim.next_seq())
        sim.advance(12, keys=(0, 0))
        self.assertNotIn(("SET_MODE", 22), sim.fc.log)

    def test_short_laser_gap_on_the_radio_is_not_laser_lost(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(16, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        sim.fc.range_gap_until = sim.now + 1.3
        sim.advance(2.0, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        self.assertEqual(sim.service.status()["last_fault"], "NONE")

    def test_long_laser_silence_still_lands(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(16, keys=(0, 0))
        sim.fc.range_gap_until = sim.now + 10
        sim.advance(3.0, keys=(0, 0))
        self.assertEqual(sim.service.status()["last_fault"], "LASER_LOST")

    def test_predictive_height_with_laggy_laser_does_not_cycle(self):
        sim = Sim(self, range_delay_s=1.0)
        sim.arm(300)
        heights = []
        for _ in range(int(25 / 0.05)):
            sim.advance(0.05, keys=(0, 0))
            heights.append(sim.fc.height_m)
        self.assertEqual(sim.phase(), "FLY")
        self.assertLess(max(heights), 0.36, "no big overshoot above 30 cm")
        self.assertAlmostEqual(heights[-1], 0.30, delta=0.04)
        settled = heights[-200:]  # last 10 s
        self.assertLess(max(settled) - min(settled), 0.03, "holds still: no up-down cycle")
        sticks = [z for kind, *rest in sim.fc.log[-400:] if kind == "MANUAL_CONTROL" for z in [rest[2]]]
        self.assertTrue(all(z == 500 for z in sticks[-20:]), "settled at the target: sticks centred")

    def test_arrow_up_moves_exactly_ten_centimetres(self):
        sim = Sim(self, range_delay_s=1.0)
        sim.arm(1000)
        sim.advance(25, keys=(0, 0))
        start = sim.fc.height_m
        for _ in range(int(10 / 0.05)):
            sim.service.browser_heartbeat(sim.token, forward=0, right=0, target_mm=1100)
            sim.advance(0.05)
        self.assertAlmostEqual(sim.fc.height_m - start, 0.10, delta=0.04)


class RunawayTests(unittest.TestCase):
    """Bad position estimate: level the drone (ALT_HOLD) and land."""

    def fly(self, **fc_kw):
        sim = Sim(self, **fc_kw)
        sim.arm(300)
        sim.advance(16, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        return sim

    def assert_levelled_and_landed(self, sim, since):
        later = sim.fc.log[since:]
        self.assertEqual(sim.service.status()["last_fault"], "RUNAWAY")
        self.assertIn(("SET_MODE", 2), later, "ALT_HOLD: level, no chasing a wrong position")
        self.assertNotIn(("SET_MODE", 5), later[later.index(("SET_MODE", 2)):], "never back to LOITER")
        self.assertNotIn(("SET_MODE", 9), later, "not the FC's LAND: it would chase the wrong position too")
        sticks = [e for e in later if e[0] == "MANUAL_CONTROL"]
        self.assertTrue(all(e[1] == 0 and e[2] == 0 for e in sticks[-20:]), "roll/pitch centred")
        sim.advance(10, keys=(0, 0))
        self.assertFalse(sim.fc.armed)
        self.assertEqual(sim.phase(), "LANDED")

    def test_held_hard_lean_levels_and_lands(self):
        sim = self.fly()
        mark = len(sim.fc.log)
        sim.fc.runaway_pitch_deg = 15.0
        sim.advance(2.0, keys=(0, 0))
        self.assert_levelled_and_landed(sim, mark)

    def test_runaway_speed_estimate_levels_and_lands(self):
        sim = self.fly()
        mark = len(sim.fc.log)
        sim.fc.ekf_speed_bias = 1.2
        sim.advance(2.0, keys=(0, 0))
        self.assert_levelled_and_landed(sim, mark)

    def test_brief_lean_is_not_a_runaway(self):
        sim = self.fly()
        sim.fc.runaway_pitch_deg = 14.0
        sim.advance(1.0, keys=(0, 0))
        sim.fc.runaway_pitch_deg = 0.0
        sim.advance(3.0, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        self.assertEqual(sim.service.status()["last_fault"], "NONE")

    def test_flying_with_keys_is_not_a_runaway(self):
        sim = self.fly()
        sim.advance(6, keys=(300, 300))
        self.assertEqual(sim.phase(), "FLY")
        self.assertEqual(sim.service.status()["last_fault"], "NONE")


class KeysAfterTakeoffTests(unittest.TestCase):
    """Keys work soon after take-off even when the hover never looks perfectly still."""

    def test_keys_work_soon_after_takeoff_even_if_the_hover_wobbles(self):
        sim = Sim(self, range_wobble_cm=5)
        sim.arm(300)
        sim.advance(14, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY", "take-off ends once near the target height")
        self.assertAlmostEqual(sim.fc.height_m, 0.30, delta=0.06)
        north0 = sim.fc.north
        sim.advance(2, keys=(300, 0))
        self.assertGreater(sim.fc.north - north0, 0.2, "W flies forward")

    def test_arrow_up_works_in_a_wobbly_hover(self):
        sim = Sim(self, range_wobble_cm=5)
        sim.arm(300)
        sim.advance(14, keys=(0, 0))
        start = sim.fc.height_m
        for _ in range(int(12 / 0.05)):
            sim.service.browser_heartbeat(sim.token, forward=0, right=0, target_mm=400)
            sim.advance(0.05)
        self.assertAlmostEqual(sim.fc.height_m - start, 0.10, delta=0.05)

    def test_take_off_still_waits_until_near_the_target(self):
        sim = Sim(self)
        sim.arm(1000)
        sim.advance(6, keys=(0, 0))
        self.assertEqual(sim.phase(), "CLIMB", "still climbing towards 1 m: keys not yet")


class BlindCameraTests(unittest.TestCase):
    """A blind floor camera reports no movement while the drone drifts: land."""

    def test_blind_floor_camera_takes_off_then_lands_at_once(self):
        # take-off is not refused; the drone lands once airborne
        sim = Sim(self, flow_quality=5)
        sim.arm(300)
        sim.advance(6, keys=(0, 0))
        self.assertEqual(sim.service.status()["last_fault"], "FLOW_LOST")
        self.assertIn(sim.phase(), {"LAND", "FINAL_DESCENT", "LANDED"})
        sim.advance(10, keys=(0, 0))
        self.assertFalse(sim.fc.armed)
        self.assertLess(sim.fc.height_m, 0.05)

    def test_camera_going_blind_in_flight_levels_and_lands(self):
        sim = Sim(self)
        sim.arm(300)
        sim.advance(16, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        mark = len(sim.fc.log)
        sim.fc.flow_quality = 5
        sim.advance(2.5, keys=(0, 0))
        self.assertEqual(sim.service.status()["last_fault"], "FLOW_LOST")
        self.assertIn(("SET_MODE", 2), sim.fc.log[mark:], "ALT_HOLD: level, not chasing a blind position")
        sim.advance(10, keys=(0, 0))
        self.assertFalse(sim.fc.armed)
        self.assertEqual(sim.phase(), "LANDED")

    def test_short_camera_dip_is_not_blind(self):
        sim = Sim(self)
        sim.arm(300)
        sim.advance(16, keys=(0, 0))
        sim.fc.flow_quality = 30
        sim.advance(0.8, keys=(0, 0))
        sim.fc.flow_quality = 150
        sim.advance(3, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        self.assertEqual(sim.service.status()["last_fault"], "NONE")


class FollowOrLandTests(unittest.TestCase):
    """Wrong height, wrong direction, drift or tilt mismatch: land at once."""

    def fly(self, target=300, **fc_kw):
        sim = Sim(self, **fc_kw)
        sim.arm(target)
        sim.advance(16, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        self.assertEqual(sim.service.status()["last_fault"], "NONE")
        return sim

    def assert_lands(self, sim, fault, level):
        self.assertEqual(sim.service.status()["last_fault"], fault)
        self.assertIn(sim.phase(), {"LAND", "FINAL_DESCENT", "EMERGENCY_DESCENT"})
        if level:
            self.assertEqual(sim.fc.mode, 2, "ALT_HOLD: level, not chasing a position it cannot trust")
        sim.advance(12, keys=(0, 0))
        self.assertFalse(sim.fc.armed)
        self.assertEqual(sim.phase(), "LANDED")

    def test_sinking_below_the_height_lands(self):
        sim = self.fly()
        sim.fc.height_drift_m_s = -0.25
        sim.advance(3, keys=(0, 0))
        sim.fc.height_drift_m_s = 0.0
        self.assert_lands(sim, "UNDER_HEIGHT", level=False)

    def test_rising_above_the_height_lands(self):
        sim = self.fly(target=600)
        sim.fc.height_drift_m_s = 0.12
        sim.advance(3, keys=(0, 0))
        sim.fc.height_drift_m_s = 0.0
        self.assert_lands(sim, "OVER_HEIGHT", level=False)

    def test_drifting_while_holding_lands(self):
        sim = self.fly()
        sim.fc.ekf_speed_bias = 0.5
        sim.advance(2.5, keys=(0, 0))
        self.assert_lands(sim, "WRONG_DIRECTION", level=True)

    def test_key_that_does_not_move_it_lands(self):
        sim = self.fly()
        sim.fc.stuck_horizontal = True
        sim.advance(3.5, keys=(300, 0))
        self.assert_lands(sim, "WRONG_DIRECTION", level=True)

    def test_tilt_it_was_not_told_lands(self):
        sim = self.fly()
        sim.fc.external_tilt_deg = 9.0
        sim.advance(1.0, keys=(0, 0))
        sim.fc.external_tilt_deg = 0.0
        self.assert_lands(sim, "TILT_MISMATCH", level=True)

    def test_new_height_target_is_not_judged_before_it_gets_there(self):
        # raising the target quickly must not count as "too low" before the climb
        sim = self.fly(target=300, range_wobble_cm=3)
        for target in (400, 500, 800, 1000, 1200, 1500, 1800, 2000):
            sim.service.browser_heartbeat(sim.token, forward=0, right=0, target_mm=target)
            sim.advance(0.1)
        for _ in range(int(6 / 0.05)):
            sim.service.browser_heartbeat(sim.token, forward=0, right=0, target_mm=2000)
            sim.advance(0.05)
        self.assertEqual(sim.service.status()["last_fault"], "NONE")
        self.assertGreater(sim.fc.height_m, 0.45, "on its way up to the new height")

    def test_rocketing_up_right_after_a_height_change_still_lands(self):
        # "too high" is never paused, even while a new target waits to be planned
        sim = self.fly(target=300, range_wobble_cm=3)
        sim.service.browser_heartbeat(sim.token, forward=0, right=0, target_mm=400)
        sim.advance(0.1)
        sim.fc.height_drift_m_s = 0.4  # rockets up at 0.4 m/s
        caught_at = None
        for _ in range(int(2.5 / 0.05)):
            sim.service.browser_heartbeat(sim.token, forward=0, right=0, target_mm=400)
            sim.advance(0.05)
            if caught_at is None and sim.service.status()["last_fault"] != "NONE":
                caught_at = sim.fc.height_m
        sim.fc.height_drift_m_s = 0.0
        self.assertEqual(sim.service.status()["last_fault"], "OVER_HEIGHT")
        self.assertLess(caught_at, 0.80, "caught within ~25 cm of the 55 cm limit (400 mm target + 150)")

    def test_following_keys_and_arrows_never_trips(self):
        sim = self.fly(target=500, range_wobble_cm=3)
        for keys in ((300, 0), (0, 0), (0, 300), (0, 0), (-300, -300), (0, 0)):
            sim.advance(3, keys=keys)
        for _ in range(int(8 / 0.05)):
            sim.service.browser_heartbeat(sim.token, forward=0, right=0, target_mm=600)
            sim.advance(0.05)
        self.assertEqual(sim.phase(), "FLY")
        self.assertEqual(sim.service.status()["last_fault"], "NONE")


class TurnTests(unittest.TestCase):
    """Left/right arrows turn on the spot; W then flies the new way."""

    def test_arrows_turn_and_w_flies_the_new_way(self):
        sim = Sim(self)
        sim.arm(300)
        sim.advance(16, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        for _ in range(int(2 / 0.05)):
            sim.service.browser_heartbeat(sim.token, forward=0, right=0, yaw=45)
            sim.advance(0.05)
        turned = sim.fc.yaw_deg
        self.assertAlmostEqual(turned, 90, delta=15, msg="45 deg/s for 2 s")
        sim.advance(2, keys=(0, 0))
        settled = sim.fc.yaw_deg
        self.assertAlmostEqual(settled, turned, delta=10, msg="release stops the turn (one link tick of overrun)")
        sim.advance(2, keys=(0, 0))
        self.assertAlmostEqual(sim.fc.yaw_deg, settled, delta=0.5, msg="then holds the new heading")
        east0 = sim.fc.east
        sim.advance(2, keys=(300, 0))
        self.assertGreater(sim.fc.east - east0, 0.2, "W now flies east: forward is where it points")
        sim.advance(2, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        self.assertEqual(sim.service.status()["last_fault"], "NONE")

    def test_landing_never_turns(self):
        sim = Sim(self)
        sim.arm(300)
        sim.advance(16, keys=(0, 0))
        sim.service.browser_heartbeat(sim.token, forward=0, right=0, yaw=45)
        sim.service.land(sim.token, sim.next_seq())
        sim.advance(1)
        self.assertEqual(sim.fc.sticks[3], 0)


class FakeHead:
    """Stands in for head_follow.HeadYawReader (the VR headset's BNO08x)."""

    def __init__(self, yaw=66.0):
        self.yaw = yaw

    def latest(self, max_age):
        return self.yaw


class HeadTurnTests(unittest.TestCase):
    """Turning the head turns the drone in place."""

    def fly_with_head(self):
        sim = Sim(self)
        head = FakeHead(66.0)
        sim.transport.head = head
        sim.arm(300)
        sim.advance(16, keys=(0, 0))
        self.assertEqual(sim.phase(), "FLY")
        return sim, head

    def test_off_by_default(self):
        sim, head = self.fly_with_head()
        head.yaw = 126.0
        sim.advance(3, keys=(0, 0))
        self.assertAlmostEqual(sim.fc.yaw_deg, 0.0, delta=1)

    def test_drone_mirrors_the_head(self):
        sim, head = self.fly_with_head()
        sim.service.set_head_follow(sim.token, True)
        sim.advance(0.5, keys=(0, 0))
        head.yaw = 66.0 + 45
        sim.advance(4, keys=(0, 0))
        self.assertAlmostEqual(sim.fc.yaw_deg, 45, delta=8, msg="head 45 deg right -> drone 45 deg right")
        head.yaw = 66.0
        sim.advance(4, keys=(0, 0))
        self.assertAlmostEqual(sim.fc.yaw_deg, 0, delta=8, msg="look back ahead -> drone turns back")
        self.assertEqual(sim.service.status()["last_fault"], "NONE")

    def test_quiet_head_sensor_holds_the_heading(self):
        sim, head = self.fly_with_head()
        sim.service.set_head_follow(sim.token, True)
        sim.advance(0.5, keys=(0, 0))
        head.yaw = None
        sim.advance(3, keys=(0, 0))
        self.assertAlmostEqual(sim.fc.yaw_deg, 0, delta=1)

    def test_arrows_win_and_the_head_recentres(self):
        sim, head = self.fly_with_head()
        sim.service.set_head_follow(sim.token, True)
        sim.advance(0.5, keys=(0, 0))
        for _ in range(int(1 / 0.05)):
            sim.service.browser_heartbeat(sim.token, forward=0, right=0, yaw=45)
            sim.advance(0.05)
        turned = sim.fc.yaw_deg
        self.assertGreater(turned, 30)
        sim.advance(3, keys=(0, 0))
        self.assertAlmostEqual(sim.fc.yaw_deg, turned, delta=10, msg="head still ahead: keeps the arrow's heading")

    def test_each_takeoff_recentres_the_head(self):
        sim, head = self.fly_with_head()
        sim.service.set_head_follow(sim.token, True)
        sim.advance(0.5, keys=(0, 0))
        head.yaw = 66.0 + 40
        sim.advance(4, keys=(0, 0))
        sim.service.land(sim.token, sim.next_seq())
        sim.advance(12, keys=(0, 0))
        self.assertEqual(sim.phase(), "LANDED")
        yaw_on_floor = sim.fc.yaw_deg
        sim.arm(300)  # head still turned 40 deg: that is "straight ahead" for the new flight
        sim.advance(16, keys=(0, 0))
        self.assertAlmostEqual(sim.fc.yaw_deg, yaw_on_floor, delta=6, msg="no jump at the next take-off")

    def test_direction_status_for_the_dashboard_dial(self):
        sim, head = self.fly_with_head()
        sim.service.set_head_follow(sim.token, True)
        sim.advance(0.5, keys=(0, 0))
        head.yaw = 66.0 + 30
        sim.advance(0.3, keys=(0, 0))
        d = sim.transport.direction_status()
        self.assertTrue(d["head_on"])
        self.assertAlmostEqual(d["head_offset"], 30, delta=0.5)
        self.assertGreater(d["turn_rate"], 0, "turning right to follow the head")

    def test_dial_shows_the_head_on_the_ground_without_turning(self):
        # the dial shows the head before take-off
        sim = Sim(self)
        head = FakeHead(66.0)
        sim.transport.head = head
        sim.service.set_head_follow(sim.token, True)
        sim.advance(0.5)
        first = sim.transport.direction_status()
        self.assertEqual(first["head_offset"], 0.0)
        self.assertEqual(first["drone_offset"], 0.0)
        head.yaw = 66.0 - 25
        sim.advance(0.5)
        later = sim.transport.direction_status()
        self.assertAlmostEqual(later["head_offset"], -25, delta=0.5)
        self.assertEqual(later["turn_rate"], 0, "on the ground: shows, never turns")

    def test_recenter_mid_flight_stops_chasing_the_old_angle(self):
        sim, head = self.fly_with_head()
        sim.service.set_head_follow(sim.token, True)
        sim.advance(0.5, keys=(0, 0))
        head.yaw = 66.0 + 60
        sim.advance(0.6, keys=(0, 0))       # starts turning towards +60
        sim.service.recenter_head(sim.token)  # "this is straight ahead now"
        sim.advance(0.3, keys=(0, 0))
        stopped_at = sim.fc.yaw_deg
        sim.advance(3, keys=(0, 0))
        self.assertAlmostEqual(sim.fc.yaw_deg, stopped_at, delta=3, msg="no more turning after recentring")
        self.assertLess(stopped_at, 45, "it had not finished the old turn")

    def test_switch_off_stops_following(self):
        sim, head = self.fly_with_head()
        sim.service.set_head_follow(sim.token, True)
        sim.advance(0.5, keys=(0, 0))
        sim.service.set_head_follow(sim.token, False)
        head.yaw = 126.0
        sim.advance(3, keys=(0, 0))
        self.assertAlmostEqual(sim.fc.yaw_deg, 0.0, delta=1)


class MappingTests(unittest.TestCase):
    def test_throttle_mapping_matches_fc_deadzone(self):
        self.assertEqual(throttle_for_climb(0), 500)
        self.assertEqual(throttle_for_climb(0.5), 1000)
        self.assertEqual(throttle_for_climb(-0.5), 0)
        self.assertGreater(throttle_for_climb(0.05), 600)
        self.assertLess(throttle_for_climb(-0.05), 400)

    def test_malformed_lines_rejected(self):
        sim = Sim(self)
        with self.assertRaises(ValueError):
            sim.transport.send("ARM NOW")


if __name__ == "__main__":
    unittest.main()

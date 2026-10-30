"""follow_guard.FollowGuard on its own: what counts as not following, and what does not."""
import unittest
from types import SimpleNamespace as NS

from drone_flight.flight.follow_guard import FollowGuard


def run(guard, check, seconds, step=0.1, **kw):
    t, fault = 0.0, None
    while t <= seconds and fault is None:
        fault = check(guard, t, **kw)
        t += step
    return fault, t


class FollowGuardTests(unittest.TestCase):
    def test_too_high_is_always_judged(self):
        g = FollowGuard()
        f, t = run(g, lambda g, t: g.height(t, 600, ceiling_mm=450, floor_mm=None), 5)
        self.assertEqual(f, "OVER_HEIGHT")
        self.assertGreaterEqual(t, 1.0)

    def test_too_low_only_when_a_floor_is_given(self):
        g = FollowGuard()
        self.assertIsNone(run(g, lambda g, t: g.height(t, 100, ceiling_mm=450, floor_mm=None), 5)[0])
        self.assertEqual(run(g, lambda g, t: g.height(t, 100, ceiling_mm=450, floor_mm=150), 5)[0], "UNDER_HEIGHT")

    def test_inside_the_band_is_fine(self):
        g = FollowGuard()
        f, _ = run(g, lambda g, t: g.height(t, 420, ceiling_mm=450, floor_mm=150), 5)
        self.assertIsNone(f)

    def test_new_key_gets_time_to_take_effect(self):
        g = FollowGuard()
        f, _ = run(g, lambda g, t: g.direction(t, (300, 0), (0.0, 0.0)), 1.4)
        self.assertIsNone(f, "1.5 s to get going")
        g2 = FollowGuard()
        f, t = run(g2, lambda g, t: g.direction(t, (300, 0), (0.0, 0.0)), 5)
        self.assertEqual(f, "WRONG_DIRECTION")

    def test_going_the_asked_way_is_fine_and_sideways_is_not(self):
        g = FollowGuard()
        self.assertIsNone(run(g, lambda g, t: g.direction(t, (300, 0), (250.0, 40.0)), 6)[0])
        g = FollowGuard()
        self.assertEqual(run(g, lambda g, t: g.direction(t, (300, 0), (250.0, 400.0)), 6)[0], "WRONG_DIRECTION")

    def test_no_speed_data_never_lands(self):
        g = FollowGuard()
        self.assertIsNone(run(g, lambda g, t: g.direction(t, (300, 0), (None, None)), 6)[0])
        self.assertIsNone(run(g, lambda g, t: g.tilt(t, None, NS(nav_roll=0.0, nav_pitch=0.0)), 6)[0])

    def test_tilt_mismatch_needs_half_a_second(self):
        g = FollowGuard()
        att, cmd = NS(roll=0.0, pitch=0.157), NS(nav_roll=0.0, nav_pitch=0.0)  # 9 deg vs 0 commanded
        f, t = run(g, lambda g, t: g.tilt(t, att, cmd), 3)
        self.assertEqual(f, "TILT_MISMATCH")
        self.assertGreaterEqual(t, 0.5)
        g = FollowGuard()
        self.assertIsNone(run(g, lambda g, t: g.tilt(t, NS(roll=0.0, pitch=0.07), cmd), 3)[0], "4 deg off is normal")


if __name__ == "__main__":
    unittest.main()

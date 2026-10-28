"""always_land.lua in a Lua runtime with stand-ins for the ArduPilot scripting bindings."""
import unittest
from pathlib import Path

try:
    from lupa import LuaRuntime
except ImportError:  # pragma: no cover
    LuaRuntime = None

SCRIPT = (Path(__file__).resolve().parent.parent / "fc_scripts" / "always_land.lua").read_text()

STAND_INS = r"""
S = {now = 0, armed = true, flying = true, mode = 5, height = 0.30, laser_ok = true, roll = 0, pitch = 0,
     att_err = 1.0, quality = 90, params = {}, modes = {}, texts = {}, disarms = 0}
param = {}
function param:add_table(key, prefix, n) return true end
function param:add_param(key, index, name, default) S.params["ALAND_" .. name] = default return true end
function Parameter(name)
  return {get = function(self) return S.params[name] end}
end
rangefinder = {}
function rangefinder:has_data_orient(o) return S.laser_ok end
function rangefinder:status_orient(o) if S.laser_ok then return 4 end return 1 end
function rangefinder:distance_orient(o) return S.height end
ahrs = {}
function ahrs:get_roll_rad() return math.rad(S.roll) end
function ahrs:get_pitch_rad() return math.rad(S.pitch) end
AC_AttitudeControl = {}
function AC_AttitudeControl:get_att_error_angle_deg() return S.att_err end
optical_flow = {}
function optical_flow:quality() return S.quality end
arming = {}
function arming:is_armed() return S.armed end
function arming:disarm() S.disarms = S.disarms + 1 S.armed = false return true end
vehicle = {}
function vehicle:get_likely_flying() return S.flying end
function vehicle:get_mode() return S.mode end
function vehicle:set_mode(m) S.modes[#S.modes + 1] = m S.mode = m return true end
gcs = {}
function gcs:send_text(sev, text) S.texts[#S.texts + 1] = text end
function millis() return {tofloat = function(self) return S.now end} end
"""


@unittest.skipIf(LuaRuntime is None, "lupa not installed")
class AlwaysLandLuaTests(unittest.TestCase):
    def setUp(self):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.lua.execute(STAND_INS)
        self.S = self.lua.globals().S
        self.step, _ = self.lua.execute(SCRIPT)

    def run_for(self, seconds):
        for _ in range(int(round(seconds * 10))):
            self.S.now = self.S.now + 100
            self.step()

    def modes(self):
        return list(self.S.modes.values())

    def texts(self):
        return list(self.S.texts.values())

    def test_normal_hover_never_acts(self):
        self.run_for(10)
        self.assertEqual(self.modes(), [])
        self.assertEqual(self.S.disarms, 0)
        self.assertEqual(self.texts(), ["ALWAYS-LAND running"])

    def test_too_high_lands(self):
        self.S.height = 2.5
        self.run_for(0.4)
        self.assertEqual(self.modes(), [], "needs 0.5 s")
        self.run_for(0.3)
        self.assertEqual(self.modes(), [9])
        self.assertTrue(any("too high" in t for t in self.texts()))

    def test_laser_lost_lands(self):
        self.S.laser_ok = False
        self.run_for(1.2)
        self.assertEqual(self.modes(), [9])

    def test_big_tilt_lands(self):
        self.S.roll = 30
        self.run_for(0.7)
        self.assertEqual(self.modes(), [9])

    def test_not_following_lands(self):
        self.S.att_err = 9.0
        self.run_for(0.7)
        self.assertEqual(self.modes(), [9])
        self.assertTrue(any("not following" in t for t in self.texts()))

    def test_normal_attitude_error_is_fine(self):
        self.S.att_err = 4.5
        self.run_for(5)
        self.assertEqual(self.modes(), [])

    def test_blind_camera_lands_only_in_loiter(self):
        self.S.quality = 3
        self.S.mode = 2  # ALT_HOLD
        self.run_for(3)
        self.assertEqual(self.modes(), [])
        self.S.mode = 5
        self.run_for(1.5)
        self.assertEqual(self.modes(), [], "needs 2 s")
        self.run_for(1.0)
        self.assertEqual(self.modes(), [9])

    def test_land_that_climbs_stops_the_motors(self):
        self.S.mode = 9
        self.S.height = 0.5
        self.run_for(1)
        self.S.height = 0.3
        self.run_for(1)
        self.S.height = 0.8
        self.run_for(0.8)
        self.assertEqual(self.S.disarms, 0, "needs 1 s")
        self.run_for(0.5)
        self.assertEqual(self.S.disarms, 1)

    def test_normal_land_is_never_cut(self):
        self.S.mode = 9
        for tenth in range(40):
            self.S.height = max(0.02, 0.6 - tenth * 0.02)
            self.run_for(0.1)
        self.assertEqual(self.S.disarms, 0)

    def test_leg_contact_on_the_floor_is_not_a_fault(self):
        self.S.height = 0.03  # on the floor
        self.S.roll, self.S.att_err, self.S.quality = 30, 15.0, 3
        self.run_for(4)
        self.assertEqual(self.modes(), [])

    def test_on_the_ground_nothing_happens(self):
        self.S.armed, self.S.flying, self.S.height = False, False, 5.0
        self.run_for(3)
        self.assertEqual(self.modes(), [])

    def test_test_mode_only_reports_even_disarmed(self):
        self.S.params["ALAND_TEST"] = 1
        self.S.params["ALAND_MAX_HGT"] = 0.01
        self.S.armed, self.S.flying, self.S.height = False, False, 0.02
        self.run_for(1)
        self.assertEqual(self.modes(), [])
        self.assertTrue(any("TEST would LAND: too high" in t for t in self.texts()))

    def test_switched_off_does_nothing(self):
        self.S.params["ALAND_ENABLE"] = 0
        self.S.height = 3.0
        self.run_for(2)
        self.assertEqual(self.modes(), [])


if __name__ == "__main__":
    unittest.main()

"""Tiny ArduPilot stand-in for testing ArduPilotFlightTransport without hardware.

Models what the transport relies on: modes, arming rules (FLOWHOLD, idle
throttle), a 2 s spool-up, climb/descent from MANUAL_CONTROL throttle under
ALT_HOLD-style rules (THR_DZ=100, 0.5 m/s), LAND descent with self-disarm,
forced disarm, FLOWHOLD lean -> horizontal speed, LOITER stick -> speed with
brake-and-hold on release (LOIT_SPEED_MS 0.6, LOIT_ACC_MAX_M 1.5), motors at
idle once on the floor while descending, and the MTF-01P 2 cm floor reading.
"""
from __future__ import annotations

import math
from types import SimpleNamespace


class Msg(SimpleNamespace):
    def __init__(self, kind, src=(1, 1), **fields):
        super().__init__(**fields)
        self._kind, self._src = kind, src

    def get_type(self):
        return self._kind

    def get_srcSystem(self):
        return self._src[0]

    def get_srcComponent(self):
        return self._src[1]


class FakeMav:
    def __init__(self, fc):
        self.fc = fc

    def heartbeat_send(self, *a):
        self.fc.gcs_heartbeats += 1

    def manual_control_send(self, target, x, y, z, r, buttons):
        self.fc.sticks = (x, y, z, r)
        self.fc.log.append(("MANUAL_CONTROL", x, y, z))

    def set_mode_send(self, target, base, mode):
        self.fc.log.append(("SET_MODE", mode))
        if self.fc.accept_modes and (mode != 5 or self.fc.loiter_ok):
            self.fc.mode = mode

    def command_long_send(self, ts, tc, cmd, conf, p1, p2, *rest):
        self.fc.log.append(("COMMAND", cmd, p1, p2))
        if self.fc.drop_commands > 0:
            self.fc.drop_commands -= 1
            return
        if cmd == 400:
            self.fc.arm_request(bool(p1), p2 == 21196)


class FakeConnection:
    def __init__(self, fc):
        self.fc, self.mav = fc, FakeMav(fc)

    def recv_match(self, **kw):
        return None

    def close(self):
        pass


class FakeArduPilot:
    SPOOL_S = 2.0

    def __init__(self):
        self.mode, self.armed = 0, False
        self.sticks = (0, 0, 500, 0)
        self.height_m, self.north, self.east = 0.02, 0.0, 0.0
        self.vn = self.ve = 0.0
        self.vz, self.armed_at, self.on_ground_since = 0.0, None, None
        self.accept_modes, self.can_lift, self.flow_quality = True, True, 150
        self.gcs_heartbeats, self.log = 0, []
        self.forced_disarms = 0
        self.range_alive = True
        self.range_stuck_cm = None  # e.g. 2: laser blinded by sunlight
        self.takeoff_started = False  # ArduPilot clears land_complete at takeoff start, before lift-off
        self.ekf_z_bias = 0.0  # baro drift before arming; ArduPilot resets the height datum on arming
        self.drop_commands = 0  # COMMAND_LONGs lost by the radio
        self.land_climbs = False  # LAND climbs on a wrong height estimate
        self.loiter_ok = True     # False: the FC refuses LOITER (no position estimate)
        self.range_delay_s = 0.0  # laser news arrives this late (radio + filtering lag)
        self.range_gap_until = 0.0  # no laser news over the radio until this time (crowded link)
        # bad EKF speed estimate: LOITER leans back at its limit chasing it
        self.runaway_pitch_deg = 0.0
        self.ekf_speed_bias = 0.0   # m/s added to the reported north speed (wrong EKF estimate)
        self.range_wobble_cm = 0    # laser swings +-this every 0.3 s
        # follow-or-land faults
        self.height_drift_m_s = 0.0   # sinks (<0) or rises (>0) on its own while flying
        self.stuck_horizontal = False  # LOITER sticks have no effect (cannot move where the keys say)
        self.external_tilt_deg = 0.0   # tilted by something else (tether pull, bump): not commanded
        self.yaw_deg = 0.0             # heading; the yaw stick turns it at PILOT_Y_RATE (202.5 deg/s full stick)
        self.yaw_drift_deg_s = 0.0     # heading creeps on its own in the air (gyro drift, no compass)
        self.flow_alive = True         # False: no OPTICAL_FLOW messages at all
        self._height_hist = []
        self._now = 0.0

    def connection(self, *a, **kw):
        return FakeConnection(self)

    def arm_request(self, arm, force):
        if not arm:
            if self.armed:
                self.forced_disarms += int(force)
            self.armed = False
            return
        if self.mode in (2, 5, 22) and self.sticks[2] == 0 and self.height_m < 0.1:
            self.armed, self.armed_at = True, None
            self.ekf_z_bias = 0.0

    def step(self, dt, now):
        if self.armed and self.armed_at is None:
            self.armed_at = now
        z = self.sticks[2]
        spooled = self.armed and now - self.armed_at >= self.SPOOL_S
        if not self.armed:
            self.takeoff_started = False
        elif spooled and self.mode in (2, 5, 22) and z > 600:
            self.takeoff_started = True
        descending = self.mode == 9 or z < 400
        if not self.armed:
            rate = -0.5 if self.height_m > 0.02 else 0.0
        elif self.land_climbs and descending:
            rate = 0.5
        elif self.mode == 9:
            rate = -0.5
        elif now - self.armed_at < self.SPOOL_S or not self.can_lift:
            rate = 0.0
        elif z > 600:
            rate = (z - 600) / 400 * 0.5
        elif z < 400:
            rate = -(400 - z) / 400 * 0.5
        else:
            rate = 0.0
        if self.armed and self.height_m > 0.03 and self.mode != 9:
            # ArduPilot shapes the climb rate with PILOT_ACC_Z (0.5 m/s/s): it speeds up and slows down
            step = 0.5 * dt
            rate = self.vz + max(-step, min(step, rate - self.vz))
        self.vz = rate
        if self.armed and self.height_m > 0.05:
            rate += self.height_drift_m_s
        self.height_m = max(0.02, self.height_m + rate * dt)
        airborne = self.armed and self.height_m > 0.05
        if airborne and self.mode in (2, 5):
            self.yaw_deg += (self.sticks[3] / 1000 * 202.5 + self.yaw_drift_deg_s) * dt
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        if airborne and self.mode == 5:
            # LOITER: stick = desired speed in the drone's own frame (full stick 0.6 m/s); centred = brake, hold
            fwd, right = (0.0, 0.0) if self.stuck_horizontal else (self.sticks[0] / 1000 * 0.6, self.sticks[1] / 1000 * 0.6)
            for axis, want in (("vn", fwd * c - right * s), ("ve", fwd * s + right * c)):
                have = getattr(self, axis)
                step = 1.5 * dt
                setattr(self, axis, have + max(-step, min(step, want - have)))
            self.north += self.vn * dt
            self.east += self.ve * dt
        elif airborne:
            # lean accelerates; FLOWHOLD brakes to a stop when the sticks are centred
            acc_f, acc_r = self.sticks[0] / 1000 * 10.0, self.sticks[1] / 1000 * 10.0
            acc_n, acc_e = acc_f * c - acc_r * s, acc_f * s + acc_r * c  # the lean is in the drone's own frame
            drag = 0.8
            brake = 2.0 if self.mode == 22 else 0.0  # FLOWHOLD brakes on centred sticks, ALT_HOLD does not
            centred = self.sticks[0] == 0 and self.sticks[1] == 0
            self.vn += (acc_n - drag * self.vn - (brake * self.vn if centred else 0)) * dt
            self.ve += (acc_e - drag * self.ve - (brake * self.ve if centred else 0)) * dt
            self.north += self.vn * dt
            self.east += self.ve * dt
        else:
            self.vn = self.ve = 0.0
        self._now = now
        self._height_hist = [h for h in self._height_hist if h[0] >= now - self.range_delay_s - 0.2]
        self._height_hist.append((now, self.height_m))
        if self.armed and self.mode == 9 and self.height_m <= 0.02:
            self.on_ground_since = self.on_ground_since or now
            if now - self.on_ground_since > 1.0:
                self.armed = False
        else:
            self.on_ground_since = None

    def motor_pwm(self):
        if not self.armed:
            return 1000
        if self.height_m <= 0.02 and (self.mode == 9 or self.sticks[2] < 400):
            return 1100  # on the floor and told to descend: ArduPilot spools down to idle
        return 1500

    def messages(self):
        pwm = self.motor_pwm()
        out = [Msg("HEARTBEAT", base_mode=128 if self.armed else 0, custom_mode=self.mode),
               Msg("ATTITUDE", roll=math.radians(self.external_tilt_deg),
                   pitch=math.radians(self.runaway_pitch_deg) if self.mode == 5 else 0.0,
                   yaw=math.radians(self.yaw_deg)),
               # what the FC itself commands: a runaway lean is commanded (LOITER chasing), a pull is not
               Msg("NAV_CONTROLLER_OUTPUT", nav_roll=0.0, nav_pitch=self.runaway_pitch_deg if self.mode == 5 else 0.0),
               Msg("LOCAL_POSITION_NED", x=self.north, y=self.east, z=-self.height_m + self.ekf_z_bias,
                   vx=self.vn + self.ekf_speed_bias, vy=self.ve, vz=-self.vz),
               Msg("SERVO_OUTPUT_RAW", servo1_raw=pwm, servo2_raw=pwm, servo3_raw=pwm, servo4_raw=pwm),
               Msg("SYSTEM_TIME", time_boot_ms=123000)]
        if self.flow_alive:
            out.append(Msg("OPTICAL_FLOW", quality=self.flow_quality))
        if self.range_alive and self._now >= self.range_gap_until:
            seen = self.height_m
            for t, h in self._height_hist:
                if t <= self._now - self.range_delay_s:
                    seen = h
            cm = self.range_stuck_cm if self.range_stuck_cm is not None else int(round(seen * 100))
            if self.range_wobble_cm and self.armed and self.height_m > 0.05:
                cm += self.range_wobble_cm if int(self._now / 0.3) % 2 == 0 else -self.range_wobble_cm
            out.append(Msg("DISTANCE_SENSOR", current_distance=cm))
        in_air = self.armed and (self.takeoff_started or self.height_m > 0.1)
        out.append(Msg("EXTENDED_SYS_STATE", landed_state=2 if in_air else 1))
        return out

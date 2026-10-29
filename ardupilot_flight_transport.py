"""Flight transport for the ArduPilot drone: takes the service's command lines (STANDBY, CONFIG, TAKEOFF,
FLY, MOVE, HOLD, LAND, LOCK, HEAD, PING) and flies them with MANUAL_CONTROL over the SiK radio.

- Take-off in LOITER (ALT_HOLD if refused). A held key is a speed; releasing it brakes and holds.
- Height changes are planned ahead (height_planner.py) because the laser arrives late over the radio.
- follow_guard.py and runaway_guard.py land the drone when it stops following its commands.
- LAND is a gentle laser landing (0.15 m/s, then 0.1 m/s below 10 cm); ArduPilot's LAND is the backup.
"""
from __future__ import annotations

import math
import re
import threading
import time
from typing import Callable, Optional

from ardupilot_link import MODE_ALT_HOLD, MODE_LAND, MODE_LOITER, ArduPilotLink
from flight_sticks import MAX_SPEED_MM_S, lean_for_speed, plan_for, throttle_for_climb, yaw_stick  # noqa: F401
from follow_guard import HEIGHT_BAND_MM, FollowGuard
from head_follow import HeadFollow, HeadSteering
from height_planner import LASER_STEADY_S, HeightPlanner
from runaway_guard import RunawayGuard, speed_mm_s, tilt_deg
from assisted_flight_protocol import (
    FLAG_ARMED, FLAG_CALIBRATED, FLAG_FLOW_OK, FLAG_LASER_OK, FLAG_LINK_OK, FLAG_MOVE_DONE,
    FLAG_POS_HOLD, ConfigTelemetry, FlightTelemetry,
)

_ALLOWED_LINE = re.compile(
    r"^(LOCK|STANDBY|HOLD|LAND|PING|STATUS|TAKEOFF \d{3,4}|FLY \d{3,4} -?\d{1,3} -?\d{1,3}( -?\d{1,2})?"
    r"|MOVE -?\d{1,4} -?\d{1,4}|CONFIG \d{3,5} [0-7]|HEAD [01])$"
)
FIRMWARE_VERSION = 1
TICK_SECONDS = 0.1
RANGE_FRESH_SECONDS = 0.8          # DISTANCE_SENSOR at 5 Hz
FLOW_MIN_QUALITY = 50          # floor camera quality 0-255
FOLLOW_TILT_ABOVE_MM = 100      # tilt check starts at this height
FLOW_BLIND_S = 1.5             # blind this long in the air: level landing
RANGE_VALID_MM = 200           # laser readings below this are not trusted in the air
LIFT_DETECT_MM = 40            # this far above the floor = lifted off
ARM_TIMEOUT_S = 4.0
SPINUP_LOITER_S = 2.0          # then try ALT_HOLD
LIFT_TIMEOUT_S = 9.0           # covers the 5 s spool-up
# Laser vs FC altitude cross-check (only matters if the FC falls back to the barometer)
EKF_AIRBORNE_M = 0.35
EKF_MAYBE_AIRBORNE_M = 0.15    # at lift timeout: LAND rather than cut above this
LASER_STUCK_MM = 100
EKF_OVERSHOOT_M = 0.6          # backup ceiling
CLIMB_TIMEOUT_S = 25.0         # 2 m takes ~16 s
AIRBORNE_LIMIT_S = 180.0
OVER_HEIGHT_MARGIN_MM = 350
ABSOLUTE_CEILING_MM = 2300
LAND_GROUND_MM = 60
LAND_CUT_AFTER_S = 1.0         # on the floor this long in LAND: stop motors
# on the floor = laser at the ground and motors at idle
LAND_IDLE_PWM = 1250
MAX_CLIMB_M_S = 0.2            # slow: the FC keeps the last stick during a dropout
LIFT_CLIMB_M_S = 0.05          # just above the deadzone
LIFT_BASELINE_S = 1.0          # FC resets its height datum on arming
CUT_REPEAT_S = 5.0             # re-send the motor cut this long, and until a fresh heartbeat says disarmed
CUT_REPEAT_EVERY_S = 0.2
# LAND must never climb: rising this much above its lowest point cuts the motors
LAND_CLIMB_CUT_MM = 400
LAND_CLIMB_CUT_S = 0.6
CLIMB_DONE_MM = 100            # take-off ends this close to the target
LASER_LOST_S = 2.0
NOT_TRACKING_MM = 300          # planned climb minus seen climb: laser stuck
MOVE_SPEED_MM_S = 200
LOITER_SPEED_MM_S = 600        # LOIT_SPEED_MS: full stick
LOITER_RETRY_S = 0.5
LOITER_ENTRY_TIMEOUT_S = 2.0
GENTLE_DESCENT_M_S = 0.15
FINAL_ENTRY_MM = 100
FINAL_DESCENT_M_S = 0.10
TOUCHDOWN_MM = 50
TOUCHDOWN_S = 0.3
GENTLE_LAND_TIMEOUT_S = 20.0   # then hand over to the FC's LAND
FINAL_TIMEOUT_S = 10.0
LANDING_PHASES = ("LAND", "FINAL_DESCENT", "EMERGENCY_DESCENT")
GROUND_PHASES = ("IDLE", "LANDED", "FAULT")  # FAULT: motors cut


def _to_body(north: float, east: float, yaw: float) -> tuple[float, float]:
    """North/east -> (forward, right) for a drone pointing at yaw (radians)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * north + s * east, -s * north + c * east


def _to_ned(forward: float, right: float, yaw: float) -> tuple[float, float]:
    """(forward, right) for a drone pointing at yaw -> north/east."""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * forward - s * right, s * forward + c * right


class ArduPilotFlightTransport(HeightPlanner, HeadSteering):
    """send(), telemetry(), status(), close() - same API as AssistedSerialTransport."""

    def __init__(
        self,
        device: str,
        *,
        link: Optional[ArduPilotLink] = None,
        clock: Callable[[], float] = time.monotonic,
        start_worker: bool = True,
        **link_kwargs,
    ) -> None:
        self.port = device
        self._clock = clock
        self.link = link if link is not None else ArduPilotLink(device, clock=clock, **link_kwargs)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._phase = "IDLE"
        self._fault = "NONE"
        self._standby = False
        self._phase_since = clock()
        self._airborne_since = 0.0
        self._target_mm = 1000
        self._velocity = (0, 0)
        self._move: Optional[dict] = None
        self._move_done = False
        self._scale_um, self._axis_map = 0, 0
        self._sequence = 0
        self._last_reply = ""
        self._last_error = ""
        self._last_stream_request = -1e9
        self._last_gcs_heartbeat = -1e9
        self._ground_cut_since = 0.0
        self._last_fc_seen = clock()
        self._ground_z: Optional[float] = None
        self._cut_since: Optional[float] = None
        self._land_low_mm: Optional[int] = None
        self._land_rise_since = 0.0
        self._last_cut_sent = -1e9
        self._ground_mm = 20
        self._loiter_asked_at: Optional[float] = None
        self._loiter_last_ask = -1e9
        self._loiter_refused = False
        self._level_only = False      # RUNAWAY: ALT_HOLD only from now on
        self._runaway = RunawayGuard()
        self._blind_since: Optional[float] = None
        self._follow = FollowGuard()
        self._cmd_body: tuple[float, float] = (0.0, 0.0)  # mm/s
        self._yaw_rate = 0  # deg/s from the arrow keys
        self.head = None    # HeadYawReader, set by the website
        self._head_on, self._head_follow, self._head_recentre = False, HeadFollow(), False
        self._head_view_zero = None
        self._touchdown_since = 0.0
        self._plan_end = 0.0
        self._plan_stick = 500
        self._plan_target: Optional[int] = None
        self._plans: list[tuple[float, float, float]] = []  # (start, mm/s signed, mm signed)
        self._settled_since = 0.0
        self._laser_hist: list[tuple[float, int]] = []
        self._last_laser_at = clock()
        self._last_laser_mm: Optional[int] = None
        self._last_laser_msg = None
        self._worker: Optional[threading.Thread] = None
        if start_worker:
            self._worker = threading.Thread(target=self._run, name="ardupilot-flight", daemon=True)
            self._worker.start()

    # ================================================================= commands
    def send(self, line: str) -> None:
        if not _ALLOWED_LINE.match(line):
            raise ValueError(f"refusing to send malformed transmitter line: {line!r}")
        if self.link.heartbeat() is None and line not in {"PING", "STATUS", "LOCK", "HEAD 0", "HEAD 1"}:
            # HEAD only sets head steering on this laptop; the page turns it on before the drone is heard
            raise RuntimeError("No heartbeat from the flight controller over the radio")
        word, *args = line.split()
        with self._lock:
            self._last_reply = f"OK {word}"
            handler = getattr(self, f"_cmd_{word.lower()}")
            handler(*[int(a) for a in args])

    def _cmd_ping(self) -> None:
        pass

    _cmd_status = _cmd_ping

    def _cmd_config(self, scale_um: int, axis_map: int) -> None:
        self._scale_um, self._axis_map = scale_um, axis_map

    def _cmd_standby(self) -> None:
        if self._phase in {"IDLE", "LANDED", "FAULT"} and not self.link.armed():
            self._standby = True
            self._cut_since = None
            self._set_phase("IDLE")

    def _cmd_lock(self) -> None:
        was_active = self._phase not in {"IDLE", "LANDED", "FAULT"}
        self._standby = False
        if self.link.armed() or was_active or self._cut_since is not None:
            self._cut_since = self._clock()
            self._send_cut(self._cut_since)
        self._velocity, self._move = (0, 0), None
        if was_active:
            self._fail("EXTERNAL_LOCK", land=False)
        else:  # a cut still waiting for "disarmed" keeps repeating (FAULT runs it; IDLE does not)
            self._set_phase("FAULT" if self._cut_since is not None else "IDLE")

    def _cmd_takeoff(self, target_mm: int) -> None:
        fault = self._takeoff_precondition(target_mm)
        if fault:
            self._fail(fault, land=False)
            return
        self._target_mm = target_mm
        self._velocity, self._move, self._move_done = (0, 0), None, False
        self._airborne_since = 0.0
        self._ground_z = self._ekf_z()
        self._ground_mm = self._height_mm() or 20
        self._loiter_asked_at, self._loiter_refused = None, False
        self._level_only, self._runaway, self._blind_since = False, RunawayGuard(), None
        self._head_follow.clear()  # re-centred every flight
        self._head_recentre = self._head_on
        self._follow, self._cmd_body = FollowGuard(), (0.0, 0.0)
        self._plan_end, self._plan_target, self._plans = 0.0, None, []
        self.link.manual_control(0, 0, 0)  # throttle idle to arm
        self.link.set_mode(MODE_LOITER)
        self._set_phase("SPINUP")

    def _takeoff_precondition(self, target_mm: int) -> str:
        if not 300 <= target_mm <= 2000:
            return "BAD_PARAMS"
        if not self._standby or self._phase not in {"IDLE", "LANDED", "FAULT"} or self.link.armed():
            return "PRECONDITION"
        height = self._height_mm()
        if height is None or height > 150:
            return "PRECONDITION"  # must be on the floor with the laser working
        return ""  # a blind floor camera does not block take-off

    def _cmd_fly(self, target_mm: int, forward: int, right: int, yaw_deg_s: int = 0) -> None:
        self._target_mm = max(300, min(2000, target_mm))
        if self._phase in {"FLY", "MOVE"}:
            if (forward, right, yaw_deg_s) != (0, 0, 0):
                self._move = None
                if self._phase == "MOVE":
                    self._set_phase("FLY")
            self._velocity, self._yaw_rate = (forward, right), yaw_deg_s

    def _cmd_move(self, forward_mm: int, right_mm: int) -> None:
        if self._phase not in {"FLY", "MOVE"}:
            return
        start, yaw = self._position_ned(), self._yaw()
        if start is None or yaw is None:
            self._fail("FLOW_LOST", land=False, hold=True)
            return
        distance = math.hypot(forward_mm, right_mm)
        # the goal is fixed on the floor at the start heading: a heading that drifts later does not bend it
        self._move = {"goal": _to_ned(forward_mm, right_mm, yaw), "start": start, "yaw": yaw,
                      "deadline": self._clock() + distance / 100.0 + 8.0}  # assume >=0.1 m/s
        self._move_done = False
        self._velocity = (0, 0)
        self._set_phase("MOVE")

    def _cmd_hold(self) -> None:
        self._velocity, self._move = (0, 0), None
        if self._phase == "MOVE":
            self._set_phase("FLY")

    def _cmd_land(self) -> None:
        """Gentle laser landing when possible, else the FC's LAND."""
        self._velocity, self._move = (0, 0), None
        if self._phase in LANDING_PHASES:
            return
        height = self._height_mm()
        gentle_ok = (self._phase in {"CLIMB", "FLY", "MOVE"} and self.link.heartbeat() is not None
                     and height is not None and height >= RANGE_VALID_MM - 50)
        if gentle_ok:
            self.link.manual_control(0, 0, throttle_for_climb(-GENTLE_DESCENT_M_S))
            self._start_landing("LAND")
        elif self.link.armed() or self._phase not in {"IDLE", "LANDED", "FAULT"}:
            self._emergency_descent()

    def _emergency_descent(self) -> None:
        """ArduPilot's LAND, used when the laser or radio cannot be trusted."""
        self._velocity, self._move = (0, 0), None
        # centre the sticks: LAND_REPOSITION would follow a held lean
        self.link.manual_control(0, 0, 500)
        self.link.set_mode(MODE_LAND)
        self._start_landing("EMERGENCY_DESCENT")

    def _level_landing(self, fault: str) -> None:
        """Bad position estimate: ALT_HOLD with centred sticks, then the gentle laser landing."""
        self._fault, self._level_only, self._loiter_refused = fault, True, True
        self._velocity, self._move = (0, 0), None
        self.link.set_mode(MODE_ALT_HOLD)
        self.link.manual_control(0, 0, throttle_for_climb(-GENTLE_DESCENT_M_S))
        self._start_landing("LAND")

    def _start_landing(self, phase: str) -> None:
        if self._phase not in LANDING_PHASES:
            self._land_low_mm, self._land_rise_since = None, 0.0
        self._touchdown_since = 0.0
        self._set_phase(phase)

    # ================================================================= helpers
    def _set_phase(self, phase: str) -> None:
        if phase != self._phase:
            self._phase, self._phase_since = phase, self._clock()

    def _fail(self, fault: str, *, land: bool, hold: bool = False) -> None:
        self._fault = fault
        if hold:
            self._velocity, self._move = (0, 0), None
            self._set_phase("FLY")
        elif land and fault in {"LASER_LOST", "LINK_LOST"}:
            self._emergency_descent()
        elif land:
            self._cmd_land()
        else:
            self._set_phase("FAULT")

    def _height_mm(self) -> Optional[int]:
        msg = self.link.latest("DISTANCE_SENSOR", RANGE_FRESH_SECONDS) or \
            self.link.latest("DISTANCE_SENSOR_RAW", RANGE_FRESH_SECONDS)
        if msg is None:
            return None
        return int(msg.current_distance) * 10  # cm -> mm

    def _velocity_body(self) -> tuple[Optional[float], Optional[float]]:
        """EKF horizontal velocity in the body frame, mm/s (forward, right)."""
        pos = self.link.latest("LOCAL_POSITION_NED", 0.8)
        att = self.link.latest("ATTITUDE", 0.8)
        if pos is None or att is None:
            return None, None
        return _to_body(pos.vx * 1000.0, pos.vy * 1000.0, att.yaw)

    def _flow(self):
        return self.link.latest("OPTICAL_FLOW", 1.0)

    def _camera_blind(self, now: float) -> bool:
        """Floor camera quality below FLOW_MIN_QUALITY, or no floor camera news, for FLOW_BLIND_S."""
        flow = self.link.latest("OPTICAL_FLOW", 2.0)
        if flow is not None and int(flow.quality) >= FLOW_MIN_QUALITY:
            self._blind_since = None
            return False
        if self._blind_since is None:
            self._blind_since = now
        return now - self._blind_since >= FLOW_BLIND_S

    def _position_ned(self) -> Optional[tuple[float, float]]:
        """EKF position, mm (north, east)."""
        pos = self.link.latest("LOCAL_POSITION_NED", 0.8)
        return None if pos is None else (pos.x * 1000.0, pos.y * 1000.0)

    def _yaw(self) -> Optional[float]:
        att = self.link.latest("ATTITUDE", 0.8)
        return None if att is None else float(att.yaw)

    def _position_body(self) -> Optional[tuple[float, float]]:
        """EKF position in the body frame, mm."""
        pos, yaw = self._position_ned(), self._yaw()
        return None if pos is None or yaw is None else _to_body(*pos, yaw)

    # ================================================================= worker
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # keep the safety loop alive
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(TICK_SECONDS)

    def tick(self) -> None:
        now = self._clock()
        if now - self._last_stream_request >= 5.0 and self.link.heartbeat() is not None:
            self.link.request_streams()
            self._last_stream_request = now
        with self._lock:
            step = getattr(self, f"_step_{self._phase.lower()}", None)
            if step is not None:
                step(now)
        # only after a step that worked: a loop that keeps failing goes quiet and the FC's GCS failsafe lands
        if now - self._last_gcs_heartbeat >= 1.0:
            self.link.send_gcs_heartbeat()
            self._last_gcs_heartbeat = now

    def _step_spinup(self, now: float) -> None:
        """Arm in LOITER, or ALT_HOLD if refused."""
        elapsed = now - self._phase_since
        if self.link.armed():
            self._set_phase("LIFT")
            return
        if elapsed > SPINUP_LOITER_S + ARM_TIMEOUT_S:
            self._fail("PRECONDITION", land=False)
            return
        want = MODE_LOITER if elapsed < SPINUP_LOITER_S else MODE_ALT_HOLD
        self.link.manual_control(0, 0, 0)
        if self.link.mode() != want:
            self.link.set_mode(want)
            return
        self.link.arm()

    def _step_lift(self, now: float) -> None:
        if not self.link.armed():
            self._fail("EXTERNAL_LOCK", land=False)
            return
        ground_z = self._ekf_z()
        if now - self._phase_since < LIFT_BASELINE_S and ground_z is not None:
            self._ground_z = ground_z
        height = self._height_mm()
        ekf_alt = self._ekf_alt_m()
        laser_on_floor = height is None or height <= LASER_STUCK_MM
        if height is not None and height >= self._ground_mm + LIFT_DETECT_MM:
            self._airborne_since = now
            self._plans = [(now, 0.0, float(LIFT_DETECT_MM))]
            self._plan_end = now
            self.link.manual_control(0, 0, 500)  # planner takes over
            self._set_phase("CLIMB")
        elif laser_on_floor and ekf_alt is not None and ekf_alt >= EKF_AIRBORNE_M:
            self._fail("LASER_LOST", land=True)
            return
        elif now - self._phase_since > LIFT_TIMEOUT_S:
            # "in air" is set from take-off start: only maybe airborne
            if ekf_alt is not None and ekf_alt >= EKF_MAYBE_AIRBORNE_M:
                self._fail("LASER_LOST", land=True)  # never cut while possibly airborne
            elif self._fc_in_air():
                self._fail("NO_LIFT", land=True)
            else:
                self.link.disarm(force=True)  # still on the floor
                self._fail("NO_LIFT", land=False)
            return
        self.link.manual_control(0, 0, throttle_for_climb(LIFT_CLIMB_M_S))

    def _step_landed(self, now: float) -> None:
        self._step_fault(now)  # re-send the cut until confirmed

    def _step_fault(self, now: float) -> None:
        if self._cut_since is None:
            return
        # a stale heartbeat proves nothing (the radio may be moving to another port): keep cutting until heard
        if now - self._cut_since > CUT_REPEAT_S and self.link.heartbeat() is not None and not self.link.armed():
            self._cut_since = None
        elif now - self._last_cut_sent >= CUT_REPEAT_EVERY_S:
            self._send_cut(now)  # one disarm can be lost in a dropout

    def _send_cut(self, now: float) -> None:
        self.link.manual_control(0, 0, 0)
        self.link.disarm(force=True)
        self._last_cut_sent = now

    def _ekf_z(self) -> Optional[float]:
        pos = self.link.latest("LOCAL_POSITION_NED", 1.0)
        return None if pos is None else float(pos.z)

    def _ekf_alt_m(self) -> Optional[float]:
        z = self._ekf_z()
        if z is None or self._ground_z is None:
            return None
        return self._ground_z - z  # NED

    def _fc_in_air(self) -> bool:
        state = self.link.latest("EXTENDED_SYS_STATE", 2.5)
        return state is not None and int(state.landed_state) == 2

    def _step_climb(self, now: float) -> None:
        if not self._airborne_checks(now):
            return
        self._keep_loiter(now)
        height = self._last_laser_mm
        stick = self._height_stick(now, height)
        near = self._recent_height_mm(now)
        if near is not None and abs(self._target_mm - near) <= CLIMB_DONE_MM:
            self._set_phase("FLY")
        elif now - self._phase_since > CLIMB_TIMEOUT_S:
            self._fail("CLIMB_TIMEOUT", land=True)
            return
        self.link.manual_control(0, 0, stick)

    def _step_fly(self, now: float) -> None:
        if not self._airborne_checks(now):
            return
        self._keep_loiter(now)
        pitch, roll = self._steer(*self._velocity)
        self.link.manual_control(pitch, roll, self._height_stick(now, self._last_laser_mm), yaw_stick(self._turn_rate()))

    def _keep_loiter(self, now: float) -> None:
        """Switch to LOITER once off the floor to hold the spot."""
        if self._loiter_refused or self.link.mode() == MODE_LOITER:
            return
        self._loiter_asked_at = self._loiter_asked_at or now
        if now - self._loiter_asked_at > LOITER_ENTRY_TIMEOUT_S:
            self._loiter_refused = True  # no position estimate
        elif now - self._loiter_last_ask >= LOITER_RETRY_S:
            self.link.set_mode(MODE_LOITER)
            self._loiter_last_ask = now

    def _steer(self, forward_mm_s: float, right_mm_s: float) -> tuple[int, int]:
        """(pitch, roll) sticks for a body speed. In LOITER the stick is the speed."""
        cap = lambda v: max(-MAX_SPEED_MM_S, min(MAX_SPEED_MM_S, v))  # noqa: E731
        self._cmd_body = (cap(forward_mm_s), cap(right_mm_s))
        if self.link.mode() == MODE_LOITER:
            def stick(v):
                v = max(-MAX_SPEED_MM_S, min(MAX_SPEED_MM_S, v))
                return int(round(v / LOITER_SPEED_MM_S * 1000))
            return stick(forward_mm_s), stick(right_mm_s)
        vf, vr = self._velocity_body()
        return lean_for_speed(forward_mm_s, vf), lean_for_speed(right_mm_s, vr)

    def _step_move(self, now: float) -> None:
        if not self._airborne_checks(now):
            return
        pos, yaw = self._position_ned(), self._yaw()
        move = self._move
        if pos is None or yaw is None or move is None:
            self._fail("FLOW_LOST", land=False, hold=True)
            return
        left_n = move["goal"][0] - (pos[0] - move["start"][0])
        left_e = move["goal"][1] - (pos[1] - move["start"][1])
        remaining = math.hypot(left_n, left_e)
        left_f, left_r = _to_body(left_n, left_e, yaw)  # steer from where it points now
        if remaining < 80:
            self._move_done, self._move = True, None
            self._set_phase("FLY")
            return
        if now > move["deadline"]:
            self._fail("MOVE_TIMEOUT", land=False, hold=True)
            return
        speed = min(MOVE_SPEED_MM_S, remaining * 1.5)
        self._keep_loiter(now)
        pitch, roll = self._steer(left_f / remaining * speed, left_r / remaining * speed)
        self.link.manual_control(pitch, roll, self._height_stick(now, self._last_laser_mm))

    def _step_land(self, now: float) -> None:
        """Descend at 0.15 m/s on the laser to 10 cm above the floor."""
        height = self._gentle_landing_checks(now)
        if height is None:
            return
        if height <= self._ground_mm + FINAL_ENTRY_MM + 50:
            self._set_phase("FINAL_DESCENT")
        elif now - self._phase_since > GENTLE_LAND_TIMEOUT_S:
            self._fail("LAND_TIMEOUT", land=False)
            self._emergency_descent()
            return
        self.link.manual_control(0, 0, throttle_for_climb(-GENTLE_DESCENT_M_S))

    def _step_final_descent(self, now: float) -> None:
        """Descend at 0.1 m/s; motors off once on the floor."""
        height = self._gentle_landing_checks(now)
        if height is None:
            return
        on_floor = height <= self._ground_mm + TOUCHDOWN_MM and self._motors_at_idle()
        if on_floor:
            self._touchdown_since = self._touchdown_since or now
            if now - self._touchdown_since >= TOUCHDOWN_S:
                self._cut_since = now
                self._send_cut(now)
                self._set_phase("LANDED")
                return
        else:
            self._touchdown_since = 0.0
        if now - self._phase_since > FINAL_TIMEOUT_S:
            self._fault = "LAND_TIMEOUT"
            self._emergency_descent()
            return
        self.link.manual_control(0, 0, throttle_for_climb(-FINAL_DESCENT_M_S))

    def _gentle_landing_checks(self, now: float) -> Optional[int]:
        """Shared landing checks. None when the step already acted."""
        if self.link.heartbeat() is None:
            self.link.manual_control(0, 0, 500)  # FC lands itself on GCS loss
            if now - self._last_fc_seen > 3.0:
                self._fault = "LINK_LOST"
                self._emergency_descent()
            return None
        self._last_fc_seen = now
        if not self.link.armed():
            self._set_phase("LANDED")
            return None
        if self.link.mode() == MODE_LAND:
            self._start_landing("EMERGENCY_DESCENT")  # FC failsafe took over
            return None
        if self._level_only and self.link.mode() != MODE_ALT_HOLD and now - self._loiter_last_ask >= LOITER_RETRY_S:
            self.link.set_mode(MODE_ALT_HOLD)  # stay level after RUNAWAY
            self._loiter_last_ask = now
        height, lost = self._laser(now)
        if lost:
            self._fault = "LASER_LOST"
            self._emergency_descent()
            return None
        if height is None:
            return None
        if self._land_is_climbing(height, now):
            self._cut_since = now
            self._send_cut(now)
            self._fail("LAND_FAILED", land=False)  # FAULT phase keeps re-sending the cut
            return None
        return height

    def _step_emergency_descent(self, now: float) -> None:
        """Watch ArduPilot's LAND: cut if it climbs, backup cut at touchdown."""
        self.link.manual_control(0, 0, 500)
        if self.link.heartbeat() is None:
            self.link.set_mode(MODE_LAND)
            return
        if not self.link.armed():
            self._set_phase("LANDED")
            return
        if self.link.mode() != MODE_LAND:
            self.link.set_mode(MODE_LAND)
        height = self._height_mm()
        if self._land_is_climbing(height, now):
            self._cut_since = now
            self._send_cut(now)
            self._fail("LAND_FAILED", land=False)  # FAULT phase keeps re-sending the cut
            return
        if height is not None and height <= LAND_GROUND_MM and self._motors_at_idle():
            self._ground_cut_since = self._ground_cut_since or now
            if now - self._ground_cut_since >= LAND_CUT_AFTER_S:
                self.link.disarm(force=True)  # FC did not disarm itself
        else:
            self._ground_cut_since = 0.0

    def _motors_at_idle(self) -> bool:
        servo = self.link.latest("SERVO_OUTPUT_RAW", 1.0)
        if servo is None:
            return False  # leave it to the FC's land detector
        pwms = [servo.servo1_raw, servo.servo2_raw, servo.servo3_raw, servo.servo4_raw]
        return sum(pwms) / len(pwms) <= LAND_IDLE_PWM

    def _land_is_climbing(self, height: Optional[int], now: float) -> bool:
        if height is None or height < RANGE_VALID_MM - 50:
            return False
        self._land_low_mm = height if self._land_low_mm is None else min(self._land_low_mm, height)
        if height - self._land_low_mm < LAND_CLIMB_CUT_MM:
            self._land_rise_since = 0.0
            return False
        self._land_rise_since = self._land_rise_since or now
        return now - self._land_rise_since >= LAND_CLIMB_CUT_S

    def _airborne_checks(self, now: float) -> bool:
        """In-air limits. False when the step already acted."""
        if self.link.heartbeat() is None:
            # FC not heard: hold, then LAND blind after 3 s
            self.link.manual_control(0, 0, 500)
            if now - self._last_fc_seen > 3.0:
                self._fault = "LINK_LOST"
                self._emergency_descent()
            return False
        self._last_fc_seen = now
        if not self.link.armed():
            self._fail("EXTERNAL_LOCK", land=False)
            return False
        if self.link.mode() == MODE_LAND:
            self._start_landing("EMERGENCY_DESCENT")  # FC is landing by itself
            self.link.manual_control(0, 0, 500)
            return False
        if self._runaway.update(now, tilt_deg(self.link.latest("ATTITUDE", 0.8)),
                                speed_mm_s(self.link.latest("LOCAL_POSITION_NED", 0.8))):
            self._level_landing("RUNAWAY")
            return False
        if self._camera_blind(now):
            self._level_landing("FLOW_LOST")
            return False
        height, lost = self._laser(now)
        if lost or height is None:
            self._fail("LASER_LOST", land=True)
            return False
        if self._expected_climb_mm(now) - (height - self._ground_mm) > NOT_TRACKING_MM:
            self._fail("LASER_LOST", land=True)  # laser stuck
            return False
        if height > min(self._target_mm + OVER_HEIGHT_MARGIN_MM, ABSOLUTE_CEILING_MM):
            self._fail("OVER_HEIGHT", land=True)
            return False
        ekf_alt = self._ekf_alt_m()
        if ekf_alt is not None and ekf_alt > self._target_mm / 1000.0 + EKF_OVERSHOOT_M:
            self._fail("OVER_HEIGHT", land=True)  # FC altitude far above the target
            return False
        if self._airborne_since and now - self._airborne_since > AIRBORNE_LIMIT_S:
            self._fail("AIRBORNE_TIMEOUT", land=True)
            return False
        fault = self._follow_fault(now, height)
        if fault in {"OVER_HEIGHT", "UNDER_HEIGHT"}:
            self._fail(fault, land=True)
            return False
        if fault:
            self._level_landing(fault)
            return False
        return True

    def _follow_fault(self, now: float, height: int) -> Optional[str]:
        """Fault name when the drone stops following its commands (follow_guard.py)."""
        # too high: always, above the higher of the old and new target. Too low: only once the target is reached.
        planned = self._target_mm if self._plan_target is None else self._plan_target
        ceiling = max(self._target_mm, planned) + HEIGHT_BAND_MM
        arrived = (self._phase in {"FLY", "MOVE"} and self._plan_target == self._target_mm
                   and now >= self._plan_end)
        floor = self._target_mm - HEIGHT_BAND_MM if arrived else None
        clear = height >= FOLLOW_TILT_ABOVE_MM  # leg contact on the floor skews the tilt
        return (self._follow.tilt(now, self.link.latest("ATTITUDE", 0.5) if clear else None,
                                  self.link.latest("NAV_CONTROLLER_OUTPUT", 0.7))
                or self._follow.direction(now, self._cmd_body, self._velocity_body())
                or self._follow.height(now, height, ceiling, floor))

    def _laser(self, now: float) -> tuple[Optional[int], bool]:
        """(height mm, lost). Short gaps reuse the last reading."""
        msg = self.link.latest("DISTANCE_SENSOR", LASER_LOST_S) or \
            self.link.latest("DISTANCE_SENSOR_RAW", LASER_LOST_S)
        if msg is not None and msg is not self._last_laser_msg:
            mm = int(msg.current_distance) * 10
            self._laser_hist = [h for h in self._laser_hist if now - h[0] <= LASER_STEADY_S] + [(now, mm)]
            self._last_laser_msg, self._last_laser_at, self._last_laser_mm = msg, now, mm
        return self._last_laser_mm, now - self._last_laser_at > LASER_LOST_S

    # ================================================================= telemetry
    def telemetry(self) -> tuple[Optional[FlightTelemetry], float, Optional[ConfigTelemetry], float]:
        hb = self.link.heartbeat()
        if hb is None:
            return None, float("inf"), None, float("inf")
        with self._lock:
            self._sequence = (self._sequence + 1) & 0xFF
            flight = self._flight_packet()
            config = ConfigTelemetry(
                self._phase, self._fault, flight.flags, self._sequence, flight.height_mm,
                self._scale_um, self._axis_map, FIRMWARE_VERSION, self._uptime_s(), 0, 0, 0,
                flight.ground_mm, 0,
            )
        return flight, 0.0, config, 0.0

    def _flight_packet(self) -> FlightTelemetry:
        height = self._height_mm()
        flow = self._flow()
        att = self.link.latest("ATTITUDE", 1.0)
        pos = self._position_body()
        quality = int(flow.quality) if flow is not None else 0
        flags = (FLAG_ARMED if self.link.armed() else 0) | FLAG_CALIBRATED
        flags |= FLAG_LASER_OK if height is not None else 0
        flags |= FLAG_FLOW_OK if quality >= FLOW_MIN_QUALITY else 0
        flags |= FLAG_LINK_OK if self._standby or self.link.armed() else 0
        flags |= FLAG_POS_HOLD if self._phase in {"FLY", "MOVE"} and quality >= FLOW_MIN_QUALITY else 0
        flags |= FLAG_MOVE_DONE if self._move_done else 0
        moved = (0, 0)
        ned = self._position_ned()
        if self._move is not None and ned is not None:
            start = self._move["start"]
            moved = _to_body(ned[0] - start[0], ned[1] - start[1], self._move["yaw"])
        return FlightTelemetry(
            self._phase, self._fault, flags, self._sequence, height or 0, self._target_mm,
            self._vertical_mm_s(), int(pos[0]) if pos else 0, int(pos[1]) if pos else 0, quality,
            math.degrees(att.roll) if att else 0.0, math.degrees(att.pitch) if att else 0.0,
            self._throttle(), 0, int(moved[0]), int(moved[1]), min(height or 0, 510),
        )

    def _vertical_mm_s(self) -> int:
        pos = self.link.latest("LOCAL_POSITION_NED", 0.8)
        return int(-pos.vz * 1000) if pos is not None else 0

    def _throttle(self) -> int:
        servo = self.link.latest("SERVO_OUTPUT_RAW", 1.0)
        if servo is None:
            return 0
        outs = [servo.servo1_raw, servo.servo2_raw, servo.servo3_raw, servo.servo4_raw]
        return int(max(0, min(1000, sum(outs) / 4 - 1000)))

    def _uptime_s(self) -> int:
        sys_time = self.link.latest("SYSTEM_TIME", 5.0)
        return int(sys_time.time_boot_ms / 1000) & 0xFFFF if sys_time is not None else 0

    def status(self) -> dict:
        with self._lock:
            return {
                "connected": self.link.heartbeat() is not None,
                "port": self.link.device,
                "last_reply": self._last_reply,
                "last_error": self._last_error or self.link.stats()["last_error"],
                "telemetry_packets": self.link.stats()["messages"],
                "autopilot": "ArduPilot via SiK",
                "fc_mode": self.link.mode(),
            }

    def flying(self) -> bool:
        """True from take-off until landed, or while the FC says the motors are armed."""
        return self._phase not in GROUND_PHASES or self.link.armed()

    def close(self, final_line: Optional[str] = "LOCK") -> None:
        if final_line:
            try:
                self.send(final_line)
            except Exception:
                pass
        self._stop.set()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=0.5)
        self.link.close()

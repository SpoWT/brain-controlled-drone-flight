"""Flight service behind the website: turns page input (arm, keys, GO, STOP, LAND, motor cut) into
command lines for the transport, runs the browser watchdog and reports telemetry."""

from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from assisted_flight_protocol import (
    ACTIVE_PHASES,
    FAULT_HELP,
    FLAG_CALIBRATED,
    FLAG_FLOW_OK,
    FLAG_LASER_OK,
    FLAG_LINK_OK,
    FLAG_MOVE_DONE,
    FLAG_POS_HOLD,
    FLYING_PHASES,
    GROUND_PHASES,
    LANDING_PHASES,
    TAKEOFF_PHASES,
)

DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "output" / "assisted_flight_calibration.json"
DEFAULT_FLIGHT_LOG_DIR = Path(__file__).resolve().parent / "outputs" / "flight_logs"
MIN_TARGET_MM, MAX_TARGET_MM = 300, 2000
MAX_SPEED_MM_S = 600
MAX_YAW_DEG_S = 60
MAX_MOVE_MM = 5000


def load_calibration(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    scale, axis = data.get("scale_um"), data.get("axis_map")
    if not isinstance(scale, int) or not 100 <= scale <= 30000:
        return None
    if not isinstance(axis, int) or not 0 <= axis <= 7:
        return None
    return data


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _number(value, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


class AssistedFlightService:
    WORKER_PERIOD_SECONDS = 0.05
    BROWSER_HOLD_SECONDS = 0.30   # page silent: hold position
    BROWSER_LAND_SECONDS = 2.00   # page silent: land
    TELEMETRY_STALE_SECONDS = 0.50
    ARM_SETTLE_SECONDS = 1.50
    EDGE_GAP_SECONDS = 0.10
    EEG_FORWARD_MM_S = 200
    ALLOWS_EEG_PREARM = True
    EEG_GO_IS_HELD_KEY = True     # EEG GO acts like holding W
    EEG_GO_FRESH_SECONDS = 0.5    # GO lasts this long after the last GO decision
    DEFAULT_TARGET_MM = 1000

    def __init__(
        self,
        transport: object,
        *,
        calibration_path: Path = DEFAULT_CALIBRATION_PATH,
        flight_log_dir: Optional[Path] = DEFAULT_FLIGHT_LOG_DIR,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        start_worker: bool = True,
        token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    ) -> None:
        self.transport = transport
        self._calibration_path = calibration_path
        self._calibration = load_calibration(calibration_path)
        self._clock = clock
        self._sleep = sleep
        self._token_factory = token_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._token = ""
        self._last_sequence = 0
        self._last_browser_at = 0.0
        self._velocity = (0, 0)
        self._yaw = 0
        self._target_mm = self.DEFAULT_TARGET_MM
        self._eeg_go = False
        self._eeg_go_since = 0.0
        self._browser_landed = False
        self._was_active = False
        self._last_command = "LOCK"
        self._service_error = ""
        self._last_fault = "NONE"
        self._log_dir = flight_log_dir
        self._log_path: Optional[Path] = None
        self._log_until = 0.0
        self._send("LOCK")
        if start_worker:
            self._thread = threading.Thread(target=self._worker, name="assisted-flight", daemon=True)
            self._thread.start()

    # ----------------------------------------------------------- telemetry
    def _telemetry(self):
        flight, flight_age, config, config_age = self.transport.telemetry()
        fresh = flight is not None and flight_age <= self.TELEMETRY_STALE_SECONDS
        return (flight if fresh else None), (config if config_age <= 2.0 else None)

    def _phase(self) -> str:
        flight, _ = self._telemetry()
        return flight.phase if flight else "NO_TELEMETRY"

    def _active(self) -> bool:
        phase = self._phase()
        return phase in ACTIVE_PHASES or (phase == "NO_TELEMETRY" and self._was_active)

    # ----------------------------------------------------------- transmitter
    def _send(self, line: str) -> bool:
        try:
            self.transport.send(line)
        except Exception as exc:
            with self._lock:
                self._service_error = f"{type(exc).__name__}: {exc}"
            return False
        with self._lock:
            if line not in {"PING", "STATUS"} and not line.startswith("FLY "):
                self._last_command = line
            self._service_error = ""
        return True

    def _fail_safe_request(self) -> None:
        with self._lock:
            self._velocity = (0, 0)
            self._yaw = 0
            self._eeg_go = False
        self._send("LAND" if self._active() else "LOCK")

    # ----------------------------------------------------------- session
    def begin_browser_session(self) -> str:
        with self._lock:
            self._token = self._token_factory()
            self._last_sequence = 0
            self._last_browser_at = self._clock()
            self._velocity = (0, 0)
            self._yaw = 0
            self._eeg_go = False
            token = self._token
        if self._active():
            self._send("LAND")
        return token

    def _token_valid(self, token: str) -> bool:
        with self._lock:
            return bool(self._token) and secrets.compare_digest(str(token or ""), self._token)

    def check_token(self, token: str) -> None:
        """Token check for read-only endpoints."""
        if not self._token_valid(token):
            raise PermissionError("Invalid or expired browser control session")

    def authorize(self, token: str) -> None:
        if not self._token_valid(token):
            self._fail_safe_request()
            raise PermissionError("Invalid or expired browser control session")

    def _accept_sequence(self, sequence) -> None:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            self._fail_safe_request()
            raise ValueError("sequence must be a positive integer")
        with self._lock:
            stale = sequence <= self._last_sequence
            if not stale:
                self._last_sequence = sequence
        if stale:
            self._fail_safe_request()
            raise ValueError("browser command sequence is stale")

    def browser_heartbeat(self, token: str, forward=0, right=0, target_mm=None, yaw=0) -> None:
        """Current key state from the page (mm/s, deg/s, mm)."""
        self.authorize(token)
        vf = int(_clamp(_number(forward, 0), -MAX_SPEED_MM_S, MAX_SPEED_MM_S))
        vr = int(_clamp(_number(right, 0), -MAX_SPEED_MM_S, MAX_SPEED_MM_S))
        turn = int(_clamp(_number(yaw, 0), -MAX_YAW_DEG_S, MAX_YAW_DEG_S))
        with self._lock:
            self._last_browser_at = self._clock()
            self._browser_landed = False
            self._velocity = (vf, vr)
            self._yaw = turn
            if target_mm is not None:
                self._target_mm = int(_clamp(_number(target_mm, self._target_mm), MIN_TARGET_MM, MAX_TARGET_MM))

    # ----------------------------------------------------------- commands
    def _require_ready(self) -> None:
        state = self.status()
        if not state["control_ready"]:
            raise ValueError(state["flight_blocker"] or "Flight controls are not ready")

    def _wait_for(self, predicate, seconds: float) -> bool:
        deadline = self._clock() + seconds
        while self._clock() < deadline:
            flight, config = self._telemetry()
            if predicate(flight, config):
                return True
            self._sleep(0.03)
        return False

    def arm_and_takeoff(self, token: str, sequence, target_mm=None) -> None:
        self.authorize(token)
        self._accept_sequence(sequence)
        try:
            self._require_ready()
        except ValueError:
            self._fail_safe_request()
            raise
        flight, _ = self._telemetry()
        if flight.phase not in GROUND_PHASES:
            raise ValueError("ARM is allowed only while the drone is on the ground")
        if not flight.flag(FLAG_LASER_OK):
            raise ValueError("The height laser is not reporting; ARM refused")
        target = int(_clamp(_number(target_mm, self._target_mm), MIN_TARGET_MM, MAX_TARGET_MM))
        with self._lock:
            self._target_mm = target
            self._velocity = (0, 0)
            self._yaw = 0
            self._eeg_go = False
            self._last_browser_at = self._clock()
        if flight.fault != "NONE":
            self._last_fault = flight.fault
        # open the link first, then confirm each step
        self._send("STANDBY")
        if not self._wait_for(lambda f, c: f is not None and f.flag(FLAG_LINK_OK), self.ARM_SETTLE_SECONDS):
            self._send("LOCK")
            raise ValueError("The drone is not receiving flight commands; motors stay off")
        self._send("STANDBY")
        self._sleep(self.EDGE_GAP_SECONDS)
        if not self._wait_for(lambda f, c: f is not None and f.phase == "IDLE", self.ARM_SETTLE_SECONDS):
            self._send("LOCK")
            raise ValueError("The drone did not acknowledge STANDBY; motors stay off")
        cal = self._calibration
        self._send(f"CONFIG {cal['scale_um']} {cal['axis_map']}")
        if not self._wait_for(
            lambda f, c: c is not None and c.scale_um == cal["scale_um"] and c.axis_map == cal["axis_map"],
            self.ARM_SETTLE_SECONDS,
        ):
            self._send("LOCK")
            raise ValueError("The drone did not confirm its calibration; motors stay off")
        self._send(f"TAKEOFF {target}")
        with self._lock:
            self._was_active = True
        self._start_flight_log(target)
        if not self._wait_for(lambda f, c: f is not None and f.phase != "IDLE", self.ARM_SETTLE_SECONDS):
            self._send("LOCK")
            raise ValueError("The drone did not start the takeoff; motors stay off")
        flight, _ = self._telemetry()
        if flight is not None and flight.phase == "FAULT":
            raise ValueError(f"Takeoff refused: {FAULT_HELP.get(flight.fault, flight.fault)}")

    def go_forward(self, token: str, sequence, distance_mm=2000) -> None:
        """Measured move of distance_mm straight ahead, then hold."""
        self.authorize(token)
        self._accept_sequence(sequence)
        distance = int(_clamp(_number(distance_mm, 2000), 100, MAX_MOVE_MM))
        if self._phase() not in FLYING_PHASES:
            raise ValueError("GO FORWARD is allowed once the drone is holding in the air")
        with self._lock:
            self._velocity = (0, 0)
            self._yaw = 0
            self._eeg_go = False
            self._last_browser_at = self._clock()
        self._send(f"MOVE {distance} 0")

    def move(self, token: str, sequence, forward_mm, right_mm) -> None:
        self.authorize(token)
        self._accept_sequence(sequence)
        f = int(_clamp(_number(forward_mm, 0), -MAX_MOVE_MM, MAX_MOVE_MM))
        r = int(_clamp(_number(right_mm, 0), -MAX_MOVE_MM, MAX_MOVE_MM))
        if (f, r) == (0, 0):
            raise ValueError("Move distance is zero")
        if self._phase() not in FLYING_PHASES:
            raise ValueError("Moves are allowed once the drone is holding in the air")
        self._send(f"MOVE {f} {r}")

    def stop_and_hover(self, token: str, sequence) -> None:
        self.authorize(token)
        self._accept_sequence(sequence)
        with self._lock:
            self._velocity = (0, 0)
            self._yaw = 0
            self._eeg_go = False
        if self._phase() not in FLYING_PHASES:
            raise ValueError("STOP & HOVER requires the drone to be flying")
        self._send("HOLD")

    def page_heard_age(self) -> Optional[float]:
        """Seconds since the flying page last checked in, or None."""
        with self._lock:
            if not self._token:
                return None
            return max(0.0, self._clock() - self._last_browser_at)

    def eeg_go_active(self) -> bool:
        """True while EEG GO is held."""
        with self._lock:
            return self._eeg_go

    def recenter_head(self, token: str) -> None:
        """Where the goggles face now becomes straight ahead (turns head steering on)."""
        self.authorize(token)
        self._send("HEAD 1")

    def set_head_follow(self, token: str, enabled) -> None:
        """Head steering on/off (HEAD 1 / HEAD 0)."""
        self.authorize(token)
        if not isinstance(enabled, bool):
            raise ValueError("Head steering must be on or off")
        self._send("HEAD 1" if enabled else "HEAD 0")

    def land(self, token: str, sequence) -> None:
        self.authorize(token)
        self._accept_sequence(sequence)
        with self._lock:
            self._velocity = (0, 0)
            self._yaw = 0
            self._eeg_go = False
        if not self._active():
            self._send("LOCK")
            raise ValueError("LAND requires the drone to be flying")
        self._send("LAND")

    def emergency_lock(self, token: str) -> None:
        self.authorize(token)
        with self._lock:
            self._velocity = (0, 0)
            self._yaw = 0
            self._eeg_go = False
        self._send("LOCK")

    # ----------------------------------------------------------- EEG hooks
    def _eeg_control_allowed_locked(self, token: str) -> bool:
        fresh = self._last_browser_at and self._clock() - self._last_browser_at <= self.BROWSER_HOLD_SECONDS
        return bool(self._token and secrets.compare_digest(str(token or ""), self._token) and fresh)

    def eeg_control_allowed(self, token: str) -> bool:
        if self._phase() not in FLYING_PHASES:
            return False
        with self._lock:
            return self._eeg_control_allowed_locked(token)

    def apply_eeg_decision(self, token: str, decision: str) -> bool:
        """EEG GO = slow forward flight, STOP = hold. Never arms or lands."""
        if decision not in {"GO", "STOP"} or self._phase() not in FLYING_PHASES:
            return False
        with self._lock:
            if not self._eeg_control_allowed_locked(token):
                return False
            self._eeg_go = decision == "GO"
            if self._eeg_go:
                self._eeg_go_since = self._clock()
        return True

    # ----------------------------------------------------------- flight log
    def _start_flight_log(self, target: int) -> None:
        if self._log_dir is None:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = self._log_dir / time.strftime("flight_%Y%m%d_%H%M%S.jsonl")
            self._log_path.write_text(json.dumps({"event": "arm", "t": time.time(), "target_mm": target,
                                                  "calibration": self._calibration}) + "\n")
        except OSError as exc:
            self._log_path = None
            with self._lock:
                self._service_error = f"flight log unavailable: {exc}"

    def _record(self, now: float, flight, sent_velocity: tuple[int, int]) -> None:
        """One log line per tick while flying and for 5 s after."""
        if self._log_path is None:
            return
        active = flight is not None and flight.phase in ACTIVE_PHASES
        if active:
            self._log_until = now + 5.0
        elif now > self._log_until:
            return
        with self._lock:
            row = {"t": time.time(), "eeg_go": self._eeg_go, "keys": self._velocity, "sent": sent_velocity,
                   "target_mm": self._target_mm, "last_command": self._last_command}
        if flight is not None:
            row.update({k: getattr(flight, k) for k in (
                "phase", "fault", "flags", "height_mm", "target_mm", "vertical_mm_s", "pos_x", "pos_y",
                "quality", "roll_deg", "pitch_deg", "throttle", "hover_throttle",
                "moved_forward_mm", "moved_right_mm", "ground_mm")})
            row["aircraft_target_mm"] = row.pop("target_mm") if "target_mm" in row else None
            row["target_mm"] = self._target_mm
        else:
            row["phase"] = "NO_TELEMETRY"
        try:
            with self._log_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        except OSError:
            pass

    # ----------------------------------------------------------- worker
    def tick(self) -> None:
        now = self._clock()
        flight, config = self._telemetry()
        phase = flight.phase if flight else "NO_TELEMETRY"
        if flight is not None:
            with self._lock:
                self._was_active = phase in ACTIVE_PHASES
                if flight.fault != "NONE":
                    self._last_fault = flight.fault
        if phase not in ACTIVE_PHASES:
            self._record(now, flight, (0, 0))
            self._send("PING")
            return
        with self._lock:
            age = now - self._last_browser_at if self._last_browser_at else float("inf")
            hold = age > self.BROWSER_HOLD_SECONDS
            land = age > self.BROWSER_LAND_SECONDS
            vf, vr = (0, 0) if hold else self._velocity
            turn = 0 if hold else self._yaw
            if self._eeg_go and now - self._eeg_go_since > self.EEG_GO_FRESH_SECONDS:
                self._eeg_go = False  # GO decisions stopped
            if self._eeg_go and not hold and (vf, vr) == (0, 0):
                vf = self.EEG_FORWARD_MM_S
            target = self._target_mm
            send_land = land and not self._browser_landed and phase not in LANDING_PHASES
            if send_land:
                self._browser_landed = True
        self._record(now, flight, (vf, vr))
        if send_land:
            self._send("LAND")
            return
        self._send(f"FLY {target} {vf} {vr} {turn}" if turn else f"FLY {target} {vf} {vr}")

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # keep the watchdog alive
                with self._lock:
                    self._service_error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.WORKER_PERIOD_SECONDS)

    # ----------------------------------------------------------- status
    def _mode(self, phase: str) -> str:
        if phase == "NO_TELEMETRY":
            return "LINK_LOST" if self._was_active else "DISARMED"
        if phase in TAKEOFF_PHASES:
            return "TAKING_OFF"
        if phase in FLYING_PHASES:
            with self._lock:
                moving = self._eeg_go or self._velocity[0] > 0
            return "FORWARD" if phase == "MOVE" or moving else "HOVERING"
        if phase in LANDING_PHASES:
            return "LANDING"
        return "DISARMED"

    def status(self) -> dict:
        transport = dict(self.transport.status())
        flight, config = self._telemetry()
        phase = flight.phase if flight else "NO_TELEMETRY"
        calibration = self._calibration
        aircraft_ok = config is not None and config.firmware_version >= 1
        blocker = ""
        if not transport.get("connected"):
            blocker = "USB transmitter is not connected"
        elif flight is None:
            blocker = "No telemetry from the drone: is it powered on and running the assisted-flight firmware?"
        elif not aircraft_ok:
            blocker = "Waiting for the drone's configuration report"
        elif calibration is None:
            blocker = "Optical-flow calibration is missing (run the hand-carry calibration)"
        with self._lock:
            velocity = self._velocity
            target = self._target_mm
            last_command = self._last_command
            service_error = self._service_error
            last_fault = self._last_fault
            eeg_go = self._eeg_go
        fault = flight.fault if flight else "NONE"
        shown_fault = fault if fault != "NONE" else last_fault
        transport.update(
            {
                "mode": self._mode(phase),
                "phase": phase,
                "fault": fault,
                "last_fault": last_fault,
                "fault_help": FAULT_HELP.get(shown_fault, ""),
                "last_command": last_command,
                "service_error": service_error,
                "control_ready": blocker == "",
                "flight_ready": blocker == "" and bool(flight and flight.flag(FLAG_CALIBRATED)),
                "owner_controls_authorized": True,
                "flight_blocker": blocker,
                "target_height_mm": target,
                "target_height_kind": "laser_measured",
                "altitude_feedback": flight is not None,
                "forward_command": "measured_optical_flow",
                "commanded_velocity_mm_s": {"forward": velocity[0], "right": velocity[1]},
                "eeg_forward": eeg_go,
                "calibration": calibration,
                "telemetry": None
                if flight is None
                else {
                    "height_mm": flight.height_mm,
                    "ground_mm": flight.ground_mm,
                    "aircraft_target_mm": flight.target_mm,
                    "vertical_mm_s": flight.vertical_mm_s,
                    "moved_forward_mm": flight.moved_forward_mm,
                    "moved_right_mm": flight.moved_right_mm,
                    "flow_quality": flight.quality,
                    "roll_deg": flight.roll_deg,
                    "pitch_deg": flight.pitch_deg,
                    "throttle": flight.throttle,
                    "hover_throttle": flight.hover_throttle,
                    "armed": flight.armed,
                    "laser_ok": flight.flag(FLAG_LASER_OK),
                    "flow_ok": flight.flag(FLAG_FLOW_OK),
                    "radio_link_ok": flight.flag(FLAG_LINK_OK),
                    "position_hold": flight.flag(FLAG_POS_HOLD),
                    "calibrated": flight.flag(FLAG_CALIBRATED),
                    "move_done": flight.flag(FLAG_MOVE_DONE),
                },
                "aircraft_config": None
                if config is None
                else {
                    "firmware_version": config.firmware_version,
                    "scale_um": config.scale_um,
                    "axis_map": config.axis_map,
                    "uptime_s": config.uptime_s,
                    "laser_errors": config.laser_errors,
                },
            }
        )
        return transport

    def launcher_health(self) -> dict:
        state = self.status()
        return {
            "service_online": True,
            "mode": state["mode"],
            "phase": state["phase"],
            "last_command": state["last_command"],
            "control_ready": state["control_ready"],
            "flight_ready": state["flight_ready"],
            "flight_blocker": state["flight_blocker"],
            "checks": {
                "arduino_connected": bool(state.get("connected")),
                "aircraft_telemetry": state["telemetry"] is not None,
                "altitude_feedback": state["altitude_feedback"],
                "calibration_loaded": state["calibration"] is not None,
            },
            "target_height_kind": "laser_measured",
        }

    def close(self) -> None:
        active = self._active()
        self._send("LAND" if active else "LOCK")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        try:
            self.transport.close(final_line="LAND" if active else "LOCK")
        except Exception:
            pass

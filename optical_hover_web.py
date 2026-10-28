#!/usr/bin/env python3
"""Flask app for the flying page: routes, browser control session and EEG control."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request

from assisted_flight_service import AssistedFlightService
from assisted_flight_transport import AssistedSerialTransport
from optical_eeg import LiveEEGService
from optical_eeg_control import EEGFlightControl


EXPECTED_FIRMWARE_MARKER = "OPTICAL_HOVER_FLIGHT_V1"
EXPECTED_AIRCRAFT_FIRMWARE_SHA256 = (
    "80135025774121e72107394b4ec426a88d33d62a2a939c75aeeb7d2348bc0938"
)
EXPECTED_AIRCRAFT_VECTOR = "c00d002001010008a1390008352b0008"
DEFAULT_AIRCRAFT_VERIFICATION_RECORD = (
    Path(__file__).resolve().parent
    / "hardware_backups"
    / "aircraft_2026-08-01_pre_optical"
    / "optical_hold_bench_verification.json"
)
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def load_aircraft_verification(record_path: Path) -> bool:
    """Accept only the exact aircraft image and bench evidence recorded here."""
    try:
        record = json.loads(Path(record_path).read_text(encoding="utf-8"))
        firmware = record["firmware"]
        sensors = record["sensors"]
        runtime = record["runtime"]
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False

    return bool(
        record.get("schema_version") == 1
        and record.get("result") == "battery_powered_propeller_off_verified"
        and firmware.get("sha256") == EXPECTED_AIRCRAFT_FIRMWARE_SHA256
        and firmware.get("post_flash_vector") == EXPECTED_AIRCRAFT_VECTOR
        and firmware.get("programmed_bytes_checked", 0) >= 28_000
        and firmware.get("mismatches") == 0
        and sensors.get("mpu6050_verified") is True
        and sensors.get("lc08_verified") is True
        and runtime.get("boot_loop_verified") is True
        and runtime.get("battery_powered_propeller_off_sequence_verified") is True
        and runtime.get("browser_loss_failsafe_verified") is True
        and runtime.get("physical_motor_lock_observed") is True
        and runtime.get("radio_channels")
        == [1500, 1500, 1000, 1000, 1000, 1000, 1000, 1000]
        and runtime.get("aircraft_unlocked") is False
        and runtime.get("motor_outputs") == [0, 0, 0, 0]
    )


def load_owner_control_authorization(record_path: Optional[Path]) -> bool:
    """Allow the original manual controls when the owner authorization record matches."""
    if record_path is None:
        return False
    try:
        record = json.loads(Path(record_path).read_text())
        evidence_bytes = Path(record['motor_test_evidence']).read_bytes()
        evidence = json.loads(evidence_bytes)
        observed = evidence['human_confirmation']
        return bool(
            record.get('schema_version') == 1
            and record.get('result') == 'owner_authorized_original_manual_controls'
            and record.get('owner_authorized') is True
            and set(record.get('allowed_commands', [])) == SerialFlightTransport.ALLOWED_COMMANDS
            and record.get('evidence_sha256') == hashlib.sha256(evidence_bytes).hexdigest()
            and evidence.get('motor_spin_test_passed') is True
            and evidence.get('transmitter_hard_lock_verified') is True
            and observed.get('all_four_motors_spun') is True
            and observed.get('all_motors_stopped_afterward') is True
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


class SerialFlightTransport:
    BAUD_RATE = 115200
    HANDSHAKE_SECONDS = 3.0
    ALLOWED_COMMANDS = {"TAKEOFF", "HOVER", "FORWARD", "LAND", "LOCK"}

    def __init__(
        self,
        port: str,
        *,
        serial_factory: Optional[Callable[..., object]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not port:
            raise ValueError("An Arduino UNO serial port is required")
        self.port = port
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._serial = None
        self._connected = False
        self._last_reply = ""
        self._last_error = ""
        self._send_count = 0

        if serial_factory is None:
            import serial

            serial_factory = serial.Serial

        try:
            self._serial = serial_factory(
                port,
                self.BAUD_RATE,
                timeout=0.03,
                write_timeout=0.20,
            )
            self._handshake()
        except Exception:
            self.close()
            raise

    @staticmethod
    def detect_uno_port() -> str:
        from serial.tools import list_ports

        matches = []
        for item in list_ports.comports():
            evidence = (
                f"{item.device} {item.description or ''} "
                f"{item.manufacturer or ''} {item.product or ''}"
            ).lower()
            if "arduino uno" in evidence or (
                item.device.startswith("/dev/cu.usbmodem")
                and item.vid in {0x2341, 0x2A03}
            ):
                matches.append(item.device)
        if len(matches) != 1:
            found = ", ".join(matches) if matches else "none"
            raise RuntimeError(
                f"Expected exactly one Arduino UNO USB port; found: {found}"
            )
        return matches[0]

    def _write_raw(self, command: str) -> None:
        if self._serial is None:
            raise RuntimeError("Arduino serial link is closed")
        self._serial.write((command + "\n").encode("ascii"))

    def _read_available(self) -> list[str]:
        replies: list[str] = []
        if self._serial is None:
            return replies
        while getattr(self._serial, "in_waiting", 0):
            raw = self._serial.readline()
            if not raw:
                break
            reply = raw.decode("utf-8", errors="replace").strip()
            if reply:
                replies.append(reply)
                self._last_reply = reply
        return replies

    def _handshake(self) -> None:
        deadline = self._clock() + self.HANDSHAKE_SECONDS
        evidence: list[str] = []
        next_lock = 0.0
        while self._clock() < deadline:
            now = self._clock()
            if now >= next_lock:
                self._write_raw("LOCK")
                self._write_raw("STATUS")
                next_lock = now + 0.10
            evidence.extend(self._read_available())
            if any(EXPECTED_FIRMWARE_MARKER in line for line in evidence):
                self._connected = True
                self._last_error = ""
                self._write_raw("LOCK")
                return
            self._sleep(0.02)
        joined = " | ".join(evidence[-5:]) or "no identifying reply"
        raise RuntimeError(
            "Arduino did not identify the optical-hover flight firmware: " + joined
        )

    def send(self, command: str) -> None:
        if command not in self.ALLOWED_COMMANDS:
            raise ValueError(
                "Transport accepts only TAKEOFF, HOVER, FORWARD, LAND, or LOCK"
            )
        with self._lock:
            if not self._connected or self._serial is None:
                raise RuntimeError("Arduino flight transport is disconnected")
            try:
                self._read_available()
                self._write_raw(command)
                self._send_count += 1
                if self._send_count % 3 == 0:
                    self._write_raw("STATUS")
                self._read_available()
                self._last_error = ""
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._connected = False
                raise

    def status(self) -> dict:
        with self._lock:
            try:
                self._read_available()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._connected = False
            return {
                "connected": self._connected,
                "port": self.port,
                "last_reply": self._last_reply,
                "last_error": self._last_error,
            }

    def close(self, final_command: Optional[str] = "LOCK") -> None:
        with self._lock:
            if self._serial is not None:
                if final_command in self.ALLOWED_COMMANDS:
                    try:
                        self._write_raw(final_command)
                    except Exception:
                        pass
                try:
                    self._serial.close()
                except Exception:
                    pass
            self._serial = None
            self._connected = False


class FlightControlService:
    BROWSER_WATCHDOG_SECONDS = 0.30
    TAKEOFF_TIMEOUT_SECONDS = 7.0
    MAX_FORWARD_SECONDS = 1.80
    WORKER_PERIOD_SECONDS = 0.05

    def __init__(
        self,
        transport: object,
        *,
        clock: Callable[[], float] = time.monotonic,
        start_worker: bool = True,
        token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
        aircraft_optical_hold_verified: bool = False,
        owner_controls_authorized: bool = False,
    ) -> None:
        self.transport = transport
        self._clock = clock
        self._token_factory = token_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._token = ""
        self._last_sequence = 0
        self._mode = "DISARMED"
        self._last_browser_at = 0.0
        self._takeoff_started_at = 0.0
        self._forward_started_at = 0.0
        self._last_command = "LOCK"
        self._service_error = ""
        self._aircraft_optical_hold_verified = bool(
            aircraft_optical_hold_verified
        )
        self._owner_controls_authorized = bool(owner_controls_authorized)

        self._send("LOCK")
        if start_worker:
            self._thread = threading.Thread(
                target=self._worker,
                name="optical-hover-flight-watchdog",
                daemon=True,
            )
            self._thread.start()

    @staticmethod
    def _firmware_mode(reply: str) -> str:
        match = re.search(r"(?:^|\s)MODE=([A-Z_]+)", reply or "")
        return match.group(1) if match else ""

    def _send(self, command: str) -> None:
        try:
            self.transport.send(command)
            with self._lock:
                self._last_command = command
                self._service_error = ""
        except Exception as exc:
            with self._lock:
                self._last_command = "UNCONFIRMED"
                self._service_error = f"{type(exc).__name__}: {exc}"
                if self._mode != "DISARMED":
                    self._mode = "LINK_LOST"

    def _reset_disarmed(self) -> None:
        self._mode = "DISARMED"
        self._takeoff_started_at = 0.0
        self._forward_started_at = 0.0

    def begin_browser_session(self) -> str:
        with self._lock:
            was_airborne = self._mode != "DISARMED"
            self._token = self._token_factory()
            self._last_sequence = 0
            self._last_browser_at = self._clock()
            if was_airborne:
                self._mode = "LANDING"
            else:
                self._reset_disarmed()
            token = self._token
        self._send("LAND" if was_airborne else "LOCK")
        return token

    def _fail_safe_request(self) -> None:
        with self._lock:
            airborne = self._mode != "DISARMED"
            if airborne:
                self._mode = "LANDING"
            else:
                self._reset_disarmed()
        self._send("LAND" if airborne else "LOCK")

    def authorize(self, token: str) -> None:
        with self._lock:
            allowed = bool(self._token) and secrets.compare_digest(
                str(token or ""), self._token
            )
        if not allowed:
            self._fail_safe_request()
            raise PermissionError("Invalid or expired browser control session")

    def _accept_sequence(self, sequence: int) -> None:
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

    def browser_heartbeat(self, token: str) -> None:
        self.authorize(token)
        with self._lock:
            self._last_browser_at = self._clock()

    def arm_and_takeoff(self, token: str, sequence: int) -> None:
        self.authorize(token)
        if not (self._aircraft_optical_hold_verified or self._owner_controls_authorized):
            self._fail_safe_request()
            raise ValueError(
                "Aircraft optical hold is not verified; motors remain locked"
            )
        self._accept_sequence(sequence)
        now = self._clock()
        with self._lock:
            allowed = self._mode == "DISARMED"
            if allowed:
                self._mode = "TAKING_OFF"
                self._takeoff_started_at = now
                self._forward_started_at = 0.0
                self._last_browser_at = now
        if not allowed:
            self._fail_safe_request()
            raise ValueError("ARM is allowed only while disarmed")

    def go_forward(self, token: str, sequence: int) -> None:
        self.authorize(token)
        self._accept_sequence(sequence)
        now = self._clock()
        with self._lock:
            allowed = self._mode == "HOVERING"
            if allowed:
                self._mode = "FORWARD"
                self._forward_started_at = now
                self._last_browser_at = now
        if not allowed:
            self._fail_safe_request()
            raise ValueError("GO is allowed only after takeoff reaches hover")

    def stop_and_hover(self, token: str, sequence: int) -> None:
        self.authorize(token)
        self._accept_sequence(sequence)
        now = self._clock()
        with self._lock:
            allowed = self._mode in {"FORWARD", "HOVERING"}
            if allowed:
                self._mode = "HOVERING"
                self._forward_started_at = 0.0
                self._last_browser_at = now
        if not allowed:
            self._fail_safe_request()
            raise ValueError("STOP/HOVER requires an airborne hover or forward state")
        self._send("HOVER")

    def land(self, token: str, sequence: int) -> None:
        self.authorize(token)
        self._accept_sequence(sequence)
        now = self._clock()
        with self._lock:
            allowed = self._mode != "DISARMED"
            if allowed:
                self._mode = "LANDING"
                self._forward_started_at = 0.0
                self._last_browser_at = now
        if not allowed:
            self._send("LOCK")
            raise ValueError("LAND requires an active takeoff or airborne state")
        self._send("LAND")

    def emergency_lock(self, token: str) -> None:
        self.authorize(token)
        with self._lock:
            self._reset_disarmed()
        self._send("LOCK")

    def _eeg_control_allowed_locked(self, token: str) -> bool:
        return bool(
            (self._aircraft_optical_hold_verified or self._owner_controls_authorized)
            and self._token and secrets.compare_digest(str(token or ''), self._token)
            and self._mode in {'HOVERING', 'FORWARD'}
            and self._last_browser_at
            and self._clock() - self._last_browser_at <= self.BROWSER_WATCHDOG_SECONDS
        )

    def eeg_control_allowed(self, token: str) -> bool:
        with self._lock:
            return self._eeg_control_allowed_locked(token)

    def apply_eeg_decision(self, token: str, decision: str) -> bool:
        """Change an already airborne mode; never arm or refresh the watchdog."""
        if decision not in {'GO', 'STOP'}:
            return False
        with self._lock:
            if not self._eeg_control_allowed_locked(token):
                return False
            if decision == 'STOP':
                self._mode = 'HOVERING'
                self._forward_started_at = 0.0
            elif self._mode == 'HOVERING':
                self._mode = 'FORWARD'
                self._forward_started_at = self._clock()
        # the flight worker sends the command and enforces its cap
        return True

    def tick(self) -> None:
        now = self._clock()
        transport_status = self.transport.status()
        firmware_mode = self._firmware_mode(transport_status.get("last_reply", ""))

        with self._lock:
            mode = self._mode
            browser_stale = (
                mode not in {"DISARMED", "LANDING", "LINK_LOST"}
                and (
                    not self._last_browser_at
                    or now - self._last_browser_at > self.BROWSER_WATCHDOG_SECONDS
                )
            )

            if browser_stale:
                self._mode = "LANDING"
                command = "LAND"
            elif mode == "TAKING_OFF":
                if firmware_mode == "HOVERING":
                    self._mode = "HOVERING"
                    command = "HOVER"
                elif firmware_mode in {"FAULT_LOCK", "LANDED_LOCK"}:
                    self._reset_disarmed()
                    command = "LOCK"
                elif now - self._takeoff_started_at > self.TAKEOFF_TIMEOUT_SECONDS:
                    self._mode = "LANDING"
                    command = "LAND"
                else:
                    command = "TAKEOFF"
            elif mode == "HOVERING":
                if firmware_mode in {"LANDING"}:
                    self._mode = "LANDING"
                    command = "LAND"
                elif firmware_mode in {"LOCKED", "FAULT_LOCK", "LANDED_LOCK"}:
                    self._reset_disarmed()
                    command = "LOCK"
                else:
                    command = "HOVER"
            elif mode == "FORWARD":
                if now - self._forward_started_at >= self.MAX_FORWARD_SECONDS:
                    self._mode = "HOVERING"
                    self._forward_started_at = 0.0
                    command = "HOVER"
                elif firmware_mode in {"LANDING"}:
                    self._mode = "LANDING"
                    command = "LAND"
                else:
                    command = "FORWARD"
            elif mode in {"LANDING", "LINK_LOST"}:
                if firmware_mode in {"LOCKED", "FAULT_LOCK", "LANDED_LOCK"}:
                    self._reset_disarmed()
                    command = "LOCK"
                else:
                    self._mode = "LANDING"
                    command = "LAND"
            else:
                command = "LOCK"

        self._send(command)

    def status(self) -> dict:
        transport_status = dict(self.transport.status())
        with self._lock:
            mode = self._mode
            last_command = self._last_command
            service_error = self._service_error
        aircraft_firmware_state = (
            "optical_hold_prop_off_verified"
            if self._aircraft_optical_hold_verified
            else "aircraft_not_programmed_remote_restored"
        )
        flight_blocker = (
            ""
            if self._aircraft_optical_hold_verified
            else "Optical hold remains unverified; original manual controls authorized by owner"
            if self._owner_controls_authorized
            else "Aircraft verification record is missing or invalid; controls remain locked"
        )
        transport_status.update(
            {
                "mode": mode,
                "firmware_mode": self._firmware_mode(
                    transport_status.get("last_reply", "")
                ),
                "last_command": last_command,
                "service_error": service_error,
                "target_height_mm": 1000,
                "altitude_feedback": False,
                "target_height_kind": "timed_nominal_estimate",
                "forward_command": "vendor_fixed_slow_velocity",
                "max_forward_ms": 2000,
                "aircraft_optical_hold_verified": (
                    self._aircraft_optical_hold_verified
                ),
                "aircraft_firmware_state": aircraft_firmware_state,
                "flight_blocker": flight_blocker,
                "flight_ready": self._aircraft_optical_hold_verified,
                "control_ready": self._aircraft_optical_hold_verified or self._owner_controls_authorized,
                "owner_controls_authorized": self._owner_controls_authorized,
                "verification_scope": "battery_powered_propeller_off",
            }
        )
        return transport_status

    def launcher_health(self) -> dict:
        state = self.status()
        connected = bool(state.get("connected"))
        return {
            "service_online": True,
            "mode": state['mode'],
            "last_command": state['last_command'],
            "checks": {
                "arduino_connected": connected,
                "transmitter_firmware_verified": connected,
                "nrf24_initialized": connected,
                # one-way radio: only the aircraft's LEDs or UART confirm the mode
                "aircraft_optical_hold_verified": state[
                    "aircraft_optical_hold_verified"
                ],
                "altitude_feedback": False,
            },
            "flight_ready": state["flight_ready"],
            "control_ready": state['control_ready'],
            "owner_controls_authorized": state['owner_controls_authorized'],
            "aircraft_firmware_state": state["aircraft_firmware_state"],
            "flight_blocker": state["flight_blocker"],
            "verification_scope": state["verification_scope"],
            "target_height_mm": 1000,
            "target_height_kind": "timed_nominal_estimate",
            "forward_command": "vendor_fixed_slow_velocity",
        }

    def _worker(self) -> None:
        while not self._stop.is_set():
            self.tick()
            self._stop.wait(self.WORKER_PERIOD_SECONDS)

    def close(self) -> None:
        with self._lock:
            airborne = self._mode != "DISARMED"
            if airborne:
                self._mode = "LANDING"
        self._send("LAND" if airborne else "LOCK")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        try:
            self.transport.close(final_command="LAND" if airborne else "LOCK")
        except TypeError:
            self.transport.close()
        except Exception:
            pass


def create_app(
    service: FlightControlService, *, eeg_service: Optional[LiveEEGService] = None
) -> Flask:
    template_dir = Path(__file__).resolve().parent / "web" / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    eeg = eeg_service if eeg_service is not None else LiveEEGService()
    brain = EEGFlightControl(service, eeg, start_worker=service._thread is not None)
    assisted = isinstance(service, AssistedFlightService)
    app.extensions["live_eeg"] = eeg
    app.extensions["eeg_flight_control"] = brain

    def flight_state():
        return {**service.status(), 'brain_control': brain.status()}

    def eeg_state():
        return {**eeg.status(), 'brain_control': brain.status()}

    @app.before_request
    def local_only():
        remote = request.remote_addr or ""
        host = urlsplit(f"//{request.host}").hostname or ""
        origin = request.headers.get("Origin")
        origin_host = urlsplit(origin).hostname if origin else None
        if (
            remote not in {"127.0.0.1", "::1"}
            or host not in LOCAL_HOSTS
            or (origin_host is not None and origin_host not in LOCAL_HOSTS)
        ):
            service._fail_safe_request()
            return jsonify(status="error", message="Local Mac only"), 403
        return None

    @app.after_request
    def safety_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; frame-ancestors 'none'"
        )
        response.headers["X-OpenBCI-Drone-App"] = "flight-launcher-v1"
        return response

    def token() -> str:
        return request.headers.get("X-Flight-Token", "")

    @app.get("/")
    def index():
        with brain._lock:
            brain.reset_session()
            control_token = service.begin_browser_session()
        state = service.status()
        return render_template(
            "optical_hover.html",
            control_token=control_token,
            flight_ready=state["flight_ready"],
            controls_ready=state['control_ready'],
            flight_blocker=state.get("flight_blocker", ""),
            assisted=assisted,
            eeg_state=eeg_state(),
            eeg_models=eeg.model_catalog(),
        )

    @app.get("/favicon.ico")
    def favicon():
        return "", 204

    @app.get("/launcher-health")
    def launcher_health():
        return jsonify(service.launcher_health())

    @app.post("/api/heartbeat")
    def heartbeat():
        try:
            if assisted:
                payload = request.get_json(silent=True) or {}
                service.browser_heartbeat(
                    token(),
                    forward=payload.get("forward", 0),
                    right=payload.get("right", 0),
                    target_mm=payload.get("target_mm"),
                    yaw=payload.get("yaw", 0),
                )
            else:
                service.browser_heartbeat(token())
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        return jsonify(flight_state())

    @app.post("/api/arm")
    def arm():
        payload = request.get_json(silent=True) or {}
        try:
            if assisted:
                service.arm_and_takeoff(token(), payload.get("sequence"), payload.get("target_mm"))
            else:
                service.arm_and_takeoff(token(), payload.get("sequence"))
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        except ValueError as exc:
            return jsonify(status="error", message=str(exc)), 409
        return jsonify(flight_state())

    @app.post("/api/head")
    def head_follow():
        payload = request.get_json(silent=True) or {}
        try:
            service.set_head_follow(token(), payload.get("enabled"))
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        except (ValueError, AttributeError) as exc:
            return jsonify(status="error", message=str(exc)), 400
        return jsonify(flight_state())

    @app.post("/api/go")
    def go():
        payload = request.get_json(silent=True) or {}
        try:
            with brain._lock:
                service.authorize(token())
                if brain.manual_go_allowed() and not (
                    brain.status()['enabled'] and service.status()['mode'] == 'FORWARD'
                ):
                    if assisted:
                        service.go_forward(token(), payload.get("sequence"), payload.get("distance_mm", 2000))
                    else:
                        service.go_forward(token(), payload.get("sequence"))
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        except ValueError as exc:
            return jsonify(status="error", message=str(exc)), 409
        return jsonify(flight_state())

    @app.post("/api/stop")
    def stop():
        payload = request.get_json(silent=True) or {}
        try:
            with brain._lock:
                service.authorize(token())
                brain.disable('Stopped from computer · enable brain control again')
                service.stop_and_hover(token(), payload.get("sequence"))
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        except ValueError as exc:
            return jsonify(status="error", message=str(exc)), 409
        return jsonify(flight_state())

    @app.post("/api/land")
    def land():
        payload = request.get_json(silent=True) or {}
        try:
            with brain._lock:
                service.authorize(token())
                brain.disable('Brain control off · landing', hover=False)
                service.land(token(), payload.get("sequence"))
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        except ValueError as exc:
            return jsonify(status="error", message=str(exc)), 409
        return jsonify(flight_state())

    @app.post("/api/emergency-lock")
    def emergency_lock():
        try:
            with brain._lock:
                service.authorize(token())
                brain.disable('Brain control off · motor cut', hover=False)
                service.emergency_lock(token())
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        return jsonify(flight_state())

    @app.post("/api/move")
    def move():
        if not assisted:
            return jsonify(status="error", message="Measured moves need the assisted-flight firmware"), 409
        payload = request.get_json(silent=True) or {}
        try:
            with brain._lock:
                service.authorize(token())
                brain.disable('Brain control off · measured move')
                service.move(token(), payload.get("sequence"), payload.get("forward_mm"), payload.get("right_mm"))
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        except ValueError as exc:
            return jsonify(status="error", message=str(exc)), 409
        return jsonify(flight_state())

    @app.get("/api/status")
    def status():
        try:
            if assisted:
                service.check_token(token())  # read-only
            else:
                service.authorize(token())
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        return jsonify(flight_state())

    @app.get("/api/eeg")
    def eeg_status():
        try:
            service.authorize(token())
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        return jsonify(eeg_state())

    @app.post("/api/eeg/connect")
    def eeg_connect():
        try:
            service.authorize(token())
            eeg.connect()
            return jsonify(eeg_state())
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        except RuntimeError as exc:
            return jsonify(status="error", message=str(exc)), 409

    @app.post("/api/eeg/disconnect")
    def eeg_disconnect():
        try:
            service.authorize(token())
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        with brain._lock:
            brain.disable('Brain control off · EEG disconnected')
            eeg.disconnect()
        return jsonify(eeg_state())

    @app.post("/api/eeg/threshold")
    def eeg_threshold():
        try:
            service.authorize(token())
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                raise ValueError("Provide a STOP threshold between 1% and 99%")
            with brain._lock:
                brain.disable('Brain control off · threshold changed; enable again')
                eeg.set_threshold(payload.get("threshold"))
            return jsonify(eeg_state())
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        except ValueError as exc:
            return jsonify(status="error", message=str(exc)), 400

    @app.post("/api/eeg/model")
    def eeg_model():
        try:
            service.authorize(token())
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                raise ValueError("Choose a model from the saved model list")
            with brain._lock:
                brain.disable('Brain control off · model changed; enable again')
                eeg.select_model(payload.get("model_id"))
            return jsonify(eeg_state())
        except PermissionError as exc:
            return jsonify(status="error", message=str(exc)), 403
        except ValueError as exc:
            return jsonify(status="error", message=str(exc)), 400

    @app.post('/api/eeg/control')
    def eeg_control():
        try:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                service.authorize(token())
                raise ValueError('Set brain control enabled to true or false')
            brain.set_enabled(token(), payload.get('enabled'))
            return jsonify(eeg_state())
        except PermissionError as exc:
            return jsonify(status='error', message=str(exc)), 403
        except ValueError as exc:
            return jsonify(status='error', message=str(exc)), 409

    return app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local optical-hover flight candidate controller"
    )
    parser.add_argument("--serial-port", default="")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument(
        "--aircraft-verification-record",
        type=Path,
        default=DEFAULT_AIRCRAFT_VERIFICATION_RECORD,
    )
    parser.add_argument('--owner-control-authorization', type=Path, default=None)
    parser.add_argument(
        "--flight-stack",
        choices=("assisted", "legacy"),
        default="assisted",
        help="assisted: onboard laser/optical-flow flight (firmware/aircraft_mission); "
        "legacy: original timed optical-hover transmitter",
    )
    args = parser.parse_args()

    serial_port = args.serial_port or SerialFlightTransport.detect_uno_port()
    if args.flight_stack == "assisted":
        service = AssistedFlightService(AssistedSerialTransport(serial_port))
    else:
        transport = SerialFlightTransport(serial_port)
        service = FlightControlService(
            transport,
            aircraft_optical_hold_verified=load_aircraft_verification(
                args.aircraft_verification_record
            ),
            owner_controls_authorized=load_owner_control_authorization(args.owner_control_authorization),
        )
    eeg = LiveEEGService()
    app = create_app(service, eeg_service=eeg)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print(
        f"DRONE FLIGHT READY ({args.flight_stack}): http://localhost:{args.port} "
        f"via {serial_port} (CONTROLS_AVAILABLE={service.status()['control_ready']}; OPTICAL_VERIFIED={service.status()['flight_ready']})"
    )
    try:
        app.run(
            host="127.0.0.1",
            port=args.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    finally:
        app.extensions['eeg_flight_control'].close()
        service.close()
        eeg.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

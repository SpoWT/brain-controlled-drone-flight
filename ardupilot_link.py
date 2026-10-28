"""MAVLink link to the flight controller over the SiK radio.

Keeps the newest copy of each message and sends as GCS 254/190, so the FC's GCS failsafe
(FS_GCS_ENABLE 5) lands the drone if this program stops. The port can change while running
(serial_devices.RadioWatch moves it to whichever port the drone answers on); with no port, sends are dropped.
"""
from __future__ import annotations

import os

# TELEM1 speaks MAVLink2; must be set before pymavlink.mavutil is imported.
os.environ.setdefault("MAVLINK20", "1")

import threading  # noqa: E402
import time  # noqa: E402
from typing import Callable, Optional  # noqa: E402

GCS_SYSTEM_ID = 254
GCS_COMPONENT_ID = 190
FC_SYSTEM_ID = 1
FC_COMPONENT_ID = 1
RANGE_SENSOR_COMPONENT_ID = 88  # MTF-01P forwarded by the FC

MODE_STABILIZE = 0
MODE_ALT_HOLD = 2
MODE_LOITER = 5
MODE_LAND = 9
MODE_FLOWHOLD = 22
FORCE_DISARM_MAGIC = 21196

# (message id, rate Hz). The SiK link carries ~2 KB/s; this uses ~800 B/s.
STREAMS = (
    (0, 1),     # HEARTBEAT (always sent)
    (30, 5),    # ATTITUDE
    (32, 5),    # LOCAL_POSITION_NED
    (132, 5),   # DISTANCE_SENSOR
    (100, 2),   # OPTICAL_FLOW
    (36, 2),    # SERVO_OUTPUT_RAW
    (62, 3),    # NAV_CONTROLLER_OUTPUT
    (245, 1),   # EXTENDED_SYS_STATE
    (2, 0.5),   # SYSTEM_TIME
    (1, 0.5),   # SYS_STATUS
)


class ArduPilotLink:
    """Owns the pymavlink connection; one reader thread, locked writes."""

    HEARTBEAT_STALE_SECONDS = 3.0

    def __init__(
        self,
        device: str,
        *,
        baud: int = 57600,
        connection_factory: Optional[Callable[..., object]] = None,
        clock: Callable[[], float] = time.monotonic,
        start_reader: bool = True,
        recorder: Optional[object] = None,
    ) -> None:
        if connection_factory is None:
            from pymavlink import mavutil

            def connection_factory(dev, **kw):
                # mavlink_connection() can take a busy "COM4" for a log file
                if dev.upper().startswith("COM") or dev.startswith("/dev/"):
                    return mavutil.mavserial(dev, baud=kw["baud"], source_system=kw["source_system"],
                                             source_component=kw["source_component"])
                return mavutil.mavlink_connection(dev, **kw)
        self.device = ""
        self._factory, self._baud = connection_factory, baud
        self._clock = clock
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: dict[str, tuple[object, float]] = {}
        self._status_texts: list[tuple[float, str]] = []
        self._last_error = ""
        self._messages = 0
        self._recorder = recorder
        self._conn = None
        self._attached_at = clock()
        self._last_fc_at = -1e9
        self._armed_at = -1e9
        if device:
            self.attach(device)
        self._reader: Optional[threading.Thread] = None
        if start_reader:
            self._reader = threading.Thread(target=self._read_loop, name="ardupilot-rx", daemon=True)
            self._reader.start()

    # ------------------------------------------------------------------ receive
    def handle(self, msg) -> None:
        """Store one received message."""
        kind = msg.get_type()
        if kind == "BAD_DATA":
            return
        if self._recorder is not None:
            self._recorder.record(msg)
        source = (msg.get_srcSystem(), msg.get_srcComponent())
        if source[0] != FC_SYSTEM_ID:
            return
        now = self._clock()
        if kind == "DISTANCE_SENSOR":
            key = "DISTANCE_SENSOR" if source[1] == FC_COMPONENT_ID else "DISTANCE_SENSOR_RAW"
        elif source[1] != FC_COMPONENT_ID:
            return
        else:
            key = kind
        with self._state_lock:
            self._latest[key] = (msg, now)
            self._messages += 1
            self._last_fc_at = now
            if kind == "HEARTBEAT" and msg.base_mode & 128:
                self._armed_at = now
            if kind == "STATUSTEXT":
                self._status_texts = (self._status_texts + [(now, str(msg.text))])[-20:]

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            conn = self._conn
            if conn is None:  # no radio port yet
                self._stop.wait(0.2)
                continue
            try:
                msg = conn.recv_match(blocking=True, timeout=0.2)
            except Exception as exc:
                if conn is self._conn:  # not just let go of
                    with self._state_lock:
                        self._last_error = f"{type(exc).__name__}: {exc}"
                    self._reopen(conn)
                self._stop.wait(0.5)
                continue
            if getattr(conn, "portdead", False):
                self._reopen(conn)
            if msg is not None and conn is self._conn:
                self.handle(msg)

    def _reopen(self, conn) -> None:
        """Reopen the serial port after a USB dropout."""
        reset = getattr(conn, "reset", None)
        if reset is None:
            return
        with self._write_lock:
            if conn is not self._conn:
                return
            try:
                reset()
            except Exception as exc:  # not back yet
                with self._state_lock:
                    self._last_error = f"reopen failed: {type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ port
    def attach(self, device: str) -> None:
        """Talk to the drone through this port from now on (closes the previous one)."""
        conn = self._factory(device, baud=self._baud, source_system=GCS_SYSTEM_ID, source_component=GCS_COMPONENT_ID)
        with self._write_lock:
            old, self._conn, self.device = self._conn, conn, device
            self._attached_at = self._clock()
        self._close(old)

    def detach(self) -> None:
        """Let go of the port so it can be checked again."""
        with self._write_lock:
            old, self._conn, self.device = self._conn, None, ""
        self._close(old)

    @staticmethod
    def _close(conn) -> None:
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            pass

    def quiet_for(self) -> float:
        """Seconds since the drone was last heard on this port (or since it was opened)."""
        with self._state_lock:
            return self._clock() - max(self._last_fc_at, self._attached_at)

    def armed_within(self, seconds: float) -> bool:
        """True if a heartbeat in the last `seconds` said the motors were armed."""
        with self._state_lock:
            return self._clock() - self._armed_at <= seconds

    def latest(self, kind: str, max_age: float) -> Optional[object]:
        with self._state_lock:
            entry = self._latest.get(kind)
        if entry is None or self._clock() - entry[1] > max_age:
            return None
        return entry[0]

    def heartbeat(self):
        return self.latest("HEARTBEAT", self.HEARTBEAT_STALE_SECONDS)

    def armed(self) -> bool:
        hb = self.heartbeat()
        return bool(hb is not None and hb.base_mode & 128)

    def mode(self) -> Optional[int]:
        hb = self.heartbeat()
        return None if hb is None else int(hb.custom_mode)

    def status_texts(self, since: float) -> list[str]:
        with self._state_lock:
            return [text for at, text in self._status_texts if at >= since]

    def stats(self) -> dict:
        with self._state_lock:
            return {"messages": self._messages, "last_error": self._last_error}

    # ------------------------------------------------------------------ send
    def _send(self, fn: Callable[[], None]) -> None:
        with self._write_lock:
            if self._conn is not None:
                fn()

    def send_gcs_heartbeat(self) -> None:
        # type GCS(6), autopilot INVALID(8)
        self._send(lambda: self._conn.mav.heartbeat_send(6, 8, 0, 0, 4))

    def request_streams(self) -> None:
        for msg_id, hz in STREAMS:
            if msg_id == 0:
                continue
            self._command_long(511, msg_id, int(1_000_000 / hz))

    def manual_control(self, pitch: int, roll: int, throttle: int, yaw: int = 0) -> None:
        """MANUAL_CONTROL: pitch/roll/yaw -1000..1000, throttle 0..1000 (500 = hold height)."""
        p = int(max(-1000, min(1000, pitch)))
        r = int(max(-1000, min(1000, roll)))
        t = int(max(0, min(1000, throttle)))
        y = int(max(-1000, min(1000, yaw)))
        self._send(lambda: self._conn.mav.manual_control_send(FC_SYSTEM_ID, p, r, t, y, 0))

    def set_mode(self, custom_mode: int) -> None:
        self._send(lambda: self._conn.mav.set_mode_send(FC_SYSTEM_ID, 1, custom_mode))

    def arm(self) -> None:
        self._command_long(400, 1)

    def disarm(self, force: bool = False) -> None:
        self._command_long(400, 0, FORCE_DISARM_MAGIC if force else 0)

    def _command_long(self, command: int, p1: float = 0, p2: float = 0) -> None:
        self._send(lambda: self._conn.mav.command_long_send(
            FC_SYSTEM_ID, FC_COMPONENT_ID, command, 0, p1, p2, 0, 0, 0, 0, 0))

    def close(self) -> None:
        self._stop.set()
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=0.5)
        self.detach()

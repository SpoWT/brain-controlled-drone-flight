"""USB serial link to the older UNO + nRF24 transmitter (kept for the original drone)."""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, Optional

from drone_flight.flight.assisted_flight_protocol import (
    TRANSMITTER_MARKER,
    ConfigTelemetry,
    FlightTelemetry,
    parse_telemetry_line,
)

_ALLOWED_LINE = re.compile(
    r"^(LOCK|STANDBY|HOLD|LAND|PING|STATUS"
    r"|TAKEOFF \d{3,4}"
    r"|FLY \d{3,4} -?\d{1,3} -?\d{1,3}( -?\d{1,2})?"
    r"|MOVE -?\d{1,4} -?\d{1,4}"
    r"|CONFIG \d{3,5} [0-7]"
    r"|HEAD [01])$"
)


class AssistedSerialTransport:
    BAUD_RATE = 115200
    HANDSHAKE_SECONDS = 4.0

    def __init__(
        self,
        port: str,
        *,
        serial_factory: Optional[Callable[..., object]] = None,
        clock: Callable[[], float] = time.monotonic,
        start_reader: bool = True,
    ) -> None:
        if not port:
            raise ValueError("An Arduino UNO serial port is required")
        self.port = port
        self._clock = clock
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._serial = None
        self._connected = False
        self._last_reply = ""
        self._last_error = ""
        self._flight: Optional[FlightTelemetry] = None
        self._flight_at = 0.0
        self._config: Optional[ConfigTelemetry] = None
        self._config_at = 0.0
        self._telemetry_count = 0
        self._reader: Optional[threading.Thread] = None

        if serial_factory is None:
            import serial

            serial_factory = serial.Serial
        try:
            self._serial = serial_factory(port, self.BAUD_RATE, timeout=0.05, write_timeout=0.2)
            self._handshake()
        except Exception:
            self.close(final_line=None)
            raise
        if start_reader:
            self._reader = threading.Thread(target=self._read_loop, name="assisted-flight-rx", daemon=True)
            self._reader.start()

    def _handshake(self) -> None:
        deadline = self._clock() + self.HANDSHAKE_SECONDS
        next_query = 0.0
        evidence: list[str] = []
        while self._clock() < deadline:
            if self._clock() >= next_query:
                self._write("LOCK")
                self._write("STATUS")
                next_query = self._clock() + 0.25
            line = self._readline()
            if line:
                evidence.append(line)
                self._handle_line(line)
                if TRANSMITTER_MARKER in line:
                    self._connected = True
                    return
        joined = " | ".join(evidence[-4:]) or "no reply"
        raise RuntimeError(f"Arduino is not running the assisted-flight transmitter: {joined}")

    def _write(self, line: str) -> None:
        if self._serial is None:
            raise RuntimeError("serial link closed")
        self._serial.write((line + "\n").encode("ascii"))

    def _readline(self) -> str:
        if self._serial is None:
            return ""
        raw = self._serial.readline()
        return raw.decode("ascii", errors="replace").strip() if raw else ""

    def _handle_line(self, line: str) -> None:
        packet = parse_telemetry_line(line)
        now = self._clock()
        with self._state_lock:
            if isinstance(packet, FlightTelemetry):
                self._flight, self._flight_at = packet, now
                self._telemetry_count += 1
            elif isinstance(packet, ConfigTelemetry):
                self._config, self._config_at = packet, now
                self._telemetry_count += 1
            elif line:
                self._last_reply = line
                if line.startswith(("ERROR", "FATAL")):
                    self._last_error = line

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = self._readline()
            except Exception as exc:
                with self._state_lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._connected = False
                self._stop.wait(0.2)
                continue
            if line:
                self._handle_line(line)

    def send(self, line: str) -> None:
        if not _ALLOWED_LINE.match(line):
            raise ValueError(f"refusing to send malformed transmitter line: {line!r}")
        if line.startswith("HEAD "):
            return  # this drone cannot turn
        if line.startswith("FLY ") and len(line.split()) == 5:
            line = line.rsplit(" ", 1)[0]  # drop the yaw rate
        with self._lock:
            if not self._connected or self._serial is None:
                raise RuntimeError("assisted-flight transmitter is disconnected")
            try:
                self._write(line)
            except Exception as exc:
                with self._state_lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._connected = False
                raise

    def telemetry(self) -> tuple[Optional[FlightTelemetry], float, Optional[ConfigTelemetry], float]:
        """Latest packets and their ages in seconds (inf when never received)."""
        now = self._clock()
        with self._state_lock:
            flight_age = now - self._flight_at if self._flight else float("inf")
            config_age = now - self._config_at if self._config else float("inf")
            return self._flight, flight_age, self._config, config_age

    def status(self) -> dict:
        with self._state_lock:
            return {
                "connected": self._connected,
                "port": self.port,
                "last_reply": self._last_reply,
                "last_error": self._last_error,
                "telemetry_packets": self._telemetry_count,
            }

    def close(self, final_line: Optional[str] = "LOCK") -> None:
        self._stop.set()
        with self._lock:
            if self._serial is not None:
                if final_line:
                    try:
                        self._write(final_line)
                    except Exception:
                        pass
                try:
                    self._serial.close()
                except Exception:
                    pass
            self._serial = None
            self._connected = False
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=0.5)

"""Head steering: the drone's heading follows the VR headset's yaw.

The headset's BNO08x sends UART-RVC packets (19 bytes: 0xAA 0xAA, index, yaw/pitch/roll x100, accel,
checksum; ~100 Hz) through a CP2102 USB adapter. It is found by those packets on any USB adapter, never by
COM number (serial_devices.py).
"""
from __future__ import annotations

import math
import struct
import threading
import time
from typing import Callable, Iterable, Optional, Union

from serial_devices import SCAN_LOCK, usb_uart_ports

RVC_BAUD = 115200
RVC_PACKET_BYTES = 19
HEAD_FRESH_S = 0.3              # older readings are ignored and the heading is held
FOLLOW_GAIN = 1.5               # deg/s per degree of difference
FOLLOW_DEADBAND_DEG = 4.0
FOLLOW_MAX_DEG_S = 45
SCAN_EVERY_S = 1.0              # while no sensor is attached
SILENT_DETACH_S = 3.0           # attached but silent this long (unplugged, moved): look again
NO_ADAPTER = "no head sensor adapter plugged in"


def parse_rvc_packet(packet: bytes) -> Optional[tuple[float, float, float]]:
    """(yaw, pitch, roll) in degrees from one UART-RVC packet, or None if invalid."""
    if len(packet) != RVC_PACKET_BYTES or packet[:2] != b"\xaa\xaa" or sum(packet[2:18]) & 0xFF != packet[18]:
        return None
    yaw, pitch, roll = struct.unpack("<hhh", packet[3:9])
    return yaw / 100, pitch / 100, roll / 100


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class HeadFollow:
    """Turn rate (deg/s) that brings the drone's heading to the head's, relative to the zero point."""

    def __init__(self) -> None:
        self._zero: Optional[tuple[float, float]] = None  # (head yaw, drone yaw)

    @property
    def active(self) -> bool:
        return self._zero is not None

    @property
    def zero_point(self) -> Optional[tuple[float, float]]:
        """(head yaw, drone yaw) that count as straight ahead, or None."""
        return self._zero

    def zero(self, head_yaw: float, drone_yaw: float) -> None:
        self._zero = (head_yaw, drone_yaw)

    def clear(self) -> None:
        self._zero = None

    def rate(self, head_yaw: Optional[float], drone_yaw: Optional[float]) -> int:
        if self._zero is None or head_yaw is None or drone_yaw is None:
            return 0
        wanted = self._zero[1] + wrap_deg(head_yaw - self._zero[0])
        error = wrap_deg(wanted - drone_yaw)
        if abs(error) < FOLLOW_DEADBAND_DEG:
            return 0
        return int(round(max(-FOLLOW_MAX_DEG_S, min(FOLLOW_MAX_DEG_S, FOLLOW_GAIN * error))))


class HeadYawReader:
    """Reads the head yaw in the background; latest(max_age) returns degrees or None."""

    def __init__(self, port: str, *, opener: Optional[Callable] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if opener is None:
            import serial
            opener = lambda: serial.Serial(port, RVC_BAUD, timeout=0.05)  # noqa: E731
        self.port, self._opener, self._clock = port, opener, clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reading: Optional[tuple[float, float]] = None  # (yaw, time)
        self.error = ""
        self._thread = threading.Thread(target=self._run, name="head-yaw", daemon=True)
        self._thread.start()

    def latest(self, max_age: float) -> Optional[float]:
        with self._lock:
            reading = self._reading
        if reading is None or self._clock() - reading[1] > max_age:
            return None
        return reading[0]

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._opener() as link:
                    self._pump(link)
            except Exception as exc:  # unplugged: keep retrying
                self.error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(1.0)

    def _pump(self, link) -> None:
        buffer = bytearray()
        while not self._stop.is_set():
            buffer += link.read(256)
            while True:
                start = buffer.find(b"\xaa\xaa")
                if start < 0 or len(buffer) < start + RVC_PACKET_BYTES:
                    break
                angles = parse_rvc_packet(bytes(buffer[start:start + RVC_PACKET_BYTES]))
                if angles is None:
                    del buffer[:start + 1]
                    continue
                with self._lock:
                    self._reading = (angles[0], self._clock())
                del buffer[:start + RVC_PACKET_BYTES]
            del buffer[:-RVC_PACKET_BYTES]

    @staticmethod
    def find(exclude: Iterable[str] = (), listen_s: float = 0.6) -> Optional[str]:
        """The CP2102 port sending RVC packets. Ports in exclude are never opened."""
        return scan(exclude, listen_s)[0]


def has_rvc_packet(data: bytes) -> bool:
    start = data.find(b"\xaa\xaa")
    while 0 <= start <= len(data) - RVC_PACKET_BYTES:
        if parse_rvc_packet(bytes(data[start:start + RVC_PACKET_BYTES])) is not None:
            return True
        start = data.find(b"\xaa\xaa", start + 1)
    return False


def adapter_ports(exclude: Iterable[str] = ()) -> list[str]:
    return usb_uart_ports(exclude)


def listen(port: str, seconds: float) -> bytes:
    """Bytes heard at the head sensor's speed; stops at the first head packet."""
    from serial_devices import listen as listen_port
    with SCAN_LOCK:
        return listen_port(port, RVC_BAUD, seconds, until=has_rvc_packet)


def scan(exclude: Iterable[str] = (), listen_s: float = 0.6, *, ports: Optional[list[str]] = None,
         listener: Callable[[str, float], bytes] = listen) -> tuple[Optional[str], str]:
    """(port sending head packets, ""), or (None, what was seen instead)."""
    ports = adapter_ports(exclude) if ports is None else ports
    if not ports:
        return None, NO_ADAPTER
    seen = []
    for port in ports:
        try:
            data = listener(port, listen_s)
        except Exception:
            seen.append(f"{port} busy")
            continue
        if has_rvc_packet(data):
            return port, ""
        rate = round(len(data) / listen_s) if listen_s > 0 else len(data)
        seen.append(f"{port} sends unreadable data ({rate} bytes/s)" if data else f"{port} sends nothing")
    return None, ", ".join(seen)


class HeadSensorWatch:
    """Attaches the head sensor whenever it appears, so it can be plugged in or fixed while the website runs."""

    def __init__(self, transport, exclude: Union[Iterable[str], Callable[[], Iterable[str]]] = (), *,
                 scanner: Callable = scan,
                 reader_factory: Callable[[str], HeadYawReader] = HeadYawReader,
                 clock: Callable[[], float] = time.monotonic, every_s: float = SCAN_EVERY_S) -> None:
        self.transport = transport
        self._exclude = exclude if callable(exclude) else (lambda fixed=set(exclude): fixed)  # e.g. the radio's port
        self._scan, self._reader_factory, self._clock, self._every_s = scanner, reader_factory, clock, every_s
        self._attached_at = 0.0
        self._stop = threading.Event()
        self.note = "looking for the head sensor"

    def start(self) -> None:
        threading.Thread(target=self._run, name="head-watch", daemon=True).start()

    def close(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return {"note": self.note}

    def step(self) -> None:
        reader = self.transport.head
        if reader is not None:
            silent = reader.latest(SILENT_DETACH_S) is None
            if silent and self._clock() - self._attached_at > SILENT_DETACH_S:
                reader.close()
                self.transport.head = None
                self.note = f"{reader.port} went silent"
            return
        with SCAN_LOCK:  # the radio search skips the port once transport.head holds it
            port, seen = self._scan({port for port in self._exclude() if port})
            if port is None:
                self.note = seen
                return
            self.transport.head = self._reader_factory(port)
        self._attached_at = self._clock()
        self.note = ""

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception as exc:
                self.note = f"head sensor search failed: {exc}"
            self._stop.wait(self._every_s)


class HeadSteering:
    """Mixed into ArduPilotFlightTransport: HEAD 1/0 lines and the turn rate while flying."""

    def _cmd_head(self, on: int) -> None:
        """On: where the head looks now becomes the drone's current heading."""
        self._head_on = bool(on)
        self._head_follow.clear()
        self._head_recentre = self._head_on
        self._head_view_zero = None

    def _head_yaw(self) -> Optional[float]:
        reader = self.head  # read once: HeadSensorWatch can drop it at any moment
        return reader.latest(HEAD_FRESH_S) if reader is not None else None

    def _turn_rate(self) -> int:
        """Arrow keys win, then head steering."""
        if self._yaw_rate:
            self._head_recentre = self._head_on  # re-zero after an arrow turn
            return self._yaw_rate
        if not self._head_on:
            return 0
        head = self._head_yaw()
        att = self.link.latest("ATTITUDE", 0.5)
        drone = None if att is None else math.degrees(att.yaw)
        if self._head_recentre and head is not None and drone is not None:
            self._head_follow.zero(head, drone)
            self._head_recentre = False
        return self._head_follow.rate(head, drone)

    def direction_status(self) -> dict:
        """Head and drone heading relative to straight ahead, and the current turn rate."""
        reader = self.head
        head = reader.latest(HEAD_FRESH_S) if reader is not None else None
        att = self.link.latest("ATTITUDE", 0.5)
        drone = None if att is None else math.degrees(att.yaw)
        zero = self._head_follow.zero_point
        if zero is None:  # not flying: show relative to the first reading
            if self._head_view_zero is None and head is not None and drone is not None:
                self._head_view_zero = (head, drone)
            zero = self._head_view_zero
        flying = self._phase == "FLY"
        return {"head_sensor": reader is not None, "head_on": self._head_on,
                "head_offset": None if zero is None or head is None else round(wrap_deg(head - zero[0]), 1),
                "drone_offset": None if zero is None or drone is None else round(wrap_deg(drone - zero[1]), 1),
                "turn_rate": (self._yaw_rate or self._head_follow.rate(head, drone)) if flying else 0,
                "arrow_rate": self._yaw_rate if flying else 0}

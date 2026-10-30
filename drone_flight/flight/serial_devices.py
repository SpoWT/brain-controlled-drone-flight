"""Finds the radio and the head sensor by what they send, never by COM number.

Both plug in through CP210x USB adapters that can report the same serial number, so Windows hands out COM
numbers by plug order and they can swap. The radio is the port carrying heartbeats from the flight controller
(MAVLink, 57600 baud); the head sensor is the port sending UART-RVC packets (115200 baud). Probing only
listens: nothing is ever sent.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Iterable, Optional

os.environ.setdefault("MAVLINK20", "1")

USB_UART_IDS = {(0x10C4, 0xEA60), (0x0403, 0x6001), (0x0403, 0x6015)}  # CP210x, FTDI (never the EEG dongle)
RADIO_BAUD = 57600
RADIO_LISTEN_S = 2.5            # the FC sends a heartbeat every second
FC_SYSTEM_ID = 1
FC_COMPONENT_ID = 1
MAV_TYPE_GCS = 6
RADIO_SILENT_S = 5.0            # drone silent this long (and not flying): look on every adapter again
FLYING_HOLD_S = 60.0            # may be flying: keep the port this long after the drone went silent
                                # (by then the FC's failsafe has landed it; Windows may refuse to reopen
                                # the same COM name until the old handle is closed)
WATCH_EVERY_S = 1.0
NO_RADIO = "no USB radio plugged in"
# One probe at a time: the radio and head sensor searches never open the same port together.
SCAN_LOCK = threading.RLock()


def usb_uart_ports(exclude: Iterable[str] = ()) -> list[str]:
    """USB serial adapters that could be the radio or the head sensor, minus ports in exclude."""
    from serial.tools import list_ports
    skip = {port.upper() for port in exclude if port}
    return [info.device for info in list_ports.comports()
            if (info.vid, info.pid) in USB_UART_IDS and info.device.upper() not in skip]


def listen(port: str, baud: int, seconds: float, until: Optional[Callable[[bytes], bool]] = None) -> bytes:
    """Bytes heard on a port (read-only). Stops early once until(all bytes so far) is true."""
    import serial
    with serial.Serial(port, baud, timeout=0.05) as link:
        data, end = bytearray(), time.monotonic() + seconds
        while time.monotonic() < end:
            data += link.read(256)
            if until is not None and data and until(bytes(data)):
                break
    return bytes(data)


def has_fc_heartbeat(data: bytes) -> bool:
    """True if the bytes hold a valid heartbeat from the flight controller (checksum verified)."""
    from pymavlink.dialects.v20 import ardupilotmega as mavlink
    parser = mavlink.MAVLink(None)
    parser.robust_parsing = True
    for start in range(0, len(data), 256):
        for msg in parser.parse_buffer(data[start:start + 256]) or []:
            if (msg.get_type() == "HEARTBEAT" and msg.get_srcSystem() == FC_SYSTEM_ID
                    and msg.get_srcComponent() == FC_COMPONENT_ID and msg.type != MAV_TYPE_GCS):
                return True
    return False


def probe_radio(port: str, seconds: float = RADIO_LISTEN_S) -> bool:
    """True if the drone answers on this port."""
    with SCAN_LOCK:
        return has_fc_heartbeat(listen(port, RADIO_BAUD, seconds, until=has_fc_heartbeat))


def find_radio(exclude: Iterable[str] = (), *, prefer: str = "", ports: Optional[list[str]] = None,
               prober: Callable[[str], bool] = probe_radio) -> tuple[Optional[str], str]:
    """(port the drone answers on, ""), or (None, what was seen instead)."""
    ports = usb_uart_ports(exclude) if ports is None else ports
    if not ports:
        return None, NO_RADIO
    ports = sorted(ports, key=lambda port: port.upper() != prefer.upper())  # the last radio port first
    seen = []
    for port in ports:
        try:
            if prober(port):
                return port, ""
            seen.append(port)
        except Exception:
            seen.append(f"{port} busy")
    return None, f"drone not heard on {', '.join(seen)}"


class RadioWatch:
    """Keeps the drone link on whichever port the drone answers on, checking every second.

    - Drone answering: nothing changes.
    - Silent for RADIO_SILENT_S and not flying: the port is let go and every adapter not used by the head
      sensor is checked for drone heartbeats, again and again, until one answers.
    - While it may be flying it keeps its port (the link reopens it after a USB dropout), but once the drone has
      been silent for RADIO_SILENT_S it keeps checking the other adapters, because after a USB glitch the radio
      can come back under another COM name while the old name now belongs to the head sensor. It moves only to a
      port the drone answers on, opening it before closing the old one: never without a replacement.
    - Silent for FLYING_HOLD_S even though it may be flying: the FC has landed by itself by then, so it is
      treated like the ground case (let go, check every port).
    """

    def __init__(self, link, *, busy: Callable[[], set] = set, flying: Callable[[], bool] = lambda: False,
                 finder: Callable[..., tuple[Optional[str], str]] = find_radio,
                 clock: Callable[[], float] = time.monotonic, every_s: float = WATCH_EVERY_S) -> None:
        self.link, self._busy, self._flying, self._find = link, busy, flying, finder
        self._clock, self._every_s = clock, every_s
        self._last_port = link.device
        self._stop = threading.Event()
        self.note = "looking for the radio"

    def start(self) -> None:
        threading.Thread(target=self._run, name="radio-watch", daemon=True).start()

    def close(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return {"port": self.link.device, "note": self.note}

    def step(self) -> None:
        link = self.link
        if link.heartbeat() is not None:
            self._last_port, self.note = link.device, ""
            return
        if link.device:
            may_fly = self._flying() or link.armed_within(FLYING_HOLD_S)
            if may_fly and link.quiet_for() < FLYING_HOLD_S:
                self._while_flying()
                return
            if link.quiet_for() < RADIO_SILENT_S:
                return
            link.detach()
        with SCAN_LOCK:  # the head sensor search must not grab the port between finding and opening it
            port, seen = self._find(exclude=self._busy(), prefer=self._last_port)
            if port is None:
                self.note = seen
                return
            link.attach(port)
        self._last_port, self.note = port, ""

    def _while_flying(self) -> None:
        link = self.link
        note = f"drone silent on {link.device} while it may be flying: keeping the port"
        if link.quiet_for() < RADIO_SILENT_S:
            self.note = note
            return
        with SCAN_LOCK:
            port, seen = self._find(exclude=set(self._busy()) | {link.device}, prefer=self._last_port)
            if port is None:
                self.note = f"{note} ({seen})"
                return
            link.attach(port)  # opens the new port, then closes the old one
        self._last_port, self.note = port, ""

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception as exc:
                self.note = f"radio search failed: {type(exc).__name__}: {exc}"
            self._stop.wait(self._every_s)

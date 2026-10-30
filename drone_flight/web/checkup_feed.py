"""GET /api/checkup: read-only status of everything connected plus the connection lights.

No token and no commands. The Drone App passes ?recording=<seconds> so the page can show REC.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Optional

from drone_flight.web.drone_checks import build_checks

BATTERY_FRESH_S = 5.0       # SYS_STATUS every 2 s
HEAD_FRESH_S = 0.5
USB_EVERY_S = 2.0
APP_RECORDING_FRESH_S = 3.0
MAX_RECORDING_S = 7 * 24 * 3600
EEG_FIELDS = ("enabled", "state", "message", "sample_count", "age_ms", "model_name", "model_ready",
              "model_loading", "prediction", "stop_threshold")


def battery(link) -> Optional[dict]:
    """Battery volts (and %) from SYS_STATUS, or None."""
    msg = link.latest("SYS_STATUS", BATTERY_FRESH_S)
    if msg is None or msg.voltage_battery in (0, 0xFFFF):
        return None
    remaining = int(msg.battery_remaining)
    return {"volts": round(msg.voltage_battery / 1000.0, 2), "percent": remaining if remaining >= 0 else None}


def head(transport, watch=None) -> dict:
    reader = getattr(transport, "head", None)
    if reader is None:
        return {"found": False, "note": watch.note if watch is not None else ""}
    return {"found": True, "port": reader.port, "reading": reader.latest(HEAD_FRESH_S) is not None,
            "error": reader.error, "steering_on": bool(getattr(transport, "_head_on", False))}


def radio(transport, watch=None) -> dict:
    """Port the drone link uses now ("" while searching) and what the search saw."""
    return {"port": transport.link.device, "note": watch.note if watch is not None else ""}


def eeg(eeg_service, brain) -> dict:
    status = eeg_service.status()
    picked = {key: status.get(key) for key in EEG_FIELDS}
    picked["brain_control"] = brain.status() if brain is not None else None
    return picked


class UsbPorts:
    """Serial ports present on this laptop (not opened), refreshed every 2 s."""

    def __init__(self, lister=None, clock=time.monotonic) -> None:
        self._lister, self._clock = lister, clock
        self._lock = threading.Lock()
        self._ports: list[str] = []
        self._at = -1e9

    def ports(self) -> list[str]:
        with self._lock:
            if self._clock() - self._at >= USB_EVERY_S:
                self._ports, self._at = self._list(), self._clock()
            return list(self._ports)

    def _list(self) -> list[str]:
        if self._lister is not None:
            return self._lister()
        try:
            from serial.tools import list_ports
            return [port.device for port in list_ports.comports()]
        except Exception:
            return []


class AppRecording:
    """Recording time last reported by the Drone App."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._heard: Optional[tuple[int, float]] = None

    def report(self, raw: Optional[str]) -> None:
        if raw is None:
            return
        try:
            seconds = int(raw)
        except ValueError:
            return
        if 0 <= seconds <= MAX_RECORDING_S:
            self._heard = (seconds, self._clock())

    def status(self) -> Optional[dict]:
        heard = self._heard
        if heard is None or self._clock() - heard[1] > APP_RECORDING_FRESH_S:
            return None
        return {"seconds": heard[0]}


def collect(service, transport, eeg_service, eyes, recorder=None, brain=None, usb: Optional[list] = None,
            app_recording: Optional[dict] = None, head_watch=None, radio_watch=None) -> dict:
    for camera in eyes.cameras:
        camera.touch()
    state = service.status()
    state.pop("calibration", None)
    data = {
        "time": time.time(),
        "service": state,
        "page_heard_s": service.page_heard_age(),
        "radio": radio(transport, radio_watch),
        "battery": battery(transport.link),
        "head": head(transport, head_watch),
        "eyes": eyes.status(),
        "eeg": eeg(eeg_service, brain),
        "brain_go": bool(service.eeg_go_active()),
        "mavlink_log": None if recorder is None else {"path": str(recorder.path), "error": recorder.error},
        "usb": usb or [],
        "app_recording": app_recording,
    }
    data["checks"] = [asdict(check) for check in build_checks("up", data, {"com_ports": data["usb"]})]
    return data


def create_checkup_blueprint(service, transport, eeg_service, eyes, recorder=None, usb: Optional[UsbPorts] = None,
                             head_watch=None, radio_watch=None):
    from flask import Blueprint, current_app, jsonify, request

    blueprint = Blueprint("checkup_feed", __name__)
    usb = usb or UsbPorts()
    app_recording = AppRecording()

    @blueprint.get("/api/checkup")
    def checkup():
        app_recording.report(request.args.get("recording"))
        brain = current_app.extensions.get("eeg_flight_control")
        return jsonify(collect(service, transport, eeg_service, eyes, recorder, brain, usb.ports(),
                               app_recording.status(), head_watch, radio_watch))

    return blueprint

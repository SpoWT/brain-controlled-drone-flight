"""Connection checks shown as lights on the flying page and in the Drone App.

  website   "up" | "down" | "starting" | "old" (running without /api/checkup)
  checkup   /api/checkup response, or None
  usb       {"com_ports": ["COM4", ...]} (shown for reference; devices are found by what they send)
  recording {"on", "seconds", "bytes", "error"}, or None on the website
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BATTERY_GOOD_V = 23.0          # 6S
BATTERY_ARM_V = 22.2           # BATT_ARM_VOLT
FLOOR_CAMERA_MIN = 50          # FLOW_MIN_QUALITY
PAGE_IN_CONTROL_S = 2.0        # the page sends a heartbeat every 80 ms
PAGE_HIDDEN_S = 70.0           # hidden tabs are throttled
EYE_MIN_FPS = 0.5

RESTART = "Reopen the Drone App."
NO_RADIO = "no USB radio plugged in"  # serial_devices.NO_RADIO
HEAD_WIRING = ("Check the head sensor's power, GND and TX wire, and that it is in UART-RVC mode (P0 high). "
               "It connects by itself once it sends head data.")


@dataclass(frozen=True)
class Check:
    key: str
    label: str
    level: str        # ok | warn | bad | wait | off
    detail: str
    fix: str = ""
    required: bool = False


@dataclass(frozen=True)
class Verdict:
    title: str        # READY TO FLY | FLYING | NOT READY | STARTING
    level: str
    note: str


def _telemetry(checkup: Optional[dict]) -> Optional[dict]:
    return (checkup or {}).get("service", {}).get("telemetry")


def website_check(website: str) -> Check:
    if website == "up":
        return Check("website", "Flying website", "ok", "Running on this laptop", required=True)
    if website == "starting":
        return Check("website", "Flying website", "wait", "Starting... (about 20 seconds)", required=True)
    if website == "old":
        return Check("website", "Flying website", "bad", "Running an old version", RESTART, required=True)
    return Check("website", "Flying website", "bad", "Not running", RESTART, required=True)


def radio_check(checkup: Optional[dict], usb: dict) -> Check:
    """The website finds the radio on any port by the drone's heartbeats (serial_devices.RadioWatch)."""
    if checkup is None:
        return Check("radio", "Radio (USB)", "wait", "Needs the website", required=True)
    radio = checkup.get("radio") or {}
    port = radio.get("port") or ""
    if port:
        return Check("radio", "Radio (USB)", "ok", f"Drone link on {port}", required=True)
    if radio.get("note") == NO_RADIO:
        return Check("radio", "Radio (USB)", "bad", "Not plugged in",
                     "Plug the radio (small board with the antenna) into the laptop. It connects by itself.",
                     required=True)
    return Check("radio", "Radio (USB)", "warn", radio.get("note") or "Looking for the drone", required=True)


def drone_check(checkup: Optional[dict], radio_ok: bool) -> Check:
    if checkup is None:
        return Check("drone", "Drone answering", "wait", "Needs the website", required=True)
    service = checkup["service"]
    if service.get("connected"):
        return Check("drone", "Drone answering", "ok", f"Heard over the radio · {service.get('fc_mode') or ''}".strip(" ·"),
                     required=True)
    fix = "Put a charged battery in the drone. It connects by itself." if radio_ok else "Plug in the radio first."
    return Check("drone", "Drone answering", "bad", "No answer over the radio", fix, required=True)


def battery_check(checkup: Optional[dict], drone_ok: bool) -> Check:
    if not drone_ok:
        return Check("battery", "Drone battery", "wait", "Needs the drone", required=True)
    battery = checkup.get("battery")
    if battery is None:
        return Check("battery", "Drone battery", "warn", "No reading yet (a few seconds)", required=True)
    volts = battery["volts"]
    if volts >= BATTERY_GOOD_V:
        return Check("battery", "Drone battery", "ok", f"{volts:.1f} V", required=True)
    if volts >= BATTERY_ARM_V:
        return Check("battery", "Drone battery", "warn", f"{volts:.1f} V - low", "Short flight only; charge it soon.",
                     required=True)
    return Check("battery", "Drone battery", "bad", f"{volts:.1f} V - too low",
                 f"Swap in a charged battery (no take-off below {BATTERY_ARM_V} V).", required=True)


def controller_check(checkup: Optional[dict], drone_ok: bool) -> Check:
    if not drone_ok:
        return Check("controller", "Flight controller", "wait", "Needs the drone", required=True)
    service = checkup["service"]
    if service.get("flight_ready"):
        return Check("controller", "Flight controller", "ok", "Ready", required=True)
    return Check("controller", "Flight controller", "bad", service.get("flight_blocker") or "Not ready yet",
                 "Wait 10 seconds. If it stays, " + RESTART.lower(), required=True)


def laser_check(checkup: Optional[dict], drone_ok: bool) -> Check:
    telemetry = _telemetry(checkup)
    if not drone_ok or telemetry is None:
        return Check("laser", "Height laser", "wait", "Needs the drone", required=True)
    if telemetry.get("laser_ok"):
        return Check("laser", "Height laser", "ok", f"{telemetry.get('height_mm', 0) / 10:.0f} cm", required=True)
    return Check("laser", "Height laser", "bad", "Not reading",
                 "Check the sensor cable under the drone; set the drone flat on the floor.", required=True)


def floor_camera_check(checkup: Optional[dict]) -> Check:
    telemetry = _telemetry(checkup)
    if telemetry is None:
        return Check("floor", "Floor camera", "wait", "Needs the drone")
    quality = int(telemetry.get("flow_quality") or 0)
    if quality >= FLOOR_CAMERA_MIN:
        return Check("floor", "Floor camera", "ok", f"Sees the floor (quality {quality})")
    if telemetry.get("armed"):
        return Check("floor", "Floor camera", "bad", f"Blind in the air (quality {quality}) - it will land",
                     "Fly over a floor with texture and light.")
    return Check("floor", "Floor camera", "warn", f"Quality {quality} - normal while sitting on the floor")


def motors_check(checkup: Optional[dict]) -> Check:
    telemetry = _telemetry(checkup)
    if telemetry is None:
        return Check("motors", "Motors", "wait", "Needs the drone")
    if telemetry.get("armed"):
        mode = checkup["service"].get("mode", "")
        return Check("motors", "Motors", "warn", f"ON - {mode.lower()} at {telemetry.get('height_mm', 0) / 10:.0f} cm")
    return Check("motors", "Motors", "ok", "Off")


def page_check(checkup: Optional[dict]) -> Check:
    if checkup is None:
        return Check("page", "Flying page", "wait", "Needs the website", required=True)
    heard = checkup.get("page_heard_s")
    if heard is not None and heard <= PAGE_IN_CONTROL_S:
        return Check("page", "Flying page", "ok", "Open and in control", required=True)
    if heard is not None and heard <= PAGE_HIDDEN_S:
        return Check("page", "Flying page", "warn", "Open but hidden or minimised",
                     "Click on the flying page: the keys only work there.", required=True)
    return Check("page", "Flying page", "bad", "Not open", "Press OPEN FLYING PAGE.", required=True)


def head_check(checkup: Optional[dict]) -> Check:
    if checkup is None:
        return Check("head", "VR head sensor", "wait", "Needs the website")
    head = checkup.get("head") or {}
    if not head.get("found"):
        note = head.get("note") or ""
        if "unreadable" in note or "sends nothing" in note or "silent" in note:
            return Check("head", "VR head sensor", "warn", f"No head data ({note})", HEAD_WIRING)
        return Check("head", "VR head sensor", "off", "Not found",
                     "Plug in the headset's motion sensor. It connects by itself.")
    steering = "head steering ON" if head.get("steering_on") else "head steering off"
    if head.get("reading"):
        return Check("head", "VR head sensor", "ok", f"Reading ({head.get('port')}) · {steering}")
    return Check("head", "VR head sensor", "warn", f"Plugged in ({head.get('port')}), no data",
                 "Check the headset sensor's cable and power.")


def eyes_check(checkup: Optional[dict]) -> Check:
    if checkup is None:
        return Check("eyes", "Eye cameras", "wait", "Needs the website")
    eyes = checkup.get("eyes") or []
    if not eyes:
        return Check("eyes", "Eye cameras", "off", "Not found",
                     "Plug in the goggles' USB cable, then " + RESTART.lower())
    working = [e for e in eyes if e.get("fps", 0) >= EYE_MIN_FPS and not e.get("error")]
    if len(working) == len(eyes):
        return Check("eyes", "Eye cameras", "ok", f"{len(eyes)} of {len(eyes)} working")
    if not any(e.get("running") for e in eyes):
        return Check("eyes", "Eye cameras", "wait", "Starting the cameras...")
    broken = ", ".join(f"{e.get('label')}: {e.get('error') or 'no picture'}" for e in eyes if e not in working)
    return Check("eyes", "Eye cameras", "warn", f"{len(working)} of {len(eyes)} working ({broken})",
                 "Re-plug the goggles' USB cable, then " + RESTART.lower())


def eeg_check(checkup: Optional[dict]) -> Check:
    if checkup is None:
        return Check("eeg", "EEG headset", "wait", "Needs the website")
    eeg = checkup.get("eeg") or {}
    if not eeg.get("enabled"):
        return Check("eeg", "EEG headset", "off", "Not connected",
                     "Turn the headset on, then press CONNECT EEG on the flying page.")
    if eeg.get("state") != "streaming":
        return Check("eeg", "EEG headset", "warn", eeg.get("message") or eeg.get("state") or "Connecting",
                     "Keep the headset on and close to the laptop.")
    if not eeg.get("model_ready"):
        return Check("eeg", "EEG headset", "warn", "Streaming · model loading")
    decision = (eeg.get("prediction") or {}).get("decision") or "-"
    brain = eeg.get("brain_control") or {}
    control = "brain control ON" if brain.get("enabled") else "brain control off"
    return Check("eeg", "EEG headset", "ok", f"Streaming · brain says {decision} · {control}")


def recording_check(recording: dict) -> Check:
    if recording.get("error"):
        return Check("recording", "Recording", "bad", recording["error"], "Close and reopen the Drone App.")
    if not recording.get("on"):
        return Check("recording", "Recording", "wait", "Starting...")
    seconds = int(recording.get("seconds", 0))
    size = recording.get("bytes", 0) / 1e6
    return Check("recording", "Recording", "ok", f"Screen + data · {seconds // 60}:{seconds % 60:02d} · {size:.0f} MB")


def build_checks(website: str, checkup: Optional[dict], usb: dict, recording: Optional[dict] = None) -> list[Check]:
    live = checkup if website == "up" else None
    radio = radio_check(live, usb)
    drone = drone_check(live, radio.level != "bad")
    drone_ok = drone.level == "ok"
    checks = [website_check(website), radio, drone, battery_check(live, drone_ok),
              controller_check(live, drone_ok), laser_check(live, drone_ok), floor_camera_check(live),
              motors_check(live), page_check(live), head_check(live), eyes_check(live), eeg_check(live)]
    return checks if recording is None else checks + [recording_check(recording)]


def verdict(checks: list[Check], checkup: Optional[dict]) -> Verdict:
    by_key = {c.key: c for c in checks}
    if by_key["website"].level == "wait":
        return Verdict("STARTING", "wait", "Starting the flying website and the flight controller...")
    telemetry = _telemetry(checkup) if by_key["website"].level == "ok" else None
    if telemetry and telemetry.get("armed"):
        return Verdict("FLYING", "warn", "Motors are on. Keep the flying page in front. L = land.")
    blocking = [c for c in checks if c.required and c.level not in {"ok", "warn"}]
    if not blocking and by_key["page"].level == "ok":
        return Verdict("READY TO FLY", "ok", "Everything needed is connected.")
    first = blocking[0] if blocking else by_key["page"]
    count = len(blocking) or 1
    return Verdict("NOT READY", "bad", f"{count} thing{'s' if count > 1 else ''} to fix · first: "
                   f"{first.label} - {first.fix or first.detail}")


def last_landing_reason(checkup: Optional[dict]) -> str:
    service = (checkup or {}).get("service") or {}
    fault = service.get("last_fault") or ""
    if not fault or fault == "NONE":
        return ""
    return f"Last automatic landing: {fault}. {service.get('fault_help') or ''}".strip()

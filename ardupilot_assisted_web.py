"""Flying website for the ArduPilot drone over the SiK radio.

    python ardupilot_assisted_web.py                  # finds the radio by itself, on any port, again and again
    python ardupilot_assisted_web.py --mavlink COM4   # start on COM4 (still moves if the drone answers elsewhere)
    python ardupilot_assisted_web.py --mavlink tcp:127.0.0.1:5760
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ardupilot_flight_transport import ArduPilotFlightTransport
from assisted_flight_service import AssistedFlightService
from checkup_feed import create_checkup_blueprint
from dashboard_feeds import create_dashboard_blueprint
from eye_cameras import EyeCameras, create_eye_blueprint
from head_follow import HeadSensorWatch
from mavlink_recorder import MavlinkRecorder
from optical_hover_web import LiveEEGService, create_app
from serial_devices import RadioWatch

RECORDER_DIR = Path(__file__).resolve().parent / "outputs" / "mavlink_logs"


def head_port(transport) -> set:
    reader = transport.head  # read once: the head sensor watch can drop it at any moment
    return {reader.port} if reader is not None else set()


def is_serial(device: str) -> bool:
    return not device or device.upper().startswith("COM") or device.startswith("/dev/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Assisted-flight website on the ArduPilot drone")
    parser.add_argument("--mavlink", default="", help="serial device or pymavlink URL (default: find the radio)")
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()

    device = args.mavlink
    recorder = MavlinkRecorder(RECORDER_DIR)
    transport = ArduPilotFlightTransport("" if is_serial(device) else device, baud=args.baud, recorder=recorder)
    radio_watch = None
    if is_serial(device):
        if device:
            try:
                transport.link.attach(device)
            except Exception as exc:  # missing or busy: RadioWatch looks everywhere
                print(f"{device} not usable ({exc}); looking for the radio on every port")
        radio_watch = RadioWatch(transport.link, flying=transport.flying, busy=lambda: head_port(transport))
        radio_watch.start()
    head_watch = HeadSensorWatch(transport, exclude=lambda: {transport.link.device})
    head_watch.start()
    service = AssistedFlightService(transport)
    eeg = LiveEEGService()
    app = create_app(service, eeg_service=eeg)
    eyes = EyeCameras.discover()
    app.register_blueprint(create_eye_blueprint(eyes))
    app.register_blueprint(create_dashboard_blueprint(transport, service, head_watch))
    app.register_blueprint(create_checkup_blueprint(service, transport, eeg, eyes, recorder, head_watch=head_watch,
                                                    radio_watch=radio_watch))
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print(f"DRONE FLIGHT READY (ardupilot): http://localhost:{args.port} via {device or 'the radio, wherever it is'} "
          f"(CONTROLS_AVAILABLE={service.status()['control_ready']}); recording {recorder.path}; "
          f"eye cameras {[(c.label, c.device_id, c.settings_key) for c in eyes.cameras]}")
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)
    finally:
        head_watch.close()
        if radio_watch is not None:
            radio_watch.close()
        app.extensions["eeg_flight_control"].close()
        service.close()
        eeg.close()
        recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

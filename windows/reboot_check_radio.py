"""Reboot the FC through the SiK radio (disarmed only), then check the height estimate follows the laser.

    python reboot_check_radio.py          # finds the radio by the drone's heartbeats, on any port
    python reboot_check_radio.py COM7
"""
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo folder, for drone_flight
DEVICE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].lower() != "auto" else ""
if not DEVICE:
    from drone_flight.flight.serial_devices import find_radio  # noqa: E402
    DEVICE, seen = find_radio()
    if DEVICE is None:
        raise SystemExit(f"REFUSED: {seen}")
m = mavutil.mavserial(DEVICE, baud=57600, source_system=255, source_component=191)


def heartbeat(timeout):
    end = time.time() + timeout
    while time.time() < end:
        msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg and msg.get_srcSystem() == 1 and msg.get_srcComponent() == 1:
            return msg
    return None


hb = heartbeat(10)
if hb is None or hb.base_mode & 128:
    raise SystemExit("REFUSED: no heartbeat or armed")
m.mav.command_long_send(1, 1, 246, 0, 1, 0, 0, 0, 0, 0, 0)
time.sleep(8)
out = {"port": DEVICE, "back": heartbeat(40) is not None}
for msg_id, hz in ((32, 5), (132, 5), (193, 1), (30, 2), (178, 2)):
    m.mav.command_long_send(1, 1, 511, 0, msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)
z, laser, texts, ekf, att, dcm = [], [], [], None, None, None
end = time.time() + 20
while time.time() < end:
    msg = m.recv_match(blocking=True, timeout=1)
    if msg is None or msg.get_srcSystem() != 1:
        continue
    kind = msg.get_type()
    if kind == "LOCAL_POSITION_NED":
        z.append(round(msg.z, 3))
    elif kind == "DISTANCE_SENSOR" and msg.get_srcComponent() == 1:
        laser.append(msg.current_distance)
    elif kind == "EKF_STATUS_REPORT":
        ekf = {"flags": msg.flags, "vert_var": round(msg.pos_vert_variance, 3), "terrain_var": round(msg.terrain_alt_variance, 3)}
    elif kind == "STATUSTEXT":
        texts.append(msg.text)
    elif kind == "ATTITUDE":
        att = (round(math.degrees(msg.roll), 1), round(math.degrees(msg.pitch), 1))
    elif kind == "AHRS2":
        dcm = (round(math.degrees(msg.roll), 1), round(math.degrees(msg.pitch), 1))
out.update({"ekf_z_last": z[-5:], "ekf_z_span": [min(z), max(z)] if z else None, "laser_cm_last": laser[-5:],
            "ekf": ekf, "ekf_roll_pitch": att, "dcm_roll_pitch": dcm, "statustext": texts})
print(json.dumps(out))

"""Install always_land.lua on the flight controller over USB and check it runs. Never arms.

1. Upload to /APM/scripts with MAVFTP and read it back.
2. SCR_ENABLE=1, reboot, check the ALAND_* parameters exist.
3. ALAND_TEST=1 with ALAND_MAX_HGT=0.01 must report "TEST would LAND: too high"; then restore 0 / 2.3.

    python always_land_install.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402
from pymavlink.mavftp import MAVFTP  # noqa: E402
from serial.tools import list_ports  # noqa: E402

LOCAL = Path(__file__).resolve().parent / "always_land.lua"
REMOTE = "/APM/scripts/always_land.lua"
MAX_HGT_FLYING = 2.3


def connect():
    port = next((p.device for p in list_ports.comports() if p.vid == 0x1209), None)
    if port is None:
        raise ConnectionError("FC USB not found")
    m = mavutil.mavlink_connection(port, baud=115200, source_system=255, source_component=191)
    hb = m.wait_heartbeat(timeout=10)
    if hb is None:
        raise ConnectionError("no heartbeat")
    if hb.base_mode & 128:
        raise SystemExit("REFUSED: FC armed")
    return m


def read_param(m, name):
    for _ in range(3):
        m.mav.param_request_read_send(1, 1, name.encode(), -1)
        end = time.time() + 1.5
        while time.time() < end:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg and msg.param_id == name:
                return round(msg.param_value, 3)
    return None


def set_param(m, name, value):
    m.param_set_send(name, value)
    time.sleep(0.6)
    return read_param(m, name)


def texts_for(m, seconds):
    out, end = [], time.time() + seconds
    while time.time() < end:
        msg = m.recv_match(type="STATUSTEXT", blocking=True, timeout=0.5)
        if msg and msg.text not in out:
            out.append(msg.text)
    return out


def upload(m):
    os.makedirs("/tmp", exist_ok=True)  # MAVFTP staging folder
    ftp = MAVFTP(m, 1, 1)
    ftp.cmd_mkdir(["/APM/scripts"])  # ok if it exists
    ftp.cmd_put([str(LOCAL), REMOTE])
    put = ftp.process_ftp_reply("CreateFile", timeout=30)
    back = Path("/tmp") / "always_land_readback.lua"
    if back.exists():
        back.unlink()
    ftp.cmd_get([REMOTE, str(back)])
    ftp.process_ftp_reply("OpenFileRO", timeout=30)
    same = back.exists() and back.read_bytes() == LOCAL.read_bytes()
    return {"put": str(getattr(put, "error_code", put)), "read_back_identical": same}


def reconnect_after_reboot(m):
    m.mav.command_long_send(1, 1, 246, 0, 1, 0, 0, 0, 0, 0, 0)  # reboot, refused if armed
    m.close()
    time.sleep(12)
    for _ in range(20):
        try:
            return connect()
        except ConnectionError:
            time.sleep(2)
    raise SystemExit("FC did not come back on USB after the reboot")


def main():
    out = {}
    m = connect()
    out["upload"] = upload(m)
    if not out["upload"]["read_back_identical"]:
        print(json.dumps(out))
        sys.exit(1)
    out["SCR_ENABLE"] = set_param(m, "SCR_ENABLE", 1)
    m = reconnect_after_reboot(m)
    out["after_reboot_messages"] = texts_for(m, 8)
    out["script_params"] = {n: read_param(m, "ALAND_" + n) for n in ("ENABLE", "MAX_HGT", "TILT", "ATT_ERR", "TEST")}
    out["test_mode"] = {"TEST": set_param(m, "ALAND_TEST", 1), "MAX_HGT": set_param(m, "ALAND_MAX_HGT", 0.01)}
    out["test_mode_messages"] = texts_for(m, 5)
    out["restored"] = {"TEST": set_param(m, "ALAND_TEST", 0), "MAX_HGT": set_param(m, "ALAND_MAX_HGT", MAX_HGT_FLYING)}
    out["proved_running"] = any("TEST would LAND: too high" in t for t in out["test_mode_messages"])
    out["restored_ok"] = out["restored"] == {"TEST": 0.0, "MAX_HGT": MAX_HGT_FLYING}
    print(json.dumps(out))


if __name__ == "__main__":
    main()

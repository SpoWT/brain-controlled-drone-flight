"""Drone App: starts the flying website and opens the flying page in its own window.

- Disarmed: closes flying pages it opened before, restarts the website and flight controller, opens the page.
- Armed or flying: changes nothing. Opening the page again would land the drone and take control from the open page.
- Never arms, flies or lands the drone.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = "http://127.0.0.1:8083"
START_TASK = "NeuralFlightAppStart"  # restarts the website and the FC
HOME = Path(os.environ.get("USERPROFILE") or Path.home())
PROFILE = HOME / "AppData" / "Local" / "DroneApp" / "browser"
BROWSERS = (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
BROWSER_FLAGS = ("--no-first-run", "--no-default-browser-check", "--start-maximized",
                 # keep the page at full speed when covered or minimised
                 "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                 "--disable-backgrounding-occluded-windows")
ERROR_LOG = HERE / "drone_app_error.log"
POLL_S = 0.25
START_WAIT_S = 60.0
OLD_SITE_GONE_S = 20.0  # answering this long after the restart: the new website is up
NO_WINDOW = 0x08000000
BUSY = "The drone is armed or flying: use the flying page that is already open."
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # no proxy for 127.0.0.1


def log_error(text: str) -> None:
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
    except OSError:
        pass


def get_json(path: str, timeout: float = 1.0):
    """(HTTP status, JSON), or (0, None) when the website does not answer."""
    try:
        with _OPENER.open(SITE + path, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:
        return 0, None


def run_start_task() -> str:
    """Empty string on success, else the error."""
    try:
        done = subprocess.run(["schtasks", "/run", "/tn", START_TASK], capture_output=True, text=True,
                              timeout=15, creationflags=NO_WINDOW if os.name == "nt" else 0)
    except Exception as exc:
        return f"Could not start {START_TASK}: {exc}"
    return "" if done.returncode == 0 else f"{START_TASK} failed: {(done.stderr or done.stdout).strip()}"


def drone_busy() -> bool:
    """True when the website answers and the drone is not DISARMED.
    Probes /launcher-health, never "/": loading "/" takes control away from an open page."""
    code, health = get_json("/launcher-health", 2.0)
    return code == 200 and (health or {}).get("mode") != "DISARMED"


def close_page_windows() -> None:
    """Close flying pages an earlier start opened: their old control token would keep the drone on the ground."""
    if os.name != "nt":
        return
    profile = str(PROFILE).replace("'", "''")
    script = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('"
              + profile + "') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                       timeout=20, creationflags=NO_WINDOW)
    except Exception as exc:
        log_error(f"closing old flying pages: {exc}")


def start_website() -> str:
    """Restart the website and FC, then wait until the website answers. Empty string, or the problem."""
    problem = run_start_task()
    if problem:
        return problem
    started, seen_down = time.monotonic(), False
    while time.monotonic() - started < START_WAIT_S:
        up = get_json("/launcher-health")[0] == 200
        seen_down = seen_down or not up
        if up and (seen_down or time.monotonic() - started > OLD_SITE_GONE_S):
            return ""
        time.sleep(POLL_S)
    return "The website did not start: see web.log in the repo folder."


def open_page() -> str:
    """The flying page in its own browser window and profile. Empty string, or the problem."""
    exe = next((path for path in BROWSERS if Path(path).is_file()), None)
    if exe is None:
        return "Chrome or Edge is not installed"
    PROFILE.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([exe, f"--app={SITE}", f"--user-data-dir={PROFILE}", *BROWSER_FLAGS])
    return ""


def tell(text: str) -> None:
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    (messagebox.showinfo if text == BUSY else messagebox.showerror)("Drone App", text)
    root.destroy()


def first_instance() -> bool:
    """False while another Drone App is still starting."""
    if os.name != "nt":
        return True
    kernel = ctypes.windll.kernel32
    first_instance.mutex = kernel.CreateMutexW(None, False, "NeuralFlightDroneApp")
    return kernel.GetLastError() != 183  # ERROR_ALREADY_EXISTS


def main() -> int:
    if not first_instance():
        return 0
    if drone_busy():
        problem = BUSY
    else:
        close_page_windows()
        problem = start_website() or open_page()
    if problem:
        log_error(problem)
        tell(problem)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        log_error("crash: " + traceback.format_exc())
        raise

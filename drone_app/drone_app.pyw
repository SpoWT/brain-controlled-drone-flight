"""Drone App: the flying page in its own window, with recording.

- Records the page window and all status until STOP RECORDING.
- Restarts the website and flight controller on open (only when disarmed) and restarts the website if it stops.
- Never arms, flies or lands the drone.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import window_capture  # noqa: E402
from drone_checks import build_checks, verdict  # noqa: E402
from drone_recording import Session  # noqa: E402

SITE = "http://127.0.0.1:8083"
START_TASK = "NeuralFlightAppStart"  # restarts the website and the FC
HOME = Path(os.environ.get("USERPROFILE") or Path.home())
RECORDINGS = HOME / "Videos" / "Drone recordings"
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
AUTO_START_EVERY_S = 45.0
PAGE_APPEAR_S = 20.0
MESSAGE_S = 30.0
NO_WINDOW = 0x08000000
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


def usb_ports() -> list[str]:
    try:
        from serial.tools import list_ports
        return [port.device for port in list_ports.comports()]
    except Exception:
        return []


def run_start_task() -> str:
    """Empty string on success, else the error."""
    try:
        done = subprocess.run(["schtasks", "/run", "/tn", START_TASK], capture_output=True, text=True,
                              timeout=15, creationflags=NO_WINDOW if os.name == "nt" else 0)
    except Exception as exc:
        return f"Could not start {START_TASK}: {exc}"
    return "" if done.returncode == 0 else f"{START_TASK} failed: {(done.stderr or done.stdout).strip()}"


def flying(snap: dict) -> bool:
    if snap.get("website") != "up":
        return False
    service = (snap.get("checkup") or {}).get("service") or {}
    return bool((service.get("telemetry") or {}).get("armed")) or service.get("mode") not in (None, "DISARMED")


class Monitor:
    """Polls the website 4x a second, records it and restarts the website if it stops."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._lock = threading.Lock()
        self._snapshot: dict = {}
        self._restart_at = None
        self._seen_down = False
        self._usb: list[str] = []
        self._usb_at = -1e9
        self._auto_at = time.monotonic()
        self._message = ("", 0.0)

    def say(self, text: str) -> None:
        self._message = (text, time.monotonic())
        if text:
            self.session.data.note(text)

    @property
    def message(self) -> str:
        text, at = self._message
        return text if self._restart_at is not None or time.monotonic() - at < MESSAGE_S else ""

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snapshot)

    def start(self) -> None:
        threading.Thread(target=self._run, name="drone-monitor", daemon=True).start()

    def restart(self, why: str) -> None:
        """Restart the website and FC, only while disarmed."""
        code, health = get_json("/launcher-health", 2.0)
        if code == 200 and (health or {}).get("mode") != "DISARMED":
            self.say(f"Not restarting: the drone is {health.get('mode')}.")
            return
        problem = run_start_task()
        self.say(problem or why)
        if not problem:
            self._restart_at, self._seen_down = time.monotonic(), False

    def _run(self) -> None:
        try:
            self.restart("Starting the flying website and the flight controller...")
        except Exception:
            log_error("first start: " + traceback.format_exc())
        while True:
            started = time.monotonic()
            try:
                self._poll()
            except Exception:
                log_error("monitor: " + traceback.format_exc())
            time.sleep(max(0.0, POLL_S - (time.monotonic() - started)))

    def _website(self, seconds: int):
        code, checkup = get_json(f"/api/checkup?recording={seconds}")
        website = "up" if code == 200 else "old" if code == 404 else "down"
        if self._restart_at is not None:
            elapsed = time.monotonic() - self._restart_at
            self._seen_down = self._seen_down or website != "up"
            if (website == "up" and (self._seen_down or elapsed > 20)) or elapsed > START_WAIT_S:
                self._restart_at = None
                self.say("" if website == "up" else "The website did not start: see web.log in NeuralFlightWeb.")
            else:
                return "starting", None
        return website, checkup if website == "up" else None

    def _poll(self) -> None:
        recording = self.session.status()
        website, checkup = self._website(int(recording["seconds"]))
        if time.monotonic() - self._usb_at >= 2.0:
            self._usb, self._usb_at = usb_ports(), time.monotonic()
        checks = build_checks(website, checkup, {"com_ports": self._usb}, recording)
        result = verdict(checks, checkup)
        self.session.record(website, checkup, checks, result)
        with self._lock:
            self._snapshot = {"website": website, "checkup": checkup, "checks": checks, "verdict": result,
                              "recording": recording}
        if website == "down" and time.monotonic() - self._auto_at >= AUTO_START_EVERY_S:
            self._auto_at = time.monotonic()
            self.restart("The website stopped: starting it again...")


class PageWindow:
    """The flying page in its own browser window and profile."""

    def __init__(self) -> None:
        self.process = None
        self.hwnd = None
        self.opened_at = None

    def open(self) -> str:
        exe = next((path for path in BROWSERS if Path(path).is_file()), None)
        if exe is None:
            return "Chrome or Edge is not installed"
        PROFILE.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen([exe, f"--app={SITE}", f"--user-data-dir={PROFILE}", *BROWSER_FLAGS])
        self.hwnd, self.opened_at = None, time.monotonic()
        return ""

    def find(self) -> None:
        if self.opened_at is None or window_capture.exists(self.hwnd):
            return
        pids = {self.process.pid} if self.process is not None else set()
        self.hwnd = window_capture.choose_window(window_capture.list_windows(), pids)

    def showing(self) -> bool:
        return window_capture.exists(self.hwnd)

    def appearing(self) -> bool:
        return self.opened_at is not None and time.monotonic() - self.opened_at < PAGE_APPEAR_S

    def grab(self):
        return window_capture.capture(self.hwnd)

    def close(self) -> None:
        window_capture.close(self.hwnd)


class DroneApp:
    """Control window, minimised while the flying page is open."""

    def __init__(self, root, monitor: Monitor, session: Session, page: PageWindow) -> None:
        import tkinter as tk
        self.tk, self.root, self.monitor, self.session, self.page = tk, root, monitor, session, page
        self.want_page, self.reload_page, self.was_showing = True, False, False
        root.title("Drone App")
        root.configure(bg="#ffffff")
        root.protocol("WM_DELETE_WINDOW", self.quit)
        label = dict(bg="#ffffff", justify="left", anchor="w")
        tk.Label(root, text="DRONE APP", font=("Arial", 13, "bold"), fg="#000000", **label).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 4))
        self.rec = tk.Label(root, text="● REC", font=("Arial", 12, "bold"), bg="#ffffff", fg="#c40000")
        self.rec.grid(row=0, column=1, sticky="e", padx=16)
        self.status = tk.Label(root, text="Starting...", font=("Arial", 16, "bold"), fg="#000000",
                               wraplength=460, **label)
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16)
        self.problems = tk.Label(root, text="", font=("Arial", 11), fg="#c40000", wraplength=460, **label)
        self.problems.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(6, 0))
        self._buttons(root)
        root.after(300, self.tick)

    def _buttons(self, root) -> None:
        style = dict(font=("Arial", 11, "bold"), bg="#ffffff", fg="#000000", activebackground="#000000",
                     activeforeground="#ffffff", relief="solid", bd=2, padx=10, pady=8, cursor="hand2")
        bar = self.tk.Frame(root, bg="#ffffff")
        bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=16, pady=14)
        buttons = (("OPEN FLYING PAGE", self.open_page), ("RECORDINGS", self.open_recordings),
                   ("STOP RECORDING", self.quit))
        for column, (text, command) in enumerate(buttons):
            self.tk.Button(bar, text=text, command=command, **style).grid(
                row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))
            bar.columnconfigure(column, weight=1)

    def tick(self) -> None:
        try:
            self._tick()
        except Exception:
            log_error("tick: " + traceback.format_exc())
        self.root.after(500, self.tick)

    def _tick(self) -> None:
        snap = self.monitor.snapshot()
        seconds = int(self.session.status()["seconds"])
        self.rec.config(text=f"{'●' if seconds % 2 == 0 else '○'} REC {seconds // 60}:{seconds % 60:02d}")
        website = snap.get("website")
        self.page.find()
        showing = self.page.showing()
        if website in {"down", "starting", "old"} and (showing or self.page.appearing()):
            self.reload_page = True  # the old page lost its session
        if website == "up":
            if self.reload_page and showing:
                self.session.data.note("Website restarted: reopening the flying page")
                self.page.close()
            elif self.reload_page or (self.want_page and not showing and not self.page.appearing()):
                self.reload_page = False
                self._open()
        if showing and not self.was_showing:
            self.session.data.note("Flying page window open")
            self.root.iconify()
        elif self.was_showing and not showing and not self.reload_page and website == "up" and not self.page.appearing():
            self.want_page = False  # closed by the user
            self.session.data.note("Flying page window closed (recording continues)")
            self.root.deiconify()
        self.was_showing = showing
        self._show(snap, showing)

    def _open(self) -> None:
        problem = self.page.open()
        self.monitor.say(problem or "Opening the flying page...")

    def _show(self, snap: dict, showing: bool) -> None:
        if not snap:
            self.status.config(text=self.monitor.message or "Starting...")
            return
        if showing:
            text = "Flying page open · recording"
        elif snap["website"] == "up":
            text = "Flying page closed · still recording"
        else:
            text = self.monitor.message or snap["verdict"].note
        self.status.config(text=text)
        bad = [c for c in snap["checks"] if c.required and c.level == "bad"]
        self.problems.config(text="\n".join(f"{c.label}: {c.fix or c.detail}" for c in bad[:4]))

    def open_page(self) -> None:
        self.want_page = True
        if self.monitor.snapshot().get("website") == "up" and not self.page.showing():
            self._open()

    def open_recordings(self) -> None:
        try:
            os.startfile(self.session.folder)  # type: ignore[attr-defined]
        except Exception as exc:
            self.monitor.say(f"Recordings are in {self.session.folder} ({exc})")

    def quit(self) -> None:
        from tkinter import messagebox
        question = "Stop recording and close the Drone App?\nThe flying page stays open."
        if flying(self.monitor.snapshot()):
            question = "THE DRONE IS FLYING.\n" + question
        if not messagebox.askyesno("Drone App", question):
            return
        self.status.config(text="Saving the recording...")
        self.root.update_idletasks()
        try:
            self.session.close()
        except Exception:
            log_error("close: " + traceback.format_exc())
        self.root.destroy()


def windows_setup() -> bool:
    """Single instance, real pixel sizes, lower CPU priority than the website."""
    if os.name != "nt":
        return True
    kernel = ctypes.windll.kernel32
    windows_setup.mutex = kernel.CreateMutexW(None, False, "NeuralFlightDroneApp")
    if kernel.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        return False
    kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
    return True


def main() -> int:
    import tkinter as tk
    from tkinter import messagebox
    if not windows_setup():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Drone App", "The Drone App is already open (look in the taskbar).")
        return 0
    page = PageWindow()
    session = Session(RECORDINGS, grab=page.grab)
    monitor = Monitor(session)
    root = tk.Tk()
    root.report_callback_exception = lambda *exc: log_error("tk: " + "".join(traceback.format_exception(*exc)))
    DroneApp(root, monitor, session, page)
    root.update_idletasks()
    root.geometry(f"+{max(0, root.winfo_screenwidth() - root.winfo_reqwidth() - 24)}+24")
    monitor.start()
    root.mainloop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        log_error("crash: " + traceback.format_exc())
        raise

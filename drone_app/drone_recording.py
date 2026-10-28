"""Session recording, one folder per session:

  screen_001.mp4 ...   flying page window, 8 fps, new file every 5 minutes
  data.jsonl           full status 4x a second
  events.txt           timeline of changes
  drone_mavlink.jsonl  MAVLink messages received during the session
  session.json         start, end, files
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional

SCREEN_FPS = 8
MAX_WIDTH = 1920
SEGMENT_S = 300


def _default_grab():
    import numpy as np
    from PIL import ImageGrab
    return np.asarray(ImageGrab.grab(all_screens=True))[:, :, ::-1]  # RGB -> BGR


def _default_writer(path: Path, fps: int, size: tuple[int, int]):
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot write video {path.name}")
    return writer


def fit(frame, max_width: int):
    """Scale down to max_width with even sides."""
    import cv2
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width)
    size = (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2)
    return frame if size == (width, height) else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def stamp(frame, text: str):
    import cv2
    import numpy as np
    frame = np.array(frame, copy=True, order="C")
    origin = (12, frame.shape[0] - 14)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 255), 2, cv2.LINE_AA)
    return frame


class ScreenRecorder:
    """Grabs frames in a thread and repeats frames to keep real time."""

    def __init__(self, folder: Path, *, grab: Optional[Callable] = None, writer: Optional[Callable] = None,
                 fps: int = SCREEN_FPS, segment_s: float = SEGMENT_S, max_width: int = MAX_WIDTH,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time) -> None:
        self.folder, self.fps, self.segment_s, self.max_width = folder, fps, segment_s, max_width
        self._grab, self._writer = grab or _default_grab, writer or _default_writer
        self._clock, self._wall = clock, wall
        self._stop = threading.Event()
        self.error = ""
        self.files: list[Path] = []
        self.frames = 0
        self._thread = threading.Thread(target=self._run, name="screen-recorder", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _run(self) -> None:
        writer, size, segment_start, written, last = None, None, 0.0, 0, None
        try:
            while not self._stop.is_set():
                raw = self._grab()
                if raw is None and last is None:
                    self._stop.wait(0.25)
                    continue
                now = self._clock()
                if writer is not None and now - segment_start >= self.segment_s:
                    writer.release()
                    writer = None
                frame = last if raw is None else fit(raw, self.max_width)
                if writer is None:
                    size = (frame.shape[1], frame.shape[0])
                    path = self.folder / f"screen_{len(self.files) + 1:03d}.mp4"
                    writer = self._writer(path, self.fps, size)
                    self.files.append(path)
                    segment_start, written = now, 0
                elif (frame.shape[1], frame.shape[0]) != size:
                    import cv2
                    frame = cv2.resize(frame, size)
                last = frame
                clock_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._wall()))
                shown = stamp(frame, clock_text + ("  REC" if raw is not None else "  REC  (page not showing)"))
                due = int((self._clock() - segment_start) * self.fps) + 1
                for _ in range(max(1, due - written)):
                    writer.write(shown)
                    written += 1
                    self.frames += 1
                self._stop.wait(max(0.0, segment_start + written / self.fps - self._clock()))
        except Exception as exc:
            self.error = f"Recording stopped: {type(exc).__name__}: {exc}"
        finally:
            if writer is not None:
                writer.release()


class DataLog:
    """data.jsonl (everything) and events.txt (changes)."""

    def __init__(self, folder: Path, wall: Callable[[], float] = time.time) -> None:
        self._wall = wall
        self._data = open(folder / "data.jsonl", "a", encoding="utf-8", buffering=1)
        self._events = open(folder / "events.txt", "a", encoding="utf-8", buffering=1)
        self._last: dict = {}
        self.error = ""

    def note(self, text: str) -> None:
        try:
            self._events.write(f"{time.strftime('%H:%M:%S', time.localtime(self._wall()))}  {text}\n")
        except Exception as exc:
            self.error = f"Event log stopped: {exc}"

    def record(self, website: str, checkup: Optional[dict], checks: list, verdict) -> None:
        try:
            line = {"t": round(self._wall(), 3), "website": website, "verdict": verdict.title,
                    "checks": {c.key: [c.level, c.detail] for c in checks}, "checkup": checkup}
            self._data.write(json.dumps(line, separators=(",", ":"), default=str) + "\n")
        except Exception as exc:
            self.error = f"Data log stopped: {exc}"
        self._events_for(checkup, checks, verdict)

    def _changed(self, key: str, value) -> bool:
        changed = key in self._last and self._last[key] != value
        first = key not in self._last
        self._last[key] = value
        return changed or (first and value not in (None, "", "NONE"))

    def _events_for(self, checkup: Optional[dict], checks: list, verdict) -> None:
        if self._changed("verdict", verdict.title):
            self.note(f"== {verdict.title} == {verdict.note}")
        for check in checks:
            noisy = check.key == "recording" or (check.key == "floor" and check.level != "bad")
            if not noisy and self._changed(f"check:{check.key}", check.level):
                self.note(f"{check.label}: {check.level.upper()} - {check.detail}")
        if checkup is None:
            return
        service = checkup.get("service") or {}
        if self._changed("mode", service.get("mode")):
            self.note(f"Drone mode: {service.get('mode')}")
        if self._changed("command", (service.get("last_command") or "").split(" ")[0]):
            self.note(f"Command sent: {service.get('last_command')}")
        if self._changed("fault", service.get("last_fault")):
            self.note(f"AUTOMATIC LANDING: {service.get('last_fault')} - {service.get('fault_help') or ''}")

    def close(self) -> None:
        for handle in (self._data, self._events):
            try:
                handle.close()
            except Exception:
                pass


class MavlinkSlice:
    """Copies the part of the website's MAVLink log written during this session."""

    def __init__(self, session_start: float) -> None:
        self._start = session_start
        self._offsets: dict[str, int] = {}

    def see(self, path: Optional[str]) -> None:
        if not path or path in self._offsets:
            return
        file = Path(path)
        try:
            new = file.stat().st_ctime >= self._start - 5
            self._offsets[path] = 0 if new else file.stat().st_size
        except OSError:
            self._offsets[path] = 0

    def copy_to(self, target: Path) -> Optional[str]:
        try:
            with open(target, "wb") as out:
                for path, offset in self._offsets.items():
                    with open(path, "rb") as source:
                        source.seek(offset)
                        shutil.copyfileobj(source, out)
            return None
        except OSError as exc:
            return f"MAVLink copy failed: {exc}"


class Session:
    def __init__(self, root: Path, *, grab: Optional[Callable] = None, screen: Optional[ScreenRecorder] = None,
                 wall: Callable[[], float] = time.time) -> None:
        self.started = wall()
        self._wall = wall
        self.folder = root / time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(self.started))
        self.folder.mkdir(parents=True, exist_ok=True)
        self.data = DataLog(self.folder, wall)
        self.screen = screen if screen is not None else ScreenRecorder(self.folder, grab=grab)
        self.mavlink = MavlinkSlice(self.started)
        self.data.note("Drone App opened: recording the flying page and all drone data")
        self.screen.start()

    def record(self, website: str, checkup: Optional[dict], checks: list, verdict) -> None:
        self.mavlink.see(((checkup or {}).get("mavlink_log") or {}).get("path"))
        self.data.record(website, checkup, checks, verdict)

    def status(self) -> dict:
        size = 0
        for file in self.folder.iterdir():
            try:
                size += file.stat().st_size
            except OSError:
                pass
        return {"on": True, "seconds": self._wall() - self.started, "bytes": size,
                "error": self.screen.error or self.data.error, "folder": str(self.folder)}

    def close(self) -> None:
        self.screen.stop()
        problem = self.mavlink.copy_to(self.folder / "drone_mavlink.jsonl")
        if problem:
            self.data.note(problem)
        self.data.note("Drone App closed")
        self.data.close()
        summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
                   "ended": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._wall())),
                   "screen_files": [p.name for p in self.screen.files], "screen_frames": self.screen.frames,
                   "screen_error": self.screen.error, "mavlink_sources": list(self.mavlink._offsets)}
        (self.folder / "session.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

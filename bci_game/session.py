"""Session storage: one continuous raw EEG file plus an event log per ride.

Labels are derived later from the event log (``labels.py``), so a labelling
rule can change without recording again.

Layout::

    eeg_recordings/<id>/eeg.f64       float64 rows: unix_time, ch1..ch4 (µV)
    eeg_recordings/<id>/events.jsonl  one JSON object per browser event
    eeg_recordings/<id>/session.json  manifest + summary, written on finish
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import CHANNELS, MAX_EVENTS_PER_BATCH, N_CHANNELS, RECORDINGS_DIR, SAMPLE_RATE

EVENT_TYPES = frozenset({
    "ride_start", "ride_end", "key", "hazard", "collision", "speed", "decoder",
})
HAZARD_KINDS = frozenset({
    "red_light", "green_light", "fallen_tree", "rockfall", "shock", "deer", "lead_car",
})
HAZARD_PHASES = frozenset({"appear", "demand_stop", "clear", "end", "timeout"})
SESSION_ID = re.compile(r"^\d{8}_\d{6}(_\d+)?$")
ROW_WIDTH = 1 + N_CHANNELS


class SessionError(ValueError):
    """A session request or stored session was invalid."""


def _clean_value(value: object) -> object:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SessionError("event values must be finite numbers")
        return value
    if isinstance(value, str):
        return value[:64]
    raise SessionError("event values must be numbers, strings, booleans, or null")


def validate_event(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise SessionError("each event must be an object")
    kind = raw.get("type")
    if kind not in EVENT_TYPES:
        raise SessionError(f"unknown event type {kind!r}")
    stamp = raw.get("t")
    if not isinstance(stamp, (int, float)) or not math.isfinite(float(stamp)):
        raise SessionError("event t must be epoch milliseconds")
    if kind == "hazard":
        if raw.get("kind") not in HAZARD_KINDS or raw.get("phase") not in HAZARD_PHASES:
            raise SessionError("hazard events need a known kind and phase")
    event = {}
    for key, value in list(raw.items())[:16]:
        if not isinstance(key, str) or len(key) > 32:
            raise SessionError("event keys must be short strings")
        event[key] = _clean_value(value)
    return event


@dataclass
class Recorder:
    directory: Path
    kind: str
    minutes: int
    source: str = "unknown"  # "ganglion" or "simulated"
    started: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _samples: int = 0
    _events: int = 0
    _closed: bool = False

    @classmethod
    def create(cls, kind: str, minutes: int, root: Path = RECORDINGS_DIR, source: str = "unknown") -> "Recorder":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = root / stamp
        suffix = 1
        while directory.exists():
            directory = root / f"{stamp}_{suffix}"
            suffix += 1
        directory.mkdir(parents=True)
        recorder = cls(directory=directory, kind=kind, minutes=minutes, source=source)
        (directory / "eeg.f64").touch()
        (directory / "events.jsonl").touch()
        return recorder

    @property
    def session_id(self) -> str:
        return self.directory.name

    def write_eeg(self, eeg: np.ndarray, stamps: np.ndarray) -> None:
        rows = np.column_stack([stamps, eeg]).astype(np.float64)
        with self._lock:
            if self._closed:
                return
            with (self.directory / "eeg.f64").open("ab") as handle:
                handle.write(rows.tobytes())
            self._samples += len(stamps)

    def write_events(self, raw_events: list[object]) -> int:
        if not isinstance(raw_events, list) or len(raw_events) > MAX_EVENTS_PER_BATCH:
            raise SessionError(f"send a list of at most {MAX_EVENTS_PER_BATCH} events")
        events = [validate_event(item) for item in raw_events]
        with self._lock:
            if self._closed:
                raise SessionError("this ride is already saved")
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, separators=(",", ":")) + "\n")
            self._events += len(events)
        return len(events)

    def status(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "kind": self.kind,
            "elapsed_s": round(time.time() - self.started, 1),
            "samples": self._samples,
            "events": self._events,
        }

    def finish(self, summary: dict[str, object]) -> dict[str, object]:
        with self._lock:
            self._closed = True
        manifest = {
            "session_id": self.session_id,
            "kind": self.kind,
            "source": self.source,
            "minutes_planned": self.minutes,
            "created": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self.started, 1),
            "sample_rate": SAMPLE_RATE,
            "channels": list(CHANNELS),
            "samples": self._samples,
            "events": self._events,
            "excluded": False,
            "summary": summary,
        }
        _write_manifest(self.directory, manifest)
        return manifest


def _write_manifest(directory: Path, manifest: dict[str, object]) -> None:
    temporary = directory / "session.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(directory / "session.json")


def session_dir(session_id: str, root: Path = RECORDINGS_DIR) -> Path:
    if not SESSION_ID.match(session_id or ""):
        raise SessionError("invalid session id")
    directory = root / session_id
    if not (directory / "session.json").exists():
        raise SessionError("session not found")
    return directory


def load_eeg(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(directory / "eeg.f64", dtype=np.float64)
    rows = raw[: len(raw) - len(raw) % ROW_WIDTH].reshape(-1, ROW_WIDTH)
    return rows[:, 1:], rows[:, 0]


def load_events(directory: Path) -> list[dict[str, object]]:
    events = []
    with (directory / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return sorted(events, key=lambda event: float(event["t"]))


def list_sessions(root: Path = RECORDINGS_DIR) -> list[dict[str, object]]:
    if not root.exists():
        return []
    manifests = []
    for path in sorted(root.glob("*/session.json"), reverse=True):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return manifests


def set_excluded(session_id: str, excluded: bool, root: Path = RECORDINGS_DIR) -> dict[str, object]:
    directory = session_dir(session_id, root)
    manifest = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    updated = {**manifest, "excluded": bool(excluded)}
    _write_manifest(directory, updated)
    return updated

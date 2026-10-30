"""Logs every MAVLink message heard over the radio, one JSON line each.
A write error stops the recorder and is reported; it never affects the flight."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional


class MavlinkRecorder:
    def __init__(self, directory: Path, *, clock=time.time) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / time.strftime("mav_%Y%m%d_%H%M%S.jsonl")
        self._clock = clock
        self._lock = threading.Lock()
        self._file: Optional[object] = open(self.path, "a", buffering=1, encoding="utf-8")
        self.error = ""

    def record(self, msg) -> None:
        if self._file is None:
            return
        try:
            if hasattr(msg, "to_dict"):
                fields = {k: v for k, v in msg.to_dict().items() if k != "mavpackettype"}
            else:
                fields = {k: v for k, v in vars(msg).items() if not k.startswith("_")}
            row = {"t": round(self._clock(), 3), "msg": msg.get_type(),
                   "src": [msg.get_srcSystem(), msg.get_srcComponent()], "f": fields}
            line = json.dumps(row, default=str) + "\n"
            with self._lock:
                if self._file is not None:
                    self._file.write(line)
        except Exception as exc:  # stop recording, keep flying
            self.error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._file = None

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

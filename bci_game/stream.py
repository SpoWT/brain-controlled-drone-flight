"""EEG sources (Ganglion, simulator) and the one background reader that owns them.

Every consumer (signal check, recorder, drive decoder) reads from the reader,
so the headset is opened exactly once and STOP can always release it.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Protocol

import numpy as np

from bci_v2 import acquisition

from .config import BUFFER_SECONDS, MAINS_HZ, N_CHANNELS, PULL_INTERVAL_S, SAMPLE_RATE

Sink = Callable[[np.ndarray, np.ndarray], None]


class Source(Protocol):
    kind: str

    def read(self) -> tuple[np.ndarray, np.ndarray]: ...
    def describe(self) -> dict[str, object]: ...
    def close(self) -> None: ...


class SourceError(RuntimeError):
    """The headset could not be reached or stopped streaming."""


class GanglionSource:
    """A Ganglion opened by ``bci_v2.acquisition`` (the same connection the flying website uses)."""

    kind = "ganglion"

    def __init__(self, handle) -> None:
        self._handle = handle

    @classmethod
    def connect(cls) -> "GanglionSource":
        handle = acquisition.connect(simulate=False)
        if handle.sample_rate != SAMPLE_RATE:
            acquisition.release(handle)
            raise SourceError(f"Expected a {SAMPLE_RATE} Hz Ganglion; board reports {handle.sample_rate} Hz")
        return cls(handle)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Drain buffered samples -> (eeg (n, 4) in µV, unix timestamps (n,))."""
        chunk = acquisition.read_chunk(self._handle)
        if chunk.size == 0:
            return np.zeros((0, N_CHANNELS)), np.zeros(0)
        return acquisition.to_eeg_matrix(self._handle, chunk).copy(), chunk[self._handle.timestamp_row].copy()

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "transport": self._handle.transport, "sample_rate": SAMPLE_RATE}

    def close(self) -> None:
        acquisition.release(self._handle)


class SimulatedSource:
    """Synthetic EEG for running the game without a headset.

    Pink-ish noise plus a 10 Hz rhythm that grows while stopped and a 20 Hz
    rhythm that grows while driving, so training has a known effect to find.
    """

    kind = "simulated"

    def __init__(self, seed: int = 7) -> None:
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._driving = False
        self._last = time.time()
        self._phase = 0.0
        self._pink = np.zeros(N_CHANNELS)

    def set_driving(self, driving: bool) -> None:
        with self._lock:
            self._driving = bool(driving)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        now = time.time()
        with self._lock:
            n = int((now - self._last) * SAMPLE_RATE)
            if n <= 0:
                return np.zeros((0, N_CHANNELS)), np.zeros(0)
            start = self._last
            self._last = start + n / SAMPLE_RATE
            driving = self._driving
        stamps = start + np.arange(1, n + 1) / SAMPLE_RATE
        t = self._phase + np.arange(n) / SAMPLE_RATE
        self._phase = float(t[-1] + 1 / SAMPLE_RATE)
        white = self._rng.normal(0.0, 6.0, size=(n, N_CHANNELS))
        pink = np.empty_like(white)
        state = self._pink
        for i in range(n):  # one-pole low-pass -> 1/f-ish background
            state = 0.97 * state + white[i]
            pink[i] = state
        self._pink = state
        alpha_amp = 4.0 if driving else 12.0
        beta_amp = 6.0 if driving else 2.0
        gains = np.array([1.0, 0.8, 1.1, 0.9])
        rhythm = (
            alpha_amp * np.sin(2 * np.pi * 10.0 * t)[:, None]
            + beta_amp * np.sin(2 * np.pi * 20.0 * t + 0.7)[:, None]
        ) * gains
        line = 1.5 * np.sin(2 * np.pi * MAINS_HZ * t)[:, None]
        return 0.25 * pink + white * 0.5 + rhythm + line, stamps

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "transport": "synthetic", "sample_rate": SAMPLE_RATE}

    def close(self) -> None:
        return None


class EEGStream:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._source: Source | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        capacity = int(BUFFER_SECONDS * SAMPLE_RATE)
        self._eeg = np.zeros((capacity, N_CHANNELS))
        self._stamps = np.zeros(capacity)
        self._filled = 0
        self._sinks: dict[str, Sink] = {}
        self._state = "disconnected"
        self._error: str | None = None
        self._total_samples = 0
        self._last_sample_at = 0.0

    # -- lifecycle -------------------------------------------------------
    def attach(self, source: Source) -> None:
        self.detach()
        with self._lock:
            self._source = source
            self._filled = 0
            self._total_samples = 0
            self._error = None
            self._state = "streaming"
            self._last_sample_at = time.time()
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="eeg-reader", daemon=True)
            self._thread.start()

    def detach(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            source, self._source, self._thread = self._source, None, None
            self._sinks.clear()
            if self._state != "error":
                self._state = "disconnected"
        if source is not None:
            source.close()

    @property
    def source(self) -> Source | None:
        return self._source

    # -- sinks -----------------------------------------------------------
    def add_sink(self, name: str, sink: Sink) -> None:
        with self._lock:
            self._sinks[name] = sink

    def remove_sink(self, name: str) -> None:
        with self._lock:
            self._sinks.pop(name, None)

    # -- reading ---------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            source = self._source
            if source is None:
                return
            try:
                eeg, stamps = source.read()
            except Exception as exc:
                with self._lock:
                    self._state = "error"
                    self._error = str(exc)
                return
            if len(stamps):
                self._append(eeg, stamps)
            elif time.time() - self._last_sample_at > 3.0:
                with self._lock:
                    self._state = "stalled"
            self._stop.wait(PULL_INTERVAL_S)

    def _append(self, eeg: np.ndarray, stamps: np.ndarray) -> None:
        with self._lock:
            n = len(stamps)
            capacity = len(self._stamps)
            if n >= capacity:
                eeg, stamps, n = eeg[-capacity:], stamps[-capacity:], capacity
            self._eeg = np.roll(self._eeg, -n, axis=0)
            self._stamps = np.roll(self._stamps, -n)
            self._eeg[-n:] = eeg
            self._stamps[-n:] = stamps
            self._filled = min(capacity, self._filled + n)
            self._total_samples += n
            self._last_sample_at = time.time()
            self._state = "streaming"
            sinks = list(self._sinks.values())
        for sink in sinks:
            sink(eeg, stamps)

    def recent(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            n = min(self._filled, int(seconds * SAMPLE_RATE))
            if n == 0:
                return np.zeros((0, N_CHANNELS)), np.zeros(0)
            return self._eeg[-n:].copy(), self._stamps[-n:].copy()

    def status(self) -> dict[str, object]:
        with self._lock:
            source = self._source
            return {
                "state": self._state,
                "error": self._error,
                "source": source.describe() if source else None,
                "samples": self._total_samples,
                "seconds_since_sample": round(time.time() - self._last_sample_at, 2) if source else None,
            }

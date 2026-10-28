"""Application state: headset, current ride, drive decoder and training job.

Flask routes stay thin; every rule about what may happen when lives here.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from bci_game_runtime import continuous_window

from . import bundles, labels, session, train
from .config import (ALLOWED_RIDE_MINUTES, BUNDLES_DIR, GO, LIVE_STEP_S, QUALITY_WINDOW_S, RECORDINGS_DIR,
                     SAMPLE_RATE, STOP, WINDOW_SAMPLES)
from .dsp import channel_quality, downsample_trace
from .stream import EEGStream, GanglionSource, SimulatedSource, Source

LOG = logging.getLogger("bci_game")
STALE_S = 0.6
FRESH_S = 1.0


class StudioError(RuntimeError):
    """A request that is not allowed in the current state."""


def _default_runtime(folder: Path):
    from bci_game_runtime import BCIGameStopGoRuntime

    return BCIGameStopGoRuntime(folder)


class LiveDecoder:
    """Drive mode: feeds the latest EEG to a bundle every ``LIVE_STEP_S``, like the
    flying website does. Anything unclear is STOP."""

    def __init__(self, stream: EEGStream, runtime) -> None:
        self._stream = stream
        self._runtime = runtime
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state = STOP
        self._p_go = 0.0
        self._reason = "warming up"
        self._updated = 0.0
        self._thread = threading.Thread(target=self._run, name="live-decoder", daemon=True)

    def start(self) -> "LiveDecoder":
        self._runtime.reset()
        self._updated = time.time()  # "warming up" until the first decision
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        with self._lock:
            self._state, self._reason = STOP, "stopped"

    def _publish(self, state: str, p_go: float, reason: str) -> None:
        with self._lock:
            self._state, self._p_go, self._reason = state, p_go, reason
            self._updated = time.time()

    def _run(self) -> None:
        samples = int(getattr(self._runtime, "WINDOW_SAMPLES", WINDOW_SAMPLES))  # 7000 for a Random Forest
        while not self._stop.wait(LIVE_STEP_S):
            eeg, stamps = self._stream.recent(samples / SAMPLE_RATE * 1.2)
            if len(stamps) < samples:
                self._publish(STOP, 0.0, f"collecting {samples / SAMPLE_RATE:g} s of EEG")
                continue
            if time.time() - stamps[-1] > STALE_S:
                self._runtime.reset()
                self._publish(STOP, 0.0, "signal stalled")
                continue
            window = eeg[-samples:]
            if not np.all(np.isfinite(window)):
                self._runtime.reset()
                self._publish(STOP, 0.0, "invalid samples")
                continue
            if not continuous_window(stamps[-samples:], SAMPLE_RATE):
                self._runtime.reset()
                self._publish(STOP, 0.0, "signal gap")
                continue
            result = self._runtime.predict(window)
            p_go = float((result.get("probabilities") or {}).get("NOT_STOP", 0.0))
            if result.get("safe_fallback"):
                self._publish(STOP, p_go, result.get("reason") or "no decision")
            else:
                self._publish(GO if result["decision"] == "NOT_STOP" else STOP, p_go, "decoding")

    def status(self) -> dict[str, object]:
        with self._lock:
            fresh = time.time() - self._updated < FRESH_S
            return {
                "state": self._state if fresh else STOP,
                "p_go": round(self._p_go, 3),
                "reason": self._reason if fresh else "decoder stale",
            }


class Studio:
    def __init__(
        self,
        recordings_dir: Path = RECORDINGS_DIR,
        bundles_dir: Path = BUNDLES_DIR,
        ganglion_factory: Callable[[], Source] = GanglionSource.connect,
        runtime_factory: Callable[[Path], object] = _default_runtime,
    ) -> None:
        self.recordings_dir = recordings_dir
        self.bundles_dir = bundles_dir
        self._ganglion_factory = ganglion_factory
        self._runtime_factory = runtime_factory
        self._lock = threading.RLock()
        self._search_lock = threading.Lock()  # one headset search at a time; STOP never waits for it
        self.stream = EEGStream()
        self.recorder: session.Recorder | None = None
        self.live: LiveDecoder | None = None
        self._train_thread: threading.Thread | None = None
        self._train_state: dict[str, object] = {"state": "idle"}
        self._connects = 0  # bumped by every connect and by STOP; a stale search releases its headset

    # -- headset ---------------------------------------------------------
    def connect(self, mode: str) -> dict[str, object]:
        if mode not in ("ganglion", "simulated"):
            raise ValueError("mode must be 'ganglion' or 'simulated'")
        with self._lock:
            if self.recorder is not None:
                raise StudioError("Finish the ride before reconnecting.")
            self.stream.detach()
            self._connects += 1
            attempt = self._connects
        # The Ganglion search can take many seconds; status and STOP stay responsive meanwhile.
        with self._search_lock:
            if attempt != self._connects:
                raise StudioError("Connection cancelled.")
            source = SimulatedSource() if mode == "simulated" else self._ganglion_factory()
            with self._lock:
                if attempt != self._connects:
                    source.close()
                    raise StudioError("Connection cancelled.")
                self.stream.attach(source)
        return self.stream.status()

    def disconnect(self) -> dict[str, object]:
        self.stop_everything()
        return self.stream.status()

    def signal(self) -> dict[str, object]:
        eeg, _ = self.stream.recent(QUALITY_WINDOW_S * 2)
        quality = channel_quality(eeg[-int(QUALITY_WINDOW_S * SAMPLE_RATE):])
        return {"channels": quality, "trace": downsample_trace(eeg), "stream": self.stream.status()}

    def _require_streaming(self) -> None:
        if self.stream.status()["state"] != "streaming":
            raise StudioError("Connect the headset (or the simulator) first.")

    # -- rides -----------------------------------------------------------
    def start_ride(self, kind: str, minutes: int) -> dict[str, object]:
        if kind not in ("ride", "drive"):
            raise ValueError("kind must be 'ride' or 'drive'")
        if minutes not in ALLOWED_RIDE_MINUTES:
            raise ValueError(f"minutes must be one of {ALLOWED_RIDE_MINUTES}")
        with self._lock:
            self._require_streaming()
            if self.recorder is not None:
                raise StudioError("A ride is already running.")
            runtime = None
            if kind == "drive":
                folder = bundles.latest(self.bundles_dir)
                if folder is None:
                    raise StudioError("Train a model before driving with your mind.")
                if bundles.latest_report(self.bundles_dir).get("simulated") and self.stream.source.kind != "simulated":
                    raise StudioError("The newest model was trained on simulator rides; drive it with the simulator.")
                try:
                    runtime = self._runtime_factory(folder)
                except Exception as exc:
                    LOG.exception("bundle %s could not be loaded", folder.name)
                    raise StudioError(f"The model in {folder.name} could not be loaded: {exc}") from exc
            recorder = session.Recorder.create(kind, minutes, self.recordings_dir, source=self.stream.source.kind)
            self.stream.add_sink("recorder", recorder.write_eeg)
            self.recorder = recorder
            if runtime is not None:
                self.live = LiveDecoder(self.stream, runtime).start()
            return recorder.status()

    def add_events(self, events: list[object]) -> dict[str, object]:
        with self._lock:
            recorder = self.recorder
        if recorder is None:
            raise StudioError("No ride is running.")
        written = recorder.write_events(events)
        source = self.stream.source
        if isinstance(source, SimulatedSource):
            # In simulation the drive key stands in for the rider's intention.
            for event in events:
                if isinstance(event, dict) and event.get("type") == "key":
                    source.set_driving(bool(event.get("down")))
        return {"written": written}

    def finish_ride(self) -> dict[str, object]:
        with self._lock:
            recorder, self.recorder = self.recorder, None
            live, self.live = self.live, None
            self.stream.remove_sink("recorder")
        if live is not None:
            live.stop()
        if recorder is None:
            raise StudioError("No ride is running.")
        summary = labels.summarize(session.load_events(recorder.directory))
        return recorder.finish(summary)

    def live_status(self) -> dict[str, object]:
        live = self.live
        if live is None:
            return {"state": STOP, "p_go": 0.0, "reason": "not driving"}
        return live.status()

    def stop_everything(self) -> None:
        with self._lock:
            self._connects += 1
        if self.recorder is not None:
            try:
                self.finish_ride()
            except Exception:
                LOG.exception("could not save the ride during STOP; raw files are kept")
        self.stream.detach()

    # -- rides on file and training ----------------------------------------
    def sessions(self) -> list[dict[str, object]]:
        return session.list_sessions(self.recordings_dir)

    def exclude(self, session_id: str, excluded: bool) -> dict[str, object]:
        return session.set_excluded(session_id, excluded, self.recordings_dir)

    def start_training(self) -> dict[str, object]:
        with self._lock:
            if self._train_thread is not None and self._train_thread.is_alive():
                raise StudioError("Training is already running.")
            self._train_state = {"state": "running", "progress": "Starting"}
            self._train_thread = threading.Thread(target=self._train, name="trainer", daemon=True)
            self._train_thread.start()
            return dict(self._train_state)

    def _progress(self, text: str) -> None:
        with self._lock:
            self._train_state = {"state": "running", "progress": text}

    def _train(self) -> None:
        try:
            report = train.train(self.recordings_dir, self.bundles_dir, progress=self._progress)
            state = {"state": "done", "report": report}
        except train.TrainingError as exc:
            state = {"state": "failed", "error": str(exc)}
        except Exception as exc:  # surfaced to the page, never swallowed
            LOG.exception("training failed")
            state = {"state": "failed", "error": f"{type(exc).__name__}: {exc}"}
        with self._lock:
            self._train_state = state

    def training(self) -> dict[str, object]:
        with self._lock:
            state = dict(self._train_state)
        state["latest"] = bundles.latest_report(self.bundles_dir)
        return state

    def status(self) -> dict[str, object]:
        recorder = self.recorder
        with self._lock:
            training = self._train_state.get("state")
        return {
            "stream": self.stream.status(),
            "ride": recorder.status() if recorder else None,
            "live": self.live_status(),
            "model": bundles.latest_report(self.bundles_dir),
            "training": training,
        }

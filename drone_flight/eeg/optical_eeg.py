"""Live four-channel OpenBCI EEG and STOP/GO model output for the flying page.
Runs off the flight thread and never commands the drone itself."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from bci_game.runtime import MAX_SAMPLE_GAP_S, continuous_window
from drone_flight import ROOT
from drone_flight.eeg.saved_eeg_models import AUTO_MODEL_ID, SavedEEGModels


DEFAULT_SETTINGS = ROOT / "output" / "optical_eeg_settings.json"
LOGGER = logging.getLogger(__name__)
NO_MODEL_MESSAGE = "No STOP / GO model: put a trained bundle in eeg_bundles/"
NO_MODEL = {"id": None, "label": "No STOP / GO model", "group": "STOP / GO", "kind": "none",
            "class_names": [], "window_samples": 200, "stop_go": False, "note": NO_MODEL_MESSAGE}


def _threshold(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.01 <= value <= 0.99
    ):
        raise ValueError("STOP threshold must be between 1% and 99%")
    return float(value)


class LiveEEGService:
    CHANNELS = 4
    SAMPLE_RATE = 200
    WINDOW_SAMPLES = 200
    TRACE_SAMPLES = 1000
    HISTORY_SAMPLES = 7000
    FRESH_SECONDS = 0.75
    INFERENCE_INTERVAL = 0.25
    RECONNECT_SECONDS = 3.0
    MAX_GAP_SECONDS = MAX_SAMPLE_GAP_S

    def __init__(
        self,
        *,
        settings_path: Optional[Path] = DEFAULT_SETTINGS,
        connector: Optional[Callable] = None,
        model_factory: Optional[Callable] = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._clock = clock
        self._wall_clock = wall_clock
        self._connector = connector
        self._model_factory = model_factory
        self._settings_path = settings_path
        self._thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._enabled = False
        self._generation = 0
        self._threshold_revision = 0
        self._model_revision = 0
        self._sample_epoch = 0
        self._models = SavedEEGModels()
        self._selected_model_id = self._models.default_id()
        self._model_selection = AUTO_MODEL_ID if self._models.recommendation() else self._selected_model_id
        self._loaded_model_id = None
        self._model_loading = False
        # The selected bundle's own threshold until the user sets the STOP confidence.
        self._threshold = self._models.stop_threshold(self._selected_model_id)
        self._threshold_chosen = False
        self._model = None
        self._model_error = ""
        self._model_threshold = None
        self._state = "disconnected"
        self._message = "Waiting for OpenBCI"
        self._sample_rate = self.SAMPLE_RATE
        self._samples = np.empty((0, self.CHANNELS))
        self._accel = np.empty((0, 3))
        self._times = np.empty(0)
        self._last_received_at: Optional[float] = None
        self._sample_count = 0
        self._prediction = None
        self._prediction_at = None
        if settings_path is not None:
            try:
                settings = json.loads(Path(settings_path).read_text())
                if not isinstance(settings, dict):
                    raise ValueError("EEG settings must be a JSON object")
                if settings.get("threshold_chosen") is not False:  # older files: a saved value was chosen
                    self._threshold = _threshold(settings["stop_threshold"])
                    self._threshold_chosen = True
                if settings.get("model_id") in self._models.specs:
                    self._selected_model_id = settings["model_id"]
                    self._model_selection = self._selected_model_id
                if settings.get('model_selection') == AUTO_MODEL_ID:
                    self._selected_model_id = self._models.resolve_selection(AUTO_MODEL_ID)
                    self._model_selection = AUTO_MODEL_ID
            except (OSError, KeyError, TypeError, ValueError):
                pass
            if not self._threshold_chosen:
                self._threshold = self._models.stop_threshold(self._selected_model_id)

    def _clear_locked(self) -> None:
        self._sample_epoch += 1
        self._samples = np.empty((0, self.CHANNELS))
        self._accel = np.empty((0, 3))
        self._times = np.empty(0)
        self._last_received_at = None
        self._sample_count = 0
        self._prediction = None
        self._prediction_at = None

    def connect(self) -> dict:
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("EEG service is closed")
            if not self._enabled:
                self._enabled = True
                self._generation += 1
                self._clear_locked()
                self._state = "connecting"
                self._message = "Looking for OpenBCI"
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._worker, name="drone-live-eeg", daemon=True)
                self._thread.start()
            if self._inference_thread is None or not self._inference_thread.is_alive():
                self._inference_thread = threading.Thread(target=self._model_worker, name="drone-eeg-inference", daemon=True)
                self._inference_thread.start()
        return self.status()

    def disconnect(self) -> dict:
        with self._lock:
            self._enabled = False
            self._generation += 1
            self._clear_locked()
            self._state = "disconnected"
            self._message = "EEG disconnected"
        return self.status()

    def set_threshold(self, value: object) -> dict:
        threshold = _threshold(value)
        with self._lock:
            self._save_settings_locked(threshold=threshold, threshold_chosen=True)
            self._threshold = threshold
            self._threshold_chosen = True
            self._threshold_revision += 1
            self._prediction = None
        return self.status()

    def _save_settings_locked(self, *, threshold=None, threshold_chosen=None, model_id=None, model_selection=None):
        if self._settings_path is not None:
            path = Path(self._settings_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"stop_threshold": self._threshold if threshold is None else threshold,
                                              "threshold_chosen": self._threshold_chosen if threshold_chosen is None else threshold_chosen,
                                              "model_id": self._selected_model_id if model_id is None else model_id,
                                              "model_selection": self._model_selection if model_selection is None else model_selection}) + "\n")
            temporary.replace(path)

    def model_catalog(self):
        return self._models.catalog()

    def select_model(self, model_id: object) -> dict:
        selected_id = self._models.resolve_selection(model_id)
        with self._lock:
            threshold = self._threshold if self._threshold_chosen else self._models.stop_threshold(selected_id)
            self._save_settings_locked(threshold=threshold, model_id=selected_id, model_selection=model_id)
            if threshold != self._threshold:
                self._threshold = threshold
                self._threshold_revision += 1
            self._selected_model_id = selected_id
            self._model_selection = model_id
            self._model_revision += 1
            self._model_loading = True
            self._model_error = ""
            self._prediction = None
            self._prediction_at = None
        return self.status()

    def _wanted(self, generation: int) -> bool:
        with self._lock:
            return self._enabled and self._generation == generation and not self._closed.is_set()

    def _publish_state(self, generation: int, state: str, message: str) -> None:
        with self._lock:
            if self._enabled and self._generation == generation:
                self._state = state
                self._message = message
                if state != "streaming":
                    self._prediction = None

    def _accept_chunk(self, generation: int, eeg: np.ndarray, times: np.ndarray, sample_rate: int, accel=None) -> None:
        eeg = np.asarray(eeg, dtype=float)
        times = np.asarray(times, dtype=float)
        accel = np.full((len(times), 3), np.nan) if accel is None else np.asarray(accel, dtype=float)
        if eeg.shape != (len(times), self.CHANNELS):
            raise ValueError("Expected four EEG channels with a timestamp for each sample")
        if accel.shape != (len(times), 3):
            raise ValueError("Accelerometer samples must match the EEG samples")
        if not len(times):
            return
        if not np.isfinite(eeg).all() or not np.isfinite(times).all():
            raise ValueError("EEG contains invalid samples")
        if np.any(np.diff(times) < 0):
            raise ValueError("EEG timestamps are out of order")
        sample_lag = self._wall_clock() - times[-1]
        if sample_lag > 2.0 or sample_lag < -2.0:
            raise ValueError("EEG samples are delayed")
        with self._lock:
            if not self._enabled or self._generation != generation:
                return
            if len(self._times) and times[-1] <= self._times[-1]:
                return  # repeated chunk
            if len(self._times) and times[0] <= self._times[-1]:
                fresh = times > self._times[-1]
                eeg, times, accel = eeg[fresh], times[fresh], accel[fresh]
            if len(self._times) and times[0] - self._times[-1] > self.MAX_GAP_SECONDS:
                self._sample_epoch += 1
                self._samples = np.empty((0, self.CHANNELS))
                self._accel = np.empty((0, 3))
                self._times = np.empty(0)
                self._prediction = None
                self._model_threshold = None  # reset voting after a gap
            self._samples = np.concatenate((self._samples, eeg))[-self.HISTORY_SAMPLES:]
            self._accel = np.concatenate((self._accel, accel))[-self.HISTORY_SAMPLES:]
            self._times = np.concatenate((self._times, times))[-self.HISTORY_SAMPLES:]
            self._last_received_at = self._clock() - max(0.0, sample_lag)
            self._sample_count += len(times)
            self._sample_rate = int(sample_rate)
            self._state = "streaming"
            self._message = "Live EEG"

    def _infer_once(self, generation: int) -> None:
        with self._lock:
            if not self._enabled or self._generation != generation:
                return
            age = None if self._last_received_at is None else self._clock() - self._last_received_at
            model = self._model
            count = getattr(model, "WINDOW_SAMPLES", self.WINDOW_SAMPLES)
            if (self._model_loading or age is None or age > self.FRESH_SECONDS or len(self._samples) < count):
                self._prediction = None
                return
            eeg = self._samples[-count:].copy()
            accel = self._accel[-count:].copy()
            times = self._times[-count:].copy()
            sample_rate = self._sample_rate
            threshold, revision = self._threshold, self._threshold_revision
            model_revision = self._model_revision
            sample_epoch = self._sample_epoch
            observed_at = self._last_received_at
        if model is None:
            return
        if self._model_threshold != threshold:
            model.set_stop_threshold(threshold)
            self._model_threshold = threshold
        if (sample_rate != self.SAMPLE_RATE or not continuous_window(times, self.SAMPLE_RATE)
                or np.any(np.std(eeg - np.median(eeg, axis=0), axis=0) < 0.05)):
            model.reset()
            prediction = {"valid": False, "decision": None, "reason": "Check the four electrode contacts"}
        else:
            result = model.predict(eeg, accel=accel) if getattr(model, "needs_accelerometer", False) else model.predict(eeg)
            if result.get("output_kind") == "score":
                score = result.get("score")
                prediction = {"valid": isinstance(score, (int, float)) and math.isfinite(score),
                              "decision": None, "output_kind": "score", "score": float(score) if isinstance(score, (int, float)) and math.isfinite(score) else None,
                              "reason": result.get("reason", ""), "quality": result.get("quality", "unknown")}
                probabilities = None
            else:
                probabilities = result.get("probabilities") or {}
            if probabilities is not None:
                valid = (
                    bool(probabilities) and not result.get("safe_fallback") and result.get("quality") != "reject"
                    and result.get("raw_decision") in probabilities
                    and all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
                    and abs(sum(probabilities.values()) - 1) < 0.002
                )
                if valid:
                    normalized = {("GO" if name == "NOT_STOP" else name): float(value) for name, value in probabilities.items()}
                    prediction = {"valid": True, "decision": "GO" if result["decision"] == "NOT_STOP" else result["decision"],
                                  "probabilities": normalized, "quality": result.get("quality", "unknown"), "reason": ""}
                    if set(normalized) == {"STOP", "GO"}:
                        prediction.update(stop_probability=normalized["STOP"], go_probability=normalized["GO"])
                else:
                    model.reset()
                    prediction = {"valid": False, "decision": None, "reason": "Check electrode contact — signal not ready for prediction"}
        with self._lock:
            age = self._clock() - self._last_received_at if self._last_received_at is not None else float("inf")
            if (self._enabled and self._generation == generation and revision == self._threshold_revision
                    and model_revision == self._model_revision and age <= self.FRESH_SECONDS
                    and sample_epoch == self._sample_epoch
                    and self._clock() - observed_at <= self.FRESH_SECONDS):
                self._prediction = prediction
                self._prediction_at = observed_at

    def _load_selected_model(self):
        with self._lock:
            model_id, revision = self._selected_model_id, self._model_revision
            if self._loaded_model_id == model_id and not self._model_loading:
                return
            self._model_loading = True
        try:
            model = self._model_factory() if self._model_factory is not None else self._models.load(model_id)
            error = ""
        except Exception as exc:
            model = None
            detail = f": {str(exc)[:160]}" if isinstance(exc, (RuntimeError, ImportError)) else ""
            error = f"This saved model could not be loaded ({type(exc).__name__}{detail})"
            LOGGER.exception("Selected EEG model could not be loaded")
        with self._lock:
            if revision == self._model_revision:
                self._model = model
                self._loaded_model_id = model_id
                self._model_loading = False
                self._model_error = error
                self._model_threshold = None
                self._prediction = None

    def _model_worker(self):
        while not self._closed.is_set():
            with self._lock:
                enabled, generation, revision = self._enabled, self._generation, self._model_revision
            if enabled:
                self._load_selected_model()
                try:
                    self._infer_once(generation)
                except Exception as exc:
                    with self._lock:
                        if self._enabled and self._generation == generation and self._model_revision == revision:
                            self._prediction = {"valid": False, "decision": None,
                                                "reason": str(exc) if isinstance(exc, ValueError) else "This saved model could not predict this EEG window"}
                            self._prediction_at = self._last_received_at
                    LOGGER.exception("Selected EEG model inference failed")
            self._closed.wait(self.INFERENCE_INTERVAL)

    def _worker(self) -> None:
        from bci_game.acquisition import connect, read_chunk, release, to_eeg_matrix

        while not self._closed.is_set():
            with self._lock:
                enabled, generation = self._enabled, self._generation
            if not enabled:
                self._closed.wait(0.1)
                continue
            handle = None
            try:
                self._publish_state(generation, "connecting", "Looking for OpenBCI")
                handle = self._connector() if self._connector is not None else connect(simulate=False)
                if handle.transport == "synthetic":
                    raise ValueError("Only a real OpenBCI stream is allowed")
                if not self._wanted(generation):
                    continue
                with self._lock:
                    self._clear_locked()
                    self._model_threshold = None
                while self._wanted(generation):
                    chunk = read_chunk(handle)
                    eeg = to_eeg_matrix(handle, chunk)
                    if len(eeg):
                        accel = chunk[list(handle.accel_rows), :].T if len(handle.accel_rows) == 3 else None
                        self._accept_chunk(generation, eeg, chunk[handle.timestamp_row], handle.sample_rate, accel=accel)
                    self._closed.wait(0.025)
            except Exception as exc:
                LOGGER.warning("OpenBCI connection waiting: %s", type(exc).__name__)
                self._publish_state(generation, "waiting", "Waiting for OpenBCI — turn on the headset")
                with self._lock:
                    if self._generation == generation:
                        self._clear_locked()
                release(handle)
                handle = None
                deadline = self._clock() + self.RECONNECT_SECONDS
                while self._wanted(generation) and self._clock() < deadline:
                    self._closed.wait(0.1)
            finally:
                release(handle)

    def status(self) -> dict:
        with self._lock:
            age = None if self._last_received_at is None else max(0.0, self._clock() - self._last_received_at)
            fresh = self._enabled and age is not None and age <= self.FRESH_SECONDS
            state, message = self._state, self._message
            if self._enabled and age is not None and not fresh:
                state, message = "stale", "Waiting for fresh EEG"
            prediction_fresh = self._prediction_at is not None and self._clock() - self._prediction_at <= self.FRESH_SECONDS
            prediction = self._prediction if fresh and prediction_fresh else None
            selected = self._models.specs.get(self._selected_model_id, NO_MODEL)
            if prediction is None:
                needed = selected["window_samples"]
                seconds = needed / self.SAMPLE_RATE
                reason = self._model_error or (NO_MODEL_MESSAGE if selected is NO_MODEL
                         else message if not self._enabled else "Loading selected model" if self._model_loading
                         else f"Collecting {seconds:g} seconds of EEG ({min(len(self._samples), needed)}/{needed} samples)" if fresh and len(self._samples) < needed
                         else "Waiting for model prediction" if fresh else message)
                prediction = {"valid": False, "decision": None, "reason": reason}
            return {
                "enabled": self._enabled, "state": state, "message": message,
                "source": "real", "sample_rate": self._sample_rate,
                "units": "µV", "channel_names": ["CH1", "CH2", "CH3", "CH4"],
                "eeg": self._samples[-self.TRACE_SAMPLES:].T.round(3).tolist() if fresh else [[], [], [], []],
                "trace_samples": self.TRACE_SAMPLES, "retained_samples": len(self._samples),
                "timestamp_span_seconds": round(float(self._times[-1] - self._times[0]), 3) if len(self._times) > 1 else 0,
                "sample_count": self._sample_count, "age_ms": None if age is None else round(age * 1000),
                "prediction": dict(prediction), "stop_threshold": self._threshold,
                "model_id": self._selected_model_id, "model_name": selected["label"],
                "model_selection": self._model_selection,
                "model_recommendation": self._models.recommendation(),
                "selected_model": self._models.public(selected),
                "model_loading": self._model_loading, "model_ready": self._model is not None and not self._model_loading,
            }

    def close(self) -> None:
        self.disconnect()
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5)


__all__ = ["LiveEEGService"]

"""Live STOP / NOT_STOP inference: quality gate, model, voting and hysteresis.

Runs every model type training can produce: XGBoost and LightGBM on the 487
window features, the Random Forest on causal band-power features of the past
35 s, and the CfC / LTC / GRU networks on the filtered window.
Outputs STOP, NOT_STOP or NO_DECISION (artifact or error).
"""

import json
import os
import time

import joblib
import numpy as np

from . import rf_features
from .features import feature_vector

NEURAL_ERROR = ""
try:
    from . import neural
except Exception as exc:  # only the CfC / LTC / GRU bundles need PyTorch; the others must load without it
    neural, NEURAL_ERROR = None, f"{type(exc).__name__}: {exc}"

ARTIFACT_THRESHOLD_UV = 150
ARTIFACT_REJECT_FRACTION = 0.3
ARTIFACT_MODERATE_FRACTION = 0.1
FLAT_STD_UV = 0.05
WINDOW_SAMPLES = 200
NEURAL_MODELS = ("cfc", "ltc", "gru")


def window_quality(eeg_window):
    """'reject', 'moderate' or 'clean' from the share of samples more than 150 µV from their channel's median
    (the median removes the electrode offset, which on a Ganglion is often 100 µV or more). A flat channel
    (a lost or railed electrode) is 'reject'."""
    if np.any(np.std(eeg_window, axis=0) < FLAT_STD_UV):
        return "reject"
    centered = eeg_window - np.median(eeg_window, axis=0)
    above = float(np.mean(np.abs(centered) > ARTIFACT_THRESHOLD_UV))
    if above >= ARTIFACT_REJECT_FRACTION:
        return "reject"
    if above >= ARTIFACT_MODERATE_FRACTION:
        return "moderate"
    return "clean"


class InferenceEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.class_names = ["NOT_STOP", "STOP"]
        self.channels = ["CH1", "CH2", "CH3", "CH4"]
        self.model_type = None
        self.accuracy = None
        self.stop_threshold = 0.5
        self.mains_hz = 60
        self.window_samples = WINDOW_SAMPLES  # samples each prediction needs
        self.uses_asr = True  # the Random Forest filters raw EEG itself
        self.vote_window = 3
        self.min_state_duration_ms = 500
        self._network = None
        self._settings, self._mean, self._std = {}, None, None
        self._vote_buffer = []
        self._last_state = "STOP"
        self._last_change_ms = 0

    def load(self, model_dir, model_id):
        """Load ``config_<id>.json`` and the files it names from ``model_dir``."""
        with open(os.path.join(model_dir, f"config_{model_id}.json")) as handle:
            config = json.load(handle)
        self.feature_names = list(config.get("feature_names", []))
        self.class_names = list(config["class_names"])
        self.channels = list(config.get("channels", self.channels))
        self.model_type = config.get("model_type")
        self.accuracy = config.get("accuracy")
        self.stop_threshold = float(config.get("stop_threshold", 0.5))
        self.mains_hz = float(config.get("mains_hz", 60))
        self.window_samples = int(config.get("window_samples", WINDOW_SAMPLES))
        self.uses_asr = bool(config.get("uses_asr", True))
        artifact = os.path.join(model_dir, config["artifact_filename"])
        if self.model_type in NEURAL_MODELS:
            self._load_network(artifact, config)
            return
        self.model = joblib.load(artifact)
        if self.model_type == "random_forest":
            if self.model.n_features_in_ != len(self.feature_names):
                raise ValueError("Model and feature list do not match")
            return
        self.scaler = joblib.load(os.path.join(model_dir, config["scaler_artifact_filename"]))
        if self.scaler.n_features_in_ != len(self.feature_names):
            raise ValueError("Scaler and feature list do not match")

    def _load_network(self, path, config):
        if neural is None:
            raise RuntimeError(f"This bundle needs PyTorch (pip install -r requirements.txt): {NEURAL_ERROR}")
        self._settings = {**neural.DEFAULTS, **config["network"]}
        self._mean = np.asarray(config["norm_mean"], dtype=np.float32)
        self._std = np.asarray(config["norm_std"], dtype=np.float32)
        self._network = neural.build_model(self.model_type, self._settings["patch_size"] * len(self._mean),
                                           self._settings)
        self._network.load_state_dict(neural.torch.load(path, map_location="cpu", weights_only=True))
        self._network.eval()

    def stop_probability(self, eeg_window):
        """P(STOP) for one window of ``window_samples`` samples, without voting."""
        if self.model_type in NEURAL_MODELS:
            windows = neural.preprocess_windows(eeg_window[None], self.mains_hz)
            patches = neural.network_input(windows, self._mean, self._std, self._settings["patch_size"])
            return float(neural.stop_probabilities(self._network, patches)[0])
        if self.model_type == "random_forest":
            features, _ = rf_features.latest_features(eeg_window, self.channels, self.mains_hz)
            if not np.isfinite(features).all():
                raise ValueError("Collecting 5 s of history for the Random Forest baseline")
            probabilities = self.model.predict_proba(features)[0]
        else:
            scaled = self.scaler.transform(feature_vector(eeg_window, self.feature_names, self.mains_hz))
            probabilities = self.model.predict_proba(scaled)[0]
        return float(probabilities[self.class_names.index("STOP")])

    def predict(self, eeg_window):
        if self.model is None and self._network is None:
            return {"decision": "NO_DECISION", "reason": "no_model"}
        quality = window_quality(eeg_window[-WINDOW_SAMPLES:])
        if quality == "reject":
            return {"decision": "NO_DECISION", "reason": "artifact", "quality": quality}
        try:
            stop = self.stop_probability(eeg_window)
        except Exception as exc:
            return {"decision": "NO_DECISION", "reason": f"{type(exc).__name__}: {exc}", "quality": quality}
        raw_decision = "STOP" if stop >= self.stop_threshold else "NOT_STOP"
        self._vote_buffer = (self._vote_buffer + [raw_decision])[-self.vote_window:]
        decision = self._vote()
        now_ms = time.time() * 1000
        if decision != self._last_state:
            if now_ms - self._last_change_ms < self.min_state_duration_ms:
                decision = self._last_state
            else:
                self._last_state, self._last_change_ms = decision, now_ms
        return {
            "decision": decision,
            "raw_decision": raw_decision,
            "confidence": stop if raw_decision == "STOP" else 1.0 - stop,
            "probabilities": {"NOT_STOP": 1.0 - stop, "STOP": stop},
            "quality": quality,
            "vote_buffer": list(self._vote_buffer),
            "model_type": self.model_type,
        }

    def _vote(self):
        if len(self._vote_buffer) < self.vote_window:
            return self._last_state
        stops = sum(vote == "STOP" for vote in self._vote_buffer)
        return "STOP" if stops > len(self._vote_buffer) / 2 else "NOT_STOP"

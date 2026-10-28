"""CfC, LTC and GRU STOP / GO classifiers on 1 s windows.

Each window is band-passed 0.5-45 Hz, mains-notched and common-average referenced,
normalized with the training rides' channel mean and std, and cut into 20 patches
of 10 samples (20 time steps of 40 values) for the recurrent network.
"""

from __future__ import annotations

import importlib
import math

import numpy as np

# PyTorch, XGBoost and LightGBM each ship an OpenMP runtime, and two runtimes in one process
# crash or deadlock on macOS. Load the boosters' first and keep PyTorch on one thread, which
# never enters OpenMP; these networks are small enough for that.
for _booster in ("xgboost", "lightgbm"):
    try:
        importlib.import_module(_booster)
    except Exception:  # not installed or not loadable: nothing to order
        pass

import torch  # noqa: E402

torch.set_num_threads(1)
import torch.nn.functional as functional
from scipy.signal import butter, filtfilt, iirnotch
from torch import nn

SAMPLE_RATE = 200
MODELS = ("cfc", "ltc", "gru")
DEFAULTS = {"hidden_size": 32, "patch_size": 10, "ode_unfolds": 6,
            "cfc_backbone_units": 128, "cfc_backbone_layers": 1, "cfc_backbone_dropout": 0.0}


def preprocess_windows(eeg, mains_hz=60):
    """(n, 200, 4) µV -> filtered, re-referenced float32 windows."""
    band_b, band_a = butter(4, [0.5, 45.0], btype="band", fs=SAMPLE_RATE)
    notch_b, notch_a = iirnotch(float(mains_hz), 30.0, SAMPLE_RATE)
    filtered = filtfilt(band_b, band_a, np.asarray(eeg, dtype=np.float64), axis=1)
    filtered = filtfilt(notch_b, notch_a, filtered, axis=1)
    filtered -= filtered.mean(axis=2, keepdims=True)
    return filtered.astype(np.float32)


def channel_stats(eeg):
    """Per-channel mean and std over every window and sample (std never below 1e-6)."""
    mean = eeg.mean(axis=(0, 1), dtype=np.float64)
    std = eeg.std(axis=(0, 1), dtype=np.float64)
    return mean.astype(np.float32), np.where(std < 1e-6, 1.0, std).astype(np.float32)


def make_patches(eeg, patch_size):
    """(n, 200, channels) -> (n, 200 / patch_size, patch_size * channels)."""
    n_windows, n_samples, n_channels = eeg.shape
    if n_samples % patch_size:
        raise ValueError(f"Window length {n_samples} is not divisible by patch size {patch_size}")
    patches = eeg.reshape(n_windows, n_samples // patch_size, patch_size, n_channels)
    return np.ascontiguousarray(patches.reshape(n_windows, n_samples // patch_size, patch_size * n_channels))


def network_input(windows, mean, std, patch_size):
    """Preprocessed windows -> normalized patches ready for the network."""
    return make_patches(((windows - mean) / std).astype(np.float32), patch_size)


class DenseLTCCell(nn.Module):
    """Liquid time-constant cell (Hasani et al.), fused semi-implicit solver."""

    def __init__(self, input_size, hidden_size, ode_unfolds=6):
        super().__init__()
        if ode_unfolds < 1:
            raise ValueError("ode_unfolds must be at least 1")
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds
        self.input_map = nn.Linear(input_size, hidden_size)
        self.recurrent_map = nn.Linear(hidden_size, hidden_size, bias=False)
        self.raw_tau = nn.Parameter(torch.full((hidden_size,), math.log(math.expm1(1.0))))  # tau starts at 1
        self.reversal_potential = nn.Parameter(torch.empty(hidden_size))
        nn.init.xavier_uniform_(self.input_map.weight, gain=0.5)
        nn.init.zeros_(self.input_map.bias)
        nn.init.orthogonal_(self.recurrent_map.weight, gain=0.5)
        nn.init.uniform_(self.reversal_potential, -1.0, 1.0)

    def advance(self, projected_input, state):
        dt = 1.0 / self.ode_unfolds
        inverse_tau = (functional.softplus(self.raw_tau) + 1e-4).reciprocal()
        for _ in range(self.ode_unfolds):
            conductance = torch.sigmoid(projected_input + self.recurrent_map(state))
            state = (state + dt * conductance * self.reversal_potential) / (1.0 + dt * (inverse_tau + conductance))
        return state


class LTCClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, ode_unfolds=6, n_classes=2):
        super().__init__()
        self.cell = DenseLTCCell(input_size, hidden_size, ode_unfolds)
        self.output = nn.Linear(hidden_size, n_classes)

    def forward(self, sequence):
        state = sequence.new_zeros((sequence.shape[0], self.cell.hidden_size))
        projected = self.cell.input_map(sequence)
        for step in range(projected.shape[1]):
            state = self.cell.advance(projected[:, step], state)
        return self.output(state)


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes=2):
        super().__init__()
        self.recurrent = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, n_classes)

    def forward(self, sequence):
        _, state = self.recurrent(sequence)
        return self.output(state[-1])


class CfCClassifier(nn.Module):
    """Closed-form continuous-time network from the ncps package."""

    def __init__(self, input_size, hidden_size, backbone_units=128, backbone_layers=1, backbone_dropout=0.0,
                 n_classes=2):
        super().__init__()
        from ncps.torch import CfC

        self.recurrent = CfC(input_size=input_size, units=hidden_size, return_sequences=False, batch_first=True,
                             mixed_memory=False, mode="default", backbone_units=backbone_units,
                             backbone_layers=backbone_layers, backbone_dropout=backbone_dropout)
        self.output = nn.Linear(hidden_size, n_classes)

    def forward(self, sequence):
        final, _ = self.recurrent(sequence)
        return self.output(final)


def build_model(name, input_size, settings):
    """A fresh network; ``settings`` holds the ``DEFAULTS`` keys."""
    if name == "cfc":
        return CfCClassifier(input_size, settings["hidden_size"], settings["cfc_backbone_units"],
                             settings["cfc_backbone_layers"], settings["cfc_backbone_dropout"])
    if name == "ltc":
        return LTCClassifier(input_size, settings["hidden_size"], settings["ode_unfolds"])
    if name == "gru":
        return GRUClassifier(input_size, settings["hidden_size"])
    raise ValueError(f"Unknown network {name!r}")


def stop_probabilities(model, patches, batch_size=256):
    """P(STOP) for each row of ``patches``."""
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(patches), batch_size):
            logits = model(torch.from_numpy(patches[start:start + batch_size]))
            chunks.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

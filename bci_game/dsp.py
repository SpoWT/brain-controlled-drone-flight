"""Signal check: per-channel quality and the live trace."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import CHANNELS, FLAT_UV, LINE_RATIO_MAX, MAINS_HZ, NOISY_UV, SAMPLE_RATE


@lru_cache(maxsize=None)
def _bandpass_sos(low: float, high: float) -> np.ndarray:
    from scipy import signal  # lazy: keeps server start-up fast
    return signal.butter(4, [low, high], btype="bandpass", fs=SAMPLE_RATE, output="sos")


@lru_cache(maxsize=None)
def _notch() -> tuple[np.ndarray, np.ndarray]:
    from scipy import signal
    return signal.iirnotch(float(MAINS_HZ), 30.0, fs=SAMPLE_RATE)


def clean(eeg: np.ndarray, low: float = 1.0, high: float = 40.0) -> np.ndarray:
    """Zero-phase mains notch + band-pass. ``eeg`` is (n_samples, n_channels)."""
    from scipy import signal
    if eeg.shape[0] < 3 * SAMPLE_RATE // 4:
        return eeg - eeg.mean(axis=0, keepdims=True)
    b, a = _notch()
    out = signal.filtfilt(b, a, eeg, axis=0)
    return signal.sosfiltfilt(_bandpass_sos(low, high), out, axis=0)


def channel_quality(eeg: np.ndarray) -> list[dict[str, object]]:
    """Grade each channel CLEAN / NOISY / FLAT / WAIT from the last seconds of raw data."""
    from scipy import signal
    if eeg.shape[0] < SAMPLE_RATE:
        return [{"name": name, "state": "WAIT", "rms_uv": None, "line_ratio": None} for name in CHANNELS]
    filtered = clean(eeg)
    rms = filtered.std(axis=0)
    freqs, psd = signal.welch(eeg - eeg.mean(axis=0), fs=SAMPLE_RATE, nperseg=SAMPLE_RATE, axis=0)
    line = psd[(freqs >= MAINS_HZ - 2) & (freqs <= MAINS_HZ + 2)].mean(axis=0)
    brain = psd[(freqs >= 2) & (freqs <= 40)].mean(axis=0) + 1e-12
    ratios = line / brain
    report = []
    for index, name in enumerate(CHANNELS):
        value = float(rms[index])
        ratio = float(ratios[index])
        if value < FLAT_UV:
            state = "FLAT"
        elif value > NOISY_UV or ratio > LINE_RATIO_MAX:
            state = "NOISY"
        else:
            state = "CLEAN"
        report.append({"name": name, "state": state, "rms_uv": round(value, 1), "line_ratio": round(ratio, 2)})
    return report


def downsample_trace(eeg: np.ndarray, points: int = 400) -> list[list[float]]:
    """Per-channel thinning for the live trace display."""
    if eeg.shape[0] == 0:
        return [[] for _ in CHANNELS]
    centered = eeg - np.median(eeg, axis=0)
    step = max(1, eeg.shape[0] // points)
    thinned = centered[::step]
    return [np.round(thinned[:, ch], 1).tolist() for ch in range(eeg.shape[1])]

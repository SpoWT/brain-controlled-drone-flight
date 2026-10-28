"""Causal EEG features for the Random Forest STOP / GO model.

Band power and time features of the latest second, plus the band powers relative
to their median over the past 30 s. Every feature at a decision point uses only
samples at or before it, so training and live use share this code. Decisions are
on a 125 ms grid; the filter runs on raw EEG (no ASR).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal, stats

SAMPLE_RATE = 200
WINDOW_SAMPLES = 200
STEP_SAMPLES = 25  # a decision every 125 ms
BANDPASS_HZ = (0.5, 40.0)
BASELINE_PAST_POINTS = 240  # 30 s of grid points
BASELINE_SKIP_POINTS = 16  # leave out the most recent 2 s
BASELINE_MIN_POINTS = 40  # at least 5 s of history
HISTORY_SAMPLES = 7000  # 35 s: a full baseline plus the window
BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta1", 13.0, 20.0),
    ("beta2", 20.0, 30.0),
    ("gamma", 30.0, 40.0),
)
EPS = 1e-12
_PAIRS = ((0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3))


def causal_filter(eeg, mains_hz=60):
    """Band-pass and mains notch, forward only: output sample k uses input samples <= k."""
    band = signal.butter(2, BANDPASS_HZ, btype="bandpass", fs=SAMPLE_RATE, output="sos")
    notch_b, notch_a = signal.iirnotch(float(mains_hz), Q=30.0, fs=SAMPLE_RATE)
    sos = np.vstack([band, signal.tf2sos(notch_b, notch_a)])
    zi = signal.sosfilt_zi(sos)[:, :, None] * eeg[0][None, None, :]
    filtered, _ = signal.sosfilt(sos, eeg, axis=0, zi=zi)
    return filtered


def grid_ends(n_samples):
    """Exclusive end index of every window on the grid of a recording ``n_samples`` long."""
    return np.arange(WINDOW_SAMPLES, n_samples + 1, STEP_SAMPLES)


def _windows(filtered, end_index):
    """(n_points, n_channels, WINDOW_SAMPLES), each ending just before ``end_index``."""
    if end_index.min() < WINDOW_SAMPLES or end_index.max() > len(filtered):
        raise ValueError("Decision point without a full past window")
    return sliding_window_view(filtered, WINDOW_SAMPLES, axis=0)[end_index - WINDOW_SAMPLES]


def _band_features(windows):
    freqs, psd = signal.welch(windows, fs=SAMPLE_RATE, nperseg=100, noverlap=50, axis=-1)
    total = psd[..., (freqs >= 1.0) & (freqs < 40.0)].sum(axis=-1) + EPS
    columns, names = [], []
    for band, low, high in BANDS:
        power = psd[..., (freqs >= low) & (freqs < high)].sum(axis=-1) + EPS
        columns += [np.log10(power), power / total]
        names += [f"{band}_logpow", f"{band}_relpow"]
    return columns, names


def _hjorth(windows):
    d1 = np.diff(windows, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var0, var1, var2 = windows.var(-1) + EPS, d1.var(-1) + EPS, d2.var(-1) + EPS
    mobility = np.sqrt(var1 / var0)
    return mobility, np.sqrt(var2 / var1) / mobility


def _half_beta_change(windows):
    half = WINDOW_SAMPLES // 2
    taper = np.hanning(half)
    freqs = np.fft.rfftfreq(half, d=1.0 / SAMPLE_RATE)
    beta = (freqs >= 13.0) & (freqs < 30.0)

    def beta_power(part):
        spectrum = np.abs(np.fft.rfft(part * taper, axis=-1)) ** 2
        return np.log10(spectrum[..., beta].sum(axis=-1) + EPS)

    return beta_power(windows[..., half:]) - beta_power(windows[..., :half])


def _time_features(windows):
    mobility, complexity = _hjorth(windows)
    segments = windows.reshape(*windows.shape[:-1], 4, -1).mean(axis=-1)
    with warnings.catch_warnings():  # a flat channel has no shape: its skew and kurtosis become 0
        warnings.filterwarnings("ignore", message="Precision loss", category=RuntimeWarning)
        skew = np.nan_to_num(stats.skew(windows, axis=-1))
        kurtosis = np.nan_to_num(stats.kurtosis(windows, axis=-1))
    columns = [
        np.log10(windows.var(-1) + EPS),
        np.log10(np.abs(np.diff(windows, axis=-1)).mean(-1) + EPS),
        np.log10(np.ptp(windows, axis=-1) + EPS),
        mobility,
        complexity,
        skew,
        kurtosis,
        _half_beta_change(windows),
    ] + [segments[..., k] for k in range(4)]
    names = ["logvar", "loglinelen", "logptp", "mobility", "complexity", "skew", "kurtosis",
             "beta_change"] + [f"segmean{k}" for k in range(4)]
    return columns, names


def _cross_features(windows, channels):
    centered = windows - windows.mean(-1, keepdims=True)
    norm = np.sqrt((centered ** 2).sum(-1)) + EPS
    columns, names = [], []
    for a, b in _PAIRS:
        columns.append((centered[:, a] * centered[:, b]).sum(-1) / (norm[:, a] * norm[:, b]))
        names.append(f"corr_{channels[a]}_{channels[b]}")
    lateral = windows[:, 2] - windows[:, 3]
    columns += [np.log10(lateral.var(-1) + EPS), lateral.mean(-1)]
    names += [f"{channels[2]}minus{channels[3]}_logvar", f"{channels[2]}minus{channels[3]}_mean"]
    return np.column_stack(columns), names


def window_features(windows, channels):
    band_cols, band_names = _band_features(windows)
    time_cols, time_names = _time_features(windows)
    per_channel = np.stack(band_cols + time_cols, axis=-1)  # (n, channels, features)
    names = [f"{channel}_{feature}" for channel in channels for feature in band_names + time_names]
    cross, cross_names = _cross_features(windows, channels)
    return np.hstack([per_channel.reshape(len(windows), -1), cross]), names + cross_names


def trailing_baseline(values):
    """Median of each column over past points [i - 240, i - 16]; NaN with fewer than 40 of them."""
    span = BASELINE_PAST_POINTS - BASELINE_SKIP_POINTS + 1

    def one_column(column):
        padded = np.concatenate([np.full(BASELINE_PAST_POINTS, np.nan), column])
        past = sliding_window_view(padded, span)[: len(column)]
        enough = np.isfinite(past).sum(axis=-1) >= BASELINE_MIN_POINTS
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            median = np.nanmedian(past, axis=-1)
        return np.where(enough, median, np.nan)

    return np.column_stack([one_column(values[:, j]) for j in range(values.shape[1])])


def _baseline_columns(names):
    keys = tuple(f"{band}_logpow" for band, _, _ in BANDS) + ("logvar",)
    return [i for i, name in enumerate(names) if name.endswith(keys)]


def stream_features(eeg, end_index, channels, mains_hz=60):
    """Features at consecutive grid points of one continuous raw recording.

    ``end_index`` must be a contiguous grid (``STEP_SAMPLES`` apart) so the baseline
    sees the same past points in training and live use. Rows without 5 s of history
    have NaN in the relative features.
    """
    end_index = np.asarray(end_index)
    if np.any(np.diff(end_index) != STEP_SAMPLES):
        raise ValueError("end_index must be a contiguous grid with STEP_SAMPLES spacing")
    windows = _windows(causal_filter(np.asarray(eeg, dtype=float), mains_hz), end_index)
    base, names = window_features(windows, channels)
    cols = _baseline_columns(names)
    relative = base[:, cols] - trailing_baseline(base[:, cols])
    return np.hstack([base, relative]), names + [f"{names[i]}_vs_past30s" for i in cols]


def latest_features(eeg, channels, mains_hz=60):
    """Features of the window ending at the last sample, computed over the whole history in ``eeg``."""
    aligned = eeg[(len(eeg) - WINDOW_SAMPLES) % STEP_SAMPLES:]
    features, names = stream_features(aligned, grid_ends(len(aligned)), channels, mains_hz)
    return features[-1:], names

"""Window preprocessing and the full feature set (487 values for 4 channels).

Time-domain, band power, spectral, wavelet, entropy, AR, cross-channel,
channel-difference, artifact and indicator features. Names use channel
indices (ch0..ch3), so any montage works.
"""

import antropy as ant
import numpy as np
import pywt
from scipy.integrate import trapezoid
from scipy.linalg import solve, toeplitz
from scipy.signal import butter, coherence, filtfilt, iirnotch, welch
from scipy.stats import kurtosis, skew

from .quant_features import extract_quant_features

SAMPLE_RATE = 200
FREQ_BANDS = {
    "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "gamma": (30, 45),
}
CHANNEL_PAIRS = {"lateral": (2, 3), "frontocentral": (0, 1)}
ASYMMETRY_PAIRS = (((0, 1), "front"), ((2, 3), "rear"))
ARTIFACT_THRESHOLD_UV = 150


def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=SAMPLE_RATE, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)


def notch_filter(data, freq=60.0, fs=SAMPLE_RATE, quality=30.0):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data, axis=0)


def common_average_reference(eeg):
    return eeg - np.mean(eeg, axis=1, keepdims=True)


def preprocess_window(eeg_window, fs=SAMPLE_RATE, mains_hz=60):
    """Band-pass 0.5-45 Hz, mains notch, common average reference."""
    out = eeg_window.copy()
    for ch in range(out.shape[1]):
        out[:, ch] = notch_filter(bandpass_filter(out[:, ch], fs=fs), freq=mains_hz, fs=fs)
    return common_average_reference(out)


def _band_power(signal, fs, band):
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return trapezoid(psd[mask], freqs[mask]) if mask.sum() > 0 else 0.0


def _ar_coefficients(signal, order=6):
    n = len(signal)
    if n <= order:
        return np.zeros(order)
    r = np.correlate(signal, signal, mode="full")[n - 1:] / n
    matrix = toeplitz(r[:order])
    if np.linalg.det(matrix) < 1e-10:
        return np.zeros(order)
    try:
        return solve(matrix, r[1:order + 1])
    except np.linalg.LinAlgError:
        return np.zeros(order)


def _wavelet_features(signal, wavelet="db4", level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = {}
    total = sum(np.sum(c ** 2) for c in coeffs)
    for i, c in enumerate(coeffs):
        name = "approx" if i == 0 else f"detail_{i}"
        energy = np.sum(c ** 2)
        normalized = c ** 2 / (energy + 1e-10)
        features[f"dwt_{name}_energy"] = energy
        features[f"dwt_{name}_std"] = np.std(c)
        features[f"dwt_{name}_mean"] = np.mean(np.abs(c))
        features[f"dwt_{name}_entropy"] = -np.sum(normalized * np.log2(normalized + 1e-10))
    for i, c in enumerate(coeffs):
        name = "approx" if i == 0 else f"detail_{i}"
        features[f"dwt_{name}_rel_energy"] = np.sum(c ** 2) / (total + 1e-10)
    return features


def _entropy(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception:
        return 0


def _channel_features(s, p, fs):
    features = {
        f"{p}_mean": np.mean(s), f"{p}_std": np.std(s), f"{p}_var": np.var(s),
        f"{p}_min": np.min(s), f"{p}_max": np.max(s), f"{p}_ptp": np.ptp(s),
        f"{p}_skew": skew(s), f"{p}_kurtosis": kurtosis(s),
        f"{p}_rms": np.sqrt(np.mean(s ** 2)), f"{p}_median": np.median(s),
        f"{p}_iqr": np.percentile(s, 75) - np.percentile(s, 25),
        f"{p}_abs_mean": np.mean(np.abs(s)), f"{p}_energy": np.sum(s ** 2),
        f"{p}_zcr": np.sum(np.diff(np.sign(s)) != 0) / len(s),
        f"{p}_line_length": np.sum(np.abs(np.diff(s))),
        f"{p}_mean_abs_diff": np.mean(np.abs(np.diff(s))),
    }
    d2 = np.diff(s, n=2)
    features[f"{p}_mean_abs_diff2"] = np.mean(np.abs(d2)) if len(d2) > 0 else 0

    d1 = np.diff(s)
    activity = np.var(s)
    mobility = np.sqrt(np.var(d1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(np.diff(d1)) / (np.var(d1) + 1e-10)) / (mobility + 1e-10)
    features[f"{p}_hjorth_activity"] = activity
    features[f"{p}_hjorth_mobility"] = mobility
    features[f"{p}_hjorth_complexity"] = complexity

    for band, edges in FREQ_BANDS.items():
        features[f"{p}_{band}_power"] = _band_power(s, fs, edges)
    total = sum(features[f"{p}_{band}_power"] for band in FREQ_BANDS)
    features[f"{p}_total_power"] = total
    for band in FREQ_BANDS:
        features[f"{p}_{band}_relative"] = features[f"{p}_{band}_power"] / (total + 1e-10)
    power = {band: features[f"{p}_{band}_power"] for band in FREQ_BANDS}
    beta = power["beta"] + 1e-10
    features[f"{p}_theta_beta_ratio"] = power["theta"] / beta
    features[f"{p}_alpha_theta_ratio"] = power["alpha"] / (power["theta"] + 1e-10)
    features[f"{p}_alpha_beta_ratio"] = power["alpha"] / beta
    features[f"{p}_theta_alpha_beta_ratio"] = (power["theta"] + power["alpha"]) / beta
    features[f"{p}_delta_beta_ratio"] = power["delta"] / beta

    freqs, psd = welch(s, fs=fs, nperseg=min(len(s), fs))
    total_psd = np.sum(psd) + 1e-10
    normalized = psd / total_psd
    features[f"{p}_spectral_entropy"] = -np.sum(normalized * np.log2(normalized + 1e-10))
    features[f"{p}_peak_freq"] = freqs[np.argmax(psd)] if len(freqs) > 0 else 0
    edge = np.searchsorted(np.cumsum(psd) / total_psd, 0.95)
    features[f"{p}_spectral_edge_95"] = freqs[min(edge, len(freqs) - 1)]
    mean_freq = np.sum(freqs * psd) / total_psd
    features[f"{p}_mean_freq"] = mean_freq
    features[f"{p}_spectral_bandwidth"] = np.sqrt(np.sum(((freqs - mean_freq) ** 2) * psd) / total_psd)

    for key, value in _wavelet_features(s).items():
        features[f"{p}_{key}"] = value

    features[f"{p}_approx_entropy"] = _entropy(ant.app_entropy, s)
    features[f"{p}_sample_entropy"] = _entropy(ant.sample_entropy, s)
    features[f"{p}_perm_entropy"] = _entropy(ant.perm_entropy, s, normalize=True)
    features[f"{p}_svd_entropy"] = _entropy(ant.svd_entropy, s, normalize=True)

    for i, value in enumerate(_ar_coefficients(s, order=6)):
        features[f"{p}_ar_{i}"] = value
    return features


def extract_all_features(eeg_window, fs=SAMPLE_RATE):
    """All features of one preprocessed (samples, channels) window, as name -> value."""
    features = {}
    nc = eeg_window.shape[1]
    for ch in range(nc):
        features.update(_channel_features(eeg_window[:, ch], f"ch{ch}", fs))

    for i in range(nc):
        for j in range(i + 1, nc):
            correlation = np.corrcoef(eeg_window[:, i], eeg_window[:, j])[0, 1]
            features[f"corr_ch{i}_ch{j}"] = correlation if not np.isnan(correlation) else 0
            freqs, values = coherence(eeg_window[:, i], eeg_window[:, j], fs=fs, nperseg=min(200, fs))
            alpha = (freqs >= 8) & (freqs <= 13)
            beta = (freqs >= 13) & (freqs <= 30)
            features[f"alpha_coherence_ch{i}_ch{j}"] = np.mean(values[alpha]) if alpha.sum() > 0 else 0
            features[f"beta_coherence_ch{i}_ch{j}"] = np.mean(values[beta]) if beta.sum() > 0 else 0

    for (left, right), name in ASYMMETRY_PAIRS:
        if left < nc and right < nc:
            for band in FREQ_BANDS:
                lp = features[f"ch{left}_{band}_power"]
                rp = features[f"ch{right}_{band}_power"]
                features[f"{name}_{band}_asymmetry"] = np.log(rp + 1e-10) - np.log(lp + 1e-10)
                features[f"{name}_{band}_asym_ratio"] = (rp - lp) / (rp + lp + 1e-10)

    for name, (ci, cj) in CHANNEL_PAIRS.items():
        if ci < nc and cj < nc:
            diff = eeg_window[:, ci] - eeg_window[:, cj]
            features[f"diff_{name}_mean"] = np.mean(diff)
            features[f"diff_{name}_std"] = np.std(diff)
            features[f"diff_{name}_energy"] = np.sum(diff ** 2)
            features[f"diff_{name}_zcr"] = np.sum(np.diff(np.sign(diff)) != 0) / len(diff)
            for band, edges in FREQ_BANDS.items():
                features[f"diff_{name}_{band}_power"] = _band_power(diff, fs, edges)

    magnitude = np.abs(eeg_window)
    features["artifact_max_amplitude"] = float(np.max(magnitude))
    features["artifact_mean_amplitude"] = float(np.mean(magnitude))
    features["artifact_frac_above_threshold"] = float(np.mean(magnitude > ARTIFACT_THRESHOLD_UV))
    for ch in range(nc):
        features[f"ch{ch}_artifact_frac"] = float(np.mean(magnitude[:, ch] > ARTIFACT_THRESHOLD_UV))

    features.update(extract_quant_features(eeg_window))
    return features


def feature_vector(eeg_window, names, mains_hz=60):
    """Preprocess one raw window and return the named features as a (1, n) array; non-finite -> 0."""
    features = extract_all_features(preprocess_window(np.asarray(eeg_window, dtype=float), mains_hz=mains_hz))
    vector = np.array([[features.get(name, 0.0) for name in names]], dtype=float)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

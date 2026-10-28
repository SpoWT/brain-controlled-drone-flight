"""Technical-indicator features per channel: momentum, volatility, trend and mean reversion.

Each indicator looks at the tail of one 1 s window (30 values per channel).
"""

import numpy as np


def _ema(signal, period):
    if len(signal) < period:
        return np.full_like(signal, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema = np.zeros(len(signal))
    ema[0] = signal[0]
    for i in range(1, len(signal)):
        ema[i] = alpha * signal[i] + (1 - alpha) * ema[i - 1]
    return ema


def _range(signal, period):
    recent = signal[-period:]
    return np.max(recent), np.min(recent)


# --- momentum ---------------------------------------------------------------

def rsi(signal, period=14):
    if len(signal) < period + 1:
        return 50.0
    delta = np.diff(signal)
    avg_gain = np.mean(np.where(delta > 0, delta, 0)[-period:])
    avg_loss = np.mean(np.where(delta < 0, -delta, 0)[-period:])
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return float(100 - 100 / (1 + avg_gain / avg_loss))


def macd(signal, fast=12, slow=26, signal_period=9):
    """(macd line, signal line, histogram) at the last sample."""
    if len(signal) < slow:
        return 0.0, 0.0, 0.0
    line = _ema(signal, fast) - _ema(signal, slow)
    smoothed = _ema(line, signal_period)
    return float(line[-1]), float(smoothed[-1]), float(line[-1] - smoothed[-1])


def stochastic(signal, period=14):
    """(%K, %D); %D equals %K within a single window."""
    if len(signal) < period:
        return 50.0, 50.0
    high, low = _range(signal, period)
    k = 50.0 if high == low else 100 * (signal[-1] - low) / (high - low)
    return float(k), float(k)


def roc(signal, period=10):
    if len(signal) < period + 1:
        return 0.0
    old = signal[-period - 1]
    return 0.0 if old == 0 else float((signal[-1] - old) / abs(old) * 100)


# --- volatility -------------------------------------------------------------

def bollinger(signal, period=20, num_std=2):
    """(band width, %B clipped to 0-1)."""
    if len(signal) < period:
        return 0.0, 0.5
    recent = signal[-period:]
    middle, std = np.mean(recent), np.std(recent)
    upper, lower = middle + num_std * std, middle - num_std * std
    width = (upper - lower) / (middle + 1e-10)
    percent_b = 0.5 if upper == lower else (signal[-1] - lower) / (upper - lower)
    return float(width), float(np.clip(percent_b, 0, 1))


def atr(signal, period=14):
    if len(signal) < period:
        return float(np.std(signal)) if len(signal) > 0 else 0.0
    true_range = np.abs(np.diff(signal))
    return float(np.mean(true_range[-period:]) if len(true_range) >= period else np.mean(true_range))


def normalized_atr(signal, period=14):
    return float(atr(signal, period) / (np.mean(np.abs(signal)) + 1e-10))


def keltner_width(signal, period=20, atr_mult=2):
    if len(signal) < period:
        return 0.0
    ema = _ema(signal, period)[-1]
    return float(2 * atr_mult * atr(signal, period) / (abs(ema) + 1e-10))


def volatility_ratio(signal, short_period=5, long_period=20):
    if len(signal) < long_period:
        return 1.0
    long_vol = np.std(signal[-long_period:])
    return 1.0 if long_vol == 0 else float(np.std(signal[-short_period:]) / long_vol)


def volatility_regime(signal, period=20, threshold=1.5):
    """1 when the last quarter-period is much more volatile than the period."""
    if len(signal) < period:
        return 0.0
    current = np.std(signal[-period // 4:])
    average = np.std(signal[-period:])
    if average == 0:
        return 0.0
    return 1.0 if current > threshold * average else 0.0


def chaikin_volatility(signal, period=10):
    """Change in spread between the first and second half of the window, in %."""
    if len(signal) < period * 2:
        return 0.0
    half = len(signal) // 2
    old, new = np.std(signal[:half]), np.std(signal[half:])
    return 0.0 if old == 0 else float((new - old) / old * 100)


# --- trend ------------------------------------------------------------------

def ema_features(signal, fast_period=5, slow_period=20):
    """(fast EMA, slow EMA, fast - slow) at the last sample."""
    if len(signal) < slow_period:
        last = float(signal[-1]) if len(signal) > 0 else 0.0
        return last, last, 0.0
    fast = _ema(signal, fast_period)[-1]
    slow = _ema(signal, slow_period)[-1]
    return float(fast), float(slow), float(fast - slow)


def adx(signal, period=14):
    """(ADX, +DI, -DI) over the last period, unsmoothed."""
    if len(signal) < period + 1:
        return 0.0, 0.0, 0.0
    diff = np.diff(signal)
    true_range = np.mean(np.abs(diff)[-period:]) + 1e-10
    plus_di = 100 * np.mean(np.where(diff > 0, diff, 0)[-period:]) / true_range
    minus_di = 100 * np.mean(np.where(diff < 0, -diff, 0)[-period:]) / true_range
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return float(dx), float(plus_di), float(minus_di)


def aroon(signal, period=25):
    """(up, down): position of the high and the low within the last period, 0-100."""
    if len(signal) < period:
        return 50.0, 50.0
    recent = signal[-period:]
    return (float(np.argmax(recent) / (period - 1) * 100),
            float(np.argmin(recent) / (period - 1) * 100))


# --- mean reversion / range -------------------------------------------------

def cci(signal, period=20):
    if len(signal) < period:
        return 0.0
    recent = signal[-period:]
    mean = np.mean(recent)
    mad = np.mean(np.abs(recent - mean)) + 1e-10
    return float((signal[-1] - mean) / (0.015 * mad))


def percent_r(signal, period=14):
    if len(signal) < period:
        return -50.0
    high, low = _range(signal, period)
    return -50.0 if high == low else float((high - signal[-1]) / (high - low) * -100)


def donchian_position(signal, period=20):
    if len(signal) < period:
        return 0.5
    high, low = _range(signal, period)
    return 0.5 if high == low else float((signal[-1] - low) / (high - low))


def zscore(signal, period=20):
    if len(signal) < period:
        return 0.0
    recent = signal[-period:]
    return float((signal[-1] - np.mean(recent)) / (np.std(recent) + 1e-10))


def ema_distance(signal, period=20):
    if len(signal) < period:
        return 0.0
    ema = _ema(signal, period)[-1]
    return 0.0 if ema == 0 else float((signal[-1] - ema) / abs(ema) * 100)


def range_position(signal):
    if len(signal) < 2:
        return 0.5
    high, low = np.max(signal), np.min(signal)
    return 0.5 if high == low else float((signal[-1] - low) / (high - low))


def mean_reversion_strength(signal, period=20):
    """Correlation of deviation from the mean with the next change; negative = reverting."""
    if len(signal) < period + 1:
        return 0.0
    recent = signal[-period - 1:]
    deviations = recent[:-1] - np.mean(recent)
    changes = np.diff(recent)
    if np.std(deviations) == 0 or np.std(changes) == 0:
        return 0.0
    correlation = np.corrcoef(deviations, changes)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0


def extract_quant_features(eeg_window):
    """30 indicator features per channel of a (samples, channels) window."""
    features = {}
    for ch in range(eeg_window.shape[1]):
        s = eeg_window[:, ch].astype(float)
        p = f"ch{ch}_quant"
        features[f"{p}_rsi"] = rsi(s)
        features[f"{p}_macd"], features[f"{p}_macd_signal"], features[f"{p}_macd_histogram"] = macd(s)
        features[f"{p}_stochastic_k"], features[f"{p}_stochastic_d"] = stochastic(s)
        features[f"{p}_roc"] = roc(s)
        features[f"{p}_bollinger_width"], features[f"{p}_bollinger_percent_b"] = bollinger(s)
        features[f"{p}_atr"] = atr(s)
        features[f"{p}_normalized_atr"] = normalized_atr(s)
        features[f"{p}_keltner_width"] = keltner_width(s)
        features[f"{p}_volatility_ratio"] = volatility_ratio(s)
        features[f"{p}_volatility_regime"] = volatility_regime(s)
        features[f"{p}_chaikin_volatility"] = chaikin_volatility(s)
        features[f"{p}_ema_fast"], features[f"{p}_ema_slow"], features[f"{p}_ema_crossover"] = ema_features(s)
        features[f"{p}_adx"], features[f"{p}_plus_di"], features[f"{p}_minus_di"] = adx(s)
        features[f"{p}_aroon_up"], features[f"{p}_aroon_down"] = aroon(s)
        features[f"{p}_cci"] = cci(s)
        features[f"{p}_percent_r"] = percent_r(s)
        features[f"{p}_donchian_position"] = donchian_position(s)
        features[f"{p}_zscore"] = zscore(s)
        features[f"{p}_ema_distance"] = ema_distance(s)
        features[f"{p}_range_position"] = range_position(s)
        features[f"{p}_mean_reversion_strength"] = mean_reversion_strength(s)
    return features

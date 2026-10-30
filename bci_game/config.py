"""Constants for the recording game and training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from bci_game.runtime import bundle_roots

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RECORDINGS_DIR: Final[Path] = Path(os.environ.get("BCI_GAME_RECORDINGS", PROJECT_ROOT / "eeg_recordings"))
# Trained bundles go where the flying website looks first: $EEG_BUNDLES or eeg_bundles/.
BUNDLES_DIR: Final[Path] = bundle_roots(PROJECT_ROOT)[0]

HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8790

SAMPLE_RATE: Final[int] = 200
N_CHANNELS: Final[int] = 4
DEFAULT_CHANNELS: Final[str] = "Fz,Cz,F7,F8"


def parse_channels(text: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    if len(names) != N_CHANNELS or len(set(names)) != N_CHANNELS:
        raise ValueError(f"BCI_GAME_CHANNELS needs {N_CHANNELS} different names, got {text!r}")
    return names


CHANNELS: Final[tuple[str, ...]] = parse_channels(os.environ.get("BCI_GAME_CHANNELS", DEFAULT_CHANNELS))


def parse_mains(text: str) -> int:
    if text.strip() not in ("50", "60"):
        raise ValueError(f"BCI_GAME_MAINS_HZ must be 50 or 60, got {text!r}")
    return int(text)


# Mains frequency for the notch filter and the line-noise check; saved in each bundle.
MAINS_HZ: Final[int] = parse_mains(os.environ.get("BCI_GAME_MAINS_HZ", "60"))

# Live preview buffer.
BUFFER_SECONDS: Final[float] = 40.0  # the Random Forest needs 35 s
PULL_INTERVAL_S: Final[float] = 0.05

# Signal check thresholds (µV after a 1-40 Hz band-pass).
QUALITY_WINDOW_S: Final[float] = 2.0
FLAT_UV: Final[float] = 0.5
NOISY_UV: Final[float] = 60.0
LINE_RATIO_MAX: Final[float] = 3.0

# One decision window, as in the flying website: 1 s = 200 samples.
WINDOW_S: Final[float] = 1.0
WINDOW_SAMPLES: Final[int] = int(WINDOW_S * SAMPLE_RATE)
TRAIN_STEP_S: Final[float] = 0.5

# Models training compares; the best is saved.
MODEL_NAMES: Final[tuple[str, ...]] = ("xgboost", "lightgbm", "random_forest", "cfc", "ltc", "gru")
LIVE_STEP_S: Final[float] = 0.25

GO: Final[str] = "GO"
STOP: Final[str] = "STOP"

ALLOWED_RIDE_MINUTES: Final[tuple[int, ...]] = (2, 5, 8)
MAX_EVENTS_PER_BATCH: Final[int] = 500

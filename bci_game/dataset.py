"""Rides marked IN -> labelled 1 s windows, cleaned exactly as the flying website does live."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import labels, session
from .config import CHANNELS, GO, SAMPLE_RATE, STOP, TRAIN_STEP_S, WINDOW_SAMPLES
from .ml.asr import ASRProcessor
from .ml.inference import window_quality

ASR_CALIBRATION_S = 60
CLASS_OF = {GO: "NOT_STOP", STOP: "STOP"}

Progress = Callable[[str], None]


@dataclass(frozen=True)
class Ride:
    session_id: str
    eeg: np.ndarray        # (n, 4) raw µV
    stamps: np.ndarray     # (n,) unix time of each sample
    events: list


@dataclass(frozen=True)
class Windows:
    eeg: np.ndarray        # (n, 200, 4) ASR-cleaned µV
    labels: np.ndarray     # "NOT_STOP" / "STOP"
    groups: np.ndarray     # ride id of each window
    ends: np.ndarray       # index of each window's last sample in its ride

    def subset(self, keep: np.ndarray) -> "Windows":
        return Windows(self.eeg[keep], self.labels[keep], self.groups[keep], self.ends[keep])


def usable_rides(recordings_dir: Path) -> list[dict]:
    """Manifests of saved rides marked IN, oldest first."""
    return sorted(
        (manifest for manifest in session.list_sessions(Path(recordings_dir))
         if manifest.get("kind") == "ride" and not manifest.get("excluded") and manifest.get("samples", 0) > 0),
        key=lambda manifest: str(manifest["session_id"]),
    )


def choose_rides(manifests: list[dict]) -> tuple[list[str], bool]:
    """(ride ids, simulated): Ganglion rides when there are any, else the other rides (simulator or unknown).
    The two are never mixed, and a model trained without Ganglion rides is never flown."""
    headset = [m for m in manifests if m.get("source") == "ganglion"]
    chosen = headset or manifests
    return [str(m["session_id"]) for m in chosen], bool(chosen) and not headset


def load_ride(recordings_dir: Path, session_id: str) -> Ride:
    directory = session.session_dir(session_id, Path(recordings_dir))
    eeg, stamps = session.load_eeg(directory)
    return Ride(session_id, eeg, stamps, session.load_events(directory))


def calibrate_asr(rides: list[Ride], channels: tuple[str, ...] = CHANNELS) -> ASRProcessor:
    """ASR fitted on the first minute of the oldest ride that long (else of the longest ride)."""
    samples = ASR_CALIBRATION_S * SAMPLE_RATE
    ride = next((ride for ride in rides if len(ride.eeg) >= samples), max(rides, key=lambda ride: len(ride.eeg)))
    asr = ASRProcessor(sfreq=SAMPLE_RATE, ch_names=channels)
    asr.calibrate(ride.eeg[:samples])
    return asr


def ride_windows(ride: Ride, asr: ASRProcessor) -> tuple[list[np.ndarray], list[str], list[int]]:
    """Cleaned windows, their classes and end indices; windows the live artifact gate rejects are dropped."""
    ends, window_labels = labels.windows(ride.stamps, ride.events, step_s=TRAIN_STEP_S)
    kept, classes, kept_ends = [], [], []
    for end, label in zip(ends, window_labels):
        raw = ride.eeg[end - WINDOW_SAMPLES + 1: end + 1]
        if not np.isfinite(raw).all():
            continue
        cleaned = asr.transform(raw)
        if window_quality(cleaned) == "reject":
            continue
        kept.append(np.asarray(cleaned, dtype=float))
        classes.append(CLASS_OF[label])
        kept_ends.append(int(end))
    return kept, classes, kept_ends


def collect(rides: list[Ride], asr: ASRProcessor, progress: Progress | None = None) -> Windows:
    eeg, classes, groups, ends = [], [], [], []
    for index, ride in enumerate(rides, start=1):
        if progress:
            progress(f"Cleaning ride {index}/{len(rides)}")
        kept, ride_classes, ride_ends = ride_windows(ride, asr)
        eeg.extend(kept)
        classes.extend(ride_classes)
        groups.extend([ride.session_id] * len(kept))
        ends.extend(ride_ends)
    stacked = np.stack(eeg) if eeg else np.zeros((0, WINDOW_SAMPLES, len(CHANNELS)))
    return Windows(stacked, np.array(classes, dtype=object), np.array(groups, dtype=object),
                   np.array(ends, dtype=int))

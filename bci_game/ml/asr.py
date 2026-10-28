"""Artifact Subspace Reconstruction (asrpy) with a plain pickle state file.

The state holds only numbers (filter and threshold matrices), so it loads
without this module's classes.
"""

import logging
import os
import pickle

import mne
import numpy as np

mne.set_log_level("WARNING")
LOGGER = logging.getLogger(__name__)
STATE_KEYS = ("M", "T", "A", "B", "Zi", "cov", "carry", "R")
MIN_CALIBRATION_S = 10


class ASRProcessor:
    """Calibrate on clean EEG, then clean (samples, channels) µV arrays."""

    def __init__(self, sfreq=200, cutoff=20, ch_names=None):
        self.sfreq = sfreq
        self.cutoff = cutoff
        self.ch_names = list(ch_names or ["CH1", "CH2", "CH3", "CH4"])
        self.asr = None
        self.calibrated = False
        self.calibration_samples = 0

    def _raw(self, eeg):
        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        return mne.io.RawArray(np.asarray(eeg, dtype=float).T * 1e-6, info, verbose=False)

    def calibrate(self, eeg):
        """Fit on (samples, channels) µV; asrpy picks the clean stretches itself."""
        from asrpy import ASR

        eeg = np.asarray(eeg, dtype=float)
        if eeg.ndim != 2 or eeg.shape[1] != len(self.ch_names):
            raise ValueError(f"Expected (samples, {len(self.ch_names)}) EEG, got {eeg.shape}")
        if len(eeg) < self.sfreq * MIN_CALIBRATION_S:
            raise ValueError(f"ASR needs at least {MIN_CALIBRATION_S} s of EEG, got {len(eeg) / self.sfreq:.1f} s")
        self.asr = ASR(sfreq=self.sfreq, cutoff=self.cutoff)
        self.asr.fit(self._raw(eeg))
        self.calibrated = True
        self.calibration_samples = len(eeg)

    def transform(self, eeg):
        """Cleaned copy of ``eeg``; the input unchanged when not calibrated or on error."""
        if not self.calibrated:
            return eeg
        try:
            return self.asr.transform(self._raw(eeg)).get_data().T * 1e6
        except Exception as exc:
            LOGGER.warning("ASR transform failed: %s", exc)
            return eeg

    def save(self, path):
        if not self.calibrated:
            raise RuntimeError("ASR is not calibrated")
        state = {"sfreq": self.sfreq, "cutoff": self.cutoff, "ch_names": self.ch_names,
                 "calibration_samples": self.calibration_samples,
                 **{key: getattr(self.asr, key) for key in STATE_KEYS}}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(state, handle)

    def load(self, path):
        """True when the state file was read; False (and uncalibrated) otherwise."""
        from asrpy import ASR

        try:
            with open(path, "rb") as handle:
                state = pickle.load(handle)
            self.sfreq, self.cutoff = state["sfreq"], state["cutoff"]
            self.ch_names = list(state["ch_names"])
            self.calibration_samples = state["calibration_samples"]
            self.asr = ASR(sfreq=self.sfreq, cutoff=self.cutoff)
            for key in STATE_KEYS:
                setattr(self.asr, key, state[key])
        except (OSError, KeyError, TypeError, ValueError, pickle.UnpicklingError) as exc:
            LOGGER.warning("ASR state %s could not be loaded: %s", path, exc)
            self.asr, self.calibrated = None, False
            return False
        self.calibrated = True
        return True

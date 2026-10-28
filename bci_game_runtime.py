#!/usr/bin/env python3
"""STOP / GO model bundles: discovery, verification and the live runtime.
The default decision is always STOP."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLES_ENV = "EEG_BUNDLES"
MANIFEST = "bundle.json"
BUNDLE_FORMAT = "stop-go-bundle/1"
CLASS_NAMES = ["NOT_STOP", "STOP"]
SAMPLE_RATE = 200
MAX_SAMPLE_GAP_S = 0.25  # Ganglion timestamps are BLE notification times and arrive in bursts
LOGGER = logging.getLogger(__name__)
_IMPORT_LOCK = threading.Lock()


def continuous_window(times, sample_rate: int = SAMPLE_RATE) -> bool:
    """True when a window's timestamps have no gap over 0.25 s and span 0.65-1.4x the nominal length.
    Training, the game's drive mode and the flying website all use this rule."""
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or not np.isfinite(times).all():
        return False
    nominal = (len(times) - 1) / sample_rate
    span = times[-1] - times[0]
    return bool(np.max(np.diff(times)) <= MAX_SAMPLE_GAP_S and 0.65 * nominal <= span <= 1.4 * nominal)


def bundle_roots(project_root: Path = PROJECT_ROOT) -> list[Path]:
    """$EEG_BUNDLES (os.pathsep separated), else <project>/eeg_bundles and the project folder."""
    configured = [Path(part.strip()).expanduser() for part in os.environ.get(BUNDLES_ENV, "").split(os.pathsep)
                  if part.strip()]
    return configured or [Path(project_root) / "eeg_bundles", Path(project_root)]


@dataclass(frozen=True)
class Bundle:
    folder: Path
    model_id: str
    model_dir: Path
    asr_state: Path
    code_root: Path
    engine: str = "ml.inference:InferenceEngine"
    asr: str = "server.asr_processor:ASRProcessor"
    label: str = ""
    sha256: dict = field(default_factory=dict)  # path inside the folder -> digest; empty without a manifest

    @property
    def name(self) -> str:
        return self.label or self.folder.name

    @property
    def config_path(self) -> Path:
        return self.model_dir / f"config_{self.model_id}.json"

    def config(self) -> dict:
        return json.loads(self.config_path.read_text())


def _inside(folder: Path, relative: object) -> Path:
    root = Path(folder).resolve()
    path = (root / str(relative)).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{relative!r} points outside the bundle")
    return path


def _class_ref(text: object) -> str:
    module, _, name = str(text).partition(":")
    if not name.isidentifier() or not all(part.isidentifier() for part in module.split(".")):
        raise ValueError(f"Bad class reference {text!r}")
    return f"{module}:{name}"


def read_bundle(folder: Path) -> Bundle | None:
    """The bundle stored in ``folder``, or None when the folder is not a bundle."""
    folder = Path(folder)
    if not (folder / MANIFEST).is_file():
        return _unlisted_bundle(folder)
    manifest = json.loads((folder / MANIFEST).read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Bundle manifest is not a JSON object")
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"Unknown bundle format {manifest.get('format')!r}")
    hashes = manifest["sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Bundle manifest lists no file hashes")
    return Bundle(
        folder=folder,
        model_id=str(manifest["model_id"]),
        model_dir=_inside(folder, manifest["model_dir"]),
        asr_state=_inside(folder, manifest["asr_state"]),
        code_root=_inside(folder, manifest["code"]),
        engine=_class_ref(manifest["engine"]),
        asr=_class_ref(manifest["asr"]),
        label=str(manifest.get("label", "")),
        sha256={str(path): str(digest) for path, digest in hashes.items()},
    )


def _unlisted_bundle(folder: Path) -> Bundle | None:
    """A folder without a manifest laid out as model/, dataset/asr/ and repro/."""
    configs = sorted((folder / "model").glob("config_*.json"))
    states = sorted((folder / "dataset" / "asr").glob("asr_state_*.pkl"))
    code = folder / "repro"
    if (len(configs) != 1 or len(states) != 1 or not (code / "ml" / "inference.py").is_file()
            or not (code / "server" / "asr_processor.py").is_file()):
        return None
    return Bundle(folder=folder, model_id=configs[0].stem.removeprefix("config_"),
                  model_dir=folder / "model", asr_state=states[0], code_root=code)


def find_bundles(roots=None) -> list[Bundle]:
    """Bundles directly inside each root folder (or the root itself), each folder once."""
    found, seen = [], set()
    for root in bundle_roots() if roots is None else roots:
        root = Path(root)
        try:
            candidates = [root, *sorted(path for path in root.iterdir()
                                        if path.is_dir() and not path.name.startswith("."))]
        except OSError:
            continue
        for folder in candidates:
            key = folder.resolve()
            if key in seen:
                continue
            try:
                bundle = read_bundle(folder)
            except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
                LOGGER.warning("Skipping EEG bundle %s: %s", folder.name, exc)
                continue
            if bundle is not None:
                seen.add(key)
                found.append(bundle)
    return found


def _trained_at(bundle: Bundle) -> str:
    try:
        config = bundle.config()
    except (OSError, ValueError, RecursionError):
        return ""
    return str(config.get("trained_at", "")) if isinstance(config, dict) else ""


def default_bundle(roots=None) -> Bundle:
    """The most recently trained bundle."""
    bundles = find_bundles(roots)
    if not bundles:
        searched = bundle_roots() if roots is None else roots
        raise FileNotFoundError("No STOP / GO bundle in " + ", ".join(str(path) for path in searched))
    return max(bundles, key=lambda bundle: (_trained_at(bundle), bundle.folder.name))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle: Bundle) -> None:
    """Every listed file matches its hash, and the model, ASR state and code are all listed."""
    if not bundle.sha256:
        LOGGER.info("EEG bundle %s has no manifest; checking structure only", bundle.folder.name)
        return
    for relative, expected in bundle.sha256.items():
        path = _inside(bundle.folder, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"Artifact hash mismatch for {relative}: expected {expected}, got {actual}")
    config = bundle.config()
    listed = {_inside(bundle.folder, relative) for relative in bundle.sha256}
    artifacts = [config[key] for key in ("artifact_filename", "scaler_artifact_filename") if config.get(key)]
    required = [bundle.config_path, bundle.asr_state, *(bundle.model_dir / name for name in artifacts),
                *bundle.code_root.rglob("*.py")]
    missing = [path.name for path in required if path.resolve() not in listed]
    if missing:
        raise RuntimeError("Bundle files without a recorded hash: " + ", ".join(missing))


def _import_classes(bundle: Bundle) -> tuple[type, type]:
    """Import the engine and ASR classes from the bundle's own code, isolated from other bundles."""
    packages = {ref.partition(":")[0].split(".")[0] for ref in (bundle.engine, bundle.asr)}

    def owned(name: str) -> bool:
        return name.split(".")[0] in packages

    code_root = str(bundle.code_root)
    with _IMPORT_LOCK:
        saved = {name: sys.modules.pop(name) for name in [name for name in sys.modules if owned(name)]}
        sys.path.insert(0, code_root)
        importlib.invalidate_caches()
        try:
            classes = []
            for ref in (bundle.engine, bundle.asr):
                module, _, name = ref.partition(":")
                classes.append(getattr(importlib.import_module(module), name))
            return classes[0], classes[1]
        finally:
            sys.path.remove(code_root)
            for name in [name for name in sys.modules if owned(name)]:
                del sys.modules[name]
            sys.modules.update(saved)


def _stop(reason: str) -> dict:
    return {
        "decision": "STOP",
        "raw_decision": "NO_DECISION",
        "reason": reason,
        "confidence": 1.0,
        "probabilities": {"NOT_STOP": 0.0, "STOP": 1.0},
        "quality": "reject",
        "safe_fallback": True,
    }


class BCIGameStopGoRuntime:
    """Load one STOP / NOT_STOP bundle and run it on WINDOW_SAMPLES x 4 windows of raw EEG
    (200 samples; 7000 for a Random Forest, which needs the past 35 s)."""

    WINDOW_SAMPLES = 200
    CHANNELS = 4

    def __init__(self, bundle: Bundle | Path | str | None = None, *, verify_hashes: bool = True):
        self._lock = threading.Lock()
        self.engine = None
        self.asr = None
        self.config = {}
        if bundle is None:
            bundle = default_bundle()
        elif not isinstance(bundle, Bundle):
            folder = Path(bundle)
            bundle = read_bundle(folder)
            if bundle is None:
                raise FileNotFoundError(f"{folder} is not a STOP / GO bundle")
        self.bundle = bundle
        if verify_hashes:
            verify_bundle(bundle)
        self._load()

    def _load(self) -> None:
        engine_class, asr_class = _import_classes(self.bundle)
        self.config = self.bundle.config()

        self.engine = engine_class()
        self.engine.load(model_dir=str(self.bundle.model_dir), model_id=self.bundle.model_id)
        if self.engine.class_names != CLASS_NAMES:
            raise RuntimeError(f"Unexpected class order: {self.engine.class_names}")
        if len(self.engine.feature_names) != int(self.config["n_features"]):
            raise RuntimeError(
                f"Expected {self.config['n_features']} selected features, got {len(self.engine.feature_names)}"
            )

        self.asr = asr_class()
        if not self.asr.load(str(self.bundle.asr_state)):
            raise RuntimeError("The bundle's ASR state could not be loaded")
        self.WINDOW_SAMPLES = int(getattr(self.engine, "window_samples", BCIGameStopGoRuntime.WINDOW_SAMPLES))
        self._uses_asr = bool(getattr(self.engine, "uses_asr", True))

        self.reset()

    def reset(self) -> None:
        """Reset voting and start at STOP."""
        with self._lock:
            self.engine._vote_buffer = []
            self.engine._last_state = "STOP"
            self.engine._last_change_ms = time.time() * 1000

    def set_stop_threshold(self, threshold: float) -> None:
        """Set the STOP threshold and discard old votes."""
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)
            or not 0.01 <= threshold <= 0.99
        ):
            raise ValueError("STOP threshold must be between 1% and 99%")
        with self._lock:
            self.engine.stop_threshold = float(threshold)
            self.engine._vote_buffer = []
            self.engine._last_state = "STOP"
            self.engine._last_change_ms = time.time() * 1000

    @property
    def info(self) -> dict:
        return {
            "model_id": self.config["model_id"],
            "model_type": self.config.get("model_type"),
            "trained_at": self.config["trained_at"],
            "profile": self.bundle.name,
            "bundle": self.bundle.folder.name,
            "class_names": list(self.config["class_names"]),
            "n_features": int(self.config["n_features"]),
            "stop_threshold": float(self.engine.stop_threshold),
            "accuracy": float(self.config["accuracy"]),
            "cv_mean": float(self.config["cv_mean"]),
            "asr": bool(self.asr and self.asr.calibrated and self._uses_asr),
            "window_samples": self.WINDOW_SAMPLES,
            "safe_default": "STOP",
        }

    def predict(self, eeg_window: np.ndarray) -> dict:
        """STOP, NOT_STOP, or an error result that means STOP."""
        eeg = np.asarray(eeg_window, dtype=np.float64)
        if eeg.shape != (self.WINDOW_SAMPLES, self.CHANNELS):
            return _stop(f"bad_shape:{eeg.shape}")
        if not np.isfinite(eeg).all():
            return _stop("non_finite_input")

        try:
            with self._lock:
                cleaned = self.asr.transform(eeg) if self._uses_asr else eeg
                result = self.engine.predict(cleaned)
        except Exception as exc:
            return _stop(f"{type(exc).__name__}: {exc}")

        if result.get("decision") not in {"STOP", "NOT_STOP"}:
            result["decision"] = "STOP"
            result["safe_fallback"] = True
        result["asr_applied"] = self._uses_asr
        result["model_id"] = self.bundle.model_id
        return result


__all__ = ["BCIGameStopGoRuntime", "Bundle", "bundle_roots", "continuous_window", "default_bundle", "file_sha256",
           "find_bundles", "read_bundle", "verify_bundle"]

"""Write and list bundle folders (the trained models the flying website loads).

    eeg_bundles/<name>/bundle.json                manifest: format, model id, SHA-256 of every file
    eeg_bundles/<name>/model/config_<id>.json     model type, features, classes, threshold, held-out metrics
    eeg_bundles/<name>/model/model_<id>.*         the model (.joblib, or .pt for a network)
    eeg_bundles/<name>/model/scaler_<id>.joblib   StandardScaler (XGBoost and LightGBM only)
    eeg_bundles/<name>/asr_state.pkl              ASR calibration
    eeg_bundles/<name>/code/ml/*.py               the feature and inference code used for training
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from bci_game_runtime import BUNDLE_FORMAT, MANIFEST, file_sha256

CODE_SOURCE = Path(__file__).resolve().parent / "ml"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def check_name(name: str) -> str:
    if not NAME.match(name or ""):
        raise ValueError("Bundle names use letters, digits, '.', '_' or '-' (at most 64)")
    return name


def write(bundles_dir: Path, name: str, *, artifacts, config: dict, asr, report: dict) -> Path:
    """Save a complete bundle; it appears under its name only once every file is written."""
    bundles_dir = Path(bundles_dir)
    final = bundles_dir / check_name(name)
    if final.exists():
        raise FileExistsError(f"A bundle named {name} already exists")
    staging = bundles_dir / f".{name}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    model_dir = staging / "model"
    model_dir.mkdir(parents=True)
    config = {**config, **artifacts(model_dir, config["model_id"])}
    (model_dir / f"config_{config['model_id']}.json").write_text(json.dumps(config, indent=2))
    asr.save(str(staging / "asr_state.pkl"))
    shutil.copytree(CODE_SOURCE, staging / "code" / "ml", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {path.relative_to(staging).as_posix(): file_sha256(path)
              for path in sorted(staging.rglob("*")) if path.is_file()}
    manifest = {
        "format": BUNDLE_FORMAT,
        "label": name,
        "model_id": config["model_id"],
        "model_dir": "model",
        "asr_state": "asr_state.pkl",
        "code": "code",
        "engine": "ml.inference:InferenceEngine",
        "asr": "ml.asr:ASRProcessor",
        "channels": config["channels"],
        "sample_rate": config["sample_rate"],
        "window_samples": config["window_samples"],
        "report": report,
        "sha256": hashes,
    }
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2))
    staging.rename(final)
    return final


def _manifest(folder: Path) -> dict | None:
    try:
        manifest = json.loads((folder / MANIFEST).read_text())
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
        return None
    report = manifest.get("report")
    # only bundles trained by this game carry the report the page and the list show
    if (not isinstance(report, dict) or not isinstance(report.get("sessions"), list)
            or not isinstance(report.get("method"), str) or not isinstance(report.get("trained_at"), str)
            or not isinstance(report.get("balanced_accuracy"), (int, float)) or isinstance(report.get("balanced_accuracy"), bool)):
        return None
    return manifest


def listed(bundles_dir: Path) -> list[tuple[Path, dict]]:
    """(folder, manifest) for every bundle in ``bundles_dir``, oldest first."""
    bundles_dir = Path(bundles_dir)
    if not bundles_dir.is_dir():
        return []
    found = []
    for folder in bundles_dir.iterdir():
        manifest = _manifest(folder) if folder.is_dir() and not folder.name.startswith(".") else None
        if manifest is not None:
            found.append((folder, manifest))
    return sorted(found, key=lambda item: (str(item[1]["report"].get("trained_at", "")), item[0].name))


def latest(bundles_dir: Path) -> Path | None:
    bundles = listed(bundles_dir)
    return bundles[-1][0] if bundles else None


def latest_report(bundles_dir: Path) -> dict | None:
    bundles = listed(bundles_dir)
    if not bundles:
        return None
    folder, manifest = bundles[-1]
    return {**manifest["report"], "bundle": folder.name, "folder": str(folder)}

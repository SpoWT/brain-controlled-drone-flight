"""Train every candidate model on the rides marked IN, keep the best, and save it as a bundle in eeg_bundles/.

Rides -> labelled 1 s windows (``dataset``) -> each model in ``models`` is scored
leave-one-ride-out: every ride is predicted by a model that never saw it, and all
models are scored on the same windows. The model with the best held-out balanced
accuracy is refitted on all rides and saved; the leaderboard goes in its report.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import bundles, models
from .config import BUNDLES_DIR, CHANNELS, MAINS_HZ, RECORDINGS_DIR, SAMPLE_RATE
from .dataset import Progress, calibrate_asr, choose_rides, collect, load_ride, usable_rides

MIN_RIDES = 2
MIN_WINDOWS_PER_CLASS = 40
CLASS_NAMES = ["NOT_STOP", "STOP"]


class TrainingError(ValueError):
    """Not enough usable data to train."""


def scores(truth: np.ndarray, predicted: np.ndarray, folds: list[dict]) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

    accuracies = [fold["accuracy"] for fold in folds]
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "f1_score": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "stop_recall": float(recall_score(truth, predicted, pos_label=1, zero_division=0)),
        "go_recall": float(recall_score(truth, predicted, pos_label=0, zero_division=0)),
        "cv_mean": float(np.mean(accuracies)),
        "cv_std": float(np.std(accuracies)),
        "tested_windows": int(len(truth)),
        "session_metrics": folds,
    }


def evaluate(candidate, data: models.Data, progress: Progress | None = None) -> dict:
    """Leave-one-ride-out scores of one candidate, per ride and pooled."""
    from sklearn.metrics import accuracy_score, f1_score

    groups, y = data.windows.groups, data.y
    rides = sorted(set(groups))
    truth, predicted, folds = [], [], []
    for index, ride in enumerate(rides, start=1):
        test = groups == ride
        if len(set(y[~test])) < 2:
            continue
        if progress:
            progress(f"{candidate.label}: testing on ride {index}/{len(rides)}")
        fitted = candidate.fit(data, set(rides) - {ride})
        prediction = (fitted.stop_probability(data, np.flatnonzero(test)) >= fitted.threshold).astype(int)
        folds.append({
            "session_id": str(ride),
            "n_windows": int(test.sum()),
            "accuracy": float(accuracy_score(y[test], prediction)),
            "f1_score": float(f1_score(y[test], prediction, average="weighted", zero_division=0)),
        })
        truth.extend(y[test])
        predicted.extend(prediction)
    if len(folds) < 2:
        raise TrainingError("Each ride left out needs both GO and STOP in the other rides.")
    return scores(np.array(truth), np.array(predicted), folds)


def compare(candidates: list, data: models.Data, progress: Progress | None = None) -> list[dict]:
    """Leaderboard, best first; a model that cannot be trained is listed with the reason."""
    board = []
    for candidate in candidates:
        try:
            result = evaluate(candidate, data, progress)
        except TrainingError:
            raise
        except Exception as exc:
            board.append({"model": candidate.name, "label": candidate.label, "error": f"{type(exc).__name__}: {exc}"})
            continue
        board.append({"model": candidate.name, "label": candidate.label, "scores": result})
    order = {candidate.name: index for index, candidate in enumerate(candidates)}
    return sorted(board, key=lambda row: (-row["scores"]["balanced_accuracy"] if "scores" in row else 1,
                                          order[row["model"]]))


def _prepare(recordings_dir: Path, channels, mains_hz, candidates, progress) -> tuple[models.Data, list, bool, object]:
    manifests = usable_rides(recordings_dir)
    ids, simulated = choose_rides(manifests)
    if len(ids) < MIN_RIDES:
        mixed = " Ganglion" if len(manifests) > len(ids) else ""
        raise TrainingError(f"Training needs at least {MIN_RIDES}{mixed} rides marked IN; have {len(ids)}."
                            + (" Rides without the Ganglion are never mixed in." if mixed else ""))
    rides = [load_ride(recordings_dir, session_id) for session_id in ids]
    if progress:
        progress("Calibrating ASR")
    try:
        asr = calibrate_asr(rides, channels)
    except ValueError as exc:
        raise TrainingError(f"ASR calibration failed: {exc}") from exc
    data = models.Data(collect(rides, asr, progress), rides, channels, mains_hz, progress)
    if len(data.y) and any(candidate.needs_history for candidate in candidates):
        # every model is scored on the same windows: those with a Random Forest baseline
        features = data.forest_window_features(np.arange(len(data.y)))
        data = data.restricted(np.isfinite(features).all(axis=1))
    counts = {label: int((data.windows.labels == label).sum()) for label in CLASS_NAMES}
    if min(counts.values()) < MIN_WINDOWS_PER_CLASS:
        raise TrainingError(
            f"Need at least {MIN_WINDOWS_PER_CLASS} clean windows of both GO and STOP; "
            f"have GO {counts['NOT_STOP']}, STOP {counts['STOP']}."
        )
    return data, sorted(set(data.windows.groups)), simulated, asr


def _board_row(row: dict) -> dict:
    if "error" in row:
        return {"model": row["model"], "label": row["label"], "error": row["error"]}
    return {"model": row["model"], "label": row["label"],
            **{key: round(row["scores"][key], 3) for key in ("balanced_accuracy", "accuracy", "stop_recall", "go_recall")}}


def train(recordings_dir: Path = RECORDINGS_DIR, bundles_dir: Path = BUNDLES_DIR, *,
          name: str | None = None, progress: Progress | None = None, channels: tuple[str, ...] = CHANNELS,
          mains_hz: int = MAINS_HZ, candidates: list | None = None) -> dict:
    """Compare the candidates on every ride marked IN, save the best as a bundle; returns the report."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = bundles.check_name(name or f"stopgo-{stamp.replace('_', '-')}")
    if (Path(bundles_dir) / name).exists():
        raise FileExistsError(f"A bundle named {name} already exists")
    candidates = models.candidates() if candidates is None else candidates
    if not candidates:
        raise TrainingError("Choose at least one model to train.")
    data, ids, simulated, asr = _prepare(recordings_dir, channels, mains_hz, candidates, progress)
    board = compare(candidates, data, progress)
    if "scores" not in board[0]:
        raise TrainingError("No model could be trained: " + "; ".join(f"{row['label']}: {row['error']}" for row in board))
    best = board[0]
    winner = next(candidate for candidate in candidates if candidate.name == best["model"])
    if progress:
        progress(f"Fitting the final {winner.label} on every ride")
    fitted = winner.fit(data, set(ids))
    model_id = f"stopgo_{stamp}_{winner.name}"
    result = best["scores"]
    config = {
        "model_id": model_id,
        "model_type": winner.name,
        "trained_at": stamp,
        "class_names": CLASS_NAMES,
        "artifact_format": "torch" if winner.name in ("cfc", "ltc", "gru") else "joblib",
        "evaluation_mode": "leave_one_session_out",
        "channels": list(channels),
        "mains_hz": mains_hz,
        "simulated": simulated,
        "sample_rate": SAMPLE_RATE,
        **{key: result[key] for key in ("accuracy", "balanced_accuracy", "stop_recall", "go_recall", "f1_score",
                                        "cv_mean", "cv_std", "session_metrics")},
    }
    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "leave-one-ride-out",
        "model": winner.label,
        "model_type": winner.name,
        "sessions": ids,
        "windows": {"GO": int((data.y == 0).sum()), "STOP": int((data.y == 1).sum())},
        "chance": 0.5,
        "simulated": simulated,
        **{key: round(result[key], 3) for key in ("accuracy", "balanced_accuracy", "stop_recall", "go_recall")},
        "tested_windows": result["tested_windows"],
        "leaderboard": [_board_row(row) for row in board],
    }
    folder = bundles.write(bundles_dir, name, artifacts=fitted.save, config=config, asr=asr, report=report)
    return {**report, "bundle": folder.name, "folder": str(folder)}

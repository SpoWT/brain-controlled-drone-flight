"""The models training compares. Each is trained the way it was first built; all are
scored on the same held-out windows.

  xgboost, lightgbm   300 of the 487 window features, StandardScaler, gradient-boosted trees
  random_forest       causal band-power features against the past 30 s on a 125 ms grid,
                      300 trees; settings and STOP threshold chosen on the training rides only
  cfc, ltc, gru       recurrent networks on the filtered window, 24 epochs of Adam
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from bci_game_runtime import continuous_window

from . import labels
from .config import MODEL_NAMES, SAMPLE_RATE, WINDOW_SAMPLES
from .dataset import Progress, Ride, Windows
from .ml import rf_features
from .ml.features import extract_all_features, preprocess_window
from .ml.inference import window_quality

NAMES = MODEL_NAMES
MAX_FEATURES = 300
VARIANCE_FLOOR = 0.01
EARLY_STOPPING_ROUNDS = 100
PROGRESS_EVERY = 200
XGB_PARAMS = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8,
    "colsample_bytree": 0.7, "colsample_bylevel": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.5,
    "min_child_weight": 3, "gamma": 0.2, "random_state": 42, "n_jobs": -1, "eval_metric": "logloss",
}
LGBM_PARAMS = {
    "n_estimators": 500, "max_depth": 8, "num_leaves": 40, "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.5, "min_child_samples": 15,
    "random_state": 42, "n_jobs": -1, "verbose": -1,
}
RF_GRID = (
    {"min_samples_leaf": 10, "max_features": "sqrt"},
    {"min_samples_leaf": 50, "max_features": "sqrt"},
    {"min_samples_leaf": 10, "max_features": 0.33},
    {"min_samples_leaf": 50, "max_features": 0.33},
)
RF_FIXED = {"min_samples_leaf": 50, "max_features": "sqrt"}  # when nothing can be held out to choose settings
RF_SEED = 42


# -- data -----------------------------------------------------------------------
@dataclass(frozen=True)
class ForestRide:
    features: np.ndarray   # (grid points, features) on the ride's 125 ms grid
    rows: np.ndarray       # grid points with a label, finite features and a clean last second
    y: np.ndarray          # 1 = STOP for each of ``rows``


def _history_continuous(ride: Ride, end: int) -> bool:
    """Live, the Random Forest decides only on 35 s of history without a radio gap."""
    return continuous_window(ride.stamps[max(0, end + 1 - rf_features.HISTORY_SAMPLES): end + 1], SAMPLE_RATE)


def _grid_row(end):
    """Grid point of the window whose last sample is ``end`` (None when off the grid)."""
    offset = end + 1 - rf_features.WINDOW_SAMPLES
    return offset // rf_features.STEP_SAMPLES if offset % rf_features.STEP_SAMPLES == 0 else None


class Data:
    """One training run's windows and rides; derived inputs are computed once, on first use."""

    def __init__(self, windows: Windows, rides: list[Ride], channels, mains_hz, progress: Progress | None = None):
        self.windows = windows
        self.y = (windows.labels == "STOP").astype(int)
        self.rides = {ride.session_id: ride for ride in rides}
        self.channels = tuple(channels)
        self.mains_hz = mains_hz
        self.progress = progress or (lambda text: None)
        self._features = None
        self._forest: dict[str, ForestRide] = {}
        self._network = None

    def rows(self, rides) -> np.ndarray:
        return np.flatnonzero(np.isin(self.windows.groups, list(rides)))

    def restricted(self, keep: np.ndarray) -> "Data":
        data = Data(self.windows.subset(keep), list(self.rides.values()), self.channels, self.mains_hz, self.progress)
        data._forest = self._forest
        if self._features is not None:
            data._features = (self._features[0][keep], self._features[1])
        if self._network is not None:
            data._network = self._network[keep]
        return data

    def features(self) -> tuple[np.ndarray, list[str]]:
        """(windows, 487) feature matrix of the shared windows; non-finite values become 0."""
        if self._features is None:
            rows, names = [], None
            for index, window in enumerate(self.windows.eeg):
                features = extract_all_features(preprocess_window(window, mains_hz=self.mains_hz))
                names = names or list(features)
                rows.append([features[name] for name in names])
                if (index + 1) % PROGRESS_EVERY == 0:
                    self.progress(f"Features {index + 1}/{len(self.windows.eeg)}")
            matrix = np.nan_to_num(np.array(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            self._features = (matrix, names or [])
        return self._features

    def forest(self, ride_id: str) -> ForestRide:
        if ride_id not in self._forest:
            self.progress(f"Random Forest features, ride {len(self._forest) + 1}/{len(self.rides)}")
            self._forest[ride_id] = self._forest_ride(self.rides[ride_id])
        return self._forest[ride_id]

    def _forest_ride(self, ride: Ride) -> ForestRide:
        if len(ride.eeg) < rf_features.WINDOW_SAMPLES:
            return ForestRide(np.zeros((0, len(self.forest_names()))), np.zeros(0, dtype=int), np.zeros(0, dtype=int))
        features, _ = rf_features.stream_features(ride.eeg, rf_features.grid_ends(len(ride.eeg)),
                                                  self.channels, self.mains_hz)
        step_s = rf_features.STEP_SAMPLES / SAMPLE_RATE
        ends, window_labels = labels.windows(ride.stamps, ride.events, step_s=step_s)
        rows, y = [], []
        for end, label in zip(ends, window_labels):
            row = _grid_row(int(end))
            raw = ride.eeg[end - WINDOW_SAMPLES + 1: end + 1]
            if (row is None or not np.isfinite(features[row]).all() or window_quality(raw) == "reject"
                    or not _history_continuous(ride, int(end))):
                continue
            rows.append(row)
            y.append(int(label == "STOP"))
        return ForestRide(features, np.array(rows, dtype=int), np.array(y, dtype=int))

    def forest_names(self) -> list[str]:
        return rf_features.stream_features(np.zeros((rf_features.WINDOW_SAMPLES, 4)),
                                           np.array([rf_features.WINDOW_SAMPLES]), self.channels)[1]

    def forest_undecided(self, rows: np.ndarray) -> np.ndarray:
        """True where the Random Forest would make no decision live (so STOP): the raw last second fails the
        artifact gate, or its history has a radio gap."""
        undecided = []
        for ride_id, end in zip(self.windows.groups[rows], self.windows.ends[rows]):
            ride = self.rides[ride_id]
            raw = ride.eeg[end - WINDOW_SAMPLES + 1: end + 1]
            undecided.append(window_quality(raw) == "reject" or not _history_continuous(ride, int(end)))
        return np.array(undecided, dtype=bool)

    def forest_window_features(self, rows: np.ndarray) -> np.ndarray:
        """Random Forest features of shared windows (NaN rows where the baseline is too short)."""
        out = []
        for ride_id, end in zip(self.windows.groups[rows], self.windows.ends[rows]):
            row = _grid_row(int(end))
            features = self.forest(ride_id).features
            out.append(features[row] if row is not None else np.full(features.shape[1], np.nan))
        return np.array(out, dtype=float)

    def network_windows(self) -> np.ndarray:
        if self._network is None:
            from .ml import neural

            self._network = neural.preprocess_windows(self.windows.eeg, self.mains_hz)
        return self._network


# -- gradient-boosted trees on the window features --------------------------------
def select_features(x: np.ndarray, y: np.ndarray, max_features: int = MAX_FEATURES) -> np.ndarray:
    """Mask of features that vary, reduced to the ``max_features`` with the most mutual information."""
    from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

    mask = VarianceThreshold(threshold=VARIANCE_FLOOR).fit(x).get_support()
    if mask.sum() > max_features:
        kept = np.flatnonzero(mask)
        scores = mutual_info_classif(x[:, kept], y, random_state=42)
        mask = np.zeros(x.shape[1], dtype=bool)
        mask[kept[np.argsort(scores)[-max_features:]]] = True
    return mask


def _early_stopped_trees(make, x, y, fit_probe) -> int:
    """Tree count: best round of early stopping on a stratified 20 % split, + 10 (at least 25)."""
    from sklearn.model_selection import train_test_split

    if np.bincount(y, minlength=2).min() < 2:
        return 250
    x_fit, x_val, y_fit, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    return max(int(fit_probe(make(), x_fit, y_fit, x_val, y_val) or 0) + 10, 25)


def fit_xgboost(x, y):
    import xgboost as xgb

    def probe(model, x_fit, y_fit, x_val, y_val):
        model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
        return model.best_iteration

    trees = _early_stopped_trees(lambda: xgb.XGBClassifier(early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                                                           **XGB_PARAMS), x, y, probe)
    return xgb.XGBClassifier(**{**XGB_PARAMS, "n_estimators": trees}).fit(x, y)


def fit_lightgbm(x, y):
    import lightgbm as lgb

    def probe(model, x_fit, y_fit, x_val, y_val):
        model.fit(x_fit, y_fit, eval_X=x_val, eval_y=y_val,
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        return model.best_iteration_

    trees = _early_stopped_trees(lambda: lgb.LGBMClassifier(**LGBM_PARAMS), x, y, probe)
    return lgb.LGBMClassifier(**{**LGBM_PARAMS, "n_estimators": trees}).fit(x, y)


@dataclass
class FittedBoosted:
    mask: np.ndarray
    scaler: object
    model: object
    names: list
    threshold: float = 0.5

    def stop_probability(self, data: Data, rows: np.ndarray) -> np.ndarray:
        x = data.features()[0][rows][:, self.mask]
        return self.model.predict_proba(self.scaler.transform(x))[:, 1]

    def save(self, model_dir: Path, model_id: str) -> dict:
        joblib.dump(self.model, model_dir / f"model_{model_id}.joblib")
        joblib.dump(self.scaler, model_dir / f"scaler_{model_id}.joblib")
        selected = [name for name, keep in zip(self.names, self.mask) if keep]
        return {"artifact_filename": f"model_{model_id}.joblib", "scaler_artifact_filename": f"scaler_{model_id}.joblib",
                "feature_names": selected, "n_features": len(selected), "window_samples": WINDOW_SAMPLES,
                "uses_asr": True, "stop_threshold": self.threshold}


class Boosted:
    """XGBoost or LightGBM on the 300 most informative window features."""

    needs_history = False

    def __init__(self, name: str):
        self.name = name
        self.label = {"xgboost": "XGBoost", "lightgbm": "LightGBM"}[name]

    def fit(self, data: Data, rides) -> FittedBoosted:
        from sklearn.preprocessing import StandardScaler

        x_all, names = data.features()
        rows = data.rows(rides)
        x, y = x_all[rows], data.y[rows]
        mask = select_features(x, y)
        scaler = StandardScaler().fit(x[:, mask])
        model = (fit_xgboost if self.name == "xgboost" else fit_lightgbm)(scaler.transform(x[:, mask]), y)
        return FittedBoosted(mask, scaler, model, names)


# -- Random Forest on causal band-power features ----------------------------------
def ride_balanced_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Every (ride, class) cell gets the same total weight."""
    weights = np.zeros(len(y))
    for group in np.unique(groups):
        for label in (0, 1):
            cell = (groups == group) & (y == label)
            if cell.any():
                weights[cell] = 1.0 / cell.sum()
    return weights * len(weights) / weights.sum()


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    """Threshold with the best balanced accuracy on out-of-fold training predictions."""
    from sklearn.metrics import balanced_accuracy_score

    candidates = np.unique(np.quantile(probability, np.linspace(0.01, 0.99, 197)))
    scores = [balanced_accuracy_score(y, probability >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


@dataclass
class FittedForest:
    model: object
    names: list
    params: dict
    threshold: float

    def stop_probability(self, data: Data, rows: np.ndarray) -> np.ndarray:
        x = data.forest_window_features(rows)
        if not np.isfinite(x).all():
            raise ValueError("A test window has no Random Forest baseline")
        return np.where(data.forest_undecided(rows), 1.0, self.model.predict_proba(x)[:, 1])

    def save(self, model_dir: Path, model_id: str) -> dict:
        joblib.dump(self.model, model_dir / f"model_{model_id}.joblib")
        return {"artifact_filename": f"model_{model_id}.joblib", "feature_names": list(self.names),
                "n_features": len(self.names), "window_samples": rf_features.HISTORY_SAMPLES, "uses_asr": False,
                "step_samples": rf_features.STEP_SAMPLES, "forest_params": self.params,
                "stop_threshold": self.threshold}


class Forest:
    """Random Forest STOP / GO: 300 trees, ride-balanced weights, settings picked by mean per-ride
    AUC and the threshold by balanced accuracy, both on held-out training rides only. With fewer than
    3 rides in the run there is nothing to hold out inside a fold, so every fit, scored or saved,
    uses fixed settings and 0.5."""

    name, label = "random_forest", "Random Forest"
    needs_history = True

    def __init__(self, n_trees: int = 300, grid=RF_GRID):
        self.n_trees = n_trees
        self.grid = grid

    @staticmethod
    def tunes(data: Data) -> bool:
        """Tune when at least 3 compared rides have Random Forest rows, so every fold keeps 2 to choose with."""
        return sum(len(data.forest(ride).rows) > 0 for ride in set(data.windows.groups)) >= 3

    def _table(self, data: Data, rides) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, groups = [], [], []
        for ride_id in sorted(rides):
            table = data.forest(ride_id)
            x.append(table.features[table.rows])
            y.append(table.y)
            groups.append(np.full(len(table.rows), ride_id, dtype=object))
        return np.vstack(x), np.concatenate(y), np.concatenate(groups)

    def _forest(self, x, y, groups, params):
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=self.n_trees, random_state=RF_SEED, n_jobs=-1, **params)
        return model.fit(x, y, sample_weight=ride_balanced_weights(y, groups))

    def _out_of_fold(self, x, y, groups, params) -> np.ndarray:
        probability = np.full(len(y), np.nan)
        for held in np.unique(groups):
            test = groups == held
            if len(set(y[~test])) < 2:
                continue
            probability[test] = self._forest(x[~test], y[~test], groups[~test], params).predict_proba(x[test])[:, 1]
        return probability

    def _select(self, x, y, groups, tune: bool) -> tuple[dict, float]:
        from sklearn.metrics import roc_auc_score

        if not tune or len(np.unique(groups)) < 2:
            return dict(RF_FIXED), 0.5
        best = None
        for params in self.grid:
            oof = self._out_of_fold(x, y, groups, params)
            aucs = [roc_auc_score(y[groups == g], oof[groups == g]) for g in np.unique(groups)
                    if len(set(y[groups == g])) == 2 and np.isfinite(oof[groups == g]).all()]
            if aucs and (best is None or np.mean(aucs) > best[0]):
                best = (float(np.mean(aucs)), dict(params), oof)
        if best is None:  # no held-out training ride could be scored
            return dict(RF_FIXED), 0.5
        known = np.isfinite(best[2])
        threshold = choose_threshold(y[known], best[2][known]) if len(set(y[known])) == 2 else 0.5
        return best[1], threshold

    def fit(self, data: Data, rides) -> FittedForest:
        x, y, groups = self._table(data, rides)
        if len(set(y)) < 2:
            raise ValueError("The training rides need both GO and STOP windows")
        # decided by the whole run, so the folds that are scored and the final fit follow one rule
        params, threshold = self._select(x, y, groups, tune=self.tunes(data))
        return FittedForest(self._forest(x, y, groups, params), data.forest_names(), params, threshold)


# -- CfC / LTC / GRU networks ------------------------------------------------------
@dataclass
class FittedNetwork:
    name: str
    model: object
    mean: np.ndarray
    std: np.ndarray
    settings: dict
    threshold: float = 0.5

    def stop_probability(self, data: Data, rows: np.ndarray) -> np.ndarray:
        from .ml import neural

        patches = neural.network_input(data.network_windows()[rows], self.mean, self.std, self.settings["patch_size"])
        return neural.stop_probabilities(self.model, patches)

    def save(self, model_dir: Path, model_id: str) -> dict:
        from .ml import neural

        neural.torch.save(self.model.state_dict(), model_dir / f"model_{model_id}.pt")
        return {"artifact_filename": f"model_{model_id}.pt", "feature_names": [], "n_features": 0,
                "window_samples": WINDOW_SAMPLES, "uses_asr": True, "network": dict(self.settings),
                "norm_mean": self.mean.tolist(), "norm_std": self.std.tolist(), "stop_threshold": self.threshold}


class Network:
    """CfC, LTC or GRU: Adam, class-weighted cross-entropy, gradient clipping at 1.0."""

    needs_history = False

    def __init__(self, name: str, epochs: int = 24, batch_size: int = 128, learning_rate: float = 1e-3,
                 seed: int = 17, **settings):
        from .ml.neural import DEFAULTS, MODELS

        if name not in MODELS:
            raise ValueError(f"Unknown network {name!r}")
        self.name, self.label = name, {"cfc": "CfC", "ltc": "LTC", "gru": "GRU"}[name]
        self.epochs, self.batch_size, self.learning_rate, self.seed = epochs, batch_size, learning_rate, seed
        self.settings = {**DEFAULTS, **settings}

    def fit(self, data: Data, rides) -> FittedNetwork:
        from .ml import neural

        rows = data.rows(rides)
        windows, y = data.network_windows()[rows], data.y[rows]
        if len(set(y)) < 2:
            raise ValueError("The training rides need both GO and STOP windows")
        mean, std = neural.channel_stats(windows)
        patches = neural.network_input(windows, mean, std, self.settings["patch_size"])
        model = self._train(neural, patches, y)
        return FittedNetwork(self.name, model, mean, std, dict(self.settings))

    def _train(self, neural, patches, y):
        torch = neural.torch
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        model = neural.build_model(self.name, patches.shape[2], self.settings)
        counts = np.bincount(y, minlength=2).astype(np.float32)
        weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1.0)), dtype=torch.float32)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        x_all, y_all = torch.from_numpy(patches), torch.from_numpy(y.astype(np.int64))
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            model.train()
            for batch in torch.randperm(len(y), generator=generator).split(self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x_all[batch]), y_all[batch])
                loss.backward()
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)):
                    raise RuntimeError(f"{self.label} training diverged")
                optimizer.step()
        return model.eval()


def candidates(names=NAMES) -> list:
    """Fresh candidates for the given model names, in ``NAMES`` order."""
    unknown = set(names) - set(NAMES)
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(sorted(unknown))}. Choose from {', '.join(NAMES)}")
    build = {"xgboost": lambda: Boosted("xgboost"), "lightgbm": lambda: Boosted("lightgbm"),
             "random_forest": Forest, "cfc": lambda: Network("cfc"), "ltc": lambda: Network("ltc"),
             "gru": lambda: Network("gru")}
    return [build[name]() for name in NAMES if name in names]

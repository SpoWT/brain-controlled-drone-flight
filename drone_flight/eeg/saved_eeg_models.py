"""The flying website's EEG model catalog: trained STOP / GO bundles found by bci_game.runtime.

A local ``extra_eeg_models`` module, if present, can add more models (it is not part of the repo).
"""
from __future__ import annotations

import importlib
import json
import logging
import math
from datetime import datetime
from pathlib import Path

# PyTorch, XGBoost and LightGBM each ship an OpenMP runtime; on macOS the boosters crash when PyTorch's
# loads first. The website imports this module before any model, so load the boosters' runtime here.
for _booster in ("xgboost", "lightgbm"):
    try:
        importlib.import_module(_booster)
    except Exception:  # not installed or not loadable: nothing to order
        pass

from bci_game.runtime import bundle_roots, find_bundles  # noqa: E402
from drone_flight import ROOT  # noqa: E402

try:
    import extra_eeg_models
except ModuleNotFoundError as exc:
    if exc.name != "extra_eeg_models":
        raise
    extra_eeg_models = None

AUTO_MODEL_ID = "auto"
BUNDLE_PREFIX = "bundle-"
DEFAULT_STOP_THRESHOLD = 0.5
MODEL_TYPES = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "random_forest": "Random Forest",
               "cfc": "CfC", "ltc": "LTC", "gru": "GRU"}
DATE_KEYS = ('trained_at', 'trained_utc', 'training_completed_at', 'created_utc', 'created_at', 'saved_at')
LOGGER = logging.getLogger(__name__)


def _json(path):
    return json.loads(Path(path).read_text())


def _fraction(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


class SavedEEGModels:
    """Find model weights with their configuration and preprocessing."""

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.specs = {}
        self._discover()
        for spec in self.specs.values():
            spec.update(self._catalog_details(spec))
        self._recommendation = self._recommend_stop_go()

    def _add(self, model_id, label, group, kind, path, *, labels=("GO", "STOP"), samples=200, **extra):
        path = Path(path)
        if not path.is_file():
            return
        stop_go = {str(x).upper() for x in labels} in ({"STOP", "GO"}, {"STOP", "NOT_STOP"})
        self.specs[model_id] = dict(id=model_id, label=label, group=group, kind=kind,
                                    path=path, class_names=list(labels), window_samples=samples,
                                    stop_go=stop_go and kind != "stop-only", **extra)

    def _discover(self):
        for bundle in find_bundles(bundle_roots(self.root)):
            try:
                config = bundle.config()
                weights = bundle.model_dir / config["artifact_filename"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if config.get("simulated"):
                LOGGER.info("EEG bundle %s was trained on simulator rides; not offered for flight", bundle.folder.name)
                continue
            model_type = MODEL_TYPES.get(config.get("model_type"), str(config.get("model_type", "model")))
            try:
                samples = int(config.get("window_samples", 200))
            except (TypeError, ValueError):
                continue
            self._add(BUNDLE_PREFIX + bundle.folder.name, f"{bundle.name} · STOP / GO · {model_type}", "STOP / GO",
                      "bundle", weights, labels=list(config.get("class_names", [])), samples=samples,
                      config_path=bundle.config_path, bundle=bundle, note="Trained STOP / NOT_STOP bundle")
        if extra_eeg_models is not None:
            extra_eeg_models.discover(self)

    def selected(self, model_id):
        if not isinstance(model_id, str) or model_id not in self.specs:
            raise ValueError("Choose a model from the saved model list")
        return self.specs[model_id]

    @staticmethod
    def public(spec):
        return {key: spec[key] for key in ("id", "label", "group", "kind", "class_names", "window_samples", "stop_go", "note",
                                           "accuracy", "accuracy_source", "sort_timestamp", "date_source", "date_label") if key in spec}

    def _catalog_details(self, spec):
        """Saved evaluation and date metadata, without loading weights. Dates are this computer's local time."""
        details = {'accuracy': None, 'accuracy_source': '', 'sort_timestamp': None,
                   'date_source': '', 'date_label': 'date unavailable'}
        path = spec.get('metadata_path') or spec.get('config_path') or spec.get('evaluation_path')
        try:
            metadata = _json(path) if path else {}
            if not isinstance(metadata, dict):
                metadata = {}
        except (OSError, ValueError, TypeError):
            metadata = {}
        if spec['kind'] == 'bundle' and _fraction(metadata.get('balanced_accuracy')):
            value, source = metadata['balanced_accuracy'], 'held-out sessions, balanced'
        elif spec['kind'] == 'bundle':
            value, source = metadata.get('accuracy'), 'held-out sessions'
        elif extra_eeg_models is not None:
            value, source = extra_eeg_models.accuracy(spec, metadata)
        else:
            value, source = None, ''
        # anomaly scores and losses are not accuracy
        if _fraction(value):
            details.update(accuracy=float(value), accuracy_source=source)
        sources = [(key, metadata.get(key)) for key in DATE_KEYS if isinstance(metadata.get(key), str)]
        sources.append(('file', None))
        for key, text in sources:
            try:
                if key == 'file':
                    timestamp = spec['path'].stat().st_mtime
                else:
                    stamp = datetime.strptime(text, '%Y%m%d_%H%M%S') if len(text) == 15 and '_' in text else datetime.fromisoformat(text.replace('Z', '+00:00'))
                    timestamp = stamp.timestamp()
                label = datetime.fromtimestamp(timestamp).strftime('%b %d, %Y')
            except (ValueError, OverflowError, OSError, KeyError, AttributeError):
                continue
            source = 'file' if key == 'file' else 'trained' if key.startswith('train') else 'saved'
            details.update(sort_timestamp=timestamp, date_source=source,
                           date_label=label + ' · ' + {'trained': 'trained', 'saved': 'saved', 'file': 'file date'}[source])
            break
        return details

    def catalog(self):
        return [self.public(spec) for spec in self.specs.values()]

    def recommendation(self):
        return dict(self._recommendation) if self._recommendation is not None else None

    def resolve_selection(self, model_id):
        if model_id == AUTO_MODEL_ID:
            if self._recommendation is None:
                raise ValueError('No comparable held-out STOP / GO evaluation is available for Auto')
            return self._recommendation['model_id']
        return self.selected(model_id)['id']

    def default_id(self):
        """Auto's choice, else the newest STOP / GO bundle, else None."""
        if self._recommendation is not None:
            return self._recommendation['model_id']
        bundles = [spec for spec in self.specs.values() if spec['kind'] == 'bundle']
        if not bundles:
            return None
        return max(bundles, key=lambda spec: (spec.get('sort_timestamp') or 0, spec['id']))['id']

    def stop_threshold(self, model_id):
        """The STOP threshold a bundle was trained and scored with, else 0.5."""
        spec = self.specs.get(model_id) if isinstance(model_id, str) else None
        try:
            value = _json(spec['config_path'])['stop_threshold'] if spec and spec['kind'] == 'bundle' else None
        except (OSError, ValueError, KeyError, TypeError):
            value = None
        return float(value) if _fraction(value) and 0.01 <= value <= 0.99 else DEFAULT_STOP_THRESHOLD

    @staticmethod
    def _bundle_score(spec):
        """(pooled held-out accuracy, balanced accuracy or None, {session: windows}) from a bundle's saved
        folds, or None."""
        try:
            config = _json(spec['config_path'])
            folds = config['session_metrics']
            if any(not isinstance(row['n_windows'], int) or isinstance(row['n_windows'], bool)
                   or row['n_windows'] <= 0 or not _fraction(row['accuracy'])
                   for row in folds):
                return None
            cohort = {row['session_id']: row['n_windows'] for row in folds}
            if (config['evaluation_mode'] != 'leave_one_session_out'
                    or set(config['class_names']) != {'NOT_STOP', 'STOP'}
                    or len(cohort) < 2 or len(cohort) != len(folds)):
                return None
            score = sum(row['accuracy'] * row['n_windows'] for row in folds) / sum(cohort.values())
            if not _fraction(score):
                return None
            balanced = float(config['balanced_accuracy']) if _fraction(config.get('balanced_accuracy')) else None
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return score, balanced, cohort

    def _recommend_stop_go(self):
        """Rank STOP models by complete held-out-session runs, not single folds, all on one metric:
        balanced accuracy when every run has it (bundles from the BCI game), else plain accuracy."""
        candidates = []
        for spec in self.specs.values():
            scored = self._bundle_score(spec) if spec['kind'] == 'bundle' else None
            if scored is not None:
                candidates.append({'spec': spec, 'score': scored[0], 'balanced': scored[1], 'cohort': scored[2]})
        if not candidates:
            return None
        if extra_eeg_models is not None:
            candidates += extra_eeg_models.candidates(self.specs, [row['cohort'] for row in candidates])
        metric = 'held-out accuracy'
        if all(row.get('balanced') is not None for row in candidates):
            metric = 'held-out balanced accuracy'
            candidates = [{**row, 'score': row['balanced']} for row in candidates]
        best = min(candidates, key=lambda row: (-row['score'], row['spec']['id']))
        spec, sessions = best['spec'], len(best['cohort'])
        return {
            'model_id': spec['id'], 'label': spec['label'], 'metric': metric,
            'score': best['score'], 'sessions': sessions, 'windows': sum(best['cohort'].values()),
            'compared_runs': len(candidates), 'task': 'STOP / NOT_STOP state',
            'reason': f"Suggested: {spec['label']} · {best['score']:.1%} {metric} across {sessions} sessions. Best of {len(candidates)} STOP / GO runs · offline evaluation.",
            'scope': 'STOP / GO runs scored on whole sessions they never trained on; other tasks have separate evaluations',
            'representative_checkpoint': spec['kind'] != 'bundle',
        }

    def load(self, model_id):
        spec = self.selected(model_id)
        if spec["kind"] == "bundle":
            from bci_game.runtime import BCIGameStopGoRuntime
            return BCIGameStopGoRuntime(spec["bundle"])
        if extra_eeg_models is None:
            raise ValueError("No loader for this model")
        return extra_eeg_models.load(spec, self.root)

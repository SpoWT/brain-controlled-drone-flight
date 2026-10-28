"""STOP / GO bundles: discovery, hashes, code isolation and the model catalog."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

from bci_game_runtime import (BCIGameStopGoRuntime, bundle_roots, continuous_window, default_bundle, file_sha256,
                              find_bundles, read_bundle, verify_bundle)

ENGINE = '''import json, os

class InferenceEngine:
    def __init__(self):
        self.class_names, self.feature_names, self.stop_threshold = [], [], 0.5
        self._vote_buffer, self._last_state, self._last_change_ms = [], "STOP", 0

    def load(self, model_dir, model_id):
        with open(os.path.join(model_dir, "config_" + model_id + ".json")) as handle:
            config = json.load(handle)
        self.class_names, self.feature_names = config["class_names"], config["feature_names"]
        for key in ("window_samples", "uses_asr"):
            if key in config:
                setattr(self, key, config[key])

    def predict(self, window):
        return {"decision": DECISION, "raw_decision": DECISION, "quality": "clean",
                "probabilities": {"NOT_STOP": 0.5, "STOP": 0.5}}
'''
ASR = '''class ASRProcessor:
    calibrated = False
    transforms = 0

    def load(self, path):
        with open(path) as handle:
            self.calibrated = handle.read() == "ok"
        return self.calibrated

    def transform(self, eeg):
        self.transforms += 1
        return eeg
'''


def make_bundle(folder, *, decision="NOT_STOP", trained_at="20260101_000000", unlisted=False, accuracy=0.6,
                class_names=("NOT_STOP", "STOP"), asr="ok", stop_threshold=0.5, simulated=False, model_type="xgboost",
                scaler=True, engine_settings=None, **manifest_changes):
    """A small bundle whose engine always returns ``decision``; ``unlisted`` omits the manifest.
    ``engine_settings`` adds config keys the engine reads (``window_samples``, ``uses_asr``)."""
    folder = Path(folder)
    model_id = "m_" + folder.name
    config = {"model_id": model_id, "trained_at": trained_at, "model_type": model_type,
              "class_names": list(class_names), "feature_names": ["f"], "n_features": 1,
              "accuracy": accuracy, "cv_mean": accuracy, "stop_threshold": stop_threshold, "simulated": simulated,
              "artifact_filename": f"model_{model_id}.joblib", "scaler_artifact_filename": f"scaler_{model_id}.joblib",
              "evaluation_mode": "leave_one_session_out",
              "session_metrics": [{"session_id": "a", "n_windows": 50, "accuracy": accuracy},
                                  {"session_id": "b", "n_windows": 50, "accuracy": accuracy}]}
    config.update(engine_settings or {})
    if not scaler:
        del config["scaler_artifact_filename"]
    engine = ENGINE.replace("DECISION", repr(decision))
    if unlisted:
        files = {f"model/config_{model_id}.json": json.dumps(config), f"model/model_{model_id}.joblib": "m",
                 f"model/scaler_{model_id}.joblib": "s", "dataset/asr/asr_state_x.pkl": asr,
                 "repro/ml/inference.py": engine, "repro/server/asr_processor.py": ASR}
    else:
        files = {f"model/config_{model_id}.json": json.dumps(config), f"model/model_{model_id}.joblib": "m",
                 f"model/scaler_{model_id}.joblib": "s", "asr_state.pkl": asr, "code/ml/__init__.py": "",
                 "code/ml/inference.py": engine, "code/ml/asr.py": ASR}
        if not scaler:
            del files[f"model/scaler_{model_id}.joblib"]
    for relative, text in files.items():
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / relative).write_text(text)
    if not unlisted:
        manifest = {"format": "stop-go-bundle/1", "label": folder.name, "model_id": model_id, "model_dir": "model",
                    "asr_state": "asr_state.pkl", "code": "code", "engine": "ml.inference:InferenceEngine",
                    "asr": "ml.asr:ASRProcessor",
                    "sha256": {relative: file_sha256(folder / relative) for relative in files}}
        manifest.update(manifest_changes)
        (folder / "bundle.json").write_text(json.dumps(manifest))
    return folder


WINDOW = np.random.default_rng(0).normal(0, 10, (200, 4))


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_bundles_with_and_without_manifest_are_found(self):
        make_bundle(self.root / "new")
        make_bundle(self.root / "old", unlisted=True)
        make_bundle(self.root / ".partial")
        (self.root / "not_a_bundle").mkdir()
        found = {bundle.folder.name: bundle for bundle in find_bundles([self.root])}
        self.assertEqual(set(found), {"new", "old"})
        self.assertTrue(found["new"].sha256)
        self.assertEqual(found["old"].sha256, {})
        self.assertEqual(found["old"].asr, "server.asr_processor:ASRProcessor")

    def test_a_root_can_be_a_bundle_and_is_listed_once(self):
        folder = make_bundle(self.root / "only")
        self.assertEqual([bundle.folder for bundle in find_bundles([folder, self.root])], [folder])

    def test_environment_sets_the_search_folders(self):
        with mock.patch.dict(os.environ, {"EEG_BUNDLES": os.pathsep.join(["/a", "/b"])}):
            self.assertEqual(bundle_roots(self.root), [Path("/a"), Path("/b")])
        for empty in ("", os.pathsep * 2, " "):
            with mock.patch.dict(os.environ, {"EEG_BUNDLES": empty}):
                self.assertEqual(bundle_roots(self.root), [self.root / "eeg_bundles", self.root])

    def test_default_is_the_newest_bundle(self):
        make_bundle(self.root / "older", trained_at="20260101_000000")
        make_bundle(self.root / "newer", trained_at="20260601_000000")
        broken = make_bundle(self.root / "broken")
        next((broken / "model").glob("config_*.json")).write_text("[]")
        self.assertEqual(default_bundle([self.root]).folder.name, "newer")
        with self.assertRaises(FileNotFoundError):
            default_bundle([self.root / "missing"])

    def test_a_manifest_must_be_an_object(self):
        folder = self.root / "list"
        folder.mkdir()
        (folder / "bundle.json").write_text("[]")
        with self.assertRaises(ValueError):
            read_bundle(folder)
        with self.assertLogs("bci_game_runtime", "WARNING"):
            self.assertEqual(find_bundles([self.root]), [])

    def test_manifests_cannot_point_outside_their_folder(self):
        for changes in ({"model_dir": "../elsewhere"}, {"code": "/"}, {"engine": "os:system()"},
                        {"format": "other"}, {"sha256": {}}):
            folder = make_bundle(self.root / "bad", **changes)
            with self.assertRaises(ValueError):
                read_bundle(folder)
            with self.assertLogs("bci_game_runtime", "WARNING"):
                self.assertEqual(find_bundles([self.root]), [])


class WindowRuleTests(unittest.TestCase):
    def test_gaps_and_wrong_spans_break_a_window(self):
        times = 100 + np.arange(200) / 200
        self.assertTrue(continuous_window(times))
        bursty = times.copy()
        bursty[100:] += 0.2  # BLE bursts up to 0.25 s are normal
        self.assertTrue(continuous_window(bursty))
        gap = times.copy()
        gap[100:] += 0.3
        self.assertFalse(continuous_window(gap))
        self.assertFalse(continuous_window(100 + np.arange(200) / 400))  # too short a span
        self.assertFalse(continuous_window(times[:1]))
        self.assertFalse(continuous_window(np.full(200, np.nan)))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_changed_or_unlisted_files_are_refused(self):
        folder = make_bundle(self.root / "b")
        (folder / "model" / "model_m_b.joblib").write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            BCIGameStopGoRuntime(folder)
        folder = make_bundle(self.root / "c")
        (folder / "code" / "ml" / "extra.py").write_text("")
        with self.assertRaisesRegex(RuntimeError, "without a recorded hash"):
            verify_bundle(read_bundle(folder))

    def test_each_bundle_runs_its_own_code(self):
        go = BCIGameStopGoRuntime(make_bundle(self.root / "go", decision="NOT_STOP"))
        stop = BCIGameStopGoRuntime(make_bundle(self.root / "stop", decision="STOP"))
        old = BCIGameStopGoRuntime(make_bundle(self.root / "old", decision="NOT_STOP", unlisted=True))
        self.assertEqual(go.predict(WINDOW)["decision"], "NOT_STOP")
        self.assertEqual(stop.predict(WINDOW)["decision"], "STOP")
        self.assertEqual(old.predict(WINDOW)["model_id"], "m_old")
        self.assertFalse(any(name.split(".")[0] in {"ml", "server"} for name in sys.modules))
        self.assertEqual(go.info["bundle"], "go")

    def test_bad_bundles_do_not_load(self):
        with self.assertRaisesRegex(RuntimeError, "class order"):
            BCIGameStopGoRuntime(make_bundle(self.root / "order", class_names=("STOP", "NOT_STOP")))
        with self.assertRaisesRegex(RuntimeError, "ASR"):
            BCIGameStopGoRuntime(make_bundle(self.root / "asr", asr="broken"))
        with self.assertRaises(FileNotFoundError):
            BCIGameStopGoRuntime(self.root / "missing")

    def test_window_length_and_asr_follow_the_bundle(self):
        folder = make_bundle(self.root / "forest", scaler=False, model_type="random_forest",
                             engine_settings={"window_samples": 7000, "uses_asr": False})
        forest = BCIGameStopGoRuntime(folder)
        self.assertEqual((forest.WINDOW_SAMPLES, forest.info["window_samples"]), (7000, 7000))
        self.assertEqual(forest.predict(WINDOW)["decision"], "STOP")  # a 1 s window is too short
        result = forest.predict(np.random.default_rng(1).normal(0, 10, (7000, 4)))
        self.assertEqual(result["decision"], "NOT_STOP")
        self.assertFalse(result["asr_applied"])
        self.assertEqual(forest.asr.transforms, 0)
        boosted = BCIGameStopGoRuntime(make_bundle(self.root / "boosted"))
        self.assertEqual(boosted.WINDOW_SAMPLES, 200)
        self.assertTrue(boosted.predict(WINDOW)["asr_applied"])
        self.assertEqual(BCIGameStopGoRuntime.WINDOW_SAMPLES, 200)

    def test_unclear_input_or_output_is_stop(self):
        live = BCIGameStopGoRuntime(make_bundle(self.root / "odd", decision="NO_DECISION"))
        self.assertTrue(live.predict(WINDOW)["safe_fallback"])
        self.assertEqual(live.predict(WINDOW)["decision"], "STOP")
        for window in (np.zeros((10, 4)), np.full((200, 4), np.nan)):
            self.assertEqual(live.predict(window)["decision"], "STOP")


class CatalogTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        patch = mock.patch.dict(os.environ, {"EEG_BUNDLES": str(self.root)})
        patch.start()
        self.addCleanup(patch.stop)

    def test_trained_bundles_are_listed_and_auto_picks_the_best(self):
        from saved_eeg_models import SavedEEGModels

        make_bundle(self.root / "weaker", accuracy=0.55, trained_at="20260601_000000")
        make_bundle(self.root / "stronger", accuracy=0.65)
        models = SavedEEGModels(root=self.root)
        bundles = {spec["id"]: spec for spec in models.catalog() if spec["group"] == "STOP / GO"}
        self.assertEqual(set(bundles), {"bundle-weaker", "bundle-stronger"})
        self.assertTrue(all(spec["stop_go"] and spec["kind"] == "bundle" for spec in bundles.values()))
        self.assertEqual(bundles["bundle-stronger"]["label"], "stronger · STOP / GO · XGBoost")
        self.assertEqual(models.recommendation()["model_id"], "bundle-stronger")
        self.assertEqual(models.resolve_selection("auto"), "bundle-stronger")
        self.assertEqual(models.default_id(), "bundle-stronger")
        self.assertEqual(models.load("bundle-weaker").predict(WINDOW)["decision"], "NOT_STOP")

    def test_simulator_bundles_are_never_offered_for_flight(self):
        from saved_eeg_models import SavedEEGModels

        make_bundle(self.root / "simulated", simulated=True)
        make_bundle(self.root / "headset")
        self.assertEqual({spec["id"] for spec in SavedEEGModels(root=self.root).catalog()}, {"bundle-headset"})

    def test_the_website_starts_from_the_bundle_threshold(self):
        from optical_eeg import LiveEEGService

        make_bundle(self.root / "careful", stop_threshold=0.6)
        service = LiveEEGService(settings_path=None)
        status = service.status()
        self.assertEqual((status["model_id"], status["model_selection"]), ("bundle-careful", "auto"))
        self.assertEqual(status["stop_threshold"], 0.6)
        service.close()

    def test_dates_use_this_computers_time_without_a_time_zone_database(self):
        from saved_eeg_models import SavedEEGModels

        make_bundle(self.root / "dated", trained_at="20260115_093000")
        with mock.patch("zoneinfo.ZoneInfo", side_effect=KeyError("no time zone database")):
            spec = SavedEEGModels(root=self.root).specs["bundle-dated"]
        self.assertEqual(spec["sort_timestamp"], datetime(2026, 1, 15, 9, 30, 0).timestamp())
        self.assertEqual(spec["date_label"], "Jan 15, 2026 · trained")

    def test_a_saved_threshold_wins_until_then_the_bundle_threshold_follows_the_model(self):
        from optical_eeg import LiveEEGService

        make_bundle(self.root / "best", stop_threshold=0.6, accuracy=0.7)
        make_bundle(self.root / "other", stop_threshold=0.4)
        settings = self.root / "settings.json"
        settings.write_text(json.dumps({"stop_threshold": 0.66, "model_id": "bundle-other"}))
        self.assertEqual(LiveEEGService(settings_path=settings).status()["stop_threshold"], 0.66)
        settings.unlink()
        service = LiveEEGService(settings_path=settings)
        self.assertEqual(service.status()["stop_threshold"], 0.6)
        self.assertEqual(service.select_model("bundle-other")["stop_threshold"], 0.4)
        self.assertEqual(service.select_model("auto")["stop_threshold"], 0.6)
        restored = LiveEEGService(settings_path=settings).status()
        self.assertEqual((restored["model_selection"], restored["stop_threshold"]), ("auto", 0.6))
        service.set_threshold(0.8)
        self.assertEqual(service.select_model("bundle-other")["stop_threshold"], 0.8)
        self.assertEqual(LiveEEGService(settings_path=settings).status()["stop_threshold"], 0.8)
        service.close()

    def test_shared_eeg_code_needs_no_time_zone_database(self):
        for name in ("saved_eeg_models.py", "bci_game_runtime.py", "optical_eeg.py"):
            text = (Path(__file__).resolve().parents[1] / name).read_text()
            self.assertNotIn("zoneinfo", text, name)
            self.assertNotIn("America/", text, name)

    def test_the_catalog_names_each_model_type_and_its_window(self):
        from saved_eeg_models import SavedEEGModels

        make_bundle(self.root / "forest", scaler=False, model_type="random_forest",
                    engine_settings={"window_samples": 7000, "uses_asr": False})
        make_bundle(self.root / "liquid", model_type="cfc")
        specs = SavedEEGModels(root=self.root).specs
        self.assertEqual(specs["bundle-forest"]["label"], "forest · STOP / GO · Random Forest")
        self.assertEqual(specs["bundle-forest"]["window_samples"], 7000)
        self.assertEqual((specs["bundle-liquid"]["label"], specs["bundle-liquid"]["window_samples"]),
                         ("liquid · STOP / GO · CfC", 200))

    def test_auto_ranks_every_bundle_on_one_metric(self):
        from saved_eeg_models import SavedEEGModels

        # plain accuracy and balanced accuracy rank these two in opposite orders
        make_bundle(self.root / "steady", accuracy=0.60, engine_settings={"balanced_accuracy": 0.58})
        make_bundle(self.root / "lopsided", accuracy=0.64, engine_settings={"balanced_accuracy": 0.52})
        recommendation = SavedEEGModels(root=self.root).recommendation()
        self.assertEqual((recommendation["model_id"], recommendation["metric"]),
                         ("bundle-steady", "held-out balanced accuracy"))
        make_bundle(self.root / "older", accuracy=0.62)  # no balanced accuracy: everyone on plain accuracy
        recommendation = SavedEEGModels(root=self.root).recommendation()
        self.assertEqual((recommendation["model_id"], recommendation["metric"]),
                         ("bundle-lopsided", "held-out accuracy"))
        self.assertIn("held-out accuracy across 2 sessions", recommendation["reason"])

    def test_without_bundles_the_eeg_panel_says_so(self):
        from optical_eeg import LiveEEGService

        service = LiveEEGService(settings_path=None)
        status = service.status()
        self.assertIsNone(status["model_id"])
        self.assertFalse(status["selected_model"]["stop_go"])
        self.assertIn("eeg_bundles", status["selected_model"]["note"])
        service._load_selected_model()
        self.assertIn("eeg_bundles", service.status()["prediction"]["reason"])
        service.close()



class WebsiteWindowTests(unittest.TestCase):
    """The flying website feeds each model the window length it asks for."""

    def test_a_35_second_model_gets_35_seconds_of_continuous_eeg(self):
        from optical_eeg import LiveEEGService

        now = [100.0]
        service = LiveEEGService(settings_path=None, clock=lambda: now[0], wall_clock=lambda: now[0])
        service._enabled, service._generation = True, 1
        seen = []

        class Forest:
            WINDOW_SAMPLES = 7000

            def set_stop_threshold(self, value):
                pass

            def reset(self):
                pass

            def predict(self, eeg):
                seen.append(eeg.shape)
                return {"raw_decision": "STOP", "decision": "STOP", "quality": "clean",
                        "probabilities": {"STOP": 0.7, "NOT_STOP": 0.3}}

        service._model, service._model_loading = Forest(), False
        rng = np.random.default_rng(0)

        def feed(seconds, gap=0.0):
            n = int(seconds * 200)
            times = now[0] + gap + np.arange(1, n + 1) / 200
            now[0] = times[-1]
            service._accept_chunk(1, rng.normal(0, 10, (n, 4)), times, 200)
            service._infer_once(1)
            return service.status()["prediction"]

        self.assertFalse(feed(30)["valid"])
        self.assertEqual(seen, [])
        self.assertTrue(feed(6)["valid"])
        self.assertEqual(seen, [(7000, 4)])
        self.assertFalse(feed(1, gap=0.5)["valid"])  # a radio gap inside the last 35 s
        self.assertEqual(seen, [(7000, 4)])
        # 35 s of samples, each step short, but stretched over 63 s by small dropouts
        times = now[0] + np.cumsum(np.where(np.arange(7000) % 50 == 49, 0.205, 0.005))
        now[0] = times[-1]
        service._accept_chunk(1, rng.normal(0, 10, (7000, 4)), times, 200)
        service._infer_once(1)
        self.assertEqual(service.status()["prediction"]["reason"], "Check the four electrode contacts")
        self.assertEqual(seen, [(7000, 4)])
        service.close()

    def test_a_settings_file_that_is_not_an_object_is_ignored(self):
        from optical_eeg import LiveEEGService

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("[1, 2]")
            service = LiveEEGService(settings_path=path)
            self.assertFalse(service.status()["enabled"])
            service.close()

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "xgboost", "lightgbm")),
                         "needs torch, xgboost and lightgbm")
    def test_the_website_orders_the_openmp_runtimes(self):
        code = ("import saved_eeg_models, numpy as np, torch, xgboost, lightgbm\n"
                "torch.nn.GRU(40, 8)(torch.zeros(4, 20, 40))[0].sum().backward()\n"
                "x = np.random.default_rng(0).normal(size=(400, 50)); y = (x[:, 0] > 0).astype(int)\n"
                "xgboost.XGBClassifier(n_estimators=20, n_jobs=-1).fit(x, y)\n"
                "lightgbm.LGBMClassifier(n_estimators=20, verbose=-1).fit(x, y)\n"
                "print('ok')")
        project = Path(__file__).resolve().parents[1]
        run = subprocess.run([sys.executable, "-c", code], cwd=project, capture_output=True, text=True, timeout=300)
        self.assertEqual((run.returncode, run.stdout.strip()), (0, "ok"), run.stderr[-2000:])

if __name__ == "__main__":
    unittest.main()

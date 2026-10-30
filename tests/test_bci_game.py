"""Recording game and training: storage, labels, features, training, bundles, drive mode and HTTP."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bci_game import bundles, dataset, labels, models, session, train
from bci_game.__main__ import main
from bci_game.config import GO, SAMPLE_RATE, STOP, parse_channels, parse_mains
from bci_game.dsp import channel_quality
from bci_game.ml import neural, quant_features, rf_features
from bci_game.ml.features import extract_all_features, feature_vector, preprocess_window
from bci_game.ml.inference import InferenceEngine, window_quality
from bci_game.server import create_app
from bci_game.stream import GanglionSource, SourceError
from bci_game.studio import LiveDecoder, Studio, StudioError
from bci_game.runtime import BCIGameStopGoRuntime, find_bundles, read_bundle, verify_bundle

PROJECT = Path(__file__).resolve().parents[1]

T0 = 1_800_000_000.0


def ride_events(seconds: int, period: float = 5.0, start: float = T0) -> list[dict]:
    """Alternate GO / STOP every ``period`` seconds, with a hazard before each change."""
    events = [{"type": "ride_start", "t": start * 1000}]
    down = False
    t = start + 1.0
    while t < start + seconds - 1:
        down = not down
        phase = "clear" if down else "demand_stop"
        lead = 0.5 if down else 0.4
        events.append({"type": "hazard", "t": (t - lead) * 1000, "kind": "red_light", "phase": phase, "id": f"h{t}"})
        events.append({"type": "key", "t": t * 1000, "down": down})
        t += period
    events.append({"type": "ride_end", "t": (start + seconds) * 1000})
    return events


def synthetic_eeg(seconds: int, events: list[dict], seed: int = 0, start: float = T0):
    rng = np.random.default_rng(seed)
    stamps = start + np.arange(seconds * SAMPLE_RATE) / SAMPLE_RATE
    driving = labels.driving_at(stamps, labels.key_transitions(events))
    t = np.arange(len(stamps)) / SAMPLE_RATE
    alpha = np.where(driving, 3.0, 12.0) * np.sin(2 * np.pi * 10 * t)
    eeg = rng.normal(0, 5, size=(len(stamps), 4)) + alpha[:, None] * np.array([1.0, 0.8, 1.1, 0.9])
    return eeg, stamps


def write_ride(root, seconds: int = 120, seed: int = 0) -> str:
    recorder = session.Recorder.create("ride", 2, root, source="ganglion")
    events = ride_events(seconds)
    eeg, stamps = synthetic_eeg(seconds, events, seed)
    for first in range(0, len(stamps), 37):
        recorder.write_eeg(eeg[first:first + 37], stamps[first:first + 37])
    recorder.write_events(events)
    recorder.finish(labels.summarize(events))
    return recorder.session_id


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Two synthetic rides and the bundle trained on them, once per run."""
    root = tmp_path_factory.mktemp("trained")
    recordings, bundle_dir = root / "recordings", root / "bundles"
    for seed in range(2):
        write_ride(recordings, seed=seed)
    report = train.train(recordings, bundle_dir, name="synthetic", mains_hz=60, candidates=[models.Boosted("xgboost")])
    return {"recordings": recordings, "bundles": bundle_dir, "report": report, "folder": bundle_dir / report["bundle"]}


# -- storage ----------------------------------------------------------------
def test_recorder_round_trips_eeg_and_events(tmp_path):
    recorder = session.Recorder.create("ride", 2, tmp_path)
    eeg = np.arange(40, dtype=float).reshape(10, 4)
    stamps = T0 + np.arange(10) / SAMPLE_RATE
    recorder.write_eeg(eeg, stamps)
    recorder.write_events([{"type": "key", "t": T0 * 1000, "down": True}])
    manifest = recorder.finish({"stops": 0})
    loaded_eeg, loaded_stamps = session.load_eeg(recorder.directory)
    assert np.array_equal(loaded_eeg, eeg) and np.allclose(loaded_stamps, stamps)
    assert session.load_events(recorder.directory)[0]["down"] is True
    assert manifest["samples"] == 10 and manifest["excluded"] is False


@pytest.mark.parametrize("bad", [
    {"type": "explode", "t": 1},
    {"type": "key", "t": float("nan")},
    {"type": "hazard", "t": 1, "kind": "ufo", "phase": "appear"},
    {"type": "key", "t": 1, "nested": {"a": 1}},
    "not an object",
])
def test_invalid_events_are_rejected(bad):
    with pytest.raises(session.SessionError):
        session.validate_event(bad)


def test_session_ids_cannot_escape_the_data_folder(tmp_path):
    with pytest.raises(session.SessionError):
        session.session_dir("../../etc", tmp_path)


def test_saved_ride_cannot_take_more_events(tmp_path):
    recorder = session.Recorder.create("ride", 2, tmp_path)
    recorder.finish({})
    with pytest.raises(session.SessionError):
        recorder.write_events([{"type": "key", "t": 1, "down": True}])


def test_channel_names_must_be_four_different_names():
    assert parse_channels("Fz, Cz ,F7,F8") == ("Fz", "Cz", "F7", "F8")
    for bad in ("Fz,Cz,F7", "Fz,Fz,F7,F8", ""):
        with pytest.raises(ValueError):
            parse_channels(bad)


# -- labels -----------------------------------------------------------------
def test_windows_never_straddle_a_key_change():
    events = ride_events(60)
    stamps = T0 + np.arange(60 * SAMPLE_RATE) / SAMPLE_RATE
    ends, window_labels = labels.windows(stamps, events)
    edges = [t for t, _ in labels.key_transitions(events)]
    for end in ends:
        t0, t1 = stamps[end - SAMPLE_RATE + 1], stamps[end]
        assert not any(t0 < edge <= t1 for edge in edges)
    assert set(window_labels) == {GO, STOP}


def test_windows_skip_the_settle_period_and_radio_gaps():
    events = ride_events(30)
    stamps = T0 + np.arange(30 * SAMPLE_RATE) / SAMPLE_RATE
    stamps[3000:] += 2.0  # a 2 s gap at 15 s
    stamps[4000:] += 0.3  # a 0.3 s gap at 20 s: the flying website rejects it, so training does too
    ends, _ = labels.windows(stamps, events, step_s=0.25)
    assert all(stamps[end - SAMPLE_RATE + 1] >= T0 + labels.SETTLE_S for end in ends)
    assert all(np.max(np.diff(stamps[end - SAMPLE_RATE + 1: end + 1])) <= 0.25 for end in ends)
    assert not any(end - SAMPLE_RATE + 1 < 4000 <= end for end in ends)


def test_summary_reports_reaction_times():
    summary = labels.summarize(ride_events(60))
    assert summary["stops"] > 0
    assert summary["stop_reaction_ms"] == 400
    assert summary["go_reaction_ms"] == 500
    assert summary["late"] == 0


def test_collisions_count_as_late():
    events = ride_events(30) + [{"type": "collision", "t": (T0 + 10) * 1000, "hazard_id": "x"}]
    assert labels.summarize(sorted(events, key=lambda e: e["t"]))["late"] == 1


# -- signal check -----------------------------------------------------------
def test_quality_flags_flat_and_noisy_channels():
    rng = np.random.default_rng(2)
    eeg = rng.normal(0, 8, size=(2 * SAMPLE_RATE, 4))
    eeg[:, 1] = 0.0
    eeg[:, 3] *= 30
    assert [c["state"] for c in channel_quality(eeg)] == ["CLEAN", "FLAT", "CLEAN", "NOISY"]


# -- studio -----------------------------------------------------------------
def test_simulated_ride_is_recorded(tmp_path):
    studio = Studio(tmp_path / "s", tmp_path / "b")
    with pytest.raises(StudioError):
        studio.start_ride("ride", 2)  # no headset yet
    studio.connect("simulated")
    time.sleep(0.3)
    studio.start_ride("ride", 2)
    now = time.time() * 1000
    studio.add_events([{"type": "ride_start", "t": now}, {"type": "key", "t": now + 10, "down": True}])
    time.sleep(0.6)
    manifest = studio.finish_ride()
    assert manifest["samples"] > 50 and manifest["events"] == 2
    with pytest.raises(StudioError):
        studio.start_ride("drive", 2)  # no model yet
    studio.stop_everything()
    assert studio.stream.status()["state"] == "disconnected"


# -- http -------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path):
    app = create_app(Studio(tmp_path / "s", tmp_path / "b"))
    app.testing = True
    with app.test_client() as test_client:
        test_client.token = app.config["TOKEN"]
        yield test_client
    app.config["STUDIO"].stop_everything()


def test_page_carries_token_and_strict_csp(client):
    response = client.get("/")
    assert client.token in response.get_data(as_text=True)
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]


def test_state_changes_require_the_token(client):
    assert client.post("/api/connect", json={"mode": "simulated"}).status_code == 403
    response = client.post("/api/connect", json={"mode": "simulated"}, headers={"X-Token": client.token})
    assert response.status_code == 200 and response.json["ok"] is True


def test_requests_from_other_machines_are_refused(client):
    response = client.get("/api/status", environ_base={"REMOTE_ADDR": "192.0.2.10"})
    assert response.status_code == 403


def test_bad_input_returns_a_clear_error(client):
    headers = {"X-Token": client.token}
    response = client.post("/api/ride/start", json={"kind": "ride", "minutes": 99}, headers=headers)
    assert response.status_code == 400 and "minutes" in response.json["error"]
    response = client.post("/api/ride/start", json={"kind": "ride", "minutes": 2}, headers=headers)
    assert response.status_code == 409


def test_training_without_rides_fails_with_a_message(client):
    headers = {"X-Token": client.token}
    assert client.post("/api/train", headers=headers).status_code == 202
    for _ in range(50):
        state = client.get("/api/train").json["data"]
        if state["state"] != "running":
            break
        time.sleep(0.05)
    assert state["state"] == "failed" and "at least 2 rides" in state["error"]


def test_full_http_ride(client):
    headers = {"X-Token": client.token}
    client.post("/api/connect", json={"mode": "simulated"}, headers=headers)
    time.sleep(1.2)
    signal = client.get("/api/signal").json["data"]
    assert len(signal["channels"]) == 4 and len(signal["trace"]) == 4
    assert client.post("/api/ride/start", json={"kind": "ride", "minutes": 2}, headers=headers).status_code == 200
    now = time.time() * 1000
    client.post("/api/ride/events", json={"events": [{"type": "ride_start", "t": now}]}, headers=headers)
    finished = client.post("/api/ride/finish", headers=headers).json["data"]
    assert finished["kind"] == "ride"
    assert client.get("/api/sessions").json["data"][0]["session_id"] == finished["session_id"]


# -- features and inference -----------------------------------------------
def noise(seed=0, scale=10.0):
    return np.random.default_rng(seed).normal(0, scale, (200, 4))


def test_feature_set_is_stable_and_finite():
    features = extract_all_features(preprocess_window(noise()))
    assert len(features) == 487
    assert all(np.isfinite(float(value)) for value in features.values())
    assert list(extract_all_features(preprocess_window(noise(1)))) == list(features)


def test_flat_channel_gives_finite_vector():
    window = noise()
    window[:, 2] = 0.0
    vector = feature_vector(window, list(extract_all_features(preprocess_window(noise()))))
    assert vector.shape == (1, 487) and np.isfinite(vector).all()


def test_feature_vector_follows_the_requested_names():
    names = ["ch0_alpha_power", "unknown_feature", "ch3_quant_rsi"]
    vector = feature_vector(noise(), names)
    assert vector.shape == (1, 3) and vector[0, 1] == 0.0


def test_indicators_stay_in_range():
    signal = noise()[:, 0]
    assert 0 <= quant_features.rsi(signal) <= 100
    assert -100 <= quant_features.percent_r(signal) <= 0
    assert 0 <= quant_features.donchian_position(signal) <= 1
    assert quant_features.rsi(np.ones(5)) == 50.0


@pytest.mark.parametrize("fraction, expected", [(0.0, "clean"), (0.15, "moderate"), (0.4, "reject")])
def test_quality_gate(fraction, expected):
    window = noise()
    window[: int(fraction * 200)] = 400.0
    assert window_quality(window) == expected


class FixedModel:
    def __init__(self):
        self.stop = 0.9

    def predict_proba(self, x):
        return np.array([[1 - self.stop, self.stop]])


class Identity:
    def transform(self, x):
        return x


def engine(stop=0.9):
    model = FixedModel()
    model.stop = stop
    result = InferenceEngine()
    result.model, result.scaler, result.feature_names = model, Identity(), ["ch0_mean"]
    return result


def test_artifacts_give_no_decision():
    window = noise()
    window[:100] = 400.0
    assert engine().predict(window)["decision"] == "NO_DECISION"


def test_votes_and_hysteresis_hold_a_state(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("bci_game.ml.inference.time.time", lambda: clock[0])
    live = engine(stop=0.1)
    decisions = []
    for _ in range(3):
        clock[0] += 0.25
        decisions.append(live.predict(noise())["decision"])
    assert decisions == ["STOP", "STOP", "NOT_STOP"]  # three votes needed
    live.model.stop = 0.9
    live.predict(noise())
    clock[0] += 0.1
    assert live.predict(noise())["decision"] == "NOT_STOP"  # majority flipped, but held for 500 ms
    clock[0] += 0.5
    assert live.predict(noise())["decision"] == "STOP"


def test_threshold_decides_the_raw_vote():
    live = engine(stop=0.6)
    live.stop_threshold = 0.7
    assert live.predict(noise())["raw_decision"] == "NOT_STOP"
    live.stop_threshold = 0.5
    assert live.predict(noise())["raw_decision"] == "STOP"


# -- training and bundles -------------------------------------------------
def test_report_scores_held_out_rides(trained):
    report = trained["report"]
    assert report["method"] == "leave-one-ride-out" and len(report["sessions"]) == 2
    assert report["balanced_accuracy"] > 0.85
    assert report["windows"]["GO"] >= 40 and report["windows"]["STOP"] >= 40
    assert report["bundle"] == "synthetic"


def test_bundle_has_a_complete_manifest(trained):
    folder = trained["folder"]
    manifest = json.loads((folder / "bundle.json").read_text())
    files = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()} - {"bundle.json"}
    assert set(manifest["sha256"]) == files
    assert "code/ml/inference.py" in files and "asr_state.pkl" in files
    config = read_bundle(folder).config()
    assert config["evaluation_mode"] == "leave_one_session_out"
    assert config["class_names"] == ["NOT_STOP", "STOP"]
    assert len(config["session_metrics"]) == 2 and config["n_features"] == len(config["feature_names"]) <= 300
    assert config["mains_hz"] == 60 and config["simulated"] is False
    verify_bundle(read_bundle(folder))


def test_runtime_decodes_the_trained_effect(trained, monkeypatch):
    runtime = BCIGameStopGoRuntime(trained["folder"])
    assert runtime.info["bundle"] == "synthetic" and runtime.info["asr"]
    clock = [1_000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])

    def decisions(driving):
        events = [{"type": "ride_start", "t": 0}, {"type": "key", "t": 0, "down": driving}]
        eeg, _ = synthetic_eeg(20, events, seed=9, start=0.0)
        runtime.reset()
        results = []
        for start in range(0, len(eeg) - 200, 50):  # one window every 0.25 s, as drone-flight does
            clock[0] += 0.25
            results.append(runtime.predict(eeg[start:start + 200]))
        return results

    stopped, driving = decisions(False), decisions(True)
    assert np.mean([r["probabilities"]["STOP"] for r in stopped]) > 0.7
    assert np.mean([r["probabilities"]["NOT_STOP"] for r in driving]) > 0.7
    assert np.mean([r["decision"] == "STOP" for r in stopped]) > 0.7
    assert np.mean([r["decision"] == "NOT_STOP" for r in driving]) > 0.7


def test_runtime_fails_to_stop(trained, monkeypatch):
    runtime = BCIGameStopGoRuntime(trained["folder"])
    for window in (np.zeros((10, 4)), np.full((200, 4), np.nan)):
        result = runtime.predict(window)
        assert result["decision"] == "STOP" and result["safe_fallback"] is True

    def broken(window):
        raise RuntimeError("model failed")

    monkeypatch.setattr(runtime.engine, "predict", broken)
    result = runtime.predict(np.random.default_rng(1).normal(0, 10, (200, 4)))
    assert result["decision"] == "STOP" and result["safe_fallback"] is True and "model failed" in result["reason"]


def test_tampered_bundle_is_refused(trained, tmp_path):
    copy = shutil.copytree(trained["folder"], tmp_path / "copy")
    config = next((copy / "model").glob("config_*.json"))
    config.write_text(config.read_text().replace('"stop_threshold": 0.5', '"stop_threshold": 0.05'))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        BCIGameStopGoRuntime(copy)


def test_extra_code_in_a_bundle_is_refused(trained, tmp_path):
    copy = shutil.copytree(trained["folder"], tmp_path / "copy")
    (copy / "code" / "ml" / "extra.py").write_text("x = 1\n")
    with pytest.raises(RuntimeError, match="without a recorded hash"):
        BCIGameStopGoRuntime(copy)


def test_training_needs_two_rides(tmp_path):
    write_ride(tmp_path / "s")
    with pytest.raises(train.TrainingError, match="at least 2 rides"):
        train.train(tmp_path / "s", tmp_path / "b")


def test_excluded_rides_are_not_used(tmp_path):
    ids = [write_ride(tmp_path / "s", seed=seed) for seed in range(2)]
    session.set_excluded(ids[0], True, tmp_path / "s")
    with pytest.raises(train.TrainingError, match="have 1"):
        train.train(tmp_path / "s", tmp_path / "b")


def test_rides_too_short_for_asr_are_refused(tmp_path):
    for seed in range(2):
        write_ride(tmp_path / "s", seconds=8, seed=seed)
    with pytest.raises(train.TrainingError, match="ASR calibration failed"):
        train.train(tmp_path / "s", tmp_path / "b")


def test_rides_without_both_classes_are_refused(tmp_path):
    for seed in range(2):
        recorder = session.Recorder.create("ride", 2, tmp_path / "s")
        events = [e for e in ride_events(120) if e["type"] != "key"]
        eeg, stamps = synthetic_eeg(120, events, seed)
        recorder.write_eeg(eeg, stamps)
        recorder.write_events(events)
        recorder.finish({})
    with pytest.raises(train.TrainingError, match="GO 0"):
        train.train(tmp_path / "s", tmp_path / "b")


def test_bundle_names_are_checked_and_never_overwritten(trained):
    for bad in ("../escape", "", "a b", "x" * 65):
        with pytest.raises(ValueError):
            bundles.check_name(bad)
    with pytest.raises(FileExistsError):
        train.train(trained["recordings"], trained["bundles"], name="synthetic")


def test_bundle_listing_and_discovery(trained):
    assert bundles.latest(trained["bundles"]) == trained["folder"]
    assert bundles.latest_report(trained["bundles"])["bundle"] == "synthetic"
    assert [found.folder.name for found in find_bundles([trained["bundles"]])] == ["synthetic"]


def test_drive_mode_uses_the_latest_bundle(trained):
    studio = Studio(trained["recordings"], trained["bundles"])
    studio.connect("simulated")
    time.sleep(0.3)
    studio.start_ride("drive", 2)
    time.sleep(1.8)
    live = studio.live_status()
    assert live["state"] in (GO, STOP) and live["reason"] in ("decoding", "artifact")
    manifest = studio.finish_ride()
    assert manifest["kind"] == "drive"
    studio.stop_everything()
    assert studio.live_status()["reason"] == "not driving"


def test_simulator_keys_follow_the_model_and_end_released(trained, monkeypatch, capsys):
    from bci_game import sim_keys
    from bci_game.studio import LiveDecoder

    log, decisions = [], iter([GO] * 3 + [STOP] * 2 + [GO] * 100)

    class Keyboard:
        def press(self, key):
            log.append(("press", key))

        def release(self, key):
            log.append(("release", key))

    monkeypatch.setattr(sim_keys, "keyboard_for", lambda name: (Keyboard(), name))
    monkeypatch.setattr(LiveDecoder, "status", lambda self: {"state": next(decisions), "p_go": 0.5, "reason": "test"})
    assert main(["keys", "--simulate", "--bundles", str(trained["bundles"]), "--delay", "0", "--seconds", "2",
                 "--key", "up"]) == 0
    out = capsys.readouterr().out
    assert "holds up while" in out and trained["folder"].name in out
    assert log[:3] == [("press", "up"), ("release", "up"), ("press", "up")], "GO holds the key, STOP lets go"
    assert log[-1] == ("release", "up"), "never left held down at the end"
    assert "up is released" in out
    assert main(["keys", "--simulate", "--bundles", str(trained["bundles"] / "missing"), "--delay", "0"]) == 1


def test_command_line_list_and_check(trained, capsys):
    assert main(["list", str(trained["bundles"])]) == 0
    assert "synthetic" in capsys.readouterr().out
    assert main(["check", str(trained["folder"])]) == 0
    assert "hashes OK" in capsys.readouterr().out


# -- settings, sources and cancellation -------------------------------------
def test_mains_must_be_50_or_60_and_moves_the_notch():
    assert parse_mains("50") == 50 and parse_mains(" 60") == 60
    with pytest.raises(ValueError):
        parse_mains("55")
    t = np.arange(200) / SAMPLE_RATE
    window = noise() + 40 * np.sin(2 * np.pi * 50 * t)[:, None]
    names = ["ch0_gamma_power", "ch1_gamma_power"]
    assert np.all(feature_vector(window, names, 50) < feature_vector(window, names, 60))


def test_training_saves_to_the_first_bundle_folder():
    code = "from bci_game.config import BUNDLES_DIR; print(BUNDLES_DIR)"
    env = {**os.environ, "EEG_BUNDLES": os.pathsep.join(["first", "second"])}
    run = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, env=env, capture_output=True, text=True, check=True)
    assert run.stdout.strip() == "first"
    env.pop("EEG_BUNDLES")
    run = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, env=env, capture_output=True, text=True, check=True)
    assert Path(run.stdout.strip()) == PROJECT / "eeg_bundles"


def test_recorder_saves_where_the_eeg_came_from(tmp_path):
    recorder = session.Recorder.create("ride", 2, tmp_path, source="simulated")
    assert recorder.finish({})["source"] == "simulated"


def test_simulator_rides_are_never_mixed_with_headset_rides():
    rides = [{"session_id": "a", "source": "ganglion"}, {"session_id": "b", "source": "simulated"},
             {"session_id": "c"}, {"session_id": "d", "source": "simulated"}]
    assert dataset.choose_rides(rides) == (["a"], False)
    assert dataset.choose_rides(rides[1:]) == (["b", "c", "d"], True)  # no recorded source is not a headset
    assert dataset.choose_rides([]) == ([], False)


def test_one_headset_ride_with_simulator_rides_is_refused(tmp_path):
    write_ride(tmp_path / "s")
    for seed in range(2):
        recorder = session.Recorder.create("ride", 2, tmp_path / "s", source="simulated")
        recorder.write_eeg(np.zeros((10, 4)), T0 + np.arange(10) / SAMPLE_RATE)
        recorder.finish({})
    with pytest.raises(train.TrainingError, match="2 Ganglion rides"):
        train.train(tmp_path / "s", tmp_path / "b")


def test_command_line_training_errors_are_short(tmp_path, capsys):
    assert main(["train", "--recordings", str(tmp_path / "none"), "--out", str(tmp_path / "b")]) == 1
    assert main(["train", "--recordings", str(tmp_path), "--out", str(tmp_path), "--name", "../escape"]) == 1
    assert capsys.readouterr().err.count("Not trained") == 2


class FakeBoard:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.calls = []

    def get_board_data(self):
        return self.chunks.pop(0) if self.chunks else np.zeros((6, 0))

    def stop_stream(self):
        self.calls.append("stop_stream")

    def release_session(self):
        self.calls.append("release_session")


def handle(board, rate=SAMPLE_RATE):
    return SimpleNamespace(board=board, eeg_rows=(1, 2, 3, 4), accel_rows=(), timestamp_row=5,
                           sample_rate=rate, transport="BLED112 dongle (COM6)")


def test_ganglion_source_reads_the_shared_connection(monkeypatch):
    chunk = np.vstack([np.arange(3), np.ones((4, 3)) * [[1], [2], [3], [4]], T0 + np.arange(3) / SAMPLE_RATE])
    board = FakeBoard([chunk])
    monkeypatch.setattr("bci_game.acquisition.connect", lambda simulate: handle(board))
    source = GanglionSource.connect()
    eeg, stamps = source.read()
    assert eeg.shape == (3, 4) and eeg[0].tolist() == [1, 2, 3, 4] and stamps[0] == T0
    assert source.read()[0].shape == (0, 4)
    assert source.describe()["transport"] == "BLED112 dongle (COM6)"
    source.close()
    assert board.calls == ["stop_stream", "release_session"]


def test_a_board_at_the_wrong_rate_is_released(monkeypatch):
    board = FakeBoard([])
    monkeypatch.setattr("bci_game.acquisition.connect", lambda simulate: handle(board, rate=250))
    with pytest.raises(SourceError):
        GanglionSource.connect()
    assert board.calls == ["stop_stream", "release_session"]


def test_stop_during_a_headset_search_releases_the_headset(tmp_path):
    started, finish = threading.Event(), threading.Event()
    closed = []

    class SlowSource:
        kind = "ganglion"

        def read(self):
            return np.zeros((0, 4)), np.zeros(0)

        def describe(self):
            return {"kind": self.kind}

        def close(self):
            closed.append(True)

    def slow_factory():
        started.set()
        finish.wait(5)
        return SlowSource()

    studio = Studio(tmp_path / "s", tmp_path / "b", ganglion_factory=slow_factory)
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(_connect(studio)))
    thread.start()
    assert started.wait(5)
    began = time.time()
    assert studio.status()["stream"]["state"] == "disconnected"  # not blocked by the search
    studio.stop_everything()
    assert time.time() - began < 1
    finish.set()
    thread.join(5)
    assert outcome == ["Connection cancelled."] and closed == [True]
    assert studio.stream.status()["state"] == "disconnected"


def _connect(studio):
    try:
        studio.connect("ganglion")
        return "connected"
    except StudioError as exc:
        return str(exc)


def test_drive_decoder_stops_on_a_timestamp_gap():
    stamps = time.time() - 1.0 + np.arange(240) / SAMPLE_RATE
    stamps[:120] -= 0.4  # 0.4 s gap inside the last second

    class Stream:
        def recent(self, seconds):
            return np.zeros((240, 4)), stamps

    class Runtime:
        resets = 0

        def reset(self):
            Runtime.resets += 1

        def predict(self, window):
            raise AssertionError("a broken window must not reach the model")

    decoder = LiveDecoder(Stream(), Runtime()).start()
    time.sleep(0.6)
    status = decoder.status()
    decoder.stop()
    assert status["state"] == STOP and status["reason"] == "signal gap" and Runtime.resets >= 2


class FakeGanglion:
    """Simulated samples presented as a real headset."""
    kind = "ganglion"

    def __init__(self):
        from bci_game.stream import SimulatedSource
        self._inner = SimulatedSource()

    def read(self):
        return self._inner.read()

    def describe(self):
        return {"kind": self.kind, "transport": "fake"}

    def close(self):
        pass


def test_overlapping_searches_never_run_at_once(tmp_path):
    active, peak, first = [0], [0], threading.Event()
    release = threading.Event()

    class Fake(FakeGanglion):
        def close(self):
            pass

    def factory():
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        if not first.is_set():
            first.set()
            release.wait(5)
        active[0] -= 1
        return Fake()

    studio = Studio(tmp_path / "s", tmp_path / "b", ganglion_factory=factory)
    results = []
    older = threading.Thread(target=lambda: results.append(("older", _connect(studio))))
    older.start()
    assert first.wait(5)
    newer = threading.Thread(target=lambda: results.append(("newer", _connect(studio))))
    newer.start()
    time.sleep(0.2)
    release.set()
    older.join(5)
    newer.join(5)
    studio.stop_everything()
    assert peak[0] == 1
    assert sorted(results) == [("newer", "connected"), ("older", "Connection cancelled.")]


def test_a_simulator_model_never_drives_with_a_headset(trained, tmp_path):
    bundle_dir = tmp_path / "bundles"
    copy = shutil.copytree(trained["folder"], bundle_dir / "zz-simulated")
    manifest = json.loads((copy / "bundle.json").read_text())
    manifest["report"]["simulated"] = True
    (copy / "bundle.json").write_text(json.dumps(manifest))
    studio = Studio(tmp_path / "s", bundle_dir, ganglion_factory=FakeGanglion)
    studio.connect("ganglion")
    time.sleep(0.3)
    with pytest.raises(StudioError, match="simulator"):
        studio.start_ride("drive", 2)
    studio.stop_everything()


def write_manifest(folder, manifest):
    folder.mkdir(parents=True)
    (folder / "bundle.json").write_text(manifest if isinstance(manifest, str) else json.dumps(manifest))


def test_only_bundles_with_a_game_report_are_listed(tmp_path, capsys):
    good = {"format": "stop-go-bundle/1", "report": {"trained_at": "2026-01-01 10:00:00", "method": "leave-one-ride-out",
                                                     "sessions": ["a", "b"], "balanced_accuracy": 0.61, "simulated": True}}
    write_manifest(tmp_path / "list", "[]")
    write_manifest(tmp_path / "nested", "[" * 100000)
    write_manifest(tmp_path / "no-report", {"format": "stop-go-bundle/1", "report": []})
    write_manifest(tmp_path / "odd-fields", {**good, "report": {**good["report"], "sessions": 3}})
    write_manifest(tmp_path / "good", good)
    assert [folder.name for folder, _ in bundles.listed(tmp_path)] == ["good"]
    assert bundles.latest_report(tmp_path)["folder"] == str(tmp_path / "good")
    assert main(["list", str(tmp_path)]) == 0
    assert "61.0%" in capsys.readouterr().out


def test_the_bundle_engine_uses_the_saved_mains_frequency(trained, tmp_path):
    copy = shutil.copytree(trained["folder"], tmp_path / "fifty")
    config_path = next((copy / "model").glob("config_*.json"))
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps({**config, "mains_hz": 50}))
    runtime = BCIGameStopGoRuntime(copy, verify_hashes=False)
    assert runtime.engine.mains_hz == 50
    assert BCIGameStopGoRuntime(trained["folder"]).engine.mains_hz == 60


def test_the_signal_check_follows_the_mains_setting():
    code = ("import numpy as np; from bci_game.dsp import channel_quality; "
            "t = np.arange(400) / 200; eeg = np.random.default_rng(0).normal(0, 5, (400, 4)); "
            "eeg[:, 0] += 30 * np.sin(2 * np.pi * 50 * t); print(channel_quality(eeg)[0]['state'])")
    states = {}
    for mains in ("50", "60"):
        env = {**os.environ, "BCI_GAME_MAINS_HZ": mains}
        run = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, env=env, capture_output=True, text=True, check=True)
        states[mains] = run.stdout.strip()
    assert states == {"50": "NOISY", "60": "CLEAN"}


# -- comparing every model ----------------------------------------------------
def quick_candidates():
    """All six models with settings small enough for tests."""
    return [models.Boosted("xgboost"), models.Boosted("lightgbm"), models.Forest(n_trees=40),
            models.Network("cfc"), models.Network("ltc", epochs=4), models.Network("gru")]


@pytest.fixture(scope="module")
def compared(tmp_path_factory):
    """Three synthetic rides, every model compared on them, and the data each was trained on."""
    root = tmp_path_factory.mktemp("compared")
    for seed in range(3):
        write_ride(root / "recordings", seed=seed)
    report = train.train(root / "recordings", root / "bundles", name="best", mains_hz=60, candidates=quick_candidates())
    ids, _ = dataset.choose_rides(dataset.usable_rides(root / "recordings"))
    rides = [dataset.load_ride(root / "recordings", ride) for ride in ids]
    asr = dataset.calibrate_asr(rides)
    data = models.Data(dataset.collect(rides, asr), rides, ("Fz", "Cz", "F7", "F8"), 60)
    return {"root": root, "report": report, "data": data, "asr": asr, "rides": ids}


def test_every_model_is_compared_and_the_best_is_saved(compared):
    report = compared["report"]
    board = report["leaderboard"]
    assert {row["model"] for row in board} == set(models.NAMES)
    scores = [row["balanced_accuracy"] for row in board]
    assert scores == sorted(scores, reverse=True)
    assert report["model_type"] == board[0]["model"] and report["balanced_accuracy"] == board[0]["balanced_accuracy"]
    runtime = BCIGameStopGoRuntime(Path(report["folder"]))
    assert runtime.info["model_type"] == report["model_type"]
    config = read_bundle(Path(report["folder"])).config()
    assert config["model_type"] == report["model_type"] and len(config["session_metrics"]) == 3
    by_model = {row["model"]: row["balanced_accuracy"] for row in board}
    assert min(by_model["xgboost"], by_model["lightgbm"], by_model["random_forest"], by_model["cfc"]) > 0.7


def alternating_accuracy(runtime, seconds=90):
    """Share of raw decisions that match the key on a ride alternating GO / STOP every 5 s."""
    events = ride_events(seconds, start=T0)
    eeg, stamps = synthetic_eeg(seconds, events, seed=9, start=T0)
    ends, window_labels = labels.windows(stamps, events, step_s=0.5)
    right = []
    for end, label in zip(ends, window_labels):
        if end + 1 >= runtime.WINDOW_SAMPLES:
            result = runtime.predict(eeg[end + 1 - runtime.WINDOW_SAMPLES: end + 1])
            assert result["raw_decision"] in ("STOP", "NOT_STOP"), result
            right.append(result["raw_decision"] == ("STOP" if label == STOP else "NOT_STOP"))
    return float(np.mean(right))


@pytest.mark.parametrize("name", models.NAMES)
def test_each_model_type_flies_from_its_bundle(compared, name, tmp_path):
    candidate = next(c for c in quick_candidates() if c.name == name)
    fitted = candidate.fit(compared["data"], set(compared["rides"]))
    config = {"model_id": f"m_{name}", "model_type": name, "trained_at": "20260101_000000",
              "class_names": ["NOT_STOP", "STOP"], "evaluation_mode": "leave_one_session_out",
              "channels": ["Fz", "Cz", "F7", "F8"], "mains_hz": 60, "simulated": True, "sample_rate": SAMPLE_RATE,
              "accuracy": 0.5, "f1_score": 0.5, "cv_mean": 0.5, "cv_std": 0.0, "session_metrics": []}
    report = {"trained_at": "2026-01-01 00:00:00", "method": "leave-one-ride-out", "sessions": compared["rides"],
              "balanced_accuracy": 0.5}
    folder = bundles.write(tmp_path, name, artifacts=fitted.save, config=config, asr=compared["asr"], report=report)
    runtime = BCIGameStopGoRuntime(folder)
    expected = (7000, False) if name == "random_forest" else (200, True)
    assert (runtime.WINDOW_SAMPLES, runtime.info["asr"]) == expected
    accuracy = alternating_accuracy(runtime)
    if name != "ltc":  # LTC learns slowly; 4 test epochs only check that it runs
        assert accuracy > 0.7, accuracy


def test_random_forest_features_are_causal_and_match_live_use():
    eeg, _ = synthetic_eeg(90, ride_events(90), seed=3)
    channels = ("Fz", "Cz", "F7", "F8")
    full, names = rf_features.stream_features(eeg, rf_features.grid_ends(len(eeg)), channels)
    assert full.shape[1] == len(names) == 133
    changed = eeg.copy()
    changed[10000:] += 50
    past = (10000 - rf_features.WINDOW_SAMPLES) // rf_features.STEP_SAMPLES
    altered, _ = rf_features.stream_features(changed, rf_features.grid_ends(len(eeg)), channels)
    np.testing.assert_allclose(altered[:past + 1], full[:past + 1])
    for end in (9000, 15000, 18000):
        live, _ = rf_features.latest_features(eeg[end - rf_features.HISTORY_SAMPLES:end], channels)
        np.testing.assert_allclose(live[0], full[(end - rf_features.WINDOW_SAMPLES) // rf_features.STEP_SAMPLES],
                                   rtol=1e-5, atol=1e-6)
    assert np.isnan(full[:30, -1]).all()  # no baseline in the first 5 s


def test_networks_take_patches_of_the_window():
    windows = neural.preprocess_windows(noise()[None].repeat(3, axis=0))
    mean, std = neural.channel_stats(windows)
    patches = neural.network_input(windows, mean, std, 10)
    assert patches.shape == (3, 20, 40) and patches.dtype == np.float32
    for name in neural.MODELS:
        model = neural.build_model(name, 40, neural.DEFAULTS)
        probabilities = neural.stop_probabilities(model, patches)
        assert probabilities.shape == (3,) and np.all((probabilities >= 0) & (probabilities <= 1))
    with pytest.raises(ValueError):
        neural.build_model("transformer", 40, neural.DEFAULTS)


def test_a_model_that_cannot_train_is_listed_not_fatal(compared):
    class Broken:
        name, label, needs_history = "broken", "Broken", False

        def fit(self, data, rides):
            raise RuntimeError("no luck")

    board = train.compare([Broken(), models.Boosted("xgboost")], compared["data"])
    assert [row["model"] for row in board] == ["xgboost", "broken"]
    assert board[1]["error"] == "RuntimeError: no luck"
    with pytest.raises(train.TrainingError, match="No model could be trained"):
        train.train(compared["root"] / "recordings", compared["root"] / "nothing", candidates=[Broken()])
    with pytest.raises(ValueError, match="Unknown models"):
        models.candidates(["xgboost", "svm"])


def test_boosters_and_networks_share_one_process():
    code = ("import bci_game.ml.neural as neural, numpy as np, xgboost, lightgbm, torch\n"
            "x = np.random.default_rng(0).normal(size=(400, 50)); y = (x[:, 0] > 0).astype(int)\n"
            "xgboost.XGBClassifier(n_estimators=20, n_jobs=-1).fit(x, y)\n"
            "lightgbm.LGBMClassifier(n_estimators=20, verbose=-1).fit(x, y)\n"
            "model = neural.build_model('gru', 40, neural.DEFAULTS)\n"
            "model(torch.zeros(8, 20, 40)).sum().backward()\n"
            "xgboost.XGBClassifier(n_estimators=20, n_jobs=-1).fit(x, y)\n"
            "print('ok')")
    run = subprocess.run([sys.executable, "-c", code], cwd=PROJECT, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0 and run.stdout.strip() == "ok", run.stderr[-2000:]


def test_command_line_compares_the_chosen_models(compared, tmp_path, capsys):
    assert main(["train", "--recordings", str(compared["root"] / "recordings"), "--out", str(tmp_path),
                 "--name", "cli", "--models", "xgboost,lightgbm"]) == 0
    out = capsys.readouterr().out
    assert "XGBoost" in out and "LightGBM" in out and "Saved the best" in out
    report = json.loads((tmp_path / "cli" / "bundle.json").read_text())["report"]
    assert {row["model"] for row in report["leaderboard"]} == {"xgboost", "lightgbm"}
    assert main(["train", "--recordings", str(compared["root"] / "recordings"), "--out", str(tmp_path),
                 "--models", "svm"]) == 1



def write_bundle(fitted, name, compared, folder, mains_hz=60):
    config = {"model_id": f"m_{name}", "model_type": name, "trained_at": "20260101_000000",
              "class_names": ["NOT_STOP", "STOP"], "evaluation_mode": "leave_one_session_out",
              "channels": ["Fz", "Cz", "F7", "F8"], "mains_hz": mains_hz, "simulated": True, "sample_rate": SAMPLE_RATE,
              "accuracy": 0.5, "f1_score": 0.5, "cv_mean": 0.5, "cv_std": 0.0, "session_metrics": []}
    report = {"trained_at": "2026-01-01 00:00:00", "method": "leave-one-ride-out", "sessions": compared["rides"],
              "balanced_accuracy": 0.5}
    return bundles.write(folder, name, artifacts=fitted.save, config=config, asr=compared["asr"], report=report)


@pytest.mark.parametrize("candidate, mains_hz", [
    (models.Boosted("xgboost"), 60), (models.Boosted("lightgbm"), 60), (models.Network("cfc", epochs=1), 60),
    (models.Network("ltc", epochs=1), 60), (models.Network("gru", epochs=1), 60), (models.Forest(n_trees=20), 60),
    (models.Network("gru", epochs=1), 50), (models.Forest(n_trees=20), 50),
], ids=lambda value: getattr(value, "name", str(value)))
def test_a_saved_model_scores_exactly_as_it_did_in_training(compared, candidate, mains_hz, tmp_path):
    shared = compared["data"]
    data = models.Data(shared.windows, list(shared.rides.values()), shared.channels, mains_hz)
    fitted = candidate.fit(data, set(compared["rides"]))
    runtime = BCIGameStopGoRuntime(write_bundle(fitted, candidate.name, compared, tmp_path, mains_hz))
    assert runtime.engine.stop_threshold == pytest.approx(fitted.threshold)
    rows = np.array([row for row in range(0, len(data.y), 37) if data.windows.ends[row] + 1 >= runtime.WINDOW_SAMPLES])
    expected = fitted.stop_probability(data, rows)
    undecided = data.forest_undecided(rows) if candidate.name == "random_forest" else np.zeros(len(rows), bool)
    for row, wanted, skip in zip(rows, expected, undecided):
        ride = data.rides[data.windows.groups[row]]
        end = data.windows.ends[row] + 1
        result = runtime.predict(ride.eeg[end - runtime.WINDOW_SAMPLES: end])
        if skip:
            assert wanted == 1.0 and result["decision"] == "STOP"
            continue
        assert result["probabilities"]["STOP"] == pytest.approx(wanted, abs=1e-5)


def test_the_random_forest_follows_one_rule_for_scoring_and_saving(compared, monkeypatch):
    data, rides = compared["data"], compared["rides"]
    forest = models.Forest(n_trees=20)
    tunes = []
    select = models.Forest._select
    monkeypatch.setattr(models.Forest, "_select",
                        lambda self, x, y, groups, tune: tunes.append(tune) or select(self, x, y, groups, tune))
    two = data.restricted(np.isin(data.windows.groups, rides[:2]))
    for train_rides in ([rides[0]], rides[:2]):  # a scored fold and the saved fit of a 2-ride run
        assert (forest.fit(two, set(train_rides)).params, tunes[-1]) == (models.RF_FIXED, False)
    forest.fit(data, set(rides[:2]))  # a fold of a 3-ride run
    forest.fit(data, set(rides))  # its saved fit
    assert tunes[-2:] == [True, True]
    # a ride whose raw EEG is all rejected has windows but no Random Forest rows: 2 rides cannot tune
    gutted = models.Data(data.windows, list(data.rides.values()), data.channels, data.mains_hz)
    table = data.forest(rides[2])
    gutted._forest = {**data._forest, rides[2]: models.ForestRide(table.features, table.rows[:0], table.y[:0])}
    forest.fit(gutted, set(rides[:2]))
    forest.fit(gutted, set(rides))
    assert tunes[-2:] == [False, False]


def test_the_artifact_gate_ignores_electrode_offset():
    offset = noise() + np.array([180.0, -120.0, 40.0, 0.0])
    assert window_quality(offset) == "clean"
    spiky = offset.copy()
    spiky[:80, 0] += 400.0
    spiky[:80, 1] += 400.0
    assert window_quality(spiky) in ("moderate", "reject")


def test_drive_mode_gives_a_35_second_model_35_seconds(tmp_path):
    from bci_game.stream import EEGStream

    stream = EEGStream()
    windows = []

    class Forest:
        WINDOW_SAMPLES = 7000

        def reset(self):
            pass

        def predict(self, window):
            windows.append(window.shape)
            return {"decision": "STOP", "probabilities": {"NOT_STOP": 0.4, "STOP": 0.6}}

    decoder = LiveDecoder(stream, Forest()).start()
    now = time.time()
    stream._append(np.random.default_rng(0).normal(0, 10, (6000, 4)), now - 30 + np.arange(6000) / SAMPLE_RATE)
    time.sleep(0.4)
    waiting = decoder.status()
    stream._append(np.random.default_rng(1).normal(0, 10, (1200, 4)), time.time() - 6 + np.arange(1200) / SAMPLE_RATE)
    time.sleep(0.4)
    decoder.stop()
    assert waiting["reason"] == "collecting 35 s of EEG" and windows and set(windows) == {(7000, 4)}


def test_rides_with_no_clean_windows_get_a_clear_message(tmp_path):
    for seed in range(2):
        recorder = session.Recorder.create("ride", 2, tmp_path / "s", source="ganglion")
        recorder.write_events(ride_events(20))
        eeg, stamps = synthetic_eeg(20, ride_events(20), seed=seed)
        recorder.write_eeg(eeg + 2000 * np.sign(np.sin(np.arange(len(eeg)) / 3))[:, None], stamps)
        recorder.finish({})
    with pytest.raises(train.TrainingError, match="GO 0, STOP 0"):
        train.train(tmp_path / "s", tmp_path / "b", candidates=quick_candidates())



def test_the_random_forest_says_stop_where_it_could_not_decide_live(compared, tmp_path):
    data, rides = compared["data"], compared["rides"]
    row = int(np.flatnonzero((data.windows.groups == rides[0]) & (data.windows.ends > 12000))[0])
    end = int(data.windows.ends[row]) + 1
    ride = data.rides[rides[0]]
    eeg = ride.eeg.copy()
    eeg[end - 120:end] += 600.0 * np.sign(np.sin(np.arange(120) / 2))[:, None]  # an artifact in the last second
    spoiled = models.Data(data.windows, [dataclasses.replace(ride, eeg=eeg)] + [data.rides[r] for r in rides[1:]],
                          data.channels, data.mains_hz)
    fitted = models.Forest(n_trees=20).fit(data, set(rides))
    assert spoiled.forest_undecided(np.array([row]))[0]
    assert fitted.stop_probability(spoiled, np.array([row]))[0] == 1.0
    runtime = BCIGameStopGoRuntime(write_bundle(fitted, "random_forest", compared, tmp_path))
    result = runtime.predict(eeg[end - runtime.WINDOW_SAMPLES:end])
    assert result["decision"] == "STOP" and result["safe_fallback"] and result["reason"] == "artifact"


def test_a_flat_channel_is_rejected_and_never_breaks_the_features():
    window = noise()
    window[:, 2] = 35.0
    assert window_quality(window) == "reject"
    eeg = np.random.default_rng(0).normal(0, 10, (3000, 4))
    eeg[:, 1] = -12.0
    features, _ = rf_features.stream_features(eeg, rf_features.grid_ends(len(eeg)), ("Fz", "Cz", "F7", "F8"))
    assert np.isfinite(features[-1]).all()


def test_drive_mode_says_warming_up_until_the_first_decision():
    class Quiet:
        def recent(self, seconds):
            return np.zeros((0, 4)), np.zeros(0)

    class Runtime:
        def reset(self):
            pass

    decoder = LiveDecoder(Quiet(), Runtime()).start()
    status = decoder.status()
    decoder.stop()
    assert status["reason"] == "warming up" and status["state"] == STOP

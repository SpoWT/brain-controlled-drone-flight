"""``python -m bci_game`` starts the recording game; ``train``, ``check`` and ``list`` manage bundles;
``keys`` holds a key in an FPV simulator while the newest bundle decodes GO."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np

from .config import BUNDLES_DIR, DEFAULT_PORT, HOST, MODEL_NAMES, RECORDINGS_DIR
from .sim_keys import KEYS, SIMULATED_SWITCH_S


def _warm_up() -> None:
    """Load the heavy science stack in the background once the server is up."""
    try:
        import scipy.signal  # noqa: F401
        import sklearn.feature_selection  # noqa: F401
        import xgboost  # noqa: F401
    except Exception:  # surfaced later by the feature that needs it
        logging.getLogger("bci_game").exception("background warm-up failed")


def _play(args) -> int:
    from .server import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app = create_app(port=args.port)
    url = f"http://{HOST}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    print(f"Neural Ride: {url}", flush=True)
    threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()
    try:
        app.run(host=HOST, port=args.port, threaded=True, debug=False, use_reloader=False)
    finally:
        app.config["STUDIO"].stop_everything()
    return 0


def _train(args) -> int:
    from . import models, train

    try:
        names = [name.strip() for name in args.models.split(",") if name.strip()]
        report = train.train(args.recordings, args.out, name=args.name, candidates=models.candidates(names),
                             progress=lambda text: print(text, flush=True))
    except (train.TrainingError, ValueError, FileExistsError) as exc:
        print(f"Not trained: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    print("\nHeld-out balanced accuracy, every model scored on the same windows:")
    for row in report["leaderboard"]:
        score = f"{row['balanced_accuracy']:.1%}" if "balanced_accuracy" in row else f"not trained ({row['error']})"
        print(f"  {row['label']:14} {score}")
    print(f"Saved the best, {report['model']}: {Path(args.out) / report['bundle']}")
    return 0


def _check(args) -> int:
    from bci_game.runtime import BCIGameStopGoRuntime

    runtime = BCIGameStopGoRuntime(args.bundle)
    window = np.random.default_rng(0).normal(0, 10, (runtime.WINDOW_SAMPLES, runtime.CHANNELS))
    result = runtime.predict(window)
    hashes = "hashes OK" if runtime.bundle.sha256 else "no manifest, hashes not checked"
    print(json.dumps(runtime.info, indent=2))
    print(f"Loaded, {hashes}. Noise window -> {result['decision']} (reason: {result.get('reason') or 'model'})")
    return 0


def _list(args) -> int:
    from . import bundles

    rows = bundles.listed(args.folder)
    if not rows:
        print(f"No trained bundles in {args.folder}")
    for folder, manifest in rows:
        report = manifest["report"]
        print(f"{folder.name:32} {str(report.get('model', 'XGBoost')):14} trained {report['trained_at']:19}  "
              f"held-out balanced accuracy {report['balanced_accuracy']:.1%}  rides {len(report['sessions'])}"
              + ("  simulator" if report.get("simulated") else ""))
    return 0


def _keys(args) -> int:
    from bci_game.runtime import BCIGameStopGoRuntime

    from . import bundles, sim_keys
    from .stream import EEGStream, GanglionSource, SimulatedSource
    from .studio import LiveDecoder

    report = bundles.latest_report(args.bundles)
    if report is None:
        print(f"No trained model in {args.bundles}: train one in the game first.", file=sys.stderr)
        return 1
    if report.get("simulated") and not args.simulate:
        print("The newest model was trained on simulator rides: add --simulate.", file=sys.stderr)
        return 1
    try:
        keyboard, key = sim_keys.keyboard_for(args.key)
    except ImportError as exc:
        print(f"Not started: {exc}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1
    try:
        runtime = BCIGameStopGoRuntime(Path(report["folder"]))
        source = SimulatedSource() if args.simulate else GanglionSource.connect()
    except Exception as exc:
        print(f"Not started: {exc}", file=sys.stderr)
        return 1
    holder, stop = sim_keys.KeyHolder(keyboard, key), threading.Event()
    remove_console_handler = sim_keys.release_on_console_close(holder, stop)
    stream = EEGStream()
    stream.attach(source)
    decoder = LiveDecoder(stream, runtime).start()
    timer = threading.Timer(args.delay + args.seconds, stop.set) if args.seconds > 0 else None
    if timer is not None:
        timer.start()
    if args.simulate:
        threading.Thread(target=sim_keys.alternate_driving, args=(source, stop), name="simulated-driving",
                         daemon=True).start()
    print(f"{report['bundle']}: holds {args.key} while you think GO, lets go on STOP. Click into the simulator: "
          f"it starts in {args.delay:g} s. Ctrl+C here to stop.", flush=True)
    try:
        if not stop.wait(args.delay):
            sim_keys.hold_while_go(decoder, holder, stop=stop, say=lambda text: print(text, flush=True))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        holder.release()
        remove_console_handler()
        if timer is not None:
            timer.cancel()
        decoder.stop()
        stream.detach()
    print(f"Stopped; {args.key} is released.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bci_game",
                                     description="Record STOP / GO EEG, train bundles and check them.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.set_defaults(run=_play)
    commands = parser.add_subparsers(dest="command")
    trainer = commands.add_parser("train", help="train a bundle from the rides marked IN")
    trainer.add_argument("--recordings", type=Path, default=RECORDINGS_DIR, help="ride folder (default: eeg_recordings)")
    trainer.add_argument("--out", type=Path, default=BUNDLES_DIR, help="bundle folder (default: eeg_bundles)")
    trainer.add_argument("--name", default=None, help="bundle name (default: stopgo-<date>-<time>)")
    trainer.add_argument("--models", default=",".join(MODEL_NAMES),
                         help=f"models to compare, comma separated (default: all of {','.join(MODEL_NAMES)})")
    trainer.set_defaults(run=_train)
    checker = commands.add_parser("check", help="verify a bundle's hashes and run one window through it")
    checker.add_argument("bundle", type=Path)
    checker.set_defaults(run=_check)
    lister = commands.add_parser("list", help="list trained bundles")
    lister.add_argument("folder", type=Path, nargs="?", default=BUNDLES_DIR)
    lister.set_defaults(run=_list)
    keys = commands.add_parser("keys", help="hold a key in an FPV simulator while the newest bundle decodes GO")
    keys.add_argument("--key", choices=KEYS, default="w", help="W or the up arrow (default: w)")
    keys.add_argument("--bundles", type=Path, default=BUNDLES_DIR, help="bundle folder (default: eeg_bundles)")
    keys.add_argument("--simulate", action="store_true",
                      help=f"simulated EEG that switches between GO and STOP every {SIMULATED_SWITCH_S:g} s, "
                           "instead of the Ganglion")
    keys.add_argument("--delay", type=float, default=5.0, help="seconds to click into the simulator (default: 5)")
    keys.add_argument("--seconds", type=float, default=0.0, help="stop after this long (default: until Ctrl+C)")
    keys.set_defaults(run=_keys)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())

# BCI Game Reference

Record your own STOP / GO EEG in a driving game, train six kinds of STOP / NOT_STOP model on it, keep the
best, and fly the drone with it. OpenBCI Ganglion, 4 channels at 200 Hz. Labels come from the drive key:
held = GO (`NOT_STOP`), released = `STOP`. Training saves the best model straight into `eeg_bundles/`, where
the flying website loads it.

## 1. Run

```
python -m bci_game
```

Opens `http://127.0.0.1:8790` (loopback only; every change needs the page token).

| Step | What happens |
|---|---|
| 01 HEADSET | Connect the Ganglion (BLED112 dongle, else native Bluetooth) or the simulator |
| 02 SIGNAL | Live traces, each channel graded CLEAN / NOISY / FLAT |
| 03 RIDE | 2, 5 or 8 min. Hold `↑` / `W` / `Space` to drive, release to stop. `Esc` ends and saves |
| 04 TRAIN | Every model is trained on the rides marked IN (2 minimum); the leaderboard shows each held-out score and the best is saved to `eeg_bundles/<name>/` |
| 05 DRIVE | The newest bundle drives the car, no keys. Each drive is recorded |

Then restart the flying website. It flies Auto, the bundle with the best held-out balanced accuracy (plain accuracy
if any bundle lacks it), unless another was chosen with `POST /api/eeg/model` (`model_id`: `auto` or
`bundle-<folder name>`; the choice is kept across restarts). A model trained without Ganglion rides only drives the
game and is never offered for flight.
Only one program can hold the Ganglion: press STOP in the game before connecting EEG on the flying page.

## 2. Hardware

| Part | Details |
|---|---|
| Board | OpenBCI Ganglion, 200 Hz, firmware 2 or 3 |
| Radio | BLED112 USB dongle (USB ID 2458:0001), else native Bluetooth |
| Computer | Python 3.12 and a WebGL browser. macOS: `brew install libomp` (XGBoost needs OpenMP) |

Close the OpenBCI GUI first: it holds the radio.

### Electrodes

| Ganglion input | Default name | Position |
|---|---|---|
| 1 | Fz | Frontal midline |
| 2 | Cz | Top of the head, midline |
| 3 | F7 | Left frontal |
| 4 | F8 | Right frontal |
| REF / D_G | | Ear clips |

The features include the input 1 − 2 and 3 − 4 differences, which suit this montage. Other positions
work; name them with `BCI_GAME_CHANNELS`. Wear the headset the same way for recording and flying.

## 3. Settings

| Variable | Default | Use |
|---|---|---|
| `BCI_GAME_RECORDINGS` | `eeg_recordings/` | Ride recordings |
| `EEG_BUNDLES` | `eeg_bundles/` and the project folder | Bundle folders, separated by `;` on Windows and `:` on macOS / Linux. Training writes to the first; the flying website reads all |
| `BCI_GAME_CHANNELS` | `Fz,Cz,F7,F8` | 4 different channel names |
| `BCI_GAME_MAINS_HZ` | `60` | Mains frequency, 50 or 60: notch filter and line-noise check. Saved in each bundle |

`eeg_recordings/` and `eeg_bundles/` hold your brain data. They are in `.gitignore`, but a web upload
ignores `.gitignore`: never drag them into an upload.

## 4. Recording

| Item | Value |
|---|---|
| Hazards | Red light, fallen tree, rockfall, shock, deer, braking lead car; a green light as a no-stop catch |
| Signal check | 1–40 Hz RMS: < 0.5 µV FLAT, > 60 µV NOISY; mains / EEG power > 3 NOISY |
| Use | IN / OUT on the TRAIN page includes or excludes a ride |

Each ride is one folder, `eeg_recordings/<date_time>/`:

| File | Contents |
|---|---|
| `eeg.f64` | float64 rows: unix time, ch1–ch4 (µV), continuous |
| `events.jsonl` | Key changes, hazard phases (`appear`, `demand_stop`, `clear`, `end`, `timeout`), collisions, speed |
| `session.json` | Manifest, `source` (`ganglion` / `simulated`), summary (stops, late, median reaction times), `excluded` flag |

## 5. Labels

| Rule | Value |
|---|---|
| Window | 1 s (200 samples) |
| Step | 0.5 s for the windows every model is scored on; Random Forest training rows every 125 ms; 0.25 s live |
| Label | Key state at the window end; windows that contain a key change are skipped |
| Skipped | First 2 s of a ride; windows with a sample gap over 0.25 s or a span outside 0.65–1.4 s (same rule as live) |
| Artifact gate | ≥ 30 % of the last second's samples more than 150 µV from their channel's median, after ASR (Random Forest: raw EEG): dropped in training, no decision live |

## 6. Training

Training compares six models. Each is scored leave-one-ride-out (every ride is predicted by a model that never
saw it), all on the same 1 s windows, by pooled balanced accuracy. The best is refitted on all rides and saved;
the others appear in its leaderboard. Each window is cleaned and scored exactly as the flying website does live.

| Model | Input | Training |
|---|---|---|
| XGBoost | 300 of 487 window features | Depth 6, learning rate 0.03; trees = best early-stopping round on a 20 % split + 10, at least 25 |
| LightGBM | Same features | Depth 8, 40 leaves, learning rate 0.05; trees by early stopping as above |
| Random Forest | 133 causal features on a 125 ms grid, raw EEG | 300 trees, ride-balanced weights; leaf size and features per split chosen from 4 options by mean per-ride AUC, STOP threshold by balanced accuracy, both on held-out training rides only. With fewer than 3 rides: leaf 50, √features, threshold 0.5 |
| CfC | 20 patches of 10 samples | ncps CfC, 32 units, 128-unit backbone |
| LTC | Same | Liquid time-constant cell, 32 units, 6 solver steps |
| GRU | Same | 32 units |

| Stage | Details |
|---|---|
| ASR | Calibrated on the first 60 s of the oldest ride that long, cutoff 20 SD; applied to each raw 1 s window (not for the Random Forest) |
| Window features | 0.5–45 Hz band-pass, mains notch, common average reference, then 487 features: time, band power, spectral, wavelet, entropy, AR, coherence, channel differences, artifact, indicators. Variance > 0.01, then the 300 with the most mutual information, StandardScaler |
| Random Forest features | Causal 0.5–40 Hz band-pass and mains notch on raw EEG; per channel log and relative power in 6 bands, variance, line length, range, Hjorth, skew, kurtosis, beta change, 4 segment means; 6 channel correlations, channel 3 − 4 difference; band powers and variances minus their median over the past 30 s (skipping the last 2 s, at least 5 s) |
| Networks | Filtered as the window features, normalized with the training rides' channel mean and std; Adam (learning rate 0.001), batch 128, 24 epochs, class-weighted loss, gradient clipping at 1.0, seed 17, one CPU thread |
| Scored windows | When the Random Forest is compared: only windows it can score (from about 8 s into a ride), so all models face the same test. A window the Random Forest would reject live counts as STOP for it |
| STOP threshold | 0.5, or the Random Forest's tuned value; the flying website uses it until the STOP confidence is set |
| Minimum | 2 rides marked IN, 40 clean windows each of GO and STOP |
| Simulator rides | Never mixed with Ganglion rides; a model trained without Ganglion rides only drives the game |
| Time | About 25 minutes on a laptop for five 10-minute rides |

Live decisions use a 3-window majority vote and hold each state for at least 500 ms. The Random Forest
needs the past 35 s of EEG, so it starts deciding 35 s after the headset connects.

## 7. Bundle

One folder per trained model, `eeg_bundles/<name>/`:

| File | Contents |
|---|---|
| `bundle.json` | Format `stop-go-bundle/1`, model id, channels, training report, SHA-256 of every file |
| `model/config_<id>.json` | Model type, features, classes `NOT_STOP` / `STOP`, threshold, window length, per-ride held-out metrics |
| `model/model_<id>.joblib` or `.pt` | The model (`.pt`: network weights, loaded weights-only) |
| `model/scaler_<id>.joblib` | StandardScaler (XGBoost and LightGBM) |
| `asr_state.pkl` | ASR calibration (numbers only) |
| `code/ml/*.py` | The feature, ASR, network and inference code that trained it |

A bundle written by the BCI game is refused when a listed file differs from its hash or `code/` holds an
unlisted `.py` file.
A bundle runs its own code: only load bundles you trained or trust.

## 8. Command line

| Command | Action |
|---|---|
| `python -m bci_game [--port 8790] [--no-browser]` | Start the game |
| `python -m bci_game train [--recordings DIR] [--out DIR] [--name NAME] [--models LIST]` | Compare the models (default all: `xgboost,lightgbm,random_forest,cfc,ltc,gru`) on the rides marked IN, save the best |
| `python -m bci_game list [DIR]` | List bundles with their model and held-out balanced accuracy |
| `python -m bci_game check BUNDLE` | Verify hashes, load, run one noise window |

## 9. Code

| Path | Contents |
|---|---|
| `bci_game/server.py`, `studio.py` | HTTP routes; headset, rides, drive decoder and training job |
| `bci_game/stream.py` | Ganglion (via `bci_v2/acquisition.py`) and simulator sources, background reader |
| `bci_game/session.py`, `labels.py`, `dsp.py` | Ride storage, window labels, signal check |
| `bci_game/dataset.py` | Rides to labelled, ASR-cleaned windows |
| `bci_game/models.py`, `train.py` | The six models and how each is trained; the comparison and the saved best |
| `bci_game/bundles.py` | Bundle folders |
| `bci_game/ml/` | Window features, Random Forest features, networks, ASR and inference; copied into every bundle |
| `bci_game/static/`, `templates/` | Game page and 3D road (three.js) |
| `bci_game_runtime.py` | Finds, verifies and runs bundles (flying website and drive mode) |

## 10. HTTP API

Base URL `http://127.0.0.1:8790`. POST calls need the `X-Token` header from the page.

| Method | Path | Body | Action |
|---|---|---|---|
| GET | `/` | | Game page |
| GET | `/api/status` | | Headset, ride, drive decoder, newest bundle report |
| POST | `/api/connect` | `mode: ganglion / simulated` | Open the headset |
| POST | `/api/disconnect` | | Close the headset |
| GET | `/api/signal` | | Channel grades and trace |
| POST | `/api/ride/start` | `kind: ride / drive, minutes` | Start recording |
| POST | `/api/ride/events` | `events: [...]` (≤ 500) | Append game events |
| POST | `/api/ride/finish` | | Save the ride |
| GET | `/api/live` | | Drive decoder: `state`, `p_go`, `reason` |
| GET | `/api/sessions` | | Saved rides |
| POST | `/api/sessions/<id>/exclude` | `excluded` | Include / exclude a ride |
| POST | `/api/train` | | Start training |
| GET | `/api/train` | | Training state, progress, newest report and its folder |
| POST | `/api/stop` | | Save any ride, release the headset |

# Drone Flight

Indoor flight control for an ArduPilot quadcopter from a Windows laptop: keyboard, head steering and
EEG STOP / GO, with automatic landing on the laptop and on the flight controller. Includes the BCI game
that records your own STOP / GO EEG and trains the model the drone flies with.

See [docs/COMMANDS.md](docs/COMMANDS.md) for controls, limits, safety rules, the API and the MAVLink setup,
and [docs/BCI_GAME.md](docs/BCI_GAME.md) for recording and training.

## Safety

Experimental research code, not a certified flight system or a medical device. Use it at your own risk.
Test new changes with the props off, fly indoors with no one near the drone, and keep a hand on the
EMERGENCY MOTOR CUT button. EEG control is unreliable: the automatic landings are the safety net, not the EEG.

## Layout

| Path | Contents |
|---|---|
| `ardupilot_assisted_web.py` | Entry point: website + flight transport on the SiK radio |
| `ardupilot_flight_transport.py`, `ardupilot_link.py` | Flight logic and MAVLink link |
| `follow_guard.py`, `runaway_guard.py`, `height_planner.py`, `flight_sticks.py` | Safety checks and stick math |
| `assisted_flight_service.py`, `optical_hover_web.py` | Flight service and Flask routes |
| `head_follow.py`, `eye_cameras.py`, `goggle_pupil.py` | Head steering and goggle eye cameras |
| `serial_devices.py` | Finds the radio and head sensor by what they send |
| `optical_eeg.py`, `optical_eeg_control.py`, `saved_eeg_models.py`, `bci_game_runtime.py` | EEG STOP / GO |
| `bci_v2/acquisition.py` | OpenBCI Ganglion connection |
| `bci_game/` | BCI game: records labelled STOP / GO EEG, trains models into `eeg_bundles/` |
| `checkup_feed.py`, `drone_checks.py`, `dashboard_feeds.py` | Status feeds and connection lights |
| `web/templates/optical_hover.html` | Flying page |
| `drone_app/` | Windows desktop app: opens the page and records sessions |
| `fc_scripts/always_land.lua` | Flight controller script (`/APM/scripts`) |
| `windows/` | Laptop setup and maintenance scripts |
| `hardware/` | Printed drone frames, goggles and head gear, and the parts list ([PARTS.md](hardware/PARTS.md)) |
| `tests/` | Unit tests |

## Run

```
pip install -r requirements.txt
python ardupilot_assisted_web.py --port 8083   # finds the radio on any port
```

Open `http://127.0.0.1:8083`.

To fly with your own EEG model, first record and train with the BCI game (`http://127.0.0.1:8790`):

```
python -m bci_game
```

Optional files, not in the repo:
- `models/iris_landmark.tflite` (MediaPipe iris model) for pupil detection. Without it the classic detector is used.
- `eeg_recordings/` and `eeg_bundles/`: your rides and trained STOP / GO models, written by the BCI game.
  They hold your brain data: keep them out of uploads.

## Windows setup

Repo in `$env:USERPROFILE\NeuralFlightWeb`, the `windows/` scripts and `fc_scripts/always_land.lua` in
`$env:USERPROFILE\NeuralFlightAudit`. Then, from an admin PowerShell:

```
powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\NeuralFlightAudit\install_drone_app.ps1"
```

This adds the DRONE APP desktop icon.

## Tests

```
python -m pytest tests
node --test tests/road.test.mjs
```

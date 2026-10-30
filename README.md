# Brain-Controlled Drone Flight

Fly a drone with your brain and eyes.

## How it works

- **Flying website** (`drone_flight/`) runs on the laptop at `http://127.0.0.1:8083` and flies an ArduPilot drone
  over a SiK radio. Turning your head in the FPV goggles steers the drone. Cameras inside the goggles track your
  pupils, so looking left or right with your eyes turns it a little more. Looking up or down tilts the camera gimbal
  instead of moving the drone. Your EEG decides STOP or GO. It lands the drone by itself if the radio or laser is
  lost or the drone stops following its commands.
- **BCI game** (`bci_game/`) records your STOP / GO EEG while you drive, trains six kinds of model, and saves the
  best one to `eeg_bundles/` for the website. No brain data is included: you train your own.
- **Landing script** (`fc_scripts/always_land.lua`) runs on the flight controller and lands the drone even when
  the laptop or radio is gone.
- **Drone App** (`windows/drone_app/`) is a Windows desktop icon that starts the website and opens the flying page.

## What you need

- A 5-inch ArduPilot quadcopter with a MicoAir MTF-01P (optical flow and laser), a SiK radio pair, an analog
  FPV camera and video transmitter, and a servo to tilt the camera
- FPV goggles with a BNO085 head sensor and two eye cameras
- An OpenBCI Ganglion with a BLED112 dongle
- A Windows laptop with Python 3.12

The full parts list is in [hardware/PARTS.md](hardware/PARTS.md). The frame, goggles and head gear are 3D printed:
see [hardware/](hardware/README.md).

## Setup

### 1. Laptop

```
git clone https://github.com/SpoWT/brain-controlled-drone-flight.git
cd brain-controlled-drone-flight
pip install -r requirements.txt
```

For better pupil tracking, put the MediaPipe iris model at `models/iris_landmark.tflite` (optional).

### 2. Flight controller

1. Flash ArduCopter 4.7.1.
2. Set the parameters in [docs/COMMANDS.md §10](docs/COMMANDS.md#10-flight-controller-settings), including the camera
   gimbal on the tilt servo's output. The compass stays off.
3. Plug the flight controller in by USB and install the landing script:

   ```
   python windows/always_land_install.py
   ```

### 3. Train your STOP / GO model

```
python -m bci_game
```

Open `http://127.0.0.1:8790`, connect the Ganglion, drive at least two rides, then train. The best model is saved
to `eeg_bundles/`. Electrode placement and training details: [docs/BCI_GAME.md](docs/BCI_GAME.md).

Feel free to test it in an FPV simulator (for example on Steam) first: `python -m bci_game keys` holds W
(`--key up` for the up arrow) while you think GO, so the drone climbs, and lets go on STOP, so it falls. For safety
it is suggested to check it in a simulator before flying for real, but it's up to you.

Your rides (`eeg_recordings/`) and models (`eeg_bundles/`) hold your brain data. Git ignores both folders, so they
stay out of uploads.

### 4. Fly

```
python -m drone_flight --port 8083
```

Open `http://127.0.0.1:8083`. The website finds the radio on any port. Controls, limits and automatic landings:
[docs/COMMANDS.md](docs/COMMANDS.md).

### 5. Desktop app (optional)

From an admin PowerShell in the repo folder:

```
powershell -ExecutionPolicy Bypass -File windows\install_drone_app.ps1
```

This adds the DRONE APP desktop icon.

## Safety

This is not a certified flight system or a medical device. Use it at your own risk.
Test new changes with the props off, keep people away from the drone, and keep a hand on the EMERGENCY MOTOR CUT
button.

## Project layout

| Folder | Contents |
|---|---|
| `drone_flight/` | Flying website: `flight/` (ArduPilot link, flight logic, safety guards), `head/` (head steering, eye cameras), `eeg/` (live STOP / GO), `web/` (routes, status feeds, flying page) |
| `bci_game/` | BCI game, model training, and the model runtime and Ganglion connection the website uses |
| `fc_scripts/` | Flight controller landing script |
| `windows/` | Drone App, install and start / stop scripts, flight controller tools |
| `hardware/` | Printed parts and the parts list |
| `docs/` | Command reference and BCI game reference |
| `tests/` | Tests |

## Tests

```
python -m pytest
node --test tests/road.test.mjs
```

## License

MIT. See [LICENSE](LICENSE).

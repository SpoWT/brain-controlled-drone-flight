# Drone Command Reference

ArduCopter 4.7.1 on a CORVON743V1, flown indoors and outdoors from a Windows laptop over a 915 MHz SiK radio.
Position from optical flow, height from a laser. No GPS, no compass.

## 1. Hardware

| Part | Details |
|---|---|
| Flight controller | CORVON743V1, ArduCopter 4.7.1 |
| Flow + laser | MTF-01P on SERIAL4 (MAVLink1, 115200) |
| Telemetry | SiK 915 MHz. Drone: TELEM1 / SERIAL1 (MAVLink2, 57600). Laptop: USB serial (CP210x), any COM port |
| Head sensor | BNO08x in UART-RVC mode via CP2102 USB, 115200 baud, ~100 Hz |
| Eye cameras | 2 × Innomaker U20CAM inside the goggles, 640×480 MJPG |
| EEG | OpenBCI Ganglion via BLED112 USB dongle |
| Battery | 6S |

## 2. Starting

- **Drone App** (desktop icon): while disarmed, closes the flying page it opened before, restarts the website and the
  flight controller, then opens the flying page in its own window. While armed or flying it changes nothing and says
  to use the open page (opening another page would land the drone).
- **Manual:** `python -m drone_flight --port 8083`, then open `http://127.0.0.1:8083`. `--mavlink COM7` only says where to look first.
- The website only answers on `127.0.0.1`.
- Opening the page starts a new control session. Any page opened earlier loses control.
- Before take-off: drone flat on the floor, laser reading, tick the confirmation box.

## 3. Flight controls

| Key / button | Action |
|---|---|
| `W` / `S` | Forward / back at the key speed (default 0.3 m/s) |
| `A` / `D` | Left / right |
| `↑` / `↓` | Target height +/− 0.1 m per press |
| `←` / `→` | Turn on the spot, 45 °/s |
| `SPACE` | Stop and hover |
| `L` | Land |
| `C` | Recentre head: where the goggles face and the eyes look now = straight ahead |
| ARM & TAKE OFF | Arm in LOITER (ALT_HOLD if refused), climb to the take-off height (0.3–2.0 m, default 0.3) |
| GO FORWARD | Fly the set distance straight ahead (0.1–5.0 m, default 2.0), then hover |
| STOP & HOVER | Stop and hold position |
| LAND | Gentle laser landing (FC LAND if that is not possible) |
| EMERGENCY MOTOR CUT | Motors off immediately. The drone falls. |

Keys only work while the flying page has focus. Releasing a key brakes and holds position.

**Head steering** (on by default): the drone's heading follows the goggles' yaw.
Deadband 4°, 1.5 °/s per degree of difference, max 45 °/s. Arrow keys override it. Re-centred at every take-off.

**Eye steering** (with head steering): looking left or right with the eyes adds up to 15° to the head's heading
(pupil moved 0.2 of the eye camera picture = 15°). Looking up or down with the head and eyes tilts the FPV camera
gimbal up to 30° either way instead of moving the drone. Both work while hovering or flying on keys or EEG, not
during GO FORWARD, take-off or landing; then the camera is level. Each eye uses the median of its last 0.5 s of
pupil positions, so blinks are ignored, and moves under 0.03 of the picture count as straight ahead. With the eyes
closed or the eye cameras off, only the head steers. Straight ahead for the eyes and head tilt is set at take-off and
by `C`; an arrow turn keeps it. Turning an eye picture in flight leaves that eye out until `C`. The eye cameras stay
on while head steering is on. The website levels the camera when it starts: ArduPilot keeps the last tilt it was sent.

**EEG STOP / GO** ("Use EEG STOP / GO alongside computer controls"): GO flies forward at 0.2 m/s while GO
decisions keep arriving (0.5 s). Anything else hovers. STOP, LAND, motor cut or a measured move turns it off.

**EEG models**: each folder in `eeg_bundles/` or the project folder (or in the folders listed in `EEG_BUNDLES`,
separated by `;` on Windows) is one trained STOP / GO bundle from the BCI game ([BCI_GAME.md](BCI_GAME.md)).
The website flies Auto, the bundle with the best held-out balanced accuracy (plain accuracy if any bundle lacks it),
unless `POST /api/eeg/model` chose another (`model_id`: `auto` or `bundle-<folder name>`; kept across restarts).
Bundles trained without Ganglion rides are ignored, and a bundle whose files do not match their recorded SHA-256 is
not loaded. Restart the website after training.

## 4. Limits

| Item | Limit |
|---|---|
| Target height | 0.3–2.0 m |
| Hard ceiling | 2.3 m |
| Horizontal speed | 0.4 m/s max |
| Turn rate | 45 °/s from keys, head or eyes |
| Camera tilt | 30° up or down |
| Measured move | 5 m max |
| Flight time | 180 s, then it lands |
| Take-off battery | 22.2 V min |

## 5. Automatic landing

### In the flight controller (works without the laptop or radio)

| Trigger | Action | Setting |
|---|---|---|
| No laptop heartbeat for 3 s (also sent only while the control loop runs without errors) | LAND | `FS_GCS_ENABLE 5`, `FS_GCS_TIMEOUT 3` |
| Battery 21.6 V / 21.0 V | LAND | `BATT_FS_LOW_ACT 1`, `BATT_FS_CRT_ACT 1` |
| Position estimate failing | LAND | `FS_EKF_ACTION 1` |
| Crash detected | Motors off | `FS_CRASH_CHECK 1` |
| Laser above 2.3 m for 0.5 s | LAND | `ALAND_MAX_HGT` |
| No laser reading for 1 s | LAND | always_land.lua |
| Roll or pitch over 25° for 0.5 s | LAND | `ALAND_TILT` |
| Attitude error over 6° for 0.5 s | LAND | `ALAND_ATT_ERR` |
| Floor camera quality < 50 for 2 s in LOITER | LAND | always_land.lua |
| LAND climbing 0.4 m for 1 s | Motors off | always_land.lua |

Tilt, attitude and camera checks start 10 cm above the floor.

### On the laptop (checked every 0.1 s in flight)

| Trigger | Action | Fault |
|---|---|---|
| Page silent for 0.3 s | Hold position | |
| Page silent for 2 s | Land | |
| 0.15 m above target for 1 s, or 0.35 m above at once | Land | `OVER_HEIGHT` |
| 0.15 m below target for 1 s (after reaching it) | Land | `UNDER_HEIGHT` |
| Key held 1.5 s but < 30 % of its speed that way, or > 0.3 m/s sideways, for 1 s | Level land | `WRONG_DIRECTION` |
| No key, moving faster than 0.4 m/s for 1.5 s | Level land | `WRONG_DIRECTION` |
| Lean differs from the commanded lean by > 5.5° for 0.5 s | Level land | `TILT_MISMATCH` |
| Lean ≥ 12° or estimated speed ≥ 0.8 m/s for 1.5 s | Level land | `RUNAWAY` |
| Floor camera quality < 50, or no floor camera data, for 1.5 s | Level land | `FLOW_LOST` |
| No laser reading for 2 s | FC LAND | `LASER_LOST` |
| Landing rises 0.4 m for 0.6 s | Motors off | |

Level land = ALT_HOLD (no position hold) plus the gentle laser descent.
Gentle descent: 0.15 m/s, then 0.10 m/s below 10 cm, motors off at 5 cm.

## 6. Fault codes

| Code | Meaning |
|---|---|
| `PRECONDITION` | Take-off refused: drone not flat on the floor or laser not reading |
| `BAD_PARAMS` | Height, distance or calibration value out of range |
| `NO_LIFT` | Full lift without leaving the ground: battery or props |
| `TIP_OVER` | Tilted too far, motors stopped |
| `LASER_LOST` | Laser stopped reporting |
| `FLOW_LOST` | Floor camera could not see the floor |
| `LINK_LOST` | Radio commands stopped |
| `OVER_HEIGHT` / `UNDER_HEIGHT` | Too far above / below the target height |
| `CLIMB_TIMEOUT` | Could not reach the target height |
| `MOVE_TIMEOUT` | Measured move did not finish in time |
| `AIRBORNE_TIMEOUT` | Maximum flight time reached |
| `EXTERNAL_LOCK` | Motors cut (emergency cut or tilt protection) |
| `LAND_TIMEOUT` | Touchdown took too long, motors stopped |
| `RUNAWAY` | Position estimate went wrong |
| `WRONG_DIRECTION` | Did not go where it was told, or drifted |
| `TILT_MISMATCH` | Tilted differently from what it was told |

## 7. Website API

Base URL `http://127.0.0.1:8083`. Calls marked **T** need the `X-Flight-Token` header from the open page.
Commands carry an increasing `sequence` number.

| Method | Path | Body | Action |
|---|---|---|---|
| GET | `/` | | Flying page. Starts a new control session |
| GET | `/launcher-health` | | Service up, mode, blocker |
| GET | `/api/checkup` | | Full read-only status and connection checks |
| GET | `/api/status` | | Flight state (T) |
| POST | `/api/heartbeat` | `forward, right, yaw, target_mm` | Key state every 80 ms, mm/s and °/s (T) |
| POST | `/api/arm` | `sequence, target_mm` | Arm and take off (T) |
| POST | `/api/go` | `sequence, distance_mm` | Fly forward a distance (T) |
| POST | `/api/move` | `sequence, forward_mm, right_mm` | Measured move (T) |
| POST | `/api/stop` | `sequence` | Stop and hover (T) |
| POST | `/api/land` | `sequence` | Land (T) |
| POST | `/api/emergency-lock` | | Motor cut (T) |
| POST | `/api/head` | `enabled` | Head steering on / off (T) |
| POST | `/api/head/recenter` | | Head and eyes = straight ahead (T) |
| GET | `/api/direction` | | Head and drone heading, turn rate, eye turn, camera tilt |
| GET | `/api/eyes` | | Eye camera status, pupil size and position |
| POST | `/api/eyes/<n>/rotate` | `by: 90 / -90 / 180` | Rotate an eye camera view |
| GET | `/eyes/<n>.mjpg` | | Eye camera video |
| GET | `/api/eeg` | | EEG status (T) |
| POST | `/api/eeg/connect`, `/api/eeg/disconnect` | | EEG headset (T) |
| POST | `/api/eeg/threshold` | `threshold` (0.01–0.99) | STOP confidence (T) |
| POST | `/api/eeg/model` | `model_id` | Choose the STOP / GO model (T) |
| POST | `/api/eeg/control` | `enabled` | EEG STOP / GO on / off (T) |

## 8. Command lines (website → flight transport)

| Line | Action |
|---|---|
| `LOCK` | Motors off (emergency cut when armed) |
| `STANDBY` | Ready for take-off (on the ground, disarmed) |
| `CONFIG <scale_um> <axis>` | Optical-flow calibration |
| `TAKEOFF <mm>` | Arm and climb to `<mm>` (300–2000) |
| `FLY <mm> <fwd> <right> [<yaw>]` | Target height (mm), body speed (mm/s), turn (°/s). Sent every 0.05 s in flight |
| `MOVE <fwd_mm> <right_mm>` | Measured move along the heading at the start, then hold |
| `HOLD` | Stop and hold position |
| `LAND` | Gentle landing, else FC LAND |
| `HEAD 0` / `HEAD 1` | Head steering off / on (on also re-centres) |
| `PING`, `STATUS` | Keep-alive |

Take-off sequence: `STANDBY`, `CONFIG`, `TAKEOFF`.

## 9. MAVLink

**Sent by the laptop** (system 254, component 190):

| Message | Use |
|---|---|
| `HEARTBEAT` | 1 Hz, GCS type |
| `MANUAL_CONTROL` | Pitch / roll / yaw −1000…1000, throttle 0…1000 (500 = hold height) |
| `SET_MODE` | 2 ALT_HOLD, 5 LOITER, 9 LAND |
| `COMMAND_LONG 400` | Arm / disarm (force code 21196) |
| `COMMAND_LONG 511` | Message rates |

**Received** (Hz): `ATTITUDE` 5, `LOCAL_POSITION_NED` 5, `DISTANCE_SENSOR` 5, `NAV_CONTROLLER_OUTPUT` 3,
`OPTICAL_FLOW` 2, `SERVO_OUTPUT_RAW` 2, `EXTENDED_SYS_STATE` 1, `HEARTBEAT` 1, `SYSTEM_TIME` 0.5, `SYS_STATUS` 0.5.

Stick scaling: full yaw = 202.5 °/s (`PILOT_Y_RATE`), full LOITER stick = 0.6 m/s (`LOIT_SPEED`), throttle deadzone 100.

## 10. Flight controller settings

| Parameter | Value |
|---|---|
| `SERIAL1_PROTOCOL` / `SERIAL1_BAUD` | 2 / 57 |
| `FLOW_TYPE` | 5 |
| `RNGFND1_TYPE` | 10 |
| `EK3_SRC1_POSZ` | 2 (laser height) |
| `EK3_SRC1_YAW` | 0 |
| `COMPASS_USE`, `COMPASS_USE2`, `COMPASS_USE3` | 0 |
| `FENCE_ENABLE` | 0 (needs GPS) |
| `SCR_ENABLE` | 1 |
| `BATT_ARM_VOLT` | 22.2 |
| `ALAND_ENABLE` / `ALAND_MAX_HGT` / `ALAND_TILT` / `ALAND_ATT_ERR` / `ALAND_TEST` | 1 / 2.3 / 25 / 6 / 0 |
| `MNT1_TYPE` | 1 (servo gimbal) |
| `SERVOx_FUNCTION` on the gimbal tilt servo's output | 7 (Mount1Pitch) |
| `MNT1_PITCH_MIN` / `MNT1_PITCH_MAX` | -30 / 30 |
| `SERVOx_MIN` / `SERVOx_MAX` on that output | PWM where the camera points 30° down / 30° up |

`ALAND_*` come from `/APM/scripts/always_land.lua`. `ALAND_TEST 1` only reports, never acts.
Camera level is halfway between `SERVOx_MIN` and `SERVOx_MAX`; `SERVOx_TRIM` does not move it.

## 11. Connection lights

Shown at the top of the flying page: RADIO, DRONE, BATT, FC, LASER, FLOOR CAM, HEAD, EYES, EEG.
Green = fine, amber = check, red = fix now. Hover over a light for the fix.
Devices are found by what they send, never by COM number (the radio and head sensor adapters can swap COM
numbers with plug order):

- **Radio:** the port carrying the flight controller's heartbeats. Checked every second. Silent for 5 s while
  on the ground: the website looks on every USB adapter again, until the drone answers. Never moved while flying.
- **Head sensor:** the port sending UART-RVC packets. Silent for 3 s: looked for again.
- **Eye cameras:** EYE 1 / EYE 2 follow the USB port each camera sits on inside the goggles (lower port = EYE 1),
  so re-plugging the goggles anywhere keeps them and their rotation. One camera is never shown in both slots.

## 12. Logs

The website logs every MAVLink message to `outputs/mavlink_logs/mav_<time>.jsonl`.

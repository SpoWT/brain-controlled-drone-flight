# Parts

Modules in the final build. The frame, gear and goggle shells are 3D printed (this folder).

## Drone

| Part | Model | Qty | Use |
|---|---|---:|---|
| Flight controller | AERO SELFIE H743 (board CORVON743V1), ArduPilot | 1 | Stabilizes and flies the drone |
| ESC | AERO SELFIE 60A 32-bit 4-in-1 (stack with the H743) | 1 | Drives the four motors |
| Motors | X3115 900KV brushless (DYS) | 4 | Propulsion |
| Propellers | 5-inch two-blade | 4 (2 CW, 2 CCW) | Thrust |
| Battery | OVONIC FPV 6S 22.2 V 1300 mAh 130C, XT60 | 1 | Flight power |
| Optical flow + rangefinder | MicoAir MTF-01P | 1 | Floor motion and height for indoor hover |
| FPV camera | JINJIEAN B19 Pro, analog | 1 | Video from the drone |
| Video transmitter | 5.8 GHz analog, 2000 mW, fan-cooled, with antenna | 1 | Sends the video to the goggles |
| Telemetry radio, air | YoungRC 915 MHz 100 mW SiK (HM-TRP) | 1 | MAVLink link to the laptop |
| Micro servos | Miuzei MS18 9 g | 2 | FPV camera gimbal |

## Goggles (`fpv_goggles_v1.3mf`)

| Part | Model | Qty | Use |
|---|---|---:|---|
| FPV monitor | Analog FPV monitor with built-in 5.8 GHz receiver, two SMA, with its LiPo pouch and 2 antennas | 1 | Shows the drone video |
| Eye cameras | InnoMaker U20CAM-1080P-S1, 1080p USB 2.0 UVC, 130° | 2 | One per eye, for gaze tracking |
| USB hub | Cable Matters 201309-BLK Ultra Mini 4-port USB-C, 4 ft lead | 1 | Joins the cameras and the head sensor to the laptop |
| Head orientation | GY-BNO08X (BNO080 / BNO085) board | 1 | Head steering |
| USB-UART | SEENGREAT SG-C2102-01C REV1.1 (CP2102) | 1 | Connects the head orientation board |

## EEG

| Part | Model | Qty | Use |
|---|---|---:|---|
| EEG board | OpenBCI Ganglion, 4 channels at 200 Hz | 1 | Records the STOP / GO EEG |
| Electrodes | Scalp electrodes with leads, reference and ground ear clips | 4 + 2 | Fz, Cz, F7, F8 by default |

## Laptop side

| Part | Model | Qty | Use |
|---|---|---:|---|
| Laptop | Windows | 1 | Runs the flying website and the BCI game |
| BLE dongle | Bluegiga BLED112 | 1 | Ganglion link |
| Telemetry radio, ground | YoungRC 915 MHz 100 mW SiK, USB (CP2102) | 1 | Pairs with the air radio |
| Charger | ISDT 608PD, 1–6S balance | 1 | Charges the flight batteries |

"""Eye steering and camera tilt: where the eyes look adds a little to head steering.

Left / right: the eyes add up to EYE_TURN_MAX_DEG to the heading the head asks for.
Up / down: the head and eyes tilt the FPV camera gimbal instead of moving the drone.
Both work while flying with head steering (phase FLY); otherwise the eyes add nothing and the camera is level.

The eye cameras face the eyes, so looking right moves the pupil left in the picture and looking up moves it up
(the page's rotate buttons stand each eye upright). Pupil positions are fractions of the picture.

The gimbal's tilt servo runs on ArduPilot's mount driver (MNT1_TYPE 1, the servo output on Mount1Pitch). The
website sends the angle with MAV_CMD_DO_MOUNT_CONTROL; the flight controller keeps it within
MNT1_PITCH_MIN / MNT1_PITCH_MAX.
"""
from __future__ import annotations

from typing import Optional

EYE_FRESH_S = 0.5             # older pupil positions are not used
EYE_DEADBAND = 0.03           # pupil movements smaller than this fraction of the picture are jitter
EYE_TURN_DEG_PER_FRAC = 75.0  # pupil 0.2 of the picture sideways = 15 degrees more heading
EYE_TURN_MAX_DEG = 15.0
EYE_TILT_DEG_PER_FRAC = 75.0
EYE_TILT_MAX_DEG = 15.0
HEAD_PITCH_SIGN = 1.0         # BNO08x pitch in the goggles: + = looking up
TILT_MAX_DEG = 30.0           # camera tilt either way
TILT_STEP_DEG = 2.0           # smaller changes are not sent
TILT_EVERY_S = 0.2            # at most this often over the radio
TILT_RESEND_S = 1.0           # a held tilt is sent again this often, in case the radio dropped it
TILT_SETTLE_S = 3.0           # level is repeated this long after a change, then the radio goes quiet


def _degrees(fraction: float, per_fraction: float, limit: float) -> float:
    if abs(fraction) < EYE_DEADBAND:
        return 0.0
    return max(-limit, min(limit, fraction * per_fraction))


class EyeGaze:
    """Where the eyes look, relative to where they looked when steering was centred."""

    def __init__(self) -> None:
        self._zero: dict = {}  # eye -> pupil (x, y) when looking straight ahead

    @property
    def zeroed(self) -> bool:
        return bool(self._zero)

    def zero(self, pupils: dict) -> None:
        """These pupil positions count as straight ahead. An eye not seen now is not used until the next zero."""
        self._zero = dict(pupils)

    def clear(self) -> None:
        self._zero = {}

    def offset(self, pupils: dict) -> tuple[float, float]:
        """(degrees right, degrees up) to add, averaged over the eyes seen; (0, 0) without a reading."""
        moves = [(self._zero[eye][0] - x, self._zero[eye][1] - y) for eye, (x, y) in pupils.items()
                 if eye in self._zero]
        if not moves:
            return 0.0, 0.0
        right = sum(move[0] for move in moves) / len(moves)
        up = sum(move[1] for move in moves) / len(moves)
        return (_degrees(right, EYE_TURN_DEG_PER_FRAC, EYE_TURN_MAX_DEG),
                _degrees(up, EYE_TILT_DEG_PER_FRAC, EYE_TILT_MAX_DEG))


def tilt_for(head_up_deg: float, eyes_up_deg: float) -> float:
    """Camera tilt in degrees (+ = up) for the head's and eyes' up / down."""
    return max(-TILT_MAX_DEG, min(TILT_MAX_DEG, head_up_deg + eyes_up_deg))


class CameraTilt:
    """Sends the first tilt and every change at once. A tilt is repeated every TILT_RESEND_S, level only for
    TILT_SETTLE_S, in case the radio dropped one. ArduPilot holds a MAVLink tilt until told otherwise, so the first
    one (level, on the ground) also straightens a camera that an earlier run left tilted."""

    def __init__(self) -> None:
        self._sent: Optional[float] = None
        self._sent_at = self._changed_at = -1e9

    @property
    def angle(self) -> float:
        return self._sent or 0.0

    def update(self, link, angle: float, now: float) -> None:
        if link.heartbeat() is None or now - self._sent_at < TILT_EVERY_S:
            return
        changed = self._sent is None or abs(angle - self._sent) >= TILT_STEP_DEG
        settling = angle != 0.0 or now - self._changed_at < TILT_SETTLE_S
        if not (changed or (settling and now - self._sent_at >= TILT_RESEND_S)):
            return
        if changed:
            self._changed_at = now
        link.mount_tilt(angle)
        self._sent, self._sent_at = angle, now

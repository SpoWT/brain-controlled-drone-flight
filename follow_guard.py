"""Lands the drone when it stops following its commands: wrong height, wrong direction, drifting while
holding, or tilting differently from what the flight controller commands. Pure logic."""
from __future__ import annotations

import math
from typing import Optional

HEIGHT_BAND_MM = 150           # above the target (always) or below it (once reached)
HEIGHT_BAND_S = 1.0
KEY_MIN_MM_S = 100             # slower counts as holding still
DIRECTION_SETTLE_S = 1.5       # key held this long before it is judged
DIRECTION_MIN_FRACTION = 0.3   # of the asked speed, in the asked direction
SIDEWAYS_MAX_MM_S = 300
DIRECTION_BAD_S = 1.0
HOLD_DRIFT_MM_S = 400
HOLD_DRIFT_S = 1.5
TILT_MISMATCH_DEG = 5.5
TILT_MISMATCH_S = 0.5


class Held:
    """True once a condition has been true without a break for `seconds`."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._since: Optional[float] = None

    def update(self, now: float, condition: bool) -> bool:
        if not condition:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        return now - self._since >= self.seconds


class FollowGuard:
    """Returns a fault name the moment the drone stops following its commands, else None."""

    def __init__(self) -> None:
        self._too_high = Held(HEIGHT_BAND_S)
        self._too_low = Held(HEIGHT_BAND_S)
        self._wrong_way = Held(DIRECTION_BAD_S)
        self._drifting = Held(HOLD_DRIFT_S)
        self._tilt = Held(TILT_MISMATCH_S)
        self._command: tuple[float, float] = (0.0, 0.0)
        self._command_since = 0.0

    def height(self, now: float, height_mm: Optional[int], ceiling_mm: int,
               floor_mm: Optional[int]) -> Optional[str]:
        """Above ceiling_mm always counts; below floor_mm only when a floor is given."""
        if height_mm is None:
            self._too_high.update(now, False)
            self._too_low.update(now, False)
            return None
        if self._too_high.update(now, height_mm > ceiling_mm):
            return "OVER_HEIGHT"
        if self._too_low.update(now, floor_mm is not None and height_mm < floor_mm):
            return "UNDER_HEIGHT"
        return None

    def direction(self, now: float, command: tuple[float, float],
                  measured: tuple[Optional[float], Optional[float]]) -> Optional[str]:
        """command and measured are (forward, right) body speeds in mm/s."""
        if command != self._command:
            self._command, self._command_since = command, now
        fwd, right = measured
        if fwd is None or right is None:
            return None
        asked = math.hypot(*command)
        if asked < KEY_MIN_MM_S:
            self._wrong_way.update(now, False)
            return "WRONG_DIRECTION" if self._drifting.update(now, math.hypot(fwd, right) > HOLD_DRIFT_MM_S) else None
        self._drifting.update(now, False)
        if now - self._command_since < DIRECTION_SETTLE_S:
            self._wrong_way.update(now, False)
            return None
        ux, uy = command[0] / asked, command[1] / asked
        along = fwd * ux + right * uy
        sideways = abs(-fwd * uy + right * ux)
        bad = along < DIRECTION_MIN_FRACTION * asked or sideways > SIDEWAYS_MAX_MM_S
        return "WRONG_DIRECTION" if self._wrong_way.update(now, bad) else None

    def tilt(self, now: float, actual, commanded) -> Optional[str]:
        """actual: ATTITUDE (radians); commanded: NAV_CONTROLLER_OUTPUT (nav_roll / nav_pitch, degrees)."""
        if actual is None or commanded is None:
            self._tilt.update(now, False)
            return None
        error = max(abs(math.degrees(actual.roll) - commanded.nav_roll),
                    abs(math.degrees(actual.pitch) - commanded.nav_pitch))
        return "TILT_MISMATCH" if self._tilt.update(now, error > TILT_MISMATCH_DEG) else None

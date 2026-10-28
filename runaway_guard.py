"""Detects a bad position estimate. Normal LOITER flying stays under ~9 deg and 0.6 m/s, so holding
either well past that for RUNAWAY_S means the estimate has gone wrong."""
from __future__ import annotations

import math
from typing import Optional

RUNAWAY_TILT_DEG = 12.0
RUNAWAY_SPEED_MM_S = 800
RUNAWAY_S = 1.5


def tilt_deg(attitude) -> Optional[float]:
    """Largest of roll and pitch in degrees, from a MAVLink ATTITUDE message (None if missing)."""
    if attitude is None:
        return None
    return math.degrees(max(abs(attitude.roll), abs(attitude.pitch)))


def speed_mm_s(local_position) -> Optional[float]:
    """Horizontal EKF speed in mm/s, from a MAVLink LOCAL_POSITION_NED message (None if missing)."""
    if local_position is None:
        return None
    return math.hypot(local_position.vx, local_position.vy) * 1000.0


class RunawayGuard:
    """True once the drone has leaned too hard, or believed it is going too fast, for RUNAWAY_S."""

    def __init__(self) -> None:
        self._since: Optional[float] = None

    def update(self, now: float, tilt: Optional[float], speed: Optional[float]) -> bool:
        bad = (tilt is not None and tilt >= RUNAWAY_TILT_DEG) or \
              (speed is not None and speed >= RUNAWAY_SPEED_MM_S)
        if not bad:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        return now - self._since >= RUNAWAY_S

"""Height changes planned ahead, mixed into ArduPilotFlightTransport.

The laser arrives up to ~1 s late over the radio, so chasing it overshoots. Each change climbs at a fixed
rate for distance / rate seconds, then waits for steady laser readings and corrects 80 % of what is left.
"""
from __future__ import annotations

import math
from typing import Optional

from flight_sticks import plan_for

PLAN_DEADBAND_MM = 30
PLAN_CORRECTION_GAIN = 0.8     # approach from one side, never cycle
SETTLED_VZ_M_S = 0.05
SETTLED_S = 0.6
LASER_STEADY_S = 1.2           # covers the radio lag
LASER_STEADY_MM = 15
SETTLE_GIVE_UP_S = 3.0         # not settled by then: plan from the median laser


class HeightPlanner:
    def _height_stick(self, now: float, height: Optional[int]) -> int:
        """Predictive height: run planned moves open-loop, correct only once everything is still."""
        if self._plan_target is not None and self._target_mm != self._plan_target and now < self._plan_end:
            self._plan_end = now  # target changed mid-move: settle and re-plan
        if now < self._plan_end:
            return self._plan_stick
        if height is None:
            return 500
        if not self._settled(now):
            if now - self._plan_end < SETTLE_GIVE_UP_S:
                return 500
            height = self._recent_height_mm(now)
            if height is None:
                return 500
        error = self._target_mm - height
        if abs(error) <= PLAN_DEADBAND_MM:
            self._plan_target = self._target_mm
            return 500
        distance = error if self._plan_target != self._target_mm else error * PLAN_CORRECTION_GAIN
        self._plan_stick, seconds = plan_for(distance)
        self._plan_target, self._plan_end = self._target_mm, now + seconds
        self._plans.append((now, distance / seconds if seconds > 0 else 0.0, distance))
        self._settled_since = 0.0
        return self._plan_stick

    def _recent_height_mm(self, now: float) -> Optional[int]:
        recent = sorted(mm for t, mm in self._laser_hist if now - t <= LASER_STEADY_S)
        return recent[len(recent) // 2] if recent else None

    def _expected_climb_mm(self, now: float) -> float:
        """Height gain the planned moves should have produced by now (for the stuck-laser guard)."""
        total = 0.0
        for start, rate, distance in self._plans:
            done = abs(rate) * max(0.0, now - start) if rate else abs(distance)
            total += math.copysign(min(done, abs(distance)), distance)
        return total

    def _settled(self, now: float) -> bool:
        """Still in the FC's estimate AND in the (late) laser readings, so the laser shows where it is."""
        pos = self.link.latest("LOCAL_POSITION_NED", 1.0)
        recent = [mm for t, mm in self._laser_hist if now - t <= LASER_STEADY_S]
        steady = (len(recent) >= 3 and max(recent) - min(recent) <= LASER_STEADY_MM
                  and now - self._laser_hist[0][0] >= LASER_STEADY_S * 0.8)
        if pos is None or abs(pos.vz) > SETTLED_VZ_M_S or not steady:
            self._settled_since = 0.0
            return False
        self._settled_since = self._settled_since or now
        return now - self._settled_since >= SETTLED_S

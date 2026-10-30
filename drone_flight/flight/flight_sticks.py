"""MANUAL_CONTROL stick values for climbs, planned height changes, turns and the ALT_HOLD speed loop."""
from __future__ import annotations

from typing import Optional

PILOT_SPEED_M_S = 0.5          # PILOT_SPD_UP / PILOT_SPD_DN on this FC
THROTTLE_DEADZONE = 100        # THR_DZ on this FC
PILOT_Y_RATE_DEG_S = 202.5     # PILOT_Y_RATE on this FC: full yaw stick = this turn rate
PLAN_RATE_M_S = 0.15
PLAN_FINE_RATE_M_S = 0.08      # changes of PLAN_FINE_ABOVE_MM or less
PLAN_FINE_ABOVE_MM = 100
# ALT_HOLD fallback: keys ask for a speed and a loop on the EKF velocity sets the lean.
MAX_SPEED_MM_S = 400           # cap on any requested speed
LEAN_FF_PER_MM_S = 0.10        # feed-forward stick units per mm/s
LEAN_KP_PER_MM_S = 0.30        # feedback stick units per mm/s of speed error
MAX_LEAN = 300                 # of 1000: 4.5 deg at ATC_ANGLE_MAX 15 deg


def throttle_for_climb(rate_m_s: float) -> int:
    """MANUAL_CONTROL z (0..1000) for a climb rate under ALT_HOLD-style throttle."""
    if abs(rate_m_s) < 1e-3:
        return 500
    span = 500 - THROTTLE_DEADZONE
    frac = max(-1.0, min(1.0, rate_m_s / PILOT_SPEED_M_S))
    edge = 500 + THROTTLE_DEADZONE if frac > 0 else 500 - THROTTLE_DEADZONE
    return int(round(edge + frac * span))


def plan_for(distance_mm: float) -> tuple[int, float]:
    """(throttle stick, seconds) for one planned height change at a fixed rate."""
    rate = PLAN_RATE_M_S if abs(distance_mm) > PLAN_FINE_ABOVE_MM else PLAN_FINE_RATE_M_S
    return throttle_for_climb(rate if distance_mm > 0 else -rate), abs(distance_mm) / 1000.0 / rate


def lean_for_speed(target_mm_s: float, measured_mm_s: Optional[float]) -> int:
    """Stick for one axis in the ALT_HOLD fallback: speed feedback, also braking to 0. No data -> FF only."""
    target = max(-MAX_SPEED_MM_S, min(MAX_SPEED_MM_S, target_mm_s))
    if target == 0 and measured_mm_s is None:
        return 0
    lean = target * LEAN_FF_PER_MM_S
    if measured_mm_s is not None:
        lean += (target - measured_mm_s) * LEAN_KP_PER_MM_S
    return int(max(-MAX_LEAN, min(MAX_LEAN, round(lean))))


def yaw_stick(rate_deg_s: float) -> int:
    """MANUAL_CONTROL r (-1000..1000) for a turn rate in LOITER/ALT_HOLD (stick = rate / PILOT_Y_RATE)."""
    return int(round(max(-1000.0, min(1000.0, rate_deg_s / PILOT_Y_RATE_DEG_S * 1000.0))))

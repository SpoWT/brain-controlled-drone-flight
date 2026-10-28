"""Turn a ride's event log into window labels and a summary.

The label is what the rider did: drive key held = GO, released = STOP.
Hazard events are kept so reaction times can be recomputed.
"""

from __future__ import annotations

from statistics import median

import numpy as np

from bci_game_runtime import continuous_window

from .config import GO, SAMPLE_RATE, STOP, TRAIN_STEP_S, WINDOW_S

STOP_HAZARDS = frozenset({"red_light", "fallen_tree", "rockfall", "shock", "deer", "lead_car"})
SETTLE_S = 2.0


def _seconds(event: dict[str, object]) -> float:
    return float(event["t"]) / 1000.0


def key_transitions(events: list[dict[str, object]]) -> list[tuple[float, bool]]:
    """Return (time_s, driving) at every change of the drive key, de-duplicated."""
    transitions: list[tuple[float, bool]] = []
    driving = False
    for event in events:
        if event.get("type") == "key":
            down = bool(event.get("down"))
            if down != driving:
                transitions.append((_seconds(event), down))
                driving = down
        elif event.get("type") == "ride_end" and driving:
            transitions.append((_seconds(event), False))
            driving = False
    return transitions


def ride_bounds(events: list[dict[str, object]]) -> tuple[float, float] | None:
    starts = [_seconds(e) for e in events if e.get("type") == "ride_start"]
    ends = [_seconds(e) for e in events if e.get("type") == "ride_end"]
    if not starts:
        return None
    end = ends[-1] if ends else (_seconds(events[-1]) if events else starts[0])
    return starts[0], end


def driving_at(times: np.ndarray, transitions: list[tuple[float, bool]]) -> np.ndarray:
    """Vectorised key state lookup: True where the drive key was held."""
    if not transitions:
        return np.zeros(len(times), dtype=bool)
    edges = np.array([t for t, _ in transitions])
    states = np.array([s for _, s in transitions], dtype=bool)
    index = np.searchsorted(edges, times, side="right") - 1
    result = np.zeros(len(times), dtype=bool)
    valid = index >= 0
    result[valid] = states[index[valid]]
    return result


def windows(
    stamps: np.ndarray, events: list[dict[str, object]], step_s: float = TRAIN_STEP_S
) -> tuple[np.ndarray, np.ndarray]:
    """Return (end sample indices, labels) for windows that never straddle a key change."""
    bounds = ride_bounds(events)
    if bounds is None or len(stamps) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=object)
    start, end = bounds
    transitions = key_transitions(events)
    width = int(WINDOW_S * SAMPLE_RATE)
    step = int(step_s * SAMPLE_RATE)
    edge_times = np.array([t for t, _ in transitions])
    ends, labels = [], []
    for last in range(width - 1, len(stamps), step):
        t0, t1 = stamps[last - width + 1], stamps[last]
        if t0 < start + SETTLE_S or t1 > end:
            continue
        if not continuous_window(stamps[last - width + 1: last + 1], SAMPLE_RATE):  # same rule as live
            continue
        if len(edge_times) and np.any((edge_times > t0) & (edge_times <= t1)):
            continue
        state = driving_at(np.array([t1]), transitions)[0]
        ends.append(last)
        labels.append(GO if state else STOP)
    return np.array(ends, dtype=int), np.array(labels, dtype=object)


def summarize(events: list[dict[str, object]]) -> dict[str, object]:
    """Stops demanded, late reactions, and median reaction times in ms."""
    transitions = key_transitions(events)
    releases = [t for t, down in transitions if not down]
    presses = [t for t, down in transitions if down]
    collided = {e.get("hazard_id") for e in events if e.get("type") == "collision"}
    stop_rt, go_rt = [], []
    demanded = 0
    for event in events:
        if event.get("type") != "hazard" or event.get("kind") not in STOP_HAZARDS:
            continue
        t0 = _seconds(event)
        if event.get("phase") == "demand_stop":
            demanded += 1
            if driving_at(np.array([t0]), transitions)[0]:
                after = [t for t in releases if t >= t0]
                if after and after[0] - t0 < 5.0:
                    stop_rt.append((after[0] - t0) * 1000)
        elif event.get("phase") == "clear":
            if not driving_at(np.array([t0]), transitions)[0]:
                after = [t for t in presses if t >= t0]
                if after and after[0] - t0 < 5.0:
                    go_rt.append((after[0] - t0) * 1000)
    bounds = ride_bounds(events)
    return {
        "stops": demanded,
        "late": len(collided - {None}),
        "stop_reaction_ms": round(median(stop_rt)) if stop_rt else None,
        "go_reaction_ms": round(median(go_rt)) if go_rt else None,
        "ride_s": round(bounds[1] - bounds[0], 1) if bounds else 0.0,
        "drive_share": _drive_share(transitions, bounds),
    }


def _drive_share(transitions: list[tuple[float, bool]], bounds: tuple[float, float] | None) -> float:
    if not bounds or bounds[1] <= bounds[0]:
        return 0.0
    grid = np.linspace(bounds[0], bounds[1], 2000)
    return round(float(driving_at(grid, transitions).mean()), 3)

"""Telemetry format, flags and fault codes shared by the flight service and transports."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

TRANSMITTER_MARKER = "ASSISTED_FLIGHT_TX_V1"

PHASES = (
    "IDLE",
    "SPINUP",
    "LIFT",
    "CLIMB",
    "FLY",
    "MOVE",
    "LAND",
    "FINAL_DESCENT",
    "EMERGENCY_DESCENT",
    "LANDED",
    "FAULT",
)
ACTIVE_PHASES = frozenset(PHASES[1:9])
TAKEOFF_PHASES = frozenset({"SPINUP", "LIFT", "CLIMB"})
FLYING_PHASES = frozenset({"FLY", "MOVE"})
LANDING_PHASES = frozenset({"LAND", "FINAL_DESCENT", "EMERGENCY_DESCENT"})
GROUND_PHASES = frozenset({"IDLE", "LANDED", "FAULT"})

FAULTS = (
    "NONE",
    "PRECONDITION",
    "BAD_PARAMS",
    "NO_LIFT",
    "TIP_OVER",
    "LASER_LOST",
    "FLOW_LOST",
    "LINK_LOST",
    "OVER_HEIGHT",
    "CLIMB_TIMEOUT",
    "MOVE_TIMEOUT",
    "AIRBORNE_TIMEOUT",
    "EXTERNAL_LOCK",
    "LAND_TIMEOUT",
    "NOT_CALIBRATED",
    "RUNAWAY",
    "UNDER_HEIGHT",
    "WRONG_DIRECTION",
    "TILT_MISMATCH",
)

FAULT_HELP = {
    "PRECONDITION": "Takeoff refused: put the drone flat on the floor with the laser seeing the ground.",
    "BAD_PARAMS": "A height, distance or calibration value was out of range.",
    "NO_LIFT": "Motors reached the lift limit without leaving the ground: charge or replace the battery, check props.",
    "TIP_OVER": "The drone tilted too far; motors were stopped.",
    "LASER_LOST": "The height laser stopped reporting; the drone descended and stopped.",
    "FLOW_LOST": "The floor camera could not see the floor, so the drone could not tell where it was going; it landed level. Check nothing covers or shades the camera and the floor is lit and patterned.",
    "LINK_LOST": "Radio commands stopped; the drone landed by itself.",
    "OVER_HEIGHT": "The drone rose above the allowed height; it landed.",
    "CLIMB_TIMEOUT": "The drone could not reach the target height; it landed.",
    "MOVE_TIMEOUT": "The measured move did not finish in time; the drone is holding.",
    "AIRBORNE_TIMEOUT": "Maximum flight time reached; the drone landed.",
    "EXTERNAL_LOCK": "Motors were cut (emergency cut or tilt protection).",
    "LAND_TIMEOUT": "Touchdown took longer than expected; motors were stopped.",
    "NOT_CALIBRATED": "Optical-flow calibration is missing; only height hold is available.",
    "RUNAWAY": "The drone leaned hard or thought it was racing for too long (its position estimate went wrong); it levelled out and landed.",
    "UNDER_HEIGHT": "The drone sank well below the height you set; it landed.",
    "WRONG_DIRECTION": "The drone did not go the way it was told (or drifted while holding still); it landed level.",
    "TILT_MISMATCH": "The drone tilted differently from what it was told (a pull, a bump or a motor problem); it landed level.",
}

FLAG_ARMED = 0x01
FLAG_LASER_OK = 0x02
FLAG_FLOW_OK = 0x04
FLAG_LINK_OK = 0x08
FLAG_POS_HOLD = 0x10
FLAG_CALIBRATED = 0x20
FLAG_MOVE_DONE = 0x40

AXIS_FORWARD_IS_Y = 0x01
AXIS_FORWARD_NEG = 0x02
AXIS_RIGHT_NEG = 0x04


def _name(table: tuple[str, ...], index: int) -> str:
    return table[index] if 0 <= index < len(table) else f"UNKNOWN_{index}"


@dataclass(frozen=True)
class FlightTelemetry:
    phase: str
    fault: str
    flags: int
    sequence: int
    height_mm: int
    target_mm: int
    vertical_mm_s: int
    pos_x: int
    pos_y: int
    quality: int
    roll_deg: float
    pitch_deg: float
    throttle: int
    hover_throttle: int
    moved_forward_mm: int
    moved_right_mm: int
    ground_mm: int

    @property
    def armed(self) -> bool:
        return bool(self.flags & FLAG_ARMED)

    def flag(self, mask: int) -> bool:
        return bool(self.flags & mask)


@dataclass(frozen=True)
class ConfigTelemetry:
    phase: str
    fault: str
    flags: int
    sequence: int
    height_mm: int
    scale_um: int
    axis_map: int
    firmware_version: int
    uptime_s: int
    laser_errors: int
    raw_flow_x: int
    raw_flow_y: int
    ground_mm: int
    last_command: int


def decode_telemetry(payload: bytes) -> Optional[FlightTelemetry | ConfigTelemetry]:
    """Decode one 32-byte ACK payload; None for anything malformed."""
    if len(payload) != 32 or payload[0] != ord("T"):
        return None
    kind = chr(payload[1])
    phase = _name(PHASES, payload[2])
    fault = _name(FAULTS, payload[3])
    flags, sequence = payload[4], payload[5]
    (height,) = struct.unpack(">H", payload[6:8])
    if kind == "A":
        target, vz = struct.unpack(">Hh", payload[8:12])
        pos_x, pos_y = struct.unpack(">ii", payload[12:20])
        quality = payload[20]
        roll, pitch = struct.unpack(">bb", payload[21:23])
        throttle, hover, moved_f, moved_r = struct.unpack(">HHhh", payload[23:31])
        return FlightTelemetry(
            phase, fault, flags, sequence, height, target, vz, pos_x, pos_y, quality,
            roll / 2.0, pitch / 2.0, throttle, hover, moved_f, moved_r, payload[31] * 2,
        )
    if kind == "B":
        scale, axis_map, version = struct.unpack(">HBB", payload[8:12])
        (uptime,) = struct.unpack(">H", payload[12:14])
        laser_errors = payload[14]
        raw_x, raw_y = struct.unpack(">ii", payload[15:23])
        (ground,) = struct.unpack(">H", payload[23:25])
        return ConfigTelemetry(
            phase, fault, flags, sequence, height, scale, axis_map, version, uptime,
            laser_errors, raw_x, raw_y, ground, payload[25],
        )
    return None


def parse_telemetry_line(line: str) -> Optional[FlightTelemetry | ConfigTelemetry]:
    """Parse a transmitter 'TEL <64 hex>' line."""
    if not line.startswith("TEL "):
        return None
    try:
        payload = bytes.fromhex(line[4:].strip())
    except ValueError:
        return None
    return decode_telemetry(payload)


def body_from_flow(axis_map: int, dx: float, dy: float) -> tuple[float, float]:
    """Flow-axis displacement -> (forward, right) with the firmware axis map."""
    forward_is_y = bool(axis_map & AXIS_FORWARD_IS_Y)
    forward = dy if forward_is_y else dx
    right = dx if forward_is_y else dy
    if axis_map & AXIS_FORWARD_NEG:
        forward = -forward
    if axis_map & AXIS_RIGHT_NEG:
        right = -right
    return forward, right


def calibrate_forward(dx_counts: float, dy_counts: float, distance_mm: float,
                      height_mm: float) -> tuple[int, int]:
    """Scale (um per count at 1 m) and forward-axis bits from a carried leg.

    Optical flow is angular, so millimetres per count grow with height:
    mm_per_count(h) = scale_um / 1000 * h / 1000.
    """
    if distance_mm <= 0 or height_mm <= 0:
        raise ValueError("distance and height must be positive")
    use_y = abs(dy_counts) > abs(dx_counts)
    counts = dy_counts if use_y else dx_counts
    if abs(counts) < 20:
        raise ValueError("optical flow barely moved; carry it again over a textured floor")
    other = dx_counts if use_y else dy_counts
    if abs(other) > 0.5 * abs(counts):
        raise ValueError("the carried path was not straight along one sensor axis")
    mm_per_count = distance_mm / abs(counts)
    scale_um = round(mm_per_count * 1000.0 * 1000.0 / height_mm)
    bits = (AXIS_FORWARD_IS_Y if use_y else 0) | (AXIS_FORWARD_NEG if counts < 0 else 0)
    return scale_um, bits


def calibrate_right_sign(axis_bits: int, dx_counts: float, dy_counts: float) -> int:
    """Right-axis sign bit from a carried sideways (to the right) leg."""
    counts = dx_counts if axis_bits & AXIS_FORWARD_IS_Y else dy_counts
    if abs(counts) < 20:
        raise ValueError("sideways leg barely moved the optical flow")
    return axis_bits | (AXIS_RIGHT_NEG if counts < 0 else 0)

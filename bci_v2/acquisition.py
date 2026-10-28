"""OpenBCI Ganglion connection for the live EEG panel.

BLED112 dongle first, native Bluetooth second. Only real headsets unless
``simulate`` is set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

N_EEG = 4
RING_BUFFER_SAMPLES = 45000
BLED112_HINTS = ("low energy dongle", "bled112")
BLED112_USB_ID = (0x2458, 0x0001)


class AcquisitionError(RuntimeError):
    """The headset cannot be reached or stopped streaming."""


@dataclass(frozen=True)
class BoardHandle:
    board: object
    board_id: int
    transport: str
    sample_rate: int
    eeg_rows: tuple[int, ...]
    accel_rows: tuple[int, ...]
    timestamp_row: int


def find_bled112_port() -> Optional[str]:
    """The BLED112 dongle's serial port, or None when there is not exactly one."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    matches = [port.device for port in list_ports.comports()
               if (port.vid, port.pid) == BLED112_USB_ID
               or any(hint in f"{port.device} {port.description}".lower() for hint in BLED112_HINTS)]
    return matches[0] if len(matches) == 1 else None


def _open_board(board_id: int, serial_port: str, transport: str, *, native: bool, ganglion: bool = True) -> BoardHandle:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams

    params = BrainFlowInputParams()
    if ganglion:
        params.other_info = "fw:auto"
    if serial_port and not native:
        params.serial_port = serial_port
    board = BoardShim(board_id, params)
    try:
        board.prepare_session()
        if native:
            time.sleep(1.5)  # the OS reports the device ready before its services are
        if ganglion:
            firmware = str(board.config_board("get_firmware_version")).strip()
            if firmware not in ("2", "3"):
                raise AcquisitionError(f"Ganglion firmware autodetection returned {firmware!r}")
            board.config_board("n")  # accelerometer on
        attempts = 3 if native else 1
        for attempt in range(attempts):
            try:
                board.start_stream(RING_BUFFER_SAMPLES)
                break
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(1.0)
    except Exception:
        try:
            board.release_session()
        except Exception:
            pass
        raise
    return BoardHandle(
        board=board,
        board_id=board_id,
        transport=transport,
        sample_rate=BoardShim.get_sampling_rate(board_id),
        eeg_rows=tuple(BoardShim.get_eeg_channels(board_id)[:N_EEG]),
        accel_rows=tuple(BoardShim.get_accel_channels(board_id)[:3]),
        timestamp_row=BoardShim.get_timestamp_channel(board_id),
    )


def connect(simulate: bool = False) -> BoardHandle:
    from brainflow.board_shim import BoardIds, BoardShim

    BoardShim.disable_board_logger()
    if simulate:
        return _open_board(BoardIds.SYNTHETIC_BOARD.value, "", "synthetic", native=False, ganglion=False)
    failures = []
    port = find_bled112_port()
    if port:
        try:
            return _open_board(BoardIds.GANGLION_BOARD.value, port, f"BLED112 dongle ({port})", native=False)
        except Exception as exc:
            failures.append(f"BLED112 dongle at {port}: {type(exc).__name__}: {exc}")
    else:
        failures.append("No BLED112 dongle found on any serial port.")
    try:
        return _open_board(BoardIds.GANGLION_NATIVE_BOARD.value, "", "native BLE", native=True)
    except Exception as exc:
        failures.append(f"Native BLE: {type(exc).__name__}: {exc}")
    raise AcquisitionError("Could not reach the Ganglion. " + " ".join(failures)
                           + " Check: power on (blue light), dongle plugged in, OpenBCI GUI closed, battery charged.")


def release(handle: Optional[BoardHandle]) -> None:
    """Stop and release a board session, never raising."""
    if handle is None:
        return
    for method in ("stop_stream", "release_session"):
        try:
            getattr(handle.board, method)()
        except Exception:
            pass


def read_chunk(handle: BoardHandle) -> np.ndarray:
    """Everything buffered on the board, shape (rows, n)."""
    try:
        return handle.board.get_board_data()
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise AcquisitionError(f"Lost the Ganglion mid-stream: {exc}") from exc


def to_eeg_matrix(handle: BoardHandle, chunk: np.ndarray) -> np.ndarray:
    """(n_samples, 4) µV from a raw board chunk."""
    if chunk.size == 0:
        return np.zeros((0, N_EEG))
    return chunk[list(handle.eeg_rows), :].T

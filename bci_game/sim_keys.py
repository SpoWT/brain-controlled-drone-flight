"""Simulator keys: hold W (or the up arrow) while your trained model decodes GO, let go on STOP.

In an FPV simulator where that key is throttle, thinking GO makes the drone climb and STOP lets it fall: a safe way
to see whether the model works before flying the real drone. The key goes to whichever window has focus.

    python -m bci_game keys              # newest bundle, Ganglion, holds W
    python -m bci_game keys --key up     # holds the up arrow
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable

from .config import GO, LIVE_STEP_S

KEYS = ("w", "up")
SIMULATED_SWITCH_S = 3.0  # --simulate: the simulated EEG switches between GO and STOP this often


def keyboard_for(name: str):
    """(keyboard, key) for pynput: name is "w" or "up"."""
    from pynput.keyboard import Controller, Key

    return Controller(), Key.up if name == "up" else name


class KeyHolder:
    """Holds one key down while asked to; never presses it twice."""

    def __init__(self, keyboard, key) -> None:
        self._keyboard, self._key = keyboard, key
        self.held = False

    def hold(self, on: bool) -> None:
        if on == self.held:
            return
        if not on:
            self.release()
            return
        self.held = True  # before the key-down: an interrupt right after it must still release
        self._keyboard.press(self._key)

    def release(self) -> None:
        """Key up, whatever the state: a spare key-up does no harm, a missed one leaves the throttle on."""
        self.held = False
        self._keyboard.release(self._key)


def hold_while_go(decoder, holder: KeyHolder, *, stop: threading.Event, step_s: float = LIVE_STEP_S,
                  say: Callable[[str], None] = print) -> None:
    """Hold the key while the decoder says GO. Anything else, and the end (Ctrl+C included), lets it go."""
    shown = None
    try:
        while not stop.is_set():
            status = decoder.status()
            holder.hold(status["state"] == GO)
            if (status["state"], status["reason"]) != shown:
                shown = (status["state"], status["reason"])
                say(f"{time.strftime('%H:%M:%S')} {status['state']:4} p(GO) {status['p_go']:.2f}  {status['reason']}")
            stop.wait(step_s)
    finally:
        holder.release()


def release_on_console_close(holder: KeyHolder, stop: threading.Event, *, kernel=None) -> Callable[[], None]:
    """Windows: closing the console window, Ctrl+Break, log-off and shut-down end the process without running
    Python's cleanup, so the console handler releases the key itself. Ctrl+C is left to Python.
    Returns the function that removes the handler."""
    def on_console_event(event: int) -> bool:
        if event != 0:  # CTRL_C_EVENT is 0: KeyboardInterrupt releases the key
            stop.set()
            try:
                holder.release()
            except Exception as exc:
                print(f"Could not release the key: {exc}", file=sys.stderr)
        return False  # let Windows go on with its usual action

    callback = on_console_event
    if kernel is None:
        if os.name != "nt":
            return lambda: None
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.windll.kernel32
        callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)(on_console_event)
    kernel.SetConsoleCtrlHandler(callback, True)
    return lambda: kernel.SetConsoleCtrlHandler(callback, False)  # also keeps the callback alive


def alternate_driving(source, stop: threading.Event, every_s: float = SIMULATED_SWITCH_S) -> None:
    """--simulate: switch the simulated EEG between GO and STOP, so the key visibly goes down and up."""
    driving = False
    while not stop.wait(every_s):
        driving = not driving
        source.set_driving(driving)

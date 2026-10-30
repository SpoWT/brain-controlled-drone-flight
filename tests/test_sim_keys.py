"""Simulator keys: W (or the up arrow) is held while the model decodes GO and let go on STOP."""
import threading
import unittest

from bci_game.config import GO, STOP
from bci_game.sim_keys import KeyHolder, alternate_driving, hold_while_go, release_on_console_close


class FakeKeyboard:
    def __init__(self):
        self.log = []

    def press(self, key):
        self.log.append(("press", key))

    def release(self, key):
        self.log.append(("release", key))


class FakeDecoder:
    """Hands out one status per check and sets `done` with the last one."""

    def __init__(self, states, done):
        self._states, self._done = list(states), done

    def status(self):
        state, reason = self._states.pop(0)
        if not self._states:
            self._states = [(state, reason)]
            self._done.set()
        return {"state": state, "p_go": 0.9 if state == GO else 0.1, "reason": reason}


class KeyHolderTests(unittest.TestCase):
    def test_presses_once_and_releases_once(self):
        keyboard = FakeKeyboard()
        holder = KeyHolder(keyboard, "w")
        holder.hold(True)
        holder.hold(True)
        holder.hold(False)
        holder.hold(False)
        self.assertEqual(keyboard.log, [("press", "w"), ("release", "w")])
        self.assertFalse(holder.held)


class HoldWhileGoTests(unittest.TestCase):
    def test_go_holds_the_key_and_stop_lets_go(self):
        keyboard, done = FakeKeyboard(), threading.Event()
        decoder = FakeDecoder([(GO, "decoding"), (GO, "decoding"), (STOP, "decoding"), (GO, "decoding"),
                               (STOP, "signal stalled")], done)
        said = []
        hold_while_go(decoder, KeyHolder(keyboard, "w"), stop=done, step_s=0.0, say=said.append)
        self.assertEqual(keyboard.log, [("press", "w"), ("release", "w"), ("press", "w"), ("release", "w"),
                                        ("release", "w")], "and a spare key-up at the end")
        self.assertTrue(any("signal stalled" in line for line in said), "says why it let go")

    def test_the_key_is_let_go_when_it_ends_mid_go(self):
        keyboard, stop = FakeKeyboard(), threading.Event()

        class Interrupted:
            calls = 0

            def status(self):
                Interrupted.calls += 1
                if Interrupted.calls == 3:
                    raise KeyboardInterrupt  # Ctrl+C while the key is held
                return {"state": GO, "p_go": 0.9, "reason": "decoding"}

        with self.assertRaises(KeyboardInterrupt):
            hold_while_go(Interrupted(), KeyHolder(keyboard, "w"), stop=stop, step_s=0.0, say=lambda text: None)
        self.assertEqual(keyboard.log[-1], ("release", "w"), "never left held down")

    def test_an_interrupt_right_after_the_key_down_still_releases(self):
        class Keyboard(FakeKeyboard):
            def press(self, key):
                super().press(key)
                raise KeyboardInterrupt  # Ctrl+C lands after the key went down

        keyboard, stop = Keyboard(), threading.Event()
        decoder = FakeDecoder([(GO, "decoding")], threading.Event())
        with self.assertRaises(KeyboardInterrupt):
            hold_while_go(decoder, KeyHolder(keyboard, "w"), stop=stop, step_s=0.0, say=lambda text: None)
        self.assertEqual(keyboard.log, [("press", "w"), ("release", "w")])


class ConsoleCloseTests(unittest.TestCase):
    """Windows ends the process on a console close without Python's cleanup: the handler releases the key."""

    def test_closing_the_window_releases_and_ctrl_c_is_left_to_python(self):
        class Kernel:
            def __init__(self):
                self.calls = []

            def SetConsoleCtrlHandler(self, callback, add):
                self.calls.append((callback, add))

        kernel, keyboard, stop = Kernel(), FakeKeyboard(), threading.Event()
        holder = KeyHolder(keyboard, "w")
        holder.hold(True)
        remove = release_on_console_close(holder, stop, kernel=kernel)
        handler = kernel.calls[0][0]
        self.assertFalse(handler(0), "Ctrl+C: Python's KeyboardInterrupt does it")
        self.assertEqual(keyboard.log, [("press", "w")])
        self.assertFalse(handler(2), "close: released, then Windows carries on")
        self.assertEqual(keyboard.log[-1], ("release", "w"))
        self.assertTrue(stop.is_set())
        remove()
        self.assertEqual(kernel.calls[-1], (handler, False))


class SimulatedDrivingTests(unittest.TestCase):
    def test_switches_between_go_and_stop_until_stopped(self):
        class Source:
            def __init__(self):
                self.driving = []

            def set_driving(self, driving):
                self.driving.append(driving)
                if len(self.driving) == 3:
                    stop.set()

        source, stop = Source(), threading.Event()
        alternate_driving(source, stop, every_s=0.001)
        self.assertEqual(source.driving, [True, False, True])


if __name__ == "__main__":
    unittest.main()

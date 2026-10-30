"""The radio link reopens the serial port by itself after a USB dropout."""
import time
import unittest

from drone_flight.flight.ardupilot_link import ArduPilotLink


class BlippingRadio:
    def __init__(self, dead_reads):
        self.dead_reads, self.resets, self.portdead = dead_reads, 0, False
        self.mav = self

    def recv_match(self, **kw):
        if self.dead_reads > 0:
            self.dead_reads -= 1
            self.portdead = True
            raise OSError("ClearCommError failed (device removed)")
        time.sleep(0.01)
        return None

    def reset(self):
        self.resets += 1
        if self.dead_reads > 0:
            raise OSError("could not open port")  # not plugged back in yet
        self.portdead = False
        return True

    def close(self):
        pass


class ReconnectTests(unittest.TestCase):
    def test_reopens_the_port_after_a_blip(self):
        radio = BlippingRadio(dead_reads=3)
        link = ArduPilotLink("COM4", connection_factory=lambda dev, **kw: radio, start_reader=True)
        try:
            deadline = time.time() + 5
            while time.time() < deadline and (radio.portdead or radio.resets == 0):
                time.sleep(0.05)
        finally:
            link.close()
        self.assertGreaterEqual(radio.resets, 1, "tried to reopen the radio port")
        self.assertFalse(radio.portdead, "port is open again")


if __name__ == "__main__":
    unittest.main()

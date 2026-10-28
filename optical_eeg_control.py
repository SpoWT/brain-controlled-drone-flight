"""EEG STOP/GO control, only when switched on and always within the normal flight controls."""
from __future__ import annotations

import threading


class EEGFlightControl:
    PERIOD_SECONDS = .05

    def __init__(self, flight, eeg, *, start_worker=True):
        self.flight = flight
        self.eeg = eeg
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._enabled = False
        self._token = ''
        self._model_id = None
        self._threshold = None
        self._go_latched = False
        self._flew = False
        self._message = 'Arm and reach hover, then enable brain control'
        self._thread = None
        if start_worker:
            self._thread = threading.Thread(target=self._worker, name='drone-eeg-control', daemon=True)
            self._thread.start()

    def status(self):
        with self._lock:
            return {'enabled': self._enabled, 'message': self._message}

    def reset_session(self):
        with self._lock:
            self._enabled = False
            self._token = ''
            self._go_latched = False
            self._message = 'Arm and reach hover, then enable brain control'

    def disable(self, message='Brain control off', *, hover=True):
        with self._lock:
            was_enabled = self._enabled
            self._enabled = False
            self._go_latched = False
            self._flew = False
            self._message = message
            if was_enabled and hover:
                self.flight.apply_eeg_decision(self._token, 'STOP')
            return self.status()

    @staticmethod
    def _decision(eeg):
        prediction = eeg.get('prediction', {})
        if (eeg.get('enabled') and eeg.get('state') == 'streaming'
                and eeg.get('selected_model', {}).get('stop_go')
                and prediction.get('valid') and prediction.get('decision') in {'STOP', 'GO'}
                and prediction.get('quality') != 'reject'
                and not eeg.get('model_loading')):
            return prediction['decision']
        return None

    @staticmethod
    def _streaming(eeg):
        """Live EEG with a STOP/GO model; an unclear window hovers, losing either turns control off."""
        return bool(eeg.get('enabled') and eeg.get('state') == 'streaming'
                    and eeg.get('selected_model', {}).get('stop_go') and not eeg.get('model_loading'))

    def _held_key(self):
        # ArduPilot service: GO is a held W key. Original drone: one capped GO per decision.
        return bool(getattr(self.flight, 'EEG_GO_IS_HELD_KEY', False))

    def _prearm(self):
        # may be switched on before take-off; it only steers at hover
        return bool(getattr(self.flight, 'ALLOWS_EEG_PREARM', False))

    def set_enabled(self, token, enabled):
        self.flight.authorize(token)
        if not isinstance(enabled, bool):
            raise ValueError('Brain control enabled must be true or false')
        with self._lock:
            if not enabled:
                return self.disable()
            flight, eeg = self.flight.status(), self.eeg.status()
            waiting = self._prearm() and flight['mode'] in {'DISARMED', 'TAKING_OFF'}
            if not flight.get('control_ready', flight['flight_ready']) or (
                    flight['mode'] not in {'HOVERING', 'FORWARD'} and not waiting):
                raise ValueError('Arm and reach hover before enabling brain control')
            if not waiting and not self.flight.eeg_control_allowed(token):
                raise ValueError('Keep this page connected before enabling brain control')
            if not flight.get('connected') or self._decision(eeg) is None:
                raise ValueError('Connect fresh EEG and choose a STOP / GO model first')
            self._enabled = True
            self._token = token
            self._model_id = eeg['model_id']
            self._threshold = eeg['stop_threshold']
            self._go_latched = flight['mode'] == 'FORWARD'
            self._message = ('Brain control ready · steers once the drone hovers' if waiting
                             else 'Brain control on · STOP overrides GO')
            return self.status()

    def manual_go_allowed(self):
        with self._lock:
            if not self._enabled:
                return True
            self.tick()
            if not self._enabled:
                return False  # stale EEG
            if self._decision(self.eeg.status()) != 'GO':
                self._message = 'EEG STOP overrides computer GO'
                self.flight.apply_eeg_decision(self._token, 'STOP')
                return False
            return True

    def tick(self):
        with self._lock:
            if not self._enabled:
                return
            flight, eeg = self.flight.status(), self.eeg.status()
            waiting = self._prearm() and flight['mode'] in {'DISARMED', 'TAKING_OFF'} and not self._flew
            if not flight.get('control_ready', flight['flight_ready']) or (
                    flight['mode'] not in {'HOVERING', 'FORWARD'} and not waiting):
                self.disable('Brain control off · flight is no longer in hover or forward', hover=False)
                return
            usable = self._streaming(eeg) if self._held_key() else self._decision(eeg) is not None
            if (not flight.get('connected') or not usable
                    or eeg['model_id'] != self._model_id or eeg['stop_threshold'] != self._threshold):
                self.disable('Brain control off · EEG changed or lost; enable again when ready')
                return
            if waiting:
                self._message = 'Brain control ready · steers once the drone hovers'
                return
            self._flew = True  # switches off after landing
            decision = self._decision(eeg)
            if self._held_key():
                # GO holds W; STOP or an unclear reading lets go
                applied = self.flight.apply_eeg_decision(self._token, 'GO' if decision == 'GO' else 'STOP')
                self._message = {'GO': 'Brain control on · GO (holding W)',
                                 'STOP': 'Brain control on · STOP / HOVER'}.get(
                    decision, 'Brain control on · unclear reading: hover')
            elif decision == 'STOP':
                self._go_latched = False
                applied = self.flight.apply_eeg_decision(self._token, 'STOP')
                self._message = 'Brain control on · STOP / HOVER'
            elif not self._go_latched:
                applied = self.flight.apply_eeg_decision(self._token, 'GO')
                self._go_latched = applied
                self._message = 'Brain control on · GO FORWARD'
            else:
                # a held GO cannot extend the forward cap
                applied = self.flight.eeg_control_allowed(self._token)
                self._message = ('Brain control on · GO FORWARD' if flight['mode'] == 'FORWARD'
                                 else 'GO finished · think STOP before GO again')
            if not applied:
                self.disable('Brain control off · browser or flight connection lost', hover=False)

    def _worker(self):
        while not self._closed.is_set():
            try:
                self.tick()
            except Exception:
                self.disable('Brain control off · command unavailable')
            self._closed.wait(self.PERIOD_SECONDS)

    def close(self):
        self._closed.set()
        self.disable()
        if self._thread is not None:
            self._thread.join(timeout=.5)

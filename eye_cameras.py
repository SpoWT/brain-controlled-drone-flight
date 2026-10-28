"""Goggle eye cameras: live view and pupil measurement.

Each camera runs in its own thread while someone is watching. Frames go through goggle_pupil.detect_pupil
and are served as /eyes/<n>.mjpg, with the numbers at /api/eyes.

The two cameras are the same model and can report the same serial number, so they are told apart by their
DirectShow device path and ordered by the USB port they sit on: EYE 1 / EYE 2 and their rotation settings
follow the physical camera, not the plug order. A device path is never open in two slots, and the slots are
checked for showing the same picture as a last guard.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

EYE_CAMERA_NAME = "Innomaker-U20CAM-1080p-S1"
EYE_SLOTS = 2
CAPTURE_SIZE = (640, 480)
CAPTURE_FOURCC = "MJPG"       # two uncompressed streams do not fit through the goggles' USB 2 hub
STREAM_FPS = 10
IDLE_STOP_S = 60.0            # release the camera when nobody watches
NO_PICTURE_REOPEN_S = 3.0     # no picture this long: reopen
NO_PICTURE_GIVE_UP = 5        # reopens before asking for a re-plug
MISSING_RETRY_S = 2.0         # camera not plugged in: look again this often
REFRESH_EVERY_S = 3.0         # a slot without a camera: look for cameras this often
PARTIAL_LOOKS = 3             # looks that skip placing cameras while one camera's USB port cannot be read
EYE_AIM = (0.5, 0.5)
FIRST_FRAME_S = 12.0          # max wait for a camera to stream before opening the next
FIRST_FRAMES = 3
SAME_DIFF = 4.0               # mean grey difference below this = same camera
SAME_MIN_DETAIL = 10.0        # skip dark, empty views
SAME_CHECKS = 4
MAX_REOPENS = 4
NOT_PLUGGED = "not plugged in"
MOVED = "re-plugged: finding it again"
LIST_CHANGED = "camera list changed: opening again"
NO_PICTURE_REPLUG = "No picture: re-plug the goggles' USB cable"
GAVE_UP = "Same picture as the other eye: re-plug the goggles' USB cable"
THUMB_FRESH_S = 1.5
ROTATIONS = (0, 90, 180, 270)  # clockwise
CM_DRP_LOCATION_PATHS = 0x24
DEFAULT_CONFIG = Path(__file__).resolve().parent / "output" / "eye_cameras.json"
# Opening two of these cameras at the same moment can bind both to one device: open one at a time.
_OPEN_LOCK = threading.Lock()
_KNOWN_LOCATIONS: dict[str, str] = {}  # device path -> last USB location read for it


@dataclass(frozen=True)
class EyeDevice:
    path: str           # DirectShow device path, unique per plugged-in camera
    device_id: str      # short form of the path, e.g. 183AF011 (changes with plug order)
    location: str = ""  # USB location path of the camera, e.g. PCIROOT(0)#...#USB(3)#USB(1)


class CameraMissing(RuntimeError):
    pass


def thumbnail(frame):
    """32x24 grey copy of a frame, used to spot two views of one camera."""
    import cv2
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(grey, (32, 24), interpolation=cv2.INTER_AREA).astype("float32")


def same_picture(a, b) -> bool:
    import numpy as np
    if a is None or b is None or a.shape != b.shape:
        return False
    if min(float(a.std()), float(b.std())) < SAME_MIN_DETAIL:
        return False
    return float(np.abs(a - b).mean()) < SAME_DIFF


def rotate_frame(frame, degrees: int):
    if degrees == 0:
        return frame
    import cv2
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[degrees]
    return cv2.rotate(frame, code)


def find_eye_cameras(device_names: Optional[list[str]] = None) -> list[int]:
    """DirectShow indices of the eye cameras by name only, or []."""
    if device_names is None:
        try:
            from pygrabber.dshow_graph import FilterGraph
            device_names = FilterGraph().get_input_devices()
        except Exception:
            return []
    return [index for index, name in enumerate(device_names) if name == EYE_CAMERA_NAME]


def video_devices() -> Optional[list[tuple[str, str]]]:
    """(name, device path) of each DirectShow video input in index order, or None when unavailable."""
    try:
        import comtypes
        try:
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)  # COM is needed in every thread that asks
        except OSError:
            pass  # already set up in this thread
        from comtypes import GUID, client
        from comtypes.persist import IPropertyBag
        from pygrabber.dshow_core import ICreateDevEnum
        from pygrabber.dshow_ids import DeviceCategories, clsids
        devices = client.CreateObject(clsids.CLSID_SystemDeviceEnum, interface=ICreateDevEnum).CreateClassEnumerator(
            GUID(DeviceCategories.VideoInputDevice), dwFlags=0)
        found = []
        while True:
            moniker, count = devices.Next(1)
            if count == 0:
                return found
            bag = moniker.BindToStorage(0, 0, IPropertyBag._iid_).QueryInterface(IPropertyBag)
            found.append((_read(bag, "FriendlyName"), _read(bag, "DevicePath")))
    except Exception:
        return None


def _read(bag, key: str) -> str:
    """One property of a video input, or "" (software cameras such as OBS have no DevicePath)."""
    try:
        return bag.Read(key, pErrorLog=None) or ""
    except Exception:
        return ""


def short_id(device_path: str) -> str:
    """Instance part of a device path: '\\\\?\\usb#vid_..&mi_00#a&183af011&0&0000#{...}' -> '183AF011'."""
    parts = device_path.split("#")
    if len(parts) < 3:
        return ""
    tokens = parts[2].split("&")
    return (tokens[1] if len(tokens) > 1 else tokens[0]).upper()


def instance_id(device_path: str) -> str:
    """'\\\\?\\usb#vid_0c45&pid_6366&mi_00#a&183af011&0&0000#{...}' -> 'USB\\VID_0C45&PID_6366&MI_00\\A&183AF011&0&0000'."""
    parts = device_path.split("#")
    if len(parts) < 3:
        return ""
    return "\\".join((parts[0].rsplit("\\", 1)[-1], parts[1], parts[2])).upper()


def usb_location(device_path: str) -> str:
    """USB location path of the camera behind a DirectShow device path, or "" when unknown."""
    if os.name != "nt" or not instance_id(device_path):
        return ""
    try:
        import ctypes
        from ctypes import wintypes
        cfg = ctypes.WinDLL("cfgmgr32")
        node, parent = wintypes.DWORD(), wintypes.DWORD()
        if cfg.CM_Locate_DevNodeW(ctypes.byref(node), instance_id(device_path), 0) != 0:
            return ""
        if "&MI_" in instance_id(device_path):  # one interface of a composite device: its parent is the camera
            if cfg.CM_Get_Parent(ctypes.byref(parent), node, 0) != 0:
                return ""
            node = parent
        kind, size = wintypes.ULONG(), wintypes.ULONG(0)
        cfg.CM_Get_DevNode_Registry_PropertyW(node, CM_DRP_LOCATION_PATHS, ctypes.byref(kind), None,
                                              ctypes.byref(size), 0)
        if not size.value:
            return ""
        buffer = ctypes.create_unicode_buffer(size.value // 2 + 1)
        if cfg.CM_Get_DevNode_Registry_PropertyW(node, CM_DRP_LOCATION_PATHS, ctypes.byref(kind), buffer,
                                                 ctypes.byref(size), 0) != 0:
            return ""
        return buffer.value  # the first path of the list
    except Exception:
        return ""


def usb_port(location: str) -> tuple[str, int]:
    """('...#USB(3)', 1) for '...#USB(3)#USB(1)': the hub and the port the camera sits on."""
    match = re.search(r"^(.*)#USB\((\d+)\)$", location)
    return (match.group(1), int(match.group(2))) if match else (location, 0)


def find_eye_devices(devices: Optional[list[tuple[str, str]]] = None, locate: Callable[[str], str] = usb_location,
                     known: Optional[dict[str, str]] = None) -> Optional[list[EyeDevice]]:
    """Eye cameras sorted by USB port, so EYE 1 / EYE 2 follow the physical cameras, not the plug order.
    None when the cameras cannot be listed right now. A failed location lookup reuses the last one read."""
    devices = video_devices() if devices is None else devices
    if devices is None:
        return None
    known = _KNOWN_LOCATIONS if known is None else known
    paths = [path for name, path in devices if name == EYE_CAMERA_NAME]
    fresh = {path: locate(path) for path in paths}
    found = []
    for path in paths:
        location = fresh[path]
        if location:
            known[path.lower()] = location
        else:  # lookup failed: reuse the last one, unless another camera is on that port now (paths swapped)
            cached = known.get(path.lower(), "")
            location = cached if cached not in fresh.values() else ""
        found.append(EyeDevice(path, short_id(path), location))
    return sorted(found, key=lambda device: (not device.location, usb_port(device.location), device.device_id))


def settings_key(device: EyeDevice, devices: list[EyeDevice]) -> str:
    """Rotation settings follow the camera's hub port inside the goggles (or its whole USB path)."""
    if not device.location:
        return device.device_id
    hubs = {usb_port(other.location)[0] for other in devices if other.location}
    hub, port = usb_port(device.location)
    return f"usb port {port}" if len(hubs) == 1 and port else device.location


def assign_slots(devices: list[EyeDevice], current: list[Optional[EyeDevice]],
                 saved_slot: Callable[[EyeDevice], Optional[int]]) -> list[Optional[EyeDevice]]:
    """The camera for each slot: a camera keeps the slot it is in, then gets the slot it had last time (by its
    port), then fills a free slot in port order. So EYE 1 / EYE 2 never swap while one camera is missing."""
    chosen: list[Optional[EyeDevice]] = [None] * len(current)
    left = list(devices)

    def place(slot: int, device: EyeDevice) -> None:
        chosen[slot] = device
        left.remove(device)

    def same_camera(device: EyeDevice, held: EyeDevice) -> bool:
        """Same path and same USB port: a quick re-plug can hand a camera's path to the other camera."""
        unknown = not device.location or not held.location
        return device.path.lower() == held.path.lower() and (unknown or device.location == held.location)

    for slot, held in enumerate(current):
        match = next((d for d in left if held is not None and same_camera(d, held)), None)
        if match is not None:
            place(slot, match)
    for device in list(left):
        slot = saved_slot(device)
        if isinstance(slot, int) and 0 <= slot < len(chosen) and chosen[slot] is None:
            place(slot, device)
    for device in list(left):
        free = next((slot for slot, d in enumerate(chosen) if d is None), None)
        if free is None:
            break
        place(free, device)
    return chosen


def index_of(path: str, devices: Optional[list[tuple[str, str]]] = None) -> Optional[int]:
    """DirectShow index of a device path right now, or None if it is not plugged in (or cannot be listed)."""
    devices = video_devices() if devices is None else devices
    if devices is None or not path:
        return None
    wanted = path.lower()
    return next((index for index, (_name, other) in enumerate(devices) if other.lower() == wanted), None)


def fourcc_text(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> 8 * k) & 0xFF) for k in range(4))


def measure(result, width: int) -> dict:
    """Numbers for the page from a goggle_pupil.PupilResult."""
    if not result.locked:
        return {"detected": False, "confidence": result.confidence, "brightness": result.brightness}
    if result.contour is not None:
        import cv2
        diameter = 2.0 * math.sqrt(max(cv2.contourArea(result.contour), 0.0) / math.pi)
    else:
        diameter = 2.0 * result.radius
    return {"detected": True, "x": result.x, "y": result.y, "diameter_px": round(diameter, 1),
            "diameter_frac": round(diameter / width, 4), "confidence": result.confidence,
            "brightness": result.brightness, "source": "MediaPipe iris" if result.source == "model" else "classic"}


class DeviceClaims:
    """Which slot has which camera open: one camera is never open in two slots.

    OpenCV keeps one DirectShow table for the whole program, keyed by index, so the index is claimed as well
    as the device path: after a re-plug the other slot can still hold an index that now means this camera.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paths: dict[str, int] = {}
        self._indexes: dict[int, int] = {}
        self._cameras: dict[int, "EyeCamera"] = {}

    def register(self, camera: "EyeCamera") -> None:
        self._cameras[camera.slot] = camera

    def claim(self, path: str, index: int, slot: int) -> str:
        """"" when this slot may open the camera, else why not."""
        with self._lock:
            if self._paths.get(path.lower(), slot) != slot:
                return "already open in the other slot"
            holder = self._indexes.get(index, slot)
            if holder == slot:
                self._paths[path.lower()], self._indexes[index] = slot, slot
                return ""
            other = self._cameras.get(holder)
        if other is not None:  # it holds an index that now means this camera: it must find its own again
            other.request_reopen(LIST_CHANGED)
        return LIST_CHANGED

    def release(self, slot: int) -> None:
        with self._lock:
            self._paths = {path: s for path, s in self._paths.items() if s != slot}
            self._indexes = {index: s for index, s in self._indexes.items() if s != slot}


class EyeCamera:
    def __init__(self, slot: int, label: str, *, device: Optional[EyeDevice] = None,
                 claims: Optional[DeviceClaims] = None, opener: Optional[Callable] = None,
                 detector: Optional[Callable] = None, resolver: Callable[[str], Optional[int]] = index_of,
                 locate: Callable[[str], str] = usb_location, clock: Callable[[], float] = time.monotonic) -> None:
        self.slot, self.label, self.device = slot, label, device
        self._locate = locate
        self.index: Optional[int] = None
        self.rotation = 0
        self.settings_key = device.device_id if device else ""
        self._claims = claims if claims is not None else DeviceClaims()
        self._claims.register(self)
        self._opener, self._detector, self._resolve, self._clock = opener, detector, resolver, clock
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._numbers: dict = {"detected": False}
        self._fps = 0.0
        self._error = ""
        self._format = ""
        self._watched_at = -1e9
        self._thread: Optional[threading.Thread] = None
        self._thumb = None
        self._thumb_at = -1e9
        self._reopen = threading.Event()

    @property
    def device_id(self) -> str:
        device = self.device
        return device.device_id if device else ""

    def touch(self) -> None:
        """Mark the camera as watched and start it if needed."""
        with self._lock:
            self._watched_at = self._clock()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name=f"eye-{self.slot}", daemon=True)
                self._thread.start()

    def thumb(self):
        """Latest thumbnail, or None if stale."""
        with self._lock:
            return self._thumb if self._clock() - self._thumb_at <= THUMB_FRESH_S else None

    def missing(self) -> bool:
        """No camera, or it needs finding again (EyeCameras then looks for the cameras)."""
        with self._lock:
            return self.device is None or self._error in (NOT_PLUGGED, MOVED)

    def use(self, device: Optional[EyeDevice]) -> None:
        """Show this camera in the slot from now on."""
        current = self.device
        if device is not None and current is not None and device.path.lower() == current.path.lower() \
                and (not device.location or not current.location or device.location == current.location):
            self.device = device  # same camera (its details may have been read again)
            return
        if device is None and current is None:
            return
        self.device = device
        self.request_reopen("switching camera")

    def request_reopen(self, reason: str) -> None:
        with self._lock:
            self._error, self._thumb = reason, None
        self._reopen.set()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def status(self) -> dict:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            device = self.device
            return {"label": self.label, "camera": self.index, "device": device.device_id if device else "",
                    "usb": usb_port(device.location)[1] if device else 0, "format": self._format,
                    "running": running, "fps": round(self._fps, 1), "rotation": self.rotation,
                    "error": self._error, **self._numbers}

    def _open(self, index: int):
        if self._opener is not None:
            return self._opener(index)
        import cv2
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        # size first, then MJPG: the other order leaves these cameras on uncompressed YUY2
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAPTURE_FOURCC))
        if fourcc_text(capture.get(cv2.CAP_PROP_FOURCC)) != CAPTURE_FOURCC:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAPTURE_FOURCC))
        return capture

    def _detect(self):
        if self._detector is not None:
            return self._detector
        from goggle_pupil import IrisModel, annotate, detect_pupil
        iris = IrisModel.load()  # None: classic detection only

        def run(frame, previous):
            result = detect_pupil(frame, EYE_AIM, previous, iris)
            return result, annotate(frame, result, self.label, True)
        return run

    def _start_capture(self):
        """Open this slot's camera (never one open in the other slot) and wait for its first frames."""
        with _OPEN_LOCK:
            device = self.device
            index = self._resolve(device.path) if device is not None else None
            if index is None:
                raise CameraMissing(NOT_PLUGGED)
            location = self._locate(device.path) if device.location else ""
            if location and location != device.location:  # this path now belongs to the camera on another port
                raise CameraMissing(MOVED)
            refused = self._claims.claim(device.path, index, self.slot)
            if refused:
                raise CameraMissing(refused)
            try:
                capture = self._open(index)
                if self._resolve(device.path) != index:  # the list changed while opening: maybe the other camera
                    capture.release()
                    raise CameraMissing(LIST_CHANGED)
                deadline, frames = self._clock() + FIRST_FRAME_S, 0
                while self._clock() < deadline and frames < FIRST_FRAMES:
                    if capture.read()[0]:
                        frames += 1
                    else:
                        time.sleep(0.05)
            except BaseException:
                self._claims.release(self.slot)
                raise
        get = getattr(capture, "get", None)
        with self._lock:
            self.index = index
            self._format = fourcc_text(get(6)) if get is not None else ""  # 6 = CAP_PROP_FOURCC
        return capture, device

    def _release(self, capture, device) -> None:
        if capture is not None:
            capture.release()
        if device is not None:
            self._claims.release(self.slot)

    def _watched(self) -> bool:
        return self._clock() - self._watched_at < IDLE_STOP_S

    def _run(self) -> None:
        capture = device = None
        try:
            detect = self._detect()
            previous, last, last_ok, reopens = None, self._clock(), self._clock(), 0
            while self._watched():
                stalled = capture is not None and self._clock() - last_ok > NO_PICTURE_REOPEN_S
                if capture is None or self._reopen.is_set() or stalled:
                    reopens = reopens + 1 if stalled else reopens
                    self._release(capture, device)
                    capture = device = None
                    self._reopen.clear()
                    try:
                        capture, device = self._start_capture()
                    except CameraMissing as missing:
                        with self._lock:
                            self._error, self._fps, self._thumb = str(missing), 0.0, None
                        time.sleep(MISSING_RETRY_S)
                        continue
                    previous, last_ok = None, self._clock()
                    continue
                ok, frame = capture.read()
                if not ok or frame is None:
                    with self._lock:
                        self._fps = 0.0
                        self._error = (NO_PICTURE_REPLUG if reopens >= NO_PICTURE_GIVE_UP
                                       else f"no picture: reopening ({reopens + 1})")
                    time.sleep(0.2)
                    continue
                last_ok, reopens = self._clock(), 0
                small = thumbnail(frame) if hasattr(frame, "ndim") else None
                frame = rotate_frame(frame, self.rotation)
                result, jpeg = detect(frame, previous)
                previous = (result.x, result.y) if result.locked else None
                now = self._clock()
                with self._lock:
                    self._jpeg, self._numbers, self._error = jpeg, measure(result, frame.shape[1]), ""
                    if small is not None:
                        self._thumb, self._thumb_at = small, now
                    self._fps = 0.8 * self._fps + 0.2 * (1.0 / max(now - last, 1e-3))
                last = now
                time.sleep(max(0.0, 1.0 / STREAM_FPS - (self._clock() - now)))
        except Exception as exc:  # unplugged or busy: retried on the next view
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._release(capture, device)


class EyeCameras:
    """The two eye slots. Cameras are looked for again whenever a slot has none or one is unplugged."""

    def __init__(self, cameras: list[EyeCamera], config_path: Optional[Path] = None, *,
                 finder: Optional[Callable[[], Optional[list[EyeDevice]]]] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cameras, self._config_path = cameras, config_path
        self._finder, self._clock = finder, clock
        self._settings_lock = threading.RLock()  # rotate() vs the refresh thread
        self._refresh_lock = threading.Lock()
        self._refreshing = False
        self._partial_looks = 0
        self._refreshed_at = clock()
        self._same_seen = [0] * len(cameras)
        self._reopens = [0] * len(cameras)
        self._gave_up: set[int] = set()
        self._saved = self._load()
        self._apply_settings([camera.device for camera in cameras if camera.device is not None])

    @classmethod
    def discover(cls, config_path: Path = DEFAULT_CONFIG,
                 finder: Callable[[], Optional[list[EyeDevice]]] = find_eye_devices, **camera_options) -> "EyeCameras":
        claims = DeviceClaims()
        cameras = [EyeCamera(n, f"EYE {n + 1}", claims=claims, **camera_options) for n in range(EYE_SLOTS)]
        eyes = cls(cameras, config_path, finder=finder)
        eyes.refresh()
        return eyes

    def refresh(self) -> None:
        """Look for the cameras again and put each in its slot (see assign_slots)."""
        if self._finder is None:
            return
        devices = self._finder()
        if devices is None:  # cannot list them right now: keep what is there
            return
        located = [d.location for d in devices if d.location]
        partial = bool(located) and (len(located) < len(devices) or len(set(located)) < len(located))
        self._partial_looks = self._partial_looks + 1 if partial else 0
        if 0 < self._partial_looks <= PARTIAL_LOOKS:  # one USB port could not be read: look again, don't guess
            return
        with self._settings_lock:
            keys = {device.path: settings_key(device, devices) for device in devices}
            before = [camera.device for camera in self.cameras]
            chosen = assign_slots(devices, before, lambda device: self._saved_slot(keys[device.path]))
            for camera, device in zip(self.cameras, chosen):
                camera.use(device)
            self._apply_settings(devices)
            if [d.path if d else None for d in chosen] != [d.path if d else None for d in before]:
                self._save()  # remember which slot each port is in

    def _saved_slot(self, key: str) -> Optional[int]:
        setting = self._saved.get(key)
        return setting.get("slot") if isinstance(setting, dict) else None

    def refresh_soon(self) -> None:
        """refresh() in the background (at most one at a time), so page requests never wait for it."""
        if self._finder is None:
            return
        with self._refresh_lock:
            if self._refreshing:
                return
            self._refreshing, self._refreshed_at = True, self._clock()

        def run():
            try:
                self.refresh()
            except Exception:
                pass
            finally:
                self._refreshing = False
        threading.Thread(target=run, name="eye-refresh", daemon=True).start()

    def _apply_settings(self, devices: list[EyeDevice]) -> None:
        with self._settings_lock:
            migrated = False
            for camera in self.cameras:
                if camera.device is None:
                    continue
                camera.settings_key = settings_key(camera.device, devices)
                setting = self._saved.get(camera.settings_key)
                if not isinstance(setting, dict):  # saved by an older version under the device ID
                    setting = self._saved.get(camera.device.device_id)
                    setting = setting if isinstance(setting, dict) else {}
                    migrated = migrated or bool(setting)
                rotation = setting.get("rotation", 0)
                camera.rotation = rotation if rotation in ROTATIONS else 0
            if migrated:
                self._save()

    def rotate(self, number: int, by: int) -> None:
        """Rotate one camera's picture and save the setting."""
        if by not in (90, -90, 180):
            raise ValueError("Rotate by 90, -90 or 180 degrees")
        with self._settings_lock:
            camera = self.cameras[number]
            camera.rotation = (camera.rotation + by) % 360
            self._save()

    def _load(self) -> dict:
        if self._config_path is None or not self._config_path.is_file():
            return {}
        try:
            data = json.loads(self._config_path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        for camera in self.cameras:
            if camera.device is not None:
                self._saved[camera.settings_key] = {"rotation": camera.rotation, "slot": camera.slot}
        if self._config_path is None:
            return
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(json.dumps(self._saved, indent=2))
        except OSError:
            pass

    def check_same_picture(self) -> None:
        """Reopen a camera that shows the same picture as an earlier one (up to MAX_REOPENS)."""
        for later in range(1, len(self.cameras)):
            thumb = self.cameras[later].thumb()
            same = thumb is not None and any(same_picture(self.cameras[first].thumb(), thumb)
                                             for first in range(later))
            if not same and thumb is not None:
                self._gave_up.discard(later)
                self._reopens[later] = 0
            if later in self._gave_up:
                continue
            self._same_seen[later] = self._same_seen[later] + 1 if same else 0
            if self._same_seen[later] < SAME_CHECKS:
                continue
            self._same_seen[later] = 0
            if self._reopens[later] < MAX_REOPENS:
                self._reopens[later] += 1
                self.refresh_soon()
                self.cameras[later].request_reopen(
                    f"Same picture as the other eye: reopening ({self._reopens[later]})")
            else:
                self._gave_up.add(later)

    def status(self) -> list[dict]:
        if any(camera.missing() for camera in self.cameras) and \
                self._clock() - self._refreshed_at >= REFRESH_EVERY_S:
            self.refresh_soon()
        self.check_same_picture()
        statuses = [camera.status() for camera in self.cameras]
        for index in self._gave_up:
            statuses[index]["error"] = GAVE_UP
        return statuses


def create_eye_blueprint(eyes: EyeCameras):
    from flask import Blueprint, Response, abort, jsonify, request

    blueprint = Blueprint("eye_cameras", __name__)

    @blueprint.get("/api/eyes")
    def eye_status():
        for camera in eyes.cameras:
            camera.touch()
        return jsonify(eyes=eyes.status())

    @blueprint.post("/api/eyes/<int:number>/rotate")
    def eye_rotate(number: int):
        if not 0 <= number < len(eyes.cameras):
            abort(404)
        try:
            eyes.rotate(number, (request.get_json(silent=True) or {}).get("by"))
        except ValueError as exc:
            return jsonify(status="error", message=str(exc)), 400
        return jsonify(eyes=eyes.status())

    @blueprint.get("/eyes/<int:number>.mjpg")
    def eye_stream(number: int):
        if not 0 <= number < len(eyes.cameras):
            abort(404)
        camera = eyes.cameras[number]

        def frames():
            while True:
                camera.touch()
                jpeg = camera.latest_jpeg()
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(1.0 / STREAM_FPS)

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return blueprint

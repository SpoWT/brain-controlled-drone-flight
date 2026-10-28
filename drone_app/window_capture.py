"""Find and capture the flying-page window (Windows).

PrintWindow captures the window's own pixels, even when it is covered. Win32 is loaded lazily.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Iterable, Optional

PAGE_TITLE = "Optical Hover Control"
CHROME_CLASS = "Chrome_WidgetWin_1"
PW_RENDERFULLCONTENT = 2
WM_CLOSE = 0x0010


@dataclass(frozen=True)
class Window:
    hwnd: int
    pid: int
    title: str
    cls: str
    area: int
    visible: bool


def choose_window(windows: Iterable[Window], pids: set[int]) -> Optional[int]:
    """Largest visible Chrome window of our browser, else one titled like the page."""
    chrome = [w for w in windows if w.visible and w.cls == CHROME_CLASS and w.area > 0]
    ours = [w for w in chrome if w.pid in pids] or [w for w in chrome if w.title == PAGE_TITLE]
    return max(ours, key=lambda w: w.area).hwnd if ours else None


_API = None


def _api():
    global _API
    if _API is not None:
        return _API
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32")
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    sig = [  # (function, argtypes, restype)
        (user32.EnumWindows, [enum_proc, wintypes.LPARAM], wintypes.BOOL),
        (user32.GetWindowThreadProcessId, [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
        (user32.IsWindowVisible, [wintypes.HWND], wintypes.BOOL),
        (user32.IsWindow, [wintypes.HWND], wintypes.BOOL),
        (user32.IsIconic, [wintypes.HWND], wintypes.BOOL),
        (user32.GetWindowRect, [wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
        (user32.GetWindowTextW, [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
        (user32.GetClassNameW, [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
        (user32.GetWindowDC, [wintypes.HWND], wintypes.HDC),
        (user32.ReleaseDC, [wintypes.HWND, wintypes.HDC], ctypes.c_int),
        (user32.PrintWindow, [wintypes.HWND, wintypes.HDC, wintypes.UINT], wintypes.BOOL),
        (user32.PostMessageW, [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM], wintypes.BOOL),
        (gdi32.CreateCompatibleDC, [wintypes.HDC], wintypes.HDC),
        (gdi32.CreateCompatibleBitmap, [wintypes.HDC, ctypes.c_int, ctypes.c_int], wintypes.HBITMAP),
        (gdi32.SelectObject, [wintypes.HDC, wintypes.HGDIOBJ], wintypes.HGDIOBJ),
        (gdi32.GetDIBits, [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                           ctypes.c_void_p, wintypes.UINT], ctypes.c_int),
        (gdi32.DeleteObject, [wintypes.HGDIOBJ], wintypes.BOOL),
        (gdi32.DeleteDC, [wintypes.HDC], wintypes.BOOL),
    ]
    for function, args, result in sig:
        function.argtypes, function.restype = args, result
    _API = (user32, gdi32, enum_proc, wintypes)
    return _API


def list_windows() -> list[Window]:
    user32, _, enum_proc, wintypes = _api()
    found: list[Window] = []

    def visit(hwnd, _param):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        title, cls = ctypes.create_unicode_buffer(256), ctypes.create_unicode_buffer(64)
        user32.GetWindowTextW(hwnd, title, 256)
        user32.GetClassNameW(hwnd, cls, 64)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        area = max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)
        found.append(Window(int(hwnd or 0), pid.value, title.value, cls.value, area, bool(user32.IsWindowVisible(hwnd))))
        return True

    user32.EnumWindows(enum_proc(visit), 0)
    return found


def exists(hwnd: Optional[int]) -> bool:
    return bool(hwnd) and bool(_api()[0].IsWindow(hwnd))


def close(hwnd: Optional[int]) -> None:
    if exists(hwnd):
        _api()[0].PostMessageW(hwnd, WM_CLOSE, 0, 0)


class _BitmapInfo(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
                ("biPlanes", ctypes.c_uint16), ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32), ("colors", ctypes.c_uint32 * 3)]


def capture(hwnd: Optional[int]):
    """BGR image of the window, or None when it is gone or minimised."""
    import numpy as np
    if not exists(hwnd):
        return None
    user32, gdi32, _, wintypes = _api()
    if user32.IsIconic(hwnd):
        return None
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None
    window_dc = user32.GetWindowDC(hwnd)
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    previous = gdi32.SelectObject(memory_dc, bitmap)
    try:
        if not user32.PrintWindow(hwnd, memory_dc, PW_RENDERFULLCONTENT):
            return None
        info = _BitmapInfo(biSize=40, biWidth=width, biHeight=-height, biPlanes=1, biBitCount=32, biCompression=0)
        pixels = ctypes.create_string_buffer(width * height * 4)
        if not gdi32.GetDIBits(memory_dc, bitmap, 0, height, pixels, ctypes.byref(info), 0):
            return None
        return np.frombuffer(pixels, np.uint8).reshape(height, width, 4)[:, :, :3].copy()
    finally:
        gdi32.SelectObject(memory_dc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, window_dc)

"""Capture an existing OpenCV window's client area on Windows.

This reads the displayed window, not the camera device. For an unscaled source
image, pass the original BGR frame to DeepSeekAnalyzer.analyze_frame instead.
"""

import ctypes
from ctypes import wintypes
import sys

import numpy as np


def capture_window(window_title: str) -> np.ndarray:
    """Return the named visible, restored window as a copied uint8 BGR array.

    Uses PrintWindow(PW_CLIENTONLY) rather than capturing the whole desktop.
    The title must exactly match the name passed to cv2.imshow(). Some GPU
    windows cannot be captured this way; use the original CV2 frame instead.
    """
    if sys.platform != "win32":
        raise RuntimeError("Window capture requires Windows; use analyze_frame on other systems.")
    if not window_title.strip():
        raise ValueError("A camera window title is required.")

    user = ctypes.WinDLL("user32", use_last_error = True)
    gdi = ctypes.WinDLL("gdi32", use_last_error = True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    signatures = (
        (user.EnumWindows, [callback_type, wintypes.LPARAM], wintypes.BOOL),
        (user.GetWindowTextLengthW, [wintypes.HWND], ctypes.c_int),
        (user.GetWindowTextW, [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
        (user.IsWindowVisible, [wintypes.HWND], wintypes.BOOL),
        (user.IsIconic, [wintypes.HWND], wintypes.BOOL),
        (user.IsHungAppWindow, [wintypes.HWND], wintypes.BOOL),
        (user.GetClientRect, [wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
        (user.GetDC, [wintypes.HWND], wintypes.HDC),
        (user.ReleaseDC, [wintypes.HWND, wintypes.HDC], ctypes.c_int),
        (user.PrintWindow, [wintypes.HWND, wintypes.HDC, wintypes.UINT], wintypes.BOOL),
        (gdi.CreateCompatibleDC, [wintypes.HDC], wintypes.HDC),
        (gdi.CreateCompatibleBitmap, [wintypes.HDC, ctypes.c_int, ctypes.c_int], wintypes.HBITMAP),
        (gdi.SelectObject, [wintypes.HDC, wintypes.HANDLE], wintypes.HANDLE),
        (gdi.DeleteObject, [wintypes.HANDLE], wintypes.BOOL),
        (gdi.DeleteDC, [wintypes.HDC], wintypes.BOOL),
        (gdi.GetDIBits, [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                        ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT], ctypes.c_int),
    )
    for function, arguments, result in signatures:
        function.argtypes, function.restype = arguments, result

    matches = []

    @callback_type
    def find_window(hwnd, _):
        if user.IsWindowVisible(hwnd):
            title = ctypes.create_unicode_buffer(user.GetWindowTextLengthW(hwnd) + 1)
            user.GetWindowTextW(hwnd, title, len(title))
            if title.value == window_title:
                matches.append(hwnd)
        return True

    if not user.EnumWindows(find_window, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    if len(matches) != 1:
        raise RuntimeError(f"Expected one visible window named {window_title!r}; found {len(matches)}. "
                           "Set --window-title to the exact cv2.imshow window name.")
    hwnd = matches[0]
    if user.IsIconic(hwnd) or user.IsHungAppWindow(hwnd):
        raise RuntimeError("Restore the camera window and ensure its video loop is responding.")
    rect = wintypes.RECT()
    if not user.GetClientRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0 or width * height > 32_000_000:
        raise RuntimeError("Camera window has invalid or excessive dimensions.")

    class BitmapInfoHeader(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("width", wintypes.LONG),
                    ("height", wintypes.LONG), ("planes", wintypes.WORD),
                    ("bits", wintypes.WORD), ("compression", wintypes.DWORD),
                    ("image_size", wintypes.DWORD), ("x_resolution", wintypes.LONG),
                    ("y_resolution", wintypes.LONG), ("colours", wintypes.DWORD),
                    ("important_colours", wintypes.DWORD)]

    source_dc = user.GetDC(hwnd)
    if not source_dc:
        raise ctypes.WinError(ctypes.get_last_error())
    memory_dc, bitmap, previous = None, None, None
    try:
        memory_dc = gdi.CreateCompatibleDC(source_dc)
        bitmap = gdi.CreateCompatibleBitmap(source_dc, width, height)
        if not memory_dc or not bitmap:
            raise ctypes.WinError(ctypes.get_last_error())
        previous = gdi.SelectObject(memory_dc, bitmap)
        if previous in (None, 0, ctypes.c_void_p(-1).value):
            previous = None
            raise RuntimeError("Could not allocate the camera capture surface.")
        if not user.PrintWindow(hwnd, memory_dc, 1):  # PW_CLIENTONLY
            raise RuntimeError("Camera window capture failed; pass its source frame directly instead.")
        # GetDIBits requires the bitmap to be deselected from the memory DC.
        gdi.SelectObject(memory_dc, previous)
        previous = None
        header = BitmapInfoHeader()
        header.size, header.width, header.height = ctypes.sizeof(header), width, -height
        header.planes, header.bits = 1, 32  # top-down, DWORD-aligned BGRA
        pixels = (ctypes.c_ubyte * (width * height * 4))()
        copied = gdi.GetDIBits(source_dc, bitmap, 0, height, pixels, ctypes.byref(header), 0)
        if copied != height:
            raise RuntimeError("Could not read the complete camera snapshot.")
        frame = np.ctypeslib.as_array(pixels).reshape(height, width, 4)[:, :, :3].copy()
        if not np.any(frame):
            raise RuntimeError("Camera snapshot is blank; restore the window or pass its source frame directly.")
        return frame
    finally:
        if previous and memory_dc:
            gdi.SelectObject(memory_dc, previous)
        if bitmap:
            gdi.DeleteObject(bitmap)
        if memory_dc:
            gdi.DeleteDC(memory_dc)
        user.ReleaseDC(hwnd, source_dc)

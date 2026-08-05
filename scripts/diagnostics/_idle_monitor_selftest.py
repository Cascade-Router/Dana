"""One-shot GetLastInputInfo self-test (Phase 1 compute governor)."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


def idle_seconds() -> float:
    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
        raise OSError("GetLastInputInfo failed")
    tick = ctypes.windll.kernel32.GetTickCount()
    idle_ms = (tick - lii.dwTime) & 0xFFFFFFFF
    return idle_ms / 1000.0


def main() -> None:
    s1 = idle_seconds()
    state1 = "USER_ACTIVE" if s1 < 300 else "USER_AWAY"
    print(f"sample1 idle_seconds={s1:.3f} state={state1} threshold=300s")
    time.sleep(1.2)
    s2 = idle_seconds()
    state2 = "USER_ACTIVE" if s2 < 300 else "USER_AWAY"
    print(f"sample2 idle_seconds={s2:.3f} state={state2} delta~{s2 - s1:.3f}s")
    print("OK: GetLastInputInfo readable on this machine")


if __name__ == "__main__":
    main()

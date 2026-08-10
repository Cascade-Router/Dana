"""Streaming DC blocker / high-pass filter for mic rumble, moved from ``core_agent``.

``DcBlocker`` removes constant offset and very-low-frequency energy that
otherwise inflates RMS and keeps WebRTC/Silero VAD from reaching its
silence cutoff.
"""

from __future__ import annotations

import numpy as np

from dana.core.constants import DC_BLOCKER_R


class DcBlocker:
    """Streaming first-order IIR DC blocker / high-pass for mic rumble.

    y[n] = x[n] - x[n-1] + R * y[n-1]
    Removes constant offset and very-low-frequency energy that inflates RMS and
    prevents WebRTC VAD from reaching ``silence_cutoff``.
    """

    __slots__ = ("_r", "_x_prev", "_y_prev")

    def __init__(self, r: float = DC_BLOCKER_R) -> None:
        self._r = float(np.clip(r, 0.9, 0.9999))
        self._x_prev = 0.0
        self._y_prev = 0.0

    def reset(self) -> None:
        self._x_prev = 0.0
        self._y_prev = 0.0

    def apply(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        y = np.empty_like(x)
        x_prev = self._x_prev
        y_prev = self._y_prev
        r = self._r
        for i in range(x.size):
            xi = float(x[i])
            yi = xi - x_prev + r * y_prev
            y[i] = yi
            x_prev = xi
            y_prev = yi
        self._x_prev = x_prev
        self._y_prev = y_prev
        return y


def remove_dc_offset(audio: np.ndarray, *, r: float = DC_BLOCKER_R) -> np.ndarray:
    """One-shot DC blocker for a finished buffer (Whisper / wake verify)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    # Mean subtract first (fast coarse DC kill), then light IIR for rumble.
    centered = audio - float(np.mean(audio))
    return DcBlocker(r=r).apply(centered)


__all__ = ("DcBlocker", "remove_dc_offset")

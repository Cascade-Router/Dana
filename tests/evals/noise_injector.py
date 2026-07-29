"""Adversarial desktop visual noise + latency jitter for offline OSWorld-style evals.

Seedable, injectable, and Pillow-optional so benchmark harness tests stay fast
and deterministic without GPU / network / real OSWorld downloads.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

SleepFn = Callable[[float], None]
BBox = list[float]  # xyxy pixels or normalized — caller chooses units

# Production-like jitter range (seconds).
_LATENCY_MIN_S = 0.050
_LATENCY_MAX_S = 0.300
# Deterministic / test mode: still non-zero but cheap.
_DET_LATENCY_MIN_S = 0.001
_DET_LATENCY_MAX_S = 0.005

_SHIFT_MIN_PX = 5
_SHIFT_MAX_PX = 15

_TOAST_W = 80
_TOAST_H = 28
_TOAST_RGB = (40, 40, 48)
_TOAST_ACCENT = (70, 130, 220)


@dataclass(frozen=True)
class VisualNoiseResult:
    """Screenshot after translation + toast overlays, plus bbox bookkeeping."""

    image: Any
    offset_xy: tuple[int, int]
    adjusted_bboxes: list[BBox]
    overlays: list[dict[str, Any]]


def _try_pillow():
    try:
        from PIL import Image  # noqa: F401

        return True
    except ImportError:
        return False


def _as_rgb_array(image: Any) -> tuple[list[list[tuple[int, int, int]]], int, int]:
    """Normalize to nested RGB rows (pure-Python) + (w, h)."""
    if hasattr(image, "convert") and hasattr(image, "size"):
        rgb = image.convert("RGB")
        w, h = rgb.size
        pix = list(rgb.getdata())
        rows: list[list[tuple[int, int, int]]] = []
        for y in range(h):
            row = [pix[y * w + x] for x in range(w)]
            rows.append(row)
        return rows, w, h
    # numpy ndarray HxWx3
    if hasattr(image, "shape") and len(getattr(image, "shape", ())) >= 2:
        import numpy as np

        arr = np.asarray(image)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        rows = []
        for y in range(h):
            row = []
            for x in range(w):
                px = arr[y, x]
                if getattr(px, "shape", ()) == ():
                    v = int(px)
                    row.append((v, v, v))
                else:
                    row.append((int(px[0]), int(px[1]), int(px[2])))
            rows.append(row)
        return rows, w, h
    # Nested list / tuple of RGB triples
    if isinstance(image, (list, tuple)) and image:
        h = len(image)
        w = len(image[0])
        rows = [
            [
                (int(p[0]), int(p[1]), int(p[2]))
                if not isinstance(p, int)
                else (int(p), int(p), int(p))
                for p in row
            ]
            for row in image
        ]
        return rows, w, h
    raise TypeError(f"Unsupported screenshot type: {type(image)!r}")


def _rebuild_image(
    rows: list[list[tuple[int, int, int]]],
    *,
    like: Any,
) -> Any:
    h = len(rows)
    w = len(rows[0]) if rows else 0
    if hasattr(like, "convert") and hasattr(like, "size") and _try_pillow():
        from PIL import Image

        out = Image.new("RGB", (w, h))
        out.putdata([px for row in rows for px in row])
        return out
    if hasattr(like, "shape"):
        import numpy as np

        return np.asarray(rows, dtype=np.uint8)
    return [[tuple(px) for px in row] for row in rows]


def translate_bboxes(
    bboxes: Sequence[Sequence[float]] | None,
    offset_xy: tuple[int, int],
) -> list[BBox]:
    """Shift xyxy boxes by the same pixel translation applied to the frame."""
    if not bboxes:
        return []
    dx, dy = int(offset_xy[0]), int(offset_xy[1])
    out: list[BBox] = []
    for box in bboxes:
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        out.append([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
    return out


def bbox_within_tolerance(
    got: Sequence[float],
    expected: Sequence[float],
    *,
    tol_px: float = 10.0,
) -> bool:
    """True when each corner of ``got`` is within ``tol_px`` of ``expected``."""
    if len(got) < 4 or len(expected) < 4:
        return False
    return all(abs(float(got[i]) - float(expected[i])) <= float(tol_px) for i in range(4))


class DesktopNoiseInjector:
    """Inject OS-like visual shifts, toast overlays, and execution latency jitter."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        deterministic: bool = False,
        sleep_fn: SleepFn | None = None,
    ) -> None:
        self.seed = seed
        self.deterministic = bool(deterministic)
        self._rng = random.Random(seed)
        self._sleep: SleepFn = sleep_fn if sleep_fn is not None else time.sleep
        self.last_offset_xy: tuple[int, int] = (0, 0)
        self.last_latency_s: float = 0.0
        self.last_overlays: list[dict[str, Any]] = []

    def reseed(self, seed: int | None = None) -> None:
        """Reset RNG (and optionally replace the seed)."""
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)

    def _rand_shift_px(self) -> int:
        mag = self._rng.randint(_SHIFT_MIN_PX, _SHIFT_MAX_PX)
        return mag if self._rng.random() < 0.5 else -mag

    def apply_visual_noise(
        self,
        screenshot_image: Any,
        bboxes: Sequence[Sequence[float]] | None = None,
        *,
        n_toasts: int = 1,
    ) -> VisualNoiseResult:
        """Translate the frame (±5–15px) and stamp synthetic toast overlays.

        Returns the noisy image, pixel offset applied, and bboxes shifted by the
        same offset so assertions can recover expected grounding targets.
        """
        rows, w, h = _as_rgb_array(screenshot_image)
        dx = self._rand_shift_px()
        dy = self._rand_shift_px()
        self.last_offset_xy = (dx, dy)

        # Translate: sample source at (x-dx, y-dy); fill missing with black.
        shifted: list[list[tuple[int, int, int]]] = []
        for y in range(h):
            row: list[tuple[int, int, int]] = []
            sy = y - dy
            for x in range(w):
                sx = x - dx
                if 0 <= sx < w and 0 <= sy < h:
                    row.append(rows[sy][sx])
                else:
                    row.append((0, 0, 0))
            shifted.append(row)

        overlays: list[dict[str, Any]] = []
        quadrants = (
            (0, 0, w // 2, h // 2),
            (w // 2, 0, w, h // 2),
            (0, h // 2, w // 2, h),
            (w // 2, h // 2, w, h),
        )
        for i in range(max(0, int(n_toasts))):
            qx0, qy0, qx1, qy1 = quadrants[self._rng.randrange(4)]
            qw = max(1, qx1 - qx0)
            qh = max(1, qy1 - qy0)
            tw = min(_TOAST_W, qw)
            th = min(_TOAST_H, qh)
            tx = qx0 + self._rng.randint(0, max(0, qw - tw))
            ty = qy0 + self._rng.randint(0, max(0, qh - th))
            for yy in range(ty, min(h, ty + th)):
                for xx in range(tx, min(w, tx + tw)):
                    # Accent bar on left edge of toast.
                    if xx - tx < 4:
                        shifted[yy][xx] = _TOAST_ACCENT
                    else:
                        shifted[yy][xx] = _TOAST_RGB
            overlays.append(
                {
                    "kind": "toast",
                    "index": i,
                    "bbox": [float(tx), float(ty), float(tx + tw), float(ty + th)],
                    "quadrant": (qx0, qy0, qx1, qy1),
                }
            )

        self.last_overlays = overlays
        out_img = _rebuild_image(shifted, like=screenshot_image)
        adjusted = translate_bboxes(bboxes, (dx, dy))
        return VisualNoiseResult(
            image=out_img,
            offset_xy=(dx, dy),
            adjusted_bboxes=adjusted,
            overlays=overlays,
        )

    def apply_latency_jitter(self) -> float:
        """Sleep a variable 50–300ms (or short deterministic range); return seconds slept.

        Pass ``sleep_fn`` / ``deterministic=True`` in tests to avoid wall-clock cost.
        """
        if self.deterministic:
            lo, hi = _DET_LATENCY_MIN_S, _DET_LATENCY_MAX_S
        else:
            lo, hi = _LATENCY_MIN_S, _LATENCY_MAX_S
        delay = self._rng.uniform(lo, hi)
        self._sleep(delay)
        self.last_latency_s = delay
        return delay

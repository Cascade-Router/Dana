"""Lazy YOLOv8 singleton + rolling vision frame buffer for CAMGRASPER Tracker.

Weights load on first vision use only. The Tracker thread pushes frames from
``active_vision_tool.get_frame()`` into a short deque so tools can read temporal
context without a cold screenshot.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

_yolo_lock = threading.Lock()
_yolo_model: Optional[Any] = None
_yolo_weights: Optional[str] = None

# Rolling buffer: ~2s cadence × maxlen 5 ≈ 10s of temporal context.
FRAME_BUFFER_MAXLEN = 5
FRAME_BUFFER_INTERVAL_S = 2.0


@dataclass
class FrameSample:
    """One buffered vision frame with optional YOLO dets and capture metadata."""

    frame: np.ndarray
    timestamp: float
    source: str = "screen"
    dets: tuple[tuple[Any, ...], ...] = ()
    frame_shape: tuple[int, ...] | None = None
    monitor: dict[str, int] | None = None


_buffer_lock = threading.Lock()
_frame_buffer: deque[FrameSample] = deque(maxlen=FRAME_BUFFER_MAXLEN)
_last_buffer_push_mono = 0.0


def yolo_is_loaded() -> bool:
    """True once ``YOLO(weights)`` has succeeded at least once."""
    return _yolo_model is not None


def get_yolo_model(weights: str) -> Any:
    """Return the shared YOLOv8 model, loading ``weights`` on first call only.

    Import of ``ultralytics`` and disk load of ``yolov8n.pt`` are deferred until
    Vision mode is active or a frame is explicitly processed for detection.
    """
    global _yolo_model, _yolo_weights
    if _yolo_model is not None:
        return _yolo_model
    with _yolo_lock:
        if _yolo_model is not None:
            return _yolo_model
        from ultralytics import YOLO

        _yolo_model = YOLO(weights)
        _yolo_weights = weights
        return _yolo_model


def reset_yolo_model() -> None:
    """Drop the cached model (tests / forced reload)."""
    global _yolo_model, _yolo_weights
    with _yolo_lock:
        _yolo_model = None
        _yolo_weights = None


def clear_frame_buffer() -> None:
    """Empty the rolling buffer (tests / mode reset)."""
    global _last_buffer_push_mono
    with _buffer_lock:
        _frame_buffer.clear()
        _last_buffer_push_mono = 0.0


def buffer_len() -> int:
    with _buffer_lock:
        return len(_frame_buffer)


def seconds_since_last_push() -> float:
    with _buffer_lock:
        if _last_buffer_push_mono <= 0:
            return float("inf")
        return max(0.0, time.monotonic() - _last_buffer_push_mono)


def should_push_frame(*, interval_s: float = FRAME_BUFFER_INTERVAL_S) -> bool:
    """True when enough time has elapsed since the last buffer push."""
    return seconds_since_last_push() >= float(interval_s)


def push_frame(
    frame: np.ndarray,
    *,
    source: str = "screen",
    dets: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...] | None = None,
    monitor: dict[str, int] | None = None,
    force: bool = False,
    interval_s: float = FRAME_BUFFER_INTERVAL_S,
) -> bool:
    """Append a frame to the rolling buffer (respects ~2s cadence unless ``force``).

    Returns True when the frame was stored.
    """
    global _last_buffer_push_mono
    if frame is None:
        return False
    arr = np.asarray(frame)
    if arr.size == 0:
        return False
    if not force and not should_push_frame(interval_s=interval_s):
        return False
    sample = FrameSample(
        frame=arr.copy(),
        timestamp=time.time(),
        source=str(source or "screen"),
        dets=tuple(dets or ()),
        frame_shape=tuple(int(x) for x in arr.shape),
        monitor=dict(monitor) if monitor else None,
    )
    with _buffer_lock:
        _frame_buffer.append(sample)
        _last_buffer_push_mono = time.monotonic()
    return True


def get_latest_buffered_frame() -> Optional[np.ndarray]:
    """Most recent buffered BGR frame, or None if the buffer is empty."""
    with _buffer_lock:
        if not _frame_buffer:
            return None
        return _frame_buffer[-1].frame.copy()


def get_latest_sample() -> Optional[FrameSample]:
    with _buffer_lock:
        if not _frame_buffer:
            return None
        return _frame_buffer[-1]


def get_buffered_frames(*, newest_first: bool = False) -> list[np.ndarray]:
    """Copy of all buffered frames (oldest→newest by default)."""
    with _buffer_lock:
        frames = [s.frame.copy() for s in _frame_buffer]
    if newest_first:
        frames.reverse()
    return frames


def get_temporal_context() -> dict[str, Any]:
    """Snapshot for ``analyze_visual_context`` / agent temporal reasoning."""
    with _buffer_lock:
        samples = list(_frame_buffer)
    if not samples:
        return {
            "count": 0,
            "interval_s": FRAME_BUFFER_INTERVAL_S,
            "frames": [],
            "sources": [],
            "timestamps": [],
            "latest_dets": [],
            "monitor": None,
        }
    latest = samples[-1]
    return {
        "count": len(samples),
        "interval_s": FRAME_BUFFER_INTERVAL_S,
        "frames": [s.frame.copy() for s in samples],
        "sources": [s.source for s in samples],
        "timestamps": [s.timestamp for s in samples],
        "latest_dets": list(latest.dets),
        "monitor": dict(latest.monitor) if latest.monitor else None,
        "frame_shape": latest.frame_shape,
    }


def primary_monitor_geometry() -> dict[str, int] | None:
    """Best-effort primary monitor dict ``{left, top, width, height}`` via mss."""
    try:
        import mss

        factory = getattr(mss, "mss", None) or getattr(mss, "MSS", None)
        with factory() as sct:
            if len(sct.monitors) < 2:
                return None
            mon = sct.monitors[1]
            return {
                "left": int(mon.get("left", 0)),
                "top": int(mon.get("top", 0)),
                "width": int(mon.get("width", 0)),
                "height": int(mon.get("height", 0)),
            }
    except Exception:  # noqa: BLE001
        return None


def map_box_to_screen(
    xyxy: Any,
    *,
    frame_wh: tuple[int, int] = (640, 480),
    monitor: dict[str, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """Map frame-space xyxy → absolute screen pixels for the ROI overlay."""
    try:
        x1, y1, x2, y2 = (float(v) for v in list(xyxy)[:4])
    except Exception:  # noqa: BLE001
        return None
    mon = monitor or primary_monitor_geometry()
    if not mon or mon.get("width", 0) <= 0 or mon.get("height", 0) <= 0:
        return None
    fw, fh = int(frame_wh[0]), int(frame_wh[1])
    if fw <= 0 or fh <= 0:
        return None
    sx = float(mon["width"]) / float(fw)
    sy = float(mon["height"]) / float(fh)
    left = int(mon["left"])
    top = int(mon["top"])
    return (
        int(round(left + x1 * sx)),
        int(round(top + y1 * sy)),
        int(round(left + x2 * sx)),
        int(round(top + y2 * sy)),
    )

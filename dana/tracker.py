"""Lazy YOLOv8 singleton + rolling vision frame buffer for CAMGRASPER Tracker.

Weights load on first vision use only. The Tracker thread pushes frames from
``active_vision_tool.get_frame()`` into a circular deque of low-res thumbnails
(~1 fps × 60 ≈ 60s). Only the latest full-resolution frame is retained for
live YOLO / OCR.
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

# Phase 3: ~1s cadence × maxlen 60 ≈ 60s of temporal context (thumbnails).
FRAME_BUFFER_MAXLEN = 60
FRAME_BUFFER_INTERVAL_S = 1.0
THUMB_SIZE = (160, 90)  # (width, height) — ~2.5 MiB for 60 frames


@dataclass
class FrameSample:
    """One buffered vision sample (thumbnail + metadata; optional full frame)."""

    frame: np.ndarray  # low-res thumbnail (BGR)
    timestamp: float
    source: str = "screen"
    dets: tuple[tuple[Any, ...], ...] = ()
    frame_shape: tuple[int, ...] | None = None
    monitor: dict[str, int] | None = None


_buffer_lock = threading.Lock()
_frame_buffer: deque[FrameSample] = deque(maxlen=FRAME_BUFFER_MAXLEN)
_last_buffer_push_mono = 0.0
# Single latest full-resolution frame for live OCR / YOLO (not duplicated in deque).
_latest_full_frame: Optional[np.ndarray] = None
_latest_full_meta: dict[str, Any] = {}


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
    global _last_buffer_push_mono, _latest_full_frame, _latest_full_meta
    with _buffer_lock:
        _frame_buffer.clear()
        _last_buffer_push_mono = 0.0
        _latest_full_frame = None
        _latest_full_meta = {}


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


def _make_thumbnail(arr: np.ndarray) -> np.ndarray:
    """Downscale to ``THUMB_SIZE`` (keeps rolling buffer memory bounded)."""
    tw, th = int(THUMB_SIZE[0]), int(THUMB_SIZE[1])
    if arr.ndim < 2:
        return arr.copy()
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if w == tw and h == th:
        return arr.copy()
    try:
        import cv2

        return cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)
    except Exception:  # noqa: BLE001
        # Fallback without OpenCV: crude stride subsample.
        ys = max(1, h // th)
        xs = max(1, w // tw)
        return np.ascontiguousarray(arr[::ys, ::xs][:th, :tw].copy())


def push_frame(
    frame: np.ndarray,
    *,
    source: str = "screen",
    dets: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...] | None = None,
    monitor: dict[str, int] | None = None,
    force: bool = False,
    interval_s: float = FRAME_BUFFER_INTERVAL_S,
) -> bool:
    """Append a thumbnail to the rolling buffer; retain one latest full frame.

    Returns True when the sample was stored.
    """
    global _last_buffer_push_mono, _latest_full_frame, _latest_full_meta
    if frame is None:
        return False
    arr = np.asarray(frame)
    if arr.size == 0:
        return False
    if not force and not should_push_frame(interval_s=interval_s):
        return False
    thumb = _make_thumbnail(arr)
    sample = FrameSample(
        frame=thumb,
        timestamp=time.time(),
        source=str(source or "screen"),
        dets=tuple(dets or ()),
        frame_shape=tuple(int(x) for x in arr.shape),
        monitor=dict(monitor) if monitor else None,
    )
    with _buffer_lock:
        _frame_buffer.append(sample)
        _latest_full_frame = arr.copy()
        _latest_full_meta = {
            "timestamp": sample.timestamp,
            "source": sample.source,
            "dets": sample.dets,
            "frame_shape": sample.frame_shape,
            "monitor": sample.monitor,
        }
        _last_buffer_push_mono = time.monotonic()
    return True


def get_latest_full_frame() -> Optional[np.ndarray]:
    """Most recent full-resolution BGR frame (not the thumbnail), or None."""
    with _buffer_lock:
        if _latest_full_frame is None:
            return None
        return _latest_full_frame.copy()


def get_latest_buffered_frame() -> Optional[np.ndarray]:
    """Most recent full frame when available; else latest thumbnail."""
    with _buffer_lock:
        if _latest_full_frame is not None:
            return _latest_full_frame.copy()
        if not _frame_buffer:
            return None
        return _frame_buffer[-1].frame.copy()


def get_latest_sample() -> Optional[FrameSample]:
    with _buffer_lock:
        if not _frame_buffer:
            return None
        return _frame_buffer[-1]


def get_buffered_frames(*, newest_first: bool = False) -> list[np.ndarray]:
    """Copy of buffered thumbnails (oldest→newest by default)."""
    with _buffer_lock:
        frames = [s.frame.copy() for s in _frame_buffer]
    if newest_first:
        frames.reverse()
    return frames


def get_recent_frame_sequence(seconds: float = 30.0) -> list[FrameSample]:
    """Thread-safe recent samples within the last ``seconds`` (oldest→newest).

    Returns shallow copies of ``FrameSample`` with thumbnail arrays copied.
    """
    cutoff = time.time() - max(0.0, float(seconds))
    with _buffer_lock:
        out: list[FrameSample] = []
        for s in _frame_buffer:
            if float(s.timestamp) < cutoff:
                continue
            out.append(
                FrameSample(
                    frame=s.frame.copy(),
                    timestamp=s.timestamp,
                    source=s.source,
                    dets=s.dets,
                    frame_shape=s.frame_shape,
                    monitor=dict(s.monitor) if s.monitor else None,
                )
            )
        return out


def get_temporal_context() -> dict[str, Any]:
    """Snapshot for ``analyze_visual_context`` / agent temporal reasoning."""
    with _buffer_lock:
        samples = list(_frame_buffer)
        full = None if _latest_full_frame is None else _latest_full_frame.copy()
    if not samples:
        return {
            "count": 0,
            "interval_s": FRAME_BUFFER_INTERVAL_S,
            "frames": [],
            "sources": [],
            "timestamps": [],
            "latest_dets": [],
            "monitor": None,
            "full_frame": full,
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
        "full_frame": full,
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

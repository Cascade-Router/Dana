"""JIT YOLOv8 vision capture + payload helpers for Vision Mode tools.

The ultralytics YOLO model is never imported or loaded at module import time.
Weights load on the first ``analyze_visual_context`` / ``_get_yolo_model`` call.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any, Optional

import cv2
import mss
import numpy as np

FRAME_SIZE = (640, 480)  # (width, height)

# Lazy singleton — populated only inside ``_get_yolo_model``.
_YOLO_MODEL: Any = None
_YOLO_LOCK = threading.Lock()
_YOLO_WEIGHTS: Optional[str] = None


def _default_weights() -> str:
    try:
        from dana.paths import YOLO_WEIGHTS_PATH

        return str(YOLO_WEIGHTS_PATH)
    except Exception:  # noqa: BLE001
        return "yolov8n.pt"


def _get_yolo_model(weights: str | None = None) -> Any:
    """Return the shared YOLOv8 model, loading it Just-In-Time on first use."""
    global _YOLO_MODEL, _YOLO_WEIGHTS
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    with _YOLO_LOCK:
        if _YOLO_MODEL is not None:
            return _YOLO_MODEL
        from ultralytics import YOLO

        path = weights or _default_weights()
        _YOLO_MODEL = YOLO(path)
        _YOLO_WEIGHTS = path
        return _YOLO_MODEL


def yolo_is_loaded() -> bool:
    """True once the JIT YOLO singleton has been materialized."""
    return _YOLO_MODEL is not None


def reset_yolo_model() -> None:
    """Drop the cached model (tests / forced reload)."""
    global _YOLO_MODEL, _YOLO_WEIGHTS
    with _YOLO_LOCK:
        _YOLO_MODEL = None
        _YOLO_WEIGHTS = None


def capture_screen_frame() -> Optional[np.ndarray]:
    """Grab the primary monitor via mss; drop alpha → BGR NumPy array."""
    try:
        with mss.mss() as sct:
            if len(sct.monitors) < 2:
                return None
            monitor = sct.monitors[1]  # primary
            shot = sct.grab(monitor)
            frame = np.asarray(shot, dtype=np.uint8)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif not (frame.ndim == 3 and frame.shape[2] == 3):
                return None
            w, h = FRAME_SIZE
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
            return frame
    except Exception:  # noqa: BLE001
        return None


def capture_webcam_frame(camera_index: int = 0) -> Optional[np.ndarray]:
    """Grab one webcam frame, then immediately release the device."""
    cap: Optional[cv2.VideoCapture] = None
    try:
        backends: list[int] = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        backends.append(cv2.CAP_ANY if hasattr(cv2, "CAP_ANY") else 0)

        frame: Optional[np.ndarray] = None
        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    continue
                ok, grabbed = cap.read()
                if ok and grabbed is not None:
                    frame = grabbed
                    break
                cap.release()
                cap = None
            except Exception:  # noqa: BLE001
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                    cap = None
        if frame is None:
            return None
        w, h = FRAME_SIZE
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        return frame
    except Exception:  # noqa: BLE001
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass


def _pluralize(label: str, count: int) -> str:
    name = (label or "object").strip() or "object"
    if count == 1:
        return f"1 {name}"
    if name.endswith("s") or name.endswith("ss"):
        return f"{count} {name}"
    if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
        return f"{count} {name[:-1]}ies"
    return f"{count} {name}s"


def _detections_from_results(results: Any) -> list[dict[str, Any]]:
    """Parse ultralytics results into {name, conf, xyxy} dicts."""
    out: list[dict[str, Any]] = []
    if not results:
        return out
    for result in results:
        names = getattr(result, "names", None) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
        except Exception:  # noqa: BLE001
            continue
        for i in range(len(clss)):
            cls_id = int(clss[i])
            label = str(names.get(cls_id, cls_id))
            out.append(
                {
                    "name": label,
                    "confidence": float(confs[i]),
                    "xyxy": [float(v) for v in xyxy[i].tolist()],
                }
            )
    return out


def format_vision_payload(detections: list[dict[str, Any]]) -> str:
    """Build the LLM-facing ``[Vision Output] Detected: …`` string."""
    if not detections:
        return "[Vision Output] Detected: nothing."
    counts = Counter(str(d.get("name") or "object") for d in detections)
    parts = [_pluralize(name, count) for name, count in counts.most_common()]
    return "[Vision Output] Detected: " + ", ".join(parts) + "."


def analyze_visual_context(source: str = "screen") -> str:
    """Analyze vision via Tracker rolling buffer (fallback: one cold capture).

    Prefers the latest frame from ``dana.tracker``'s maxlen-5 buffer so the
    agent gets temporal context without triggering a fresh screenshot when the
    Tracker is already sampling ``active_vision_tool``.
    """
    kind = str(source or "screen").strip().lower() or "screen"
    source_label = "webcam" if kind in {"webcam", "camera", "video"} else "screen"

    frame = None
    temporal_n = 0
    monitor = None
    try:
        from dana.tracker import (
            get_latest_buffered_frame,
            get_latest_sample,
            get_temporal_context,
        )

        ctx = get_temporal_context()
        temporal_n = int(ctx.get("count") or 0)
        sample = get_latest_sample()
        if sample is not None:
            # Prefer buffer frames that match the requested source when possible.
            src = str(sample.source or "").lower()
            if source_label == "webcam" and src in {"camera", "webcam", "video"}:
                frame = sample.frame.copy()
                monitor = sample.monitor
            elif source_label == "screen" and src in {"screen", ""}:
                frame = sample.frame.copy()
                monitor = sample.monitor
            elif temporal_n > 0 and source_label == "screen":
                # Screen queries may still use the latest buffer sample.
                frame = get_latest_buffered_frame()
                monitor = sample.monitor
    except Exception:  # noqa: BLE001
        frame = None

    if frame is None:
        if source_label == "webcam":
            frame = capture_webcam_frame()
        else:
            frame = capture_screen_frame()

    if frame is None:
        return f"[Vision Output] Detected: nothing (no {source_label} frame)."

    try:
        model = _get_yolo_model()
        results = model.predict(frame, verbose=False, conf=0.35)
    except Exception as exc:  # noqa: BLE001
        return f"[Vision Output] Detected: nothing (YOLO error: {exc})."

    detections = _detections_from_results(results)

    # Drive the live ROI overlay from agent attention (top detection).
    if detections and source_label == "screen":
        try:
            from dana.tracker import map_box_to_screen
            from dana.vision.overlay import update_roi

            best = max(detections, key=lambda d: float(d.get("confidence") or 0.0))
            xyxy = best.get("xyxy")
            name = str(best.get("name") or "object")
            shape = getattr(frame, "shape", None)
            fw = int(shape[1]) if shape is not None and len(shape) >= 2 else FRAME_SIZE[0]
            fh = int(shape[0]) if shape is not None and len(shape) >= 2 else FRAME_SIZE[1]
            screen_box = map_box_to_screen(xyxy, frame_wh=(fw, fh), monitor=monitor)
            if screen_box is not None:
                update_roi(screen_box, name)
        except Exception:  # noqa: BLE001
            pass

    payload = format_vision_payload(detections)
    if temporal_n > 1:
        payload = (
            f"{payload} "
            f"[Temporal: {temporal_n} buffered frames @ ~2s]."
        )
    # Typed objects topic for Chat / ambient awareness (not OCR).
    try:
        from dana.memory.blackboard import publish_perception_objects

        boxes_meta = []
        for d in detections[:24]:
            boxes_meta.append(
                {
                    "label": str(d.get("name") or "object"),
                    "confidence": float(d.get("confidence") or 0.0),
                    "xyxy": list(d.get("xyxy") or []),
                }
            )
        publish_perception_objects(
            payload,
            producer="analyze_visual_context",
            model="yolov8",
            boxes=boxes_meta,
        )
    except Exception:  # noqa: BLE001
        pass
    return payload


# ---------------------------------------------------------------------------
# Legacy agent classes (core_agent tracker pulls get_frame() on demand)
# ---------------------------------------------------------------------------


class ScreenAgent:
    """Lazy mss screen capture. Grabs only when get_frame() is called."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sct: Any = None
        self._monitor: Optional[dict] = None
        self._fail_streak = 0

    def _ensure_open(self) -> bool:
        if self._sct is not None and self._monitor is not None:
            return True
        try:
            # Prefer mss.mss(); fall back to mss.MSS for older package builds.
            factory = getattr(mss, "mss", None) or getattr(mss, "MSS", None)
            self._sct = factory()
            if len(self._sct.monitors) < 2:
                self._sct = None
                self._monitor = None
                return False
            self._monitor = self._sct.monitors[1]  # primary
            return True
        except Exception:
            self._sct = None
            self._monitor = None
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture one primary-monitor frame resized to 640x480 BGR."""
        with self._lock:
            if not self._ensure_open():
                return None
            assert self._sct is not None and self._monitor is not None
            try:
                shot = self._sct.grab(self._monitor)
                frame = np.asarray(shot, dtype=np.uint8)
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif not (frame.ndim == 3 and frame.shape[2] == 3):
                    raise ValueError(f"Unexpected screen frame shape: {frame.shape}")

                w, h = FRAME_SIZE
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)

                self._fail_streak = 0
                return frame
            except Exception:
                self._fail_streak += 1
                if self._fail_streak >= 5:
                    self.release()
                return None

    def release(self) -> None:
        with self._lock:
            if self._sct is not None:
                try:
                    self._sct.close()
                except Exception:  # noqa: BLE001
                    pass
            self._sct = None
            self._monitor = None


class VideoAgent:
    """Lazy webcam capture via OpenCV. Handles Windows MSMF lock failures."""

    def __init__(self, camera_index: int = 0) -> None:
        self._lock = threading.Lock()
        self._camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._backend_name = "none"
        self._fail_streak = 0
        self._last_warn = 0.0

    def _open_backends(self) -> list[tuple[str, int]]:
        backends: list[tuple[str, int]] = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(("DSHOW", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("MSMF", cv2.CAP_MSMF))
        backends.append(("ANY", cv2.CAP_ANY if hasattr(cv2, "CAP_ANY") else 0))
        return backends

    def _ensure_open(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True
        self._close_unlocked()

        for name, backend in self._open_backends():
            try:
                cap = cv2.VideoCapture(self._camera_index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                ok, _ = cap.read()
                if not ok:
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
                self._cap = cap
                self._backend_name = name
                self._fail_streak = 0
                return True
            except Exception:
                continue
        return False

    def _close_unlocked(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            self._backend_name = "none"

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture one webcam frame resized to 640x480 BGR."""
        with self._lock:
            if not self._ensure_open():
                now = time.monotonic()
                if now - self._last_warn > 5.0:
                    self._last_warn = now
                return None
            assert self._cap is not None
            try:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._fail_streak += 1
                    if self._fail_streak >= 3:
                        self._close_unlocked()
                    return None

                w, h = FRAME_SIZE
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)

                self._fail_streak = 0
                return frame
            except Exception:
                self._fail_streak += 1
                if self._fail_streak >= 3:
                    self._close_unlocked()
                return None

    def release(self) -> None:
        with self._lock:
            self._close_unlocked()
            self._fail_streak = 0

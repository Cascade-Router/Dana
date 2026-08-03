"""Compatibility shim — implementation lives in ``dana.vision_tools``."""

from __future__ import annotations

from dana.vision_tools import (  # noqa: F401
    FRAME_SIZE,
    ScreenAgent,
    VideoAgent,
    analyze_visual_context,
    capture_screen_frame,
    capture_webcam_frame,
    format_vision_payload,
    reset_yolo_model,
    yolo_is_loaded,
)

__all__ = [
    "FRAME_SIZE",
    "ScreenAgent",
    "VideoAgent",
    "analyze_visual_context",
    "capture_screen_frame",
    "capture_webcam_frame",
    "format_vision_payload",
    "reset_yolo_model",
    "yolo_is_loaded",
]

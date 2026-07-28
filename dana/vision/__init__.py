"""Vision helpers (ROI overlay, Florence-2 OCR, Tracker buffer consumers)."""

from __future__ import annotations

from dana.vision.overlay import (
    RoiOverlay,
    clear_roi,
    ensure_overlay_started,
    get_overlay,
    update_roi,
)

__all__ = (
    "RoiOverlay",
    "clear_roi",
    "ensure_overlay_started",
    "get_overlay",
    "update_roi",
)

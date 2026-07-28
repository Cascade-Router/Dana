"""Vision helpers (ROI overlay, Florence-2 OCR, Tracker buffer consumers)."""

from __future__ import annotations

from dana.vision.hybrid_grounding import HybridVisionGrounding
from dana.vision.overlay import (
    RoiOverlay,
    clear_roi,
    ensure_overlay_started,
    get_overlay,
    update_roi,
)
from dana.vision.uia_provider import Win32UIAProvider

__all__ = (
    "HybridVisionGrounding",
    "RoiOverlay",
    "Win32UIAProvider",
    "clear_roi",
    "ensure_overlay_started",
    "get_overlay",
    "update_roi",
)

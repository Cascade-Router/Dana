"""Graph helpers for hybrid UI grounding (UIA + Florence crop-and-zoom).

Thin, injectable wrappers so the ReAct corridor / macro engine can locate UI
targets without touching HITL, ToolForge gates, or ``dana.paths`` routing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from dana.vision.hybrid_grounding import HybridVisionGrounding, NormBBox

_log = logging.getLogger("dana.graph.nodes.vision")

# Module-level singleton so macros / tools share one injectable instance.
_default_grounding: HybridVisionGrounding | None = None


def get_hybrid_grounding() -> HybridVisionGrounding:
    global _default_grounding
    if _default_grounding is None:
        _default_grounding = HybridVisionGrounding()
    return _default_grounding


def set_hybrid_grounding(instance: HybridVisionGrounding | None) -> None:
    """Tests / DI: replace or clear the shared hybrid grounder."""
    global _default_grounding
    _default_grounding = instance


def locate_ui_element(
    image: Any,
    target_label: str,
    *,
    grounder: HybridVisionGrounding | None = None,
) -> Optional[NormBBox]:
    """Locate ``target_label`` via hybrid UIA → Florence crop-and-zoom.

    Returns a Florence-normalized ``[x1,y1,x2,y2]`` in ``[0,1000]``, or ``None``.
    """
    engine = grounder or get_hybrid_grounding()
    try:
        return engine.locate_ui_element(image, target_label)
    except Exception as exc:  # noqa: BLE001
        _log.debug("locate_ui_element failed: %s", exc)
        return None


def vision_ground_node(state: dict[str, Any]) -> dict[str, Any]:
    """Optional graph node: ground ``target_label`` on ``vision_image`` / screenshot.

    Reads ``target_label`` and ``vision_image`` (or ``screenshot``) from state;
    writes ``vision_bbox_norm`` + ``vision_ground_stage``. Does not alter HITL
    or mailroom routing — callers wire this node only when needed.
    """
    label = str(state.get("target_label") or state.get("vision_query") or "").strip()
    image = state.get("vision_image")
    if image is None:
        image = state.get("screenshot")
    if image is None:
        try:
            from dana.vision_tools import capture_screen_frame

            image = capture_screen_frame()
        except Exception:  # noqa: BLE001
            image = None

    grounder = get_hybrid_grounding()
    bbox = locate_ui_element(image, label, grounder=grounder) if label else None
    return {
        "vision_bbox_norm": bbox,
        "vision_ground_stage": getattr(grounder, "last_stage", ""),
        "vision_target_label": label,
    }

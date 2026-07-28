"""Visual tools: Florence-2 OCR grounding over the Tracker rolling buffer."""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np


def _frame_from_tracker_buffer() -> tuple[Optional[np.ndarray], dict[str, int] | None]:
    """Most recent Tracker buffer frame + optional monitor geometry."""
    try:
        from dana.tracker import get_latest_sample

        sample = get_latest_sample()
        if sample is None:
            return None, None
        return sample.frame.copy(), dict(sample.monitor) if sample.monitor else None
    except Exception:  # noqa: BLE001
        return None, None


def _cold_screen_frame() -> tuple[Optional[np.ndarray], dict[str, int] | None]:
    try:
        from dana.tracker import primary_monitor_geometry
        from dana.vision_tools import capture_screen_frame

        return capture_screen_frame(), primary_monitor_geometry()
    except Exception:  # noqa: BLE001
        return None, None


def _pick_region(
    regions: list[dict[str, Any]],
    *,
    query: str,
) -> dict[str, Any] | None:
    if not regions:
        return None
    q = (query or "").strip().lower()
    if not q:
        # Largest box area wins when no query.
        def _area(r: dict[str, Any]) -> float:
            box = r.get("xyxy_norm") or [0, 0, 0, 0]
            return max(0.0, float(box[2]) - float(box[0])) * max(
                0.0, float(box[3]) - float(box[1])
            )

        return max(regions, key=_area)

    # Prefer substring / token overlap with the query.
    tokens = [t for t in re.split(r"\W+", q) if len(t) >= 2]

    def _score(r: dict[str, Any]) -> float:
        text = str(r.get("text") or "").lower()
        if not text:
            return -1.0
        if q in text or text in q:
            return 100.0 + len(text)
        hits = sum(1 for t in tokens if t in text)
        return float(hits)

    best = max(regions, key=_score)
    if _score(best) < 0:
        return regions[0]
    return best


def ocr_with_region(
    *,
    query: str = "",
    task: str = "<OCR_WITH_REGION>",
) -> str:
    """Florence-2 ``<OCR_WITH_REGION>`` over the Tracker rolling frame buffer.

    Parses text + normalized boxes, maps the best match to absolute screen
    pixels, and highlights it on the live ROI overlay as ``Florence-2 OCR``.
    """
    from dana.tracker import map_box_to_screen
    from dana.vision.florence_engine import run_ocr_with_region

    frame, monitor = _frame_from_tracker_buffer()
    used_buffer = frame is not None
    if frame is None:
        frame, monitor = _cold_screen_frame()
        used_buffer = False
    if frame is None:
        return (
            "[Florence OCR] No frame available "
            "(Tracker buffer empty and screen capture failed)."
        )

    result = run_ocr_with_region(frame, task=task or "<OCR_WITH_REGION>")
    if not result.get("ok"):
        return f"[Florence OCR] ERROR: {result.get('error') or 'inference failed'}"

    regions = list(result.get("regions") or [])
    image_wh = tuple(result.get("image_wh") or (frame.shape[1], frame.shape[0]))
    picked = _pick_region(regions, query=query)

    overlay_box = None
    if picked is not None:
        norm = list(picked.get("xyxy_norm") or [])
        if len(norm) == 4:
            # post_process_generation(..., image_size=orig_wh) → frame pixels.
            overlay_box = map_box_to_screen(
                (float(norm[0]), float(norm[1]), float(norm[2]), float(norm[3])),
                frame_wh=(int(image_wh[0]), int(image_wh[1])),
                monitor=monitor,
            )
        if overlay_box is not None:
            try:
                from dana.vision.overlay import update_roi

                update_roi(overlay_box, "Florence-2 OCR")
            except Exception:  # noqa: BLE001
                pass

    # Compact MoA / ReAct observation.
    lines: list[str] = []
    src = "tracker_buffer" if used_buffer else "cold_screenshot"
    lines.append(f"[Florence OCR] source={src} regions={len(regions)}")
    if query.strip():
        lines.append(f"query={query.strip()!r}")
    if picked is not None:
        lines.append(
            f"focus_text={str(picked.get('text') or '')!r} "
            f"box_norm={picked.get('xyxy_norm')} "
            f"box_screen={list(overlay_box) if overlay_box else None}"
        )
    # Include up to 12 OCR snippets for the graph.
    for i, reg in enumerate(regions[:12]):
        lines.append(f"  [{i}] {str(reg.get('text') or '').strip()}")
    if len(regions) > 12:
        lines.append(f"  … +{len(regions) - 12} more")
    if not regions:
        lines.append("(no OCR regions detected)")
    # Parseable Florence-style labeled bbox for nav_and_click.
    if picked is not None and overlay_box is not None:
        label = str(picked.get("text") or "target").strip() or "target"
        x1, y1, x2, y2 = [int(v) for v in overlay_box]
        lines.append(f"{label} [{x1}, {y1}, {x2}, {y2}]")

    observation = "\n".join(lines)
    # Typed OCR topic — consumers (ingest/nav/ghost/jason) read this only.
    try:
        from dana.memory.blackboard import publish_perception_ocr

        boxes_meta: list[dict[str, Any]] = []
        for reg in regions:
            boxes_meta.append(
                {
                    "text": str(reg.get("text") or ""),
                    "xyxy_norm": list(reg.get("xyxy_norm") or []),
                }
            )
        if overlay_box is not None and picked is not None:
            boxes_meta.insert(
                0,
                {
                    "text": str(picked.get("text") or ""),
                    "xyxy_screen": list(overlay_box),
                },
            )
        publish_perception_ocr(
            observation,
            producer="ocr_with_region",
            model="florence-2",
            boxes=boxes_meta,
        )
    except Exception:  # noqa: BLE001
        pass
    return observation

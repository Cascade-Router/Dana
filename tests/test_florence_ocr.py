"""Unit tests for Florence OCR grounding helpers (no GPU model required)."""

from __future__ import annotations

from dana.tools.visual_tools import _pick_region
from dana.vision.florence_engine import (
    _normalize_parsed,
    _quad_to_xyxy,
    norm_box_to_screen,
)


def test_quad_to_xyxy() -> None:
    assert _quad_to_xyxy([10, 20, 30, 20, 30, 40, 10, 40]) == (10.0, 20.0, 30.0, 40.0)
    assert _quad_to_xyxy([5, 5, 50, 60]) == (5.0, 5.0, 50.0, 60.0)
    print("[PASS] quad_to_xyxy")


def test_normalize_ocr_payload() -> None:
    parsed = {
        "<OCR_WITH_REGION>": {
            "labels": ["Submit", "Cancel"],
            "quad_boxes": [[100, 100, 200, 100, 200, 150, 100, 150], [10, 10, 40, 30]],
        }
    }
    out = _normalize_parsed(parsed, "<OCR_WITH_REGION>")
    assert out["labels"] == ["Submit", "Cancel"]
    assert len(out["boxes_xyxy_norm"]) == 2
    print("[PASS] normalize OCR payload")


def test_norm_box_to_screen_1000() -> None:
    box = norm_box_to_screen(
        (0.0, 0.0, 500.0, 500.0),
        image_wh=(640, 480),
        monitor={"left": 0, "top": 0, "width": 1920, "height": 1080},
    )
    assert box is not None
    # 500/1000 * 640 = 320 → screen scale 1920/640 = 3 → 960
    assert box[2] == 960
    print("[PASS] norm_box_to_screen 0-1000 mapping")


def test_pick_region_query() -> None:
    regions = [
        {"text": "Cancel", "xyxy_norm": [0, 0, 10, 10]},
        {"text": "Submit form", "xyxy_norm": [100, 100, 200, 150]},
    ]
    picked = _pick_region(regions, query="submit button")
    assert picked is not None
    assert "Submit" in str(picked.get("text"))
    print("[PASS] pick_region query match")

"""Offline tests for hybrid Win32 UIA + Florence crop-and-zoom grounding."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest

from dana.vision.hybrid_grounding import (
    HybridVisionGrounding,
    project_roi_box_to_global,
)
from dana.vision.uia_provider import Win32UIAProvider


def _blank_image(w: int = 200, h: int = 100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_uia_hit_returns_exact_bounds_immediately() -> None:
    """Mocked UIA match → instant exact bounds; Florence never called."""
    expected = [100.0, 200.0, 300.0, 280.0]
    uia = Win32UIAProvider(
        control_tree=[
            {
                "name": "Submit",
                "automation_id": "btnSubmit",
                "control_type": "Button",
                "bounds_norm": expected,
            }
        ]
    )
    florence_calls: list[tuple[Any, str]] = []

    def florence_spy(image: Any, label: str) -> Optional[list[float]]:
        florence_calls.append((image, label))
        return [0.0, 0.0, 50.0, 50.0]

    crop_calls: list[Any] = []

    def crop_spy(image: Any, box: list[float]) -> tuple[Any, tuple[float, float, float, float]]:
        crop_calls.append(box)
        return image, (0.0, 0.0, 10.0, 10.0)

    grounder = HybridVisionGrounding(
        uia_provider=uia,
        florence_ground_fn=florence_spy,
        crop_zoom_fn=crop_spy,
    )
    got = grounder.locate_ui_element(_blank_image(), "Submit")
    assert got == expected
    assert grounder.last_stage == "uia"
    assert florence_calls == []
    assert crop_calls == []


def test_large_florence_bbox_skips_zoom() -> None:
    """Coarse box with both edges ≥ 30 in 1000-scale → no crop&zoom."""
    uia = Win32UIAProvider(control_tree=[])  # miss
    coarse = [100.0, 100.0, 400.0, 350.0]  # w=300, h=250
    calls: list[str] = []

    def florence_spy(image: Any, label: str) -> Optional[list[float]]:
        calls.append("florence")
        return list(coarse)

    def crop_spy(image: Any, box: list[float]) -> tuple[Any, tuple[float, float, float, float]]:
        calls.append("crop")
        pytest.fail("crop_zoom must not run for large coarse boxes")
        return image, (0.0, 0.0, 1.0, 1.0)

    grounder = HybridVisionGrounding(
        uia_provider=uia,
        florence_ground_fn=florence_spy,
        crop_zoom_fn=crop_spy,
    )
    got = grounder.locate_ui_element(_blank_image(), "OK")
    assert got == coarse
    assert grounder.last_stage == "coarse"
    assert calls == ["florence"]


def test_small_target_triggers_crop_zoom_and_projects() -> None:
    """Small coarse box → crop&zoom; fine ROI coords project to global [0,1000]."""
    uia = Win32UIAProvider(control_tree=[])
    # Image 200×100; coarse box 20×20 in 1000-space (< 30 → zoom).
    # Pixel coarse ≈ (40,20)-(60,40) on 200×100.
    coarse = [200.0, 200.0, 220.0, 220.0]
    # Injectable crop: pretend ROI is pixels (30,15)-(70,45) after padding.
    crop_rect = (30.0, 15.0, 70.0, 45.0)
    # Fine box centered on zoomed crop in 1000-space.
    fine = [250.0, 250.0, 750.0, 750.0]

    florence_images: list[Any] = []

    def florence_spy(image: Any, label: str) -> Optional[list[float]]:
        florence_images.append(image)
        if len(florence_images) == 1:
            return list(coarse)
        return list(fine)

    crop_calls: list[list[float]] = []

    def crop_spy(
        image: Any, box: list[float]
    ) -> tuple[Any, tuple[float, float, float, float]]:
        crop_calls.append(list(box))
        zoomed = _blank_image(80, 60)  # 2× of 40×30 crop
        return zoomed, crop_rect

    grounder = HybridVisionGrounding(
        uia_provider=uia,
        florence_ground_fn=florence_spy,
        crop_zoom_fn=crop_spy,
    )
    img = _blank_image(200, 100)
    got = grounder.locate_ui_element(img, "TinyIcon")

    assert grounder.last_stage == "zoom"
    assert len(florence_images) == 2
    assert len(crop_calls) == 1
    assert crop_calls[0] == coarse

    expected = project_roi_box_to_global(
        fine,
        crop_rect_px=crop_rect,
        image_wh=(200, 100),
    )
    assert got is not None
    assert got == pytest.approx(expected, abs=1e-6)

    # Sanity: projected box sits inside the crop's global 1000 footprint.
    crop_norm = [
        crop_rect[0] / 200.0 * 1000.0,
        crop_rect[1] / 100.0 * 1000.0,
        crop_rect[2] / 200.0 * 1000.0,
        crop_rect[3] / 100.0 * 1000.0,
    ]
    assert got[0] >= crop_norm[0] - 1e-6
    assert got[1] >= crop_norm[1] - 1e-6
    assert got[2] <= crop_norm[2] + 1e-6
    assert got[3] <= crop_norm[3] + 1e-6


def test_canvas_or_unannotated_uia_returns_none() -> None:
    uia = Win32UIAProvider(
        control_tree=[
            {
                "name": "DrawingSurface",
                "automation_id": "",
                "control_type": "Pane",
                "is_canvas": True,
                "bounds_norm": [0.0, 0.0, 500.0, 500.0],
            }
        ]
    )
    assert uia.find_element_bounds("DrawingSurface") is None


def test_uia_provider_imports_without_windows_backends() -> None:
    """Module remains importable; empty live tree → None."""
    uia = Win32UIAProvider(tree_fetcher=lambda: [])
    assert uia.find_element_bounds("Anything") is None

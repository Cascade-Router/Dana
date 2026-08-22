"""Minimal test for dana.core.react_dispatch._bbox_overlap — the pure AABB
overlap math backing analyze_bounding_box_collisions. Not a full test suite
by design; dana/plugins/freecad/tests/test_engine.py already covers the
sibling _alignment_delta helper in this same style.
"""

from __future__ import annotations

from dana.core import react_dispatch as rd


def _bbox(x_min, y_min, z_min, x_max, y_max, z_max):
    return {"x_min": x_min, "y_min": y_min, "z_min": z_min, "x_max": x_max, "y_max": y_max, "z_max": z_max}


def test_overlapping_boxes_report_collision_and_overlap_volume():
    a = _bbox(0, 0, 0, 10, 10, 10)
    b = _bbox(5, 5, 5, 15, 15, 15)
    result = rd._bbox_overlap(a, b)
    assert result["collision"] is True
    assert result["overlap_bbox"] == _bbox(5, 5, 5, 10, 10, 10)
    assert result["overlap_volume"] == 125.0


def test_disjoint_boxes_report_no_collision():
    a = _bbox(0, 0, 0, 10, 10, 10)
    b = _bbox(20, 20, 20, 30, 30, 30)
    result = rd._bbox_overlap(a, b)
    assert result == {"collision": False, "overlap_bbox": None, "overlap_volume": 0.0}


def test_touching_but_not_overlapping_boxes_report_no_collision():
    """Edges sharing a coordinate (touching, not overlapping) must not
    count as a collision — the overlap region would have zero volume."""
    a = _bbox(0, 0, 0, 10, 10, 10)
    b = _bbox(10, 0, 0, 20, 10, 10)
    result = rd._bbox_overlap(a, b)
    assert result["collision"] is False

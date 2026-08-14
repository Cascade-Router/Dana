"""Unit tests for the pure-Python pieces of dana.plugins.freecad.engine —
logic that doesn't need a real FreeCADCmd binary to verify: the alignment
delta math (align_freecad_objects) and the stdout marker extractors.
"""

from __future__ import annotations

from dana.plugins.freecad import engine


def _bbox(x_min, y_min, z_min, x_max, y_max, z_max):
    return {"x_min": x_min, "y_min": y_min, "z_min": z_min, "x_max": x_max, "y_max": y_max, "z_max": z_max}


# --------------------------------------------------------------------------
# _alignment_delta — pure arithmetic, no FreeCAD needed
# --------------------------------------------------------------------------


def test_alignment_delta_top_center_stacks_source_on_target():
    # A 60x60x5 base plate at the origin, a 40mm-radius-20 cylinder centered
    # at the origin too (so it starts overlapping the plate, not stacked).
    source = _bbox(-20, -20, 0, 20, 20, 40)
    target = _bbox(0, 0, 0, 60, 60, 5)
    dx, dy, dz = engine._alignment_delta("top_center", source, target)
    assert (dx, dy, dz) == (30.0, 30.0, 5.0)


def test_alignment_delta_bottom_center_stacks_source_below_target():
    source = _bbox(-20, -20, 0, 20, 20, 40)
    target = _bbox(0, 0, 0, 60, 60, 5)
    dx, dy, dz = engine._alignment_delta("bottom_center", source, target)
    assert (dx, dy, dz) == (30.0, 30.0, -40.0)


def test_alignment_delta_flush_left_matches_min_x_centers_other_axes():
    source = _bbox(-10, -10, -10, 10, 10, 10)
    target = _bbox(0, 0, 0, 60, 60, 5)
    dx, dy, dz = engine._alignment_delta("flush_left", source, target)
    assert dx == 10.0  # target.x_min(0) - source.x_min(-10)
    assert dy == 30.0  # centers on target's Y span
    assert dz == 2.5  # centers on target's Z span


def test_alignment_delta_flush_right_matches_max_x_centers_other_axes():
    source = _bbox(-10, -10, -10, 10, 10, 10)
    target = _bbox(0, 0, 0, 60, 60, 5)
    dx, dy, dz = engine._alignment_delta("flush_right", source, target)
    assert dx == 50.0  # target.x_max(60) - source.x_max(10)
    assert dy == 30.0
    assert dz == 2.5


def test_alignment_delta_top_center_then_bottom_center_are_mirror_pair():
    source = _bbox(0, 0, 0, 10, 10, 10)
    target = _bbox(0, 0, 0, 60, 60, 5)
    top = engine._alignment_delta("top_center", source, target)
    bottom = engine._alignment_delta("bottom_center", source, target)
    assert top[0] == bottom[0] and top[1] == bottom[1]  # same X/Y centering
    assert top[2] != bottom[2]  # opposite stacking direction


def test_alignment_delta_unknown_type_raises():
    import pytest

    with pytest.raises(ValueError):
        engine._alignment_delta("sideways", _bbox(0, 0, 0, 1, 1, 1), _bbox(0, 0, 0, 1, 1, 1))


# --------------------------------------------------------------------------
# _extract_placement — mirrors _extract_bbox's marker-line parsing
# --------------------------------------------------------------------------


def test_extract_placement_parses_marker_line():
    stdout = f"some noise\n{engine._PLACEMENT_MARKER} [1.0, 2.5, -3.0]\nmore noise\n"
    assert engine._extract_placement(stdout) == [1.0, 2.5, -3.0]


def test_extract_placement_missing_marker_returns_none():
    assert engine._extract_placement("no marker here") is None


def test_extract_placement_non_list_payload_returns_none():
    stdout = f"{engine._PLACEMENT_MARKER} not_a_list"
    assert engine._extract_placement(stdout) is None

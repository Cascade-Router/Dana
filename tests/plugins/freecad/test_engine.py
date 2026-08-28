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


# --------------------------------------------------------------------------
# _pattern_offsets — batch_pattern_array's pure offset math
# --------------------------------------------------------------------------


def test_pattern_offsets_grid_includes_original_position_at_index_zero():
    """An 8x8 grid must produce 64 TOTAL placements (the '64 tiles in one
    tool call' case this tool exists for), with the source's own existing
    position as one of them, not 64 additional copies."""
    offsets = engine._pattern_offsets("grid", count_x=8, count_y=8, spacing_x=10.0, spacing_y=10.0)
    assert len(offsets) == 64
    assert offsets[0] == (0.0, 0.0, 0.0, 0.0)
    assert offsets[-1] == (70.0, 70.0, 0.0, 0.0)


def test_pattern_offsets_circular_distributes_evenly_on_the_radius():
    offsets = engine._pattern_offsets("circular", count=4, radius=10.0)
    assert len(offsets) == 4
    xs = [round(o[0], 6) for o in offsets]
    ys = [round(o[1], 6) for o in offsets]
    assert xs == [10.0, 0.0, -10.0, 0.0]
    assert ys == [0.0, 10.0, 0.0, -10.0]
    assert [o[3] for o in offsets] == [0.0, 90.0, 180.0, 270.0]


def test_pattern_offsets_rejects_unknown_pattern_type():
    import pytest

    with pytest.raises(ValueError):
        engine._pattern_offsets("hexagonal")


# --------------------------------------------------------------------------
# Boolean scripts — the Base/Tool/Shapes consumed by a Cut/Fuse/Common must
# be hidden in the saved document itself, not just when the fit-view macro
# happens to run on GUI open.
# --------------------------------------------------------------------------


def test_boolean_cut_script_hides_base_and_tool():
    script = engine._BOOLEAN_CUT_SCRIPT.format(
        base_path="base.FCStd", tool_path="tool.FCStd", name="Cut", out_path="out.FCStd", marker="OK"
    )
    assert "obj.Base.Visibility = False" in script
    assert "obj.Tool.Visibility = False" in script


def test_boolean_fuse_common_script_hides_all_shapes():
    script = engine._BOOLEAN_FUSE_COMMON_SCRIPT.format(
        base_path="base.FCStd",
        tool_path="tool.FCStd",
        feature_type="Part::MultiFuse",
        name="Fusion",
        out_path="out.FCStd",
        marker="OK",
    )
    assert "_shape_obj.Visibility = False" in script
    assert "for _shape_obj in obj.Shapes:" in script


# --------------------------------------------------------------------------
# _sketch_edge_specs — create_sketch_extrude's pure geometry prep
# --------------------------------------------------------------------------


def test_sketch_edge_specs_builds_line_and_arc_edges_in_order():
    segments = [
        {"type": "line", "to": [10, 0]},
        {"type": "arc", "to": [10, 10], "via": [13, 5]},
        {"type": "line", "to": [0, 0]},
    ]
    specs = engine._sketch_edge_specs(segments, start=(0.0, 0.0), plane="XY")
    assert specs == [
        ("line", ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
        ("arc", ((10.0, 0.0, 0.0), (13.0, 5.0, 0.0), (10.0, 10.0, 0.0))),
        ("line", ((10.0, 10.0, 0.0), (0.0, 0.0, 0.0))),
    ]


def test_sketch_edge_specs_embeds_points_per_work_plane():
    segments = [{"type": "line", "to": [4, 5]}]
    assert engine._sketch_edge_specs(segments, start=(1.0, 2.0), plane="XY")[0][1] == (
        (1.0, 2.0, 0.0),
        (4.0, 5.0, 0.0),
    )
    assert engine._sketch_edge_specs(segments, start=(1.0, 2.0), plane="XZ")[0][1] == (
        (1.0, 0.0, 2.0),
        (4.0, 0.0, 5.0),
    )
    assert engine._sketch_edge_specs(segments, start=(1.0, 2.0), plane="YZ")[0][1] == (
        (0.0, 1.0, 2.0),
        (0.0, 4.0, 5.0),
    )

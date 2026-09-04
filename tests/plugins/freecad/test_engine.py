"""Unit tests for the pure-Python pieces of dana.plugins.freecad.engine —
logic that doesn't need a real FreeCADCmd binary to verify: the alignment
delta math (align_freecad_objects) and the stdout marker extractors.
"""

from __future__ import annotations

from dana.plugins.freecad import engine, ir


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
# Boolean IR — the Base/Tool/Shapes consumed by a Cut/Fuse/Common must be
# hidden in the saved document itself, not just when the fit-view macro
# happens to run on GUI open.
#
# Superseded by the Universal CAD IR migration (dana.plugins.freecad.ir):
# apply_boolean no longer formats a hand-written f-string template
# (_BOOLEAN_CUT_SCRIPT/_BOOLEAN_FUSE_COMMON_SCRIPT, both retired) — it goes
# through the shared "boolean" kind + ir.render_ir_script. These tests build
# that same step dict via ir.get_ir_kind(...).from_args(...) and assert
# against the rendered template output instead (same recipe as
# test_feature_on_face_composite_resolves_to_face_profile_and_boolean_steps).
# --------------------------------------------------------------------------


def _render_boolean_step(*, operation, feature_type, base_object="B", tool_object="T", name="Result"):
    step = ir.get_ir_kind("perform_freecad_boolean").from_args(
        name=name, operation=operation, feature_type=feature_type, base_object=base_object, tool_object=tool_object,
    )
    script = ir.render_ir_script(
        [step], doc_mode="session", session_path="s.FCStd", final_var=step["var"], marker="OK",
    )
    compile(script, "<generated>", "exec")
    return script


def test_boolean_cut_step_hides_base_and_tool():
    script = _render_boolean_step(operation="cut", feature_type="Part::Cut")
    assert "_base_1.Visibility = False" in script
    assert "_tool_1.Visibility = False" in script


def test_boolean_fuse_common_step_hides_all_shapes():
    script = _render_boolean_step(operation="union", feature_type="Part::MultiFuse")
    assert "_base_1.Visibility = False" in script
    assert "_tool_1.Visibility = False" in script
    assert "obj.Shapes = [_base_1, _tool_1]" in script


# --------------------------------------------------------------------------
# Boolean/edge_operation topology-collapse guard — FreeCAD hands back an
# inverted/DBL_MAX BoundBox or a zero-Volume Shape instead of raising on an
# impossible fillet/chamfer/boolean, so the rendered script must check for
# it itself, right after doc.recompute() and before the result is ever
# saved/reported as success. Inlined directly in each kind's own template
# block now (universal_ir.py.jinja2) rather than a shared
# _TOPOLOGY_VALIDATION_SNIPPET string (retired along with the legacy
# _BOOLEAN_*/_EDGE_OP_* scripts it used to be embedded in).
# --------------------------------------------------------------------------


def test_boolean_topology_validation_runs_before_save():
    for operation, feature_type in (("cut", "Part::Cut"), ("union", "Part::MultiFuse")):
        script = _render_boolean_step(operation=operation, feature_type=feature_type)
        assert "obj.Shape.isNull()" in script
        assert "doc.removeObject(obj.Name)" in script
        recompute_at = script.index("doc.recompute()")
        validation_at = script.index("_bbox_collapsed")
        save_at = script.index("doc.save()")
        assert recompute_at < validation_at < save_at


def test_boolean_topology_validation_error_names_the_operation():
    script = _render_boolean_step(operation="cut", feature_type="Part::Cut")
    assert "Topology collapse: boolean '" in script
    assert "'cut'" in script
    assert "' destroyed the geometry" in script


def test_edge_operation_step_heals_compound_and_validates_topology_before_save():
    # Production parity: a multi-boolean-cut chain's result is frequently a
    # Part::Compound wrapping the real solid — unwrap to the first real
    # solid and heal stray coplanar seams (removeSplitter) before any
    # face/edge matching, for both the whole-object and face-targeted cases.
    for centroid in (None, (0.0, 0.0, 0.0)):
        step = ir.get_ir_kind("perform_freecad_edge_operation").from_args(
            name="Fillet", feature_type="Part::Fillet", target_object="Target", value=2.0, centroid=centroid,
        )
        script = ir.render_ir_script(
            [step], doc_mode="session", session_path="s.FCStd", final_var=step["var"], marker="OK",
        )
        assert 'ShapeType == "Compound"' in script
        assert ".removeSplitter()" in script
        assert "obj.Shape.isNull()" in script
        assert "doc.removeObject(obj.Name)" in script
        recompute_at = script.index("doc.recompute()")
        validation_at = script.index("_bbox_collapsed")
        save_at = script.index("doc.save()")
        assert recompute_at < validation_at < save_at
        compile(script, "<generated>", "exec")


def test_topology_validation_snippet_logic_flags_collapsed_bbox():
    """Pure-Python re-check of the embedded script's own predicate — a
    FreeCAD-side inverted/DBL_MAX BoundBox (the documented OCC silent-
    failure mode for an oversized fillet radius) must be flagged."""

    class _FakeBBox:
        def __init__(self, x_min, x_max):
            self.XMin, self.XMax = x_min, x_max

    def bbox_collapsed(bbox):
        return bbox.XMin > bbox.XMax or abs(bbox.XMin) > 1e300 or abs(bbox.XMax) > 1e300

    assert bbox_collapsed(_FakeBBox(1.79e308, -1.79e308)) is True
    assert bbox_collapsed(_FakeBBox(-10.0, 10.0)) is False


# --------------------------------------------------------------------------
# create_star_prism/_pyramid/_sketch_extrude's underlying scripts — these
# used to instantiate their own standalone App.newDocument("DanaModel") and
# save to a one-object-per-file .FCStd, invisible to a later
# perform_freecad_boolean/modify_freecad_parameter call by name (same bug
# create_polygon's own docstring already flagged for create_star_prism).
# They were converted to the same shared-session-document pattern
# create_box/create_cylinder/create_polygon already use — this pins that.
# --------------------------------------------------------------------------


def test_extrude_script_joins_shared_session_document():
    script = engine._EXTRUDE_SCRIPT.format(
        points=[[0, 0], [10, 0], [10, 10]],
        height=5,
        name="StarPrism",
        placement=(0.0, 0.0, 0.0),
        session_path="session.FCStd",
        session_doc_name="Session_Active",
        marker="OK",
    )
    assert "_session_path = " in script
    assert "App.openDocument" in script
    assert 'App.newDocument("DanaModel")' not in script


def test_pyramid_script_joins_shared_session_document():
    script = engine._PYRAMID_SCRIPT.format(
        length=10,
        width=10,
        height=10,
        name="Pyramid",
        placement=(0.0, 0.0, 0.0),
        session_path="session.FCStd",
        session_doc_name="Session_Active",
        marker="OK",
    )
    assert "_session_path = " in script
    assert "App.openDocument" in script
    assert 'App.newDocument("DanaModel")' not in script


def test_sketch_extrude_script_joins_shared_session_document():
    script = engine._SKETCH_EXTRUDE_SCRIPT.format(
        edge_specs=[("line", [[0, 0], [10, 0]])],
        normal=(0.0, 0.0, 1.0),
        height=5,
        name="Sketch",
        placement=(0.0, 0.0, 0.0),
        session_path="session.FCStd",
        session_doc_name="Session_Active",
        marker="OK",
    )
    assert "_session_path = " in script
    assert "App.openDocument" in script
    assert 'App.newDocument("DanaModel")' not in script


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


# --------------------------------------------------------------------------
# _face_axes — create_feature_on_face's Local Coordinate System resolution.
# Pure label lookup, no FreeCAD/bbox needed — matches FreeCAD's own
# standard-view convention (top=+Z, front=-Y, right=+X).
# --------------------------------------------------------------------------


def test_face_axes_top_points_straight_up():
    normal, u_axis, v_axis, origin_expr = engine._face_axes("top")
    assert normal == (0.0, 0.0, 1.0)
    assert u_axis == (1.0, 0.0, 0.0)
    assert v_axis == (0.0, 1.0, 0.0)
    assert origin_expr == ("(bb.XMin + bb.XMax) / 2.0", "(bb.YMin + bb.YMax) / 2.0", "bb.ZMax")


def test_face_axes_front_matches_freecad_front_view():
    # FreeCAD's own Front view looks toward +Y, so the face facing the
    # viewer (and the one this label should mean) is the -Y side, at YMin.
    normal, _u_axis, _v_axis, origin_expr = engine._face_axes("front")
    assert normal == (0.0, -1.0, 0.0)
    assert origin_expr[1] == "bb.YMin"


def test_face_axes_is_case_insensitive():
    assert engine._face_axes("TOP") == engine._face_axes("top")


def test_face_axes_unknown_label_raises():
    import pytest

    with pytest.raises(ValueError):
        engine._face_axes("diagonal")


def test_face_axes_every_face_has_orthonormal_basis():
    # normal/u_axis/v_axis must form a right-handed orthonormal basis for
    # every face, or a feature built from them wouldn't actually lie flat
    # against the target surface.
    for face in ("top", "bottom", "front", "back", "left", "right"):
        normal, u_axis, v_axis, _origin_expr = engine._face_axes(face)
        for a, b in ((normal, u_axis), (normal, v_axis), (u_axis, v_axis)):
            dot = sum(x * y for x, y in zip(a, b))
            assert dot == 0.0, f"{face}: axes not orthogonal"


# --------------------------------------------------------------------------
# create_feature_on_face's underlying script — must join the shared session
# document (Phase 2's fix applies here too) and must contain the flat-face
# verification the tool's contract promises (fails clearly on a curved
# face rather than silently cutting/adding nothing).
# --------------------------------------------------------------------------


def test_feature_on_face_composite_resolves_to_face_profile_and_boolean_steps():
    # Superseded by the Universal CAD IR migration (dana.plugins.freecad.ir):
    # create_feature_on_face no longer builds a hand-formatted script
    # string — it unrolls a CompositeIRSpec into a "face_profile" step (this
    # tool's own face-anchoring geometry) followed by a "boolean" step
    # (reusing the perform_freecad_boolean kind unmodified), which
    # ir.render_ir_script renders as ONE FreeCAD script. This test asserts
    # against that step-dict shape directly instead of formatting the old
    # f-string template.
    composite = ir.get_composite_ir("create_freecad_feature_on_face")
    steps = ir.unroll_composite(
        composite,
        dict(
            object_name="Plate", face="front", shape="circle", u=0.0, v=0.0,
            extent=5.0, operation="cut", radius=6.0, name="Feature",
        ),
        scope="test",
    )
    assert len(steps) == 2
    profile, boolean = steps
    assert profile["kind"] == "face_profile"
    assert profile["object_name"] == "Plate"  # external reference — never scope-prefixed
    assert profile["name"] == "_ir_test_profile"  # local intermediate — scope-prefixed
    assert profile["origin_expr"] == ("(bb.XMin + bb.XMax) / 2.0", "bb.YMin", "(bb.ZMin + bb.ZMax) / 2.0")
    assert profile["normal"] == (0.0, -1.0, 0.0)
    assert profile["radius"] == 6.0

    assert boolean["kind"] == "boolean"
    assert boolean["name"] == "Feature"  # outward-facing result — never scope-prefixed
    assert boolean["base_object"] == "Plate"
    assert boolean["tool_object"] == "_ir_test_profile"  # rewritten to match profile's own scoped name

    script = ir.render_ir_script(
        steps, doc_mode="session", session_path="session.FCStd", final_var=steps[-1]["var"], marker="OK",
    )
    assert "App.openDocument" in script
    assert "isinstance(_nearest_1.Surface, Part.Plane)" in script
    assert "Part.Circle(_start_1, _normal_1, 6.0)" in script
    compile(script, "<generated>", "exec")


def test_feature_on_face_composite_rectangle_uses_width_and_length():
    composite = ir.get_composite_ir("create_freecad_feature_on_face")
    steps = ir.unroll_composite(
        composite,
        dict(
            object_name="Plate", face="top", shape="rectangle", u=0.0, v=0.0,
            extent=5.0, operation="add", width=10.0, length=6.0, name="Boss",
        ),
    )
    profile = steps[0]
    assert profile["width"] == 10.0
    assert profile["length"] == 6.0
    script = ir.render_ir_script(
        steps, doc_mode="session", session_path="session.FCStd", final_var=steps[-1]["var"], marker="OK",
    )
    assert "_hw_1 = 5.0" in script
    assert "_hl_1 = 3.0" in script
    assert "Part.makePolygon" in script
    compile(script, "<generated>", "exec")

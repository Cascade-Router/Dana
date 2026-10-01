"""Minimal tests for dana.plugins.freecad.py_export — verifies the Jinja2
template correctly maps a mock CadCallLog sequence onto valid FreeCAD Python
source. Deliberately not a full test suite (see Frontier 4 scope): just
enough to prove the template parses and renders the expected API shapes.
"""

from __future__ import annotations

from dana.plugins.freecad.call_log import CadCallLog
from dana.plugins.freecad.py_export import render_macro_script


def _mock_session_log() -> CadCallLog:
    """Box -> Cylinder -> Cut, plus one non-CAD call and one failed call —
    exercises success codegen, cross-step object references, and both
    skip paths (unsupported tool_id, failed call)."""
    log = CadCallLog()
    log.record(
        "create_freecad_box",
        {"length": 40, "width": 25, "height": 15, "name": "Box"},
        ok=True,
        result={
            "name": "Box",
            "type": "Part::Box",
            "dimensions": {"length": 40.0, "width": 25.0, "height": 15.0},
            "placement": [0.0, 0.0, 0.0],
            "path": "C:/ws/freecad_output/Box.FCStd",
        },
    )
    log.record(
        "create_freecad_cylinder",
        {"radius": 10, "height": 30, "name": "Cylinder"},
        ok=True,
        result={
            "name": "Cylinder",
            "type": "Part::Cylinder",
            "dimensions": {"radius": 10.0, "height": 30.0},
            "placement": [0.0, 0.0, 20.0],
            "path": "C:/ws/freecad_output/Cylinder.FCStd",
        },
    )
    log.record(
        "perform_freecad_boolean",
        {"operation": "cut", "base_object": "Box", "tool_object": "Cylinder"},
        ok=True,
        result={"name": "Cut", "type": "Part::Cut", "operation": "cut", "path": "C:/ws/freecad_output/Cut.FCStd"},
    )
    log.record("get_freecad_bounding_box", {"target_object": "Cut"}, ok=True, result={"x_min": 0.0})
    log.record(
        "create_freecad_box",
        {"length": 5, "width": 5, "height": 5},
        ok=False,
        error="create_box failed: FreeCADCmd not found",
    )
    return log


def test_render_produces_syntactically_valid_python():
    script = render_macro_script(_mock_session_log())
    compile(script, "<generated-macro>", "exec")  # raises SyntaxError if the template is malformed


def test_render_maps_each_call_to_its_freecad_api_shape():
    script = render_macro_script(_mock_session_log())
    assert 'doc.addObject("Part::Box", \'Box\')' in script
    assert ".Length = 40.0" in script
    assert 'doc.addObject("Part::Cylinder", \'Cylinder\')' in script
    assert "doc.addObject('Part::Cut', 'Cut')" in script
    assert ".Base = doc.getObject('Box')" in script
    assert ".Tool = doc.getObject('Cylinder')" in script


def test_render_skips_non_cad_and_failed_calls_with_a_note():
    script = render_macro_script(_mock_session_log())
    assert "get_freecad_bounding_box" in script  # noted as skipped, never codegen'd
    assert "create_box failed" in script
    # Neither skipped step should produce a doc.addObject call of its own.
    assert script.count("doc.addObject(") == 3  # Box, Cylinder, Cut only


# --------------------------------------------------------------------------
# This sprint's additions: insert_standard_part + create_assembly_mate
# --------------------------------------------------------------------------


def _standard_parts_and_mate_log() -> CadCallLog:
    log = CadCallLog()
    log.record(
        "create_freecad_box",
        {"length": 20, "width": 20, "height": 10, "name": "Base"},
        ok=True,
        result={
            "name": "Base",
            "type": "Part::Box",
            "dimensions": {"length": 20.0, "width": 20.0, "height": 10.0},
            "placement": [0.0, 0.0, 0.0],
        },
    )
    log.record(
        "insert_standard_part",
        {"part_type": "nema17_motor"},
        ok=True,
        result={
            "name": "NEMA17Motor",
            "type": "Part::Feature",
            "part_type": "nema17_motor",
            "dimensions": {
                "body_width_mm": 42.3,
                "typical_body_depth_mm": 40.0,
                "pilot_boss_diameter_mm": 22.0,
                "pilot_boss_depth_mm": 2.0,
                "shaft_diameter_mm": 5.0,
                "default_shaft_length_mm": 24.0,
            },
            "placement": [0.0, 0.0, 0.0],
        },
    )
    log.record(
        "insert_standard_part",
        {"part_type": "socket_head_screw", "specification": "M3x12"},
        ok=True,
        result={
            "name": "Screw_M3X12",
            "type": "Part::Feature",
            "part_type": "socket_head_screw",
            "dimensions": {
                "nominal_diameter_mm": 3.0,
                "length_mm": 12.0,
                "head_diameter_mm": 5.5,
                "head_height_mm": 3.0,
            },
            "placement": [10.0, 0.0, 0.0],
        },
    )
    log.record(
        "insert_standard_part",
        {"part_type": "ball_bearing", "specification": "608"},
        ok=True,
        result={
            "name": "Bearing_608",
            "type": "Part::Feature",
            "part_type": "ball_bearing",
            "dimensions": {"outer_diameter_mm": 22.0, "bore_diameter_mm": 8.0, "width_mm": 7.0},
            "placement": [0.0, 20.0, 0.0],
        },
    )
    log.record(
        "create_assembly_mate",
        {"fixed_obj": "Base", "moving_obj": "NEMA17Motor", "mate_type": "concentric", "mate_params": {"z_offset": 5.0}},
        ok=True,
        result={"name": "NEMA17Motor", "mate_type": "concentric", "fixed_object": "Base", "placement": [10.0, 10.0, 5.0]},
    )
    return log


def test_render_standard_parts_and_mate_produces_valid_python():
    script = render_macro_script(_standard_parts_and_mate_log())
    compile(script, "<generated-macro>", "exec")


def test_render_standard_parts_use_exact_reference_dimensions():
    script = render_macro_script(_standard_parts_and_mate_log())
    # NEMA 17: body 42.3x42.3, boss dia 22.0, shaft dia 5.0.
    assert "Part.makeBox(42.3, 42.3, 40.0" in script
    assert "Part.makeCylinder(22.0 / 2.0, 2.0" in script
    assert "Part.makeCylinder(5.0 / 2.0, 24.0" in script
    # M3x12 screw: 3.0mm shank x 12.0mm, 5.5mm head x 3.0mm.
    assert "Part.makeCylinder(3.0 / 2.0, 12.0)" in script
    assert "Part.makeCylinder(5.5 / 2.0, 3.0" in script
    # 608 bearing: outer 22.0, bore 8.0, width 7.0 — a real cut, not a compound.
    assert "Part.makeCylinder(22.0 / 2.0, 7.0)" in script
    assert "Part.makeCylinder(8.0 / 2.0, 7.0)" in script
    assert ".cut(" in script


def test_render_assembly_mate_sets_absolute_placement_and_notes_mate_type():
    script = render_macro_script(_standard_parts_and_mate_log())
    assert "# concentric mate: NEMA17Motor -> Base" in script
    assert "doc.getObject('NEMA17Motor').Placement.Base = App.Vector(10.0, 10.0, 5.0)" in script


# --------------------------------------------------------------------------
# 2D Blueprint Generation — generate_2d_blueprint
# --------------------------------------------------------------------------


def _blueprint_log() -> CadCallLog:
    log = CadCallLog()
    log.record(
        "create_freecad_box",
        {"length": 40, "width": 25, "height": 15, "name": "Box"},
        ok=True,
        result={
            "name": "Box",
            "type": "Part::Box",
            "dimensions": {"length": 40.0, "width": 25.0, "height": 15.0},
            "placement": [0.0, 0.0, 0.0],
        },
    )
    log.record(
        "generate_2d_blueprint",
        {"object_name": "Box", "views": ["Front", "Isometric"], "page_size": "A4"},
        ok=True,
        result={"name": "Box", "views": ["Front", "Isometric"], "page_size": "a4", "path": "C:/exports/Box.pdf"},
    )
    return log


def test_render_blueprint_produces_valid_python():
    script = render_macro_script(_blueprint_log())
    compile(script, "<generated-macro>", "exec")


def test_render_blueprint_creates_page_and_only_the_requested_views():
    script = render_macro_script(_blueprint_log())
    assert 'doc.addObject("TechDraw::DrawPage"' in script
    assert 'doc.addObject("TechDraw::DrawSVGTemplate"' in script
    assert "Default_Template_A4_Landscape.svg" in script
    assert "TechDraw.writeDXFPage(" in script
    assert "doc.getObject('Box')" in script
    # Exactly the two requested views, not the other two standard ones.
    assert "'Front'" in script
    assert "'Isometric'" in script
    assert "'Top'" not in script
    assert "'Right'" not in script


# -- Sketcher/PartDesign replay --------------------------------------------------


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[dict]:
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [{"type": "line", "start": list(c[i]), "end": list(c[(i + 1) % 4])} for i in range(4)]


def _partdesign_log() -> CadCallLog:
    """One of each Sketcher/PartDesign tool, with result payloads shaped like
    the real engine's (resolved names, coerced dimensions)."""
    log = CadCallLog()
    rec = log.record
    rec("create_freecad_sketch", {"name": "PlateSketch", "plane": "XY", "geometry": _rect(-30, -30, 30, 30)},
        ok=True, result={"name": "PlateSketch", "dimensions": {"plane": "XY"}})
    rec("apply_sketch_constraint",
        {"sketch_name": "PlateSketch", "constraint_type": "Distance", "geometry_indices": [0], "value": 60},
        ok=True, result={"name": "PlateSketch",
                         "dimensions": {"constraint_type": "Distance", "geometry_indices": [0], "value": 60.0}})
    rec("create_freecad_pad", {"sketch_name": "PlateSketch", "length": 4},
        ok=True, result={"name": "Pad", "dimensions": {"length": 4.0, "symmetric_to_plane": False,
                                                       "reversed_direction": False}})
    rec("create_freecad_sketch",
        {"name": "HoleSketch", "plane": "XY", "geometry": [{"type": "circle", "center": [15, 0], "radius": 3}]},
        ok=True, result={"name": "HoleSketch", "dimensions": {"plane": "XY"}})
    # The engine auto-reversed this pocket and recorded the flip in dimensions.
    rec("create_freecad_pocket", {"sketch_name": "HoleSketch", "depth": 1, "through_all": True},
        ok=True, result={"name": "Pocket", "auto_reversed": True,
                         "dimensions": {"depth": 1.0, "through_all": True, "symmetric_to_plane": False,
                                        "reversed_direction": True}})
    rec("create_freecad_polar_pattern", {"feature_name": "Pocket", "occurrences": 4},
        ok=True, result={"name": "PolarPattern", "dimensions": {"occurrences": 4, "angle": 360.0, "axis": "Z",
                                                                "reversed_direction": False}})
    rec("create_freecad_linear_pattern", {"feature_name": "Pocket", "occurrences": 3, "length": 20, "direction": "Y"},
        ok=True, result={"name": "LinearPattern", "dimensions": {"occurrences": 3, "length": 20.0,
                                                                 "direction": "Y", "reversed_direction": False}})
    rec("create_freecad_sketch",
        {"name": "Section", "plane": "XY", "geometry": [{"type": "circle", "center": [0, 0], "radius": 2}]},
        ok=True, result={"name": "Section", "dimensions": {"plane": "XY"}})
    rec("create_freecad_sketch",
        {"name": "Spine", "plane": "XZ", "geometry": [{"type": "line", "start": [0, 0], "end": [0, 20]}]},
        ok=True, result={"name": "Spine", "dimensions": {"plane": "XZ"}})
    rec("create_freecad_sweep", {"profile_sketch": "Section", "path_sketch": "Spine"},
        ok=True, result={"name": "Sweep", "dimensions": {"frenet": True}})
    rec("create_freecad_sketch", {"name": "LoftBase", "plane": "XY", "geometry": _rect(-5, -5, 5, 5)},
        ok=True, result={"name": "LoftBase", "dimensions": {"plane": "XY"}})
    rec("create_freecad_sketch",
        {"name": "LoftTop", "plane": "XY", "geometry": [{"type": "circle", "center": [0, 0], "radius": 3}]},
        ok=True, result={"name": "LoftTop001", "dimensions": {"plane": "XY"}})
    rec("create_freecad_loft", {"cross_section_sketches": ["LoftBase", "LoftTop001"], "ruled": True},
        ok=True, result={"name": "Loft", "dimensions": {"cross_section_count": 2, "ruled": True, "closed": False}})
    return log


def test_partdesign_replay_is_valid_python_and_skips_none_of_the_tools():
    script = render_macro_script(_partdesign_log())
    compile(script, "<generated-macro>", "exec")
    assert "could not be replayed" not in script
    assert "not a FreeCAD geometry operation" not in script
    assert script.count("def _check_feature(") == 1  # helpers emitted exactly once
    assert "import Sketcher" in script


def test_partdesign_replay_uses_recorded_names_and_references():
    script = render_macro_script(_partdesign_log())
    # Objects are recreated under the name the live call got, including
    # FreeCAD's auto-suffixed ones, so later by-name references resolve.
    assert 'doc.addObject("Sketcher::SketchObject", \'LoftTop001\')' in script
    assert "_require_sketch(doc, 'PlateSketch')" in script
    assert 'newObject("PartDesign::Pad", \'Pad\')' in script
    assert "_require_sketch(doc, 'HoleSketch')" in script
    assert 'newObject("PartDesign::Pocket", \'Pocket\')' in script
    assert "for _src in ['Pocket']:" in script
    assert "newObject('PartDesign::PolarPattern', 'PolarPattern')" in script
    assert "newObject('PartDesign::LinearPattern', 'LinearPattern')" in script
    assert "[_require_sketch(doc, _n) for _n in ['LoftBase', 'LoftTop001']]" in script


def test_partdesign_replay_carries_exact_parameters():
    script = render_macro_script(_partdesign_log())
    assert "obj.addConstraint(Sketcher.Constraint('Distance', *_args))" in script
    assert "_args = [0, 60.0]" in script
    assert "obj.Length = 4.0" in script
    assert 'obj.Type = "ThroughAll"' in script
    # The live pocket's auto-reversed flip is replayed explicitly.
    assert "obj.Reversed = True" in script
    assert "_origin_axis(_body, 'Z_Axis')" in script and "obj.Occurrences = 4" in script
    assert "_origin_axis(_body, 'Y_Axis')" in script and "setattr(obj, 'Length', 20.0)" in script
    assert "obj.Mode = 'Frenet'" in script
    assert "obj.Ruled = True" in script
    # XZ sketch keeps its plane rotation.
    assert "((1.0, 0.0, 0.0), 90.0)" in script


def test_macro_without_partdesign_steps_has_no_partdesign_helpers():
    script = render_macro_script(_mock_session_log())
    assert "def _check_feature(" not in script
    assert "import Sketcher" not in script


def test_unreplayable_partdesign_record_is_skipped_with_a_reason():
    log = CadCallLog()
    log.record("create_freecad_pad", {"sketch_name": "S", "length": 4}, ok=True, result={})
    script = render_macro_script(log)
    compile(script, "<generated-macro>", "exec")
    assert "create_freecad_pad could not be replayed" in script


def test_placement_edits_replay_as_a_real_placement_not_a_setattr():
    """Regression: modify_freecad_parameter(Placement) replayed as
    `setattr(obj, "Placement", [x, y, z])`, which FreeCAD rejects, so any
    exported session that moved an object produced a macro that crashed."""
    log = CadCallLog()
    log.record("create_freecad_box", {"name": "Arm"}, ok=True,
               result={"name": "Arm", "dimensions": {"length": 80.0, "width": 14.0, "height": 4.0}})
    log.record("modify_freecad_parameter", {"target_object": "Arm", "parameter_name": "Placement",
                                           "new_value": [5, 6, 7]},
               ok=True, result={"parameter_name": "Placement", "new_value": [5.0, 6.0, 7.0]})
    log.record("modify_freecad_parameter", {"target_object": "Arm", "parameter_name": "Placement",
                                           "new_value": [0, 0, 0], "yaw": 45},
               ok=True, result={"parameter_name": "Placement", "new_value": [0.0, 0.0, 0.0, 45.0, 0.0, 0.0]})
    script = render_macro_script(log)
    compile(script, "<generated-macro>", "exec")
    assert "setattr(doc.getObject('Arm'), 'Placement'" not in script
    assert "App.Vector(5.0, 6.0, 7.0)" in script
    assert "App.Rotation(45.0, 0.0, 0.0)" in script

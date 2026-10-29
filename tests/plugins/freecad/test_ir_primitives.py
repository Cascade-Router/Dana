"""Universal CAD IR coverage for pyramid / pipe / helix: one step dict per
call, rendered the same way by live execution, the macro export and the
parametric rebuild."""

from __future__ import annotations

from dana.plugins.freecad import ir
from dana.plugins.freecad.call_log import CadCallLog
from dana.plugins.freecad.py_export import build_replay_steps, render_macro_script


def _render(step: dict) -> str:
    return ir.render_ir_script([step], doc_mode="session", session_path="s.FCStd", marker="OK")


def test_pyramid_renders_as_a_faceted_solid_in_the_session_document() -> None:
    script = _render(ir.get_ir_kind("create_freecad_pyramid").from_args(name="Spire", length=20, width=10, height=30))
    assert "App.openDocument" in script and 'doc.addObject("Part::Feature", \'Spire\')' in script
    assert "Part.Shell" in script


def test_pipe_helpers_are_named_after_the_pipe() -> None:
    kind = ir.get_ir_kind("create_freecad_pipe")
    straight = _render(kind.from_args(name="Rod", pipe_radius=2, path_type="straight", length_or_angle=40))
    arc = _render(kind.from_args(name="Elbow", pipe_radius=4, path_type="arc", length_or_angle=90))
    assert "'Rod_Profile'" in straight and "'Rod_Path'" in straight and "Part::Line" in straight
    assert "Frenet = False" in straight
    assert "'Elbow_Path'" in arc and "Angle2 = 90.0" in arc and "Frenet = True" in arc
    assert "Radius = 20.0" in arc  # bend radius follows ir.pipe_arc_radius


def test_helix_angle_offset_is_a_rotation_about_z() -> None:
    script = _render(
        ir.get_ir_kind("create_freecad_helix").from_args(
            name="Coil", coil_radius=10, pitch=5, height=20, pipe_radius=1.5, angle_offset=45
        )
    )
    assert "App.Rotation(45.0, 0.0, 0.0)" in script
    assert "import math" in script


def test_records_replay_through_ir_and_into_the_macro() -> None:
    log = CadCallLog()
    log.record("create_freecad_pyramid", {}, ok=True,
               result={"name": "Spire", "dimensions": {"length": 20, "width": 20, "height": 30}})
    log.record("create_freecad_pipe", {}, ok=True,
               result={"name": "Elbow", "dimensions": {"pipe_radius": 4, "path_type": "arc", "length_or_angle": 90}})
    log.record("create_freecad_helix", {}, ok=True,
               result={"name": "Coil", "placement": [0, 60, 0], "dimensions": {
                   "coil_radius": 10, "pitch": 5, "height": 20, "pipe_radius": 1.5, "angle_offset": 30}})
    steps, skipped = build_replay_steps(log.records)
    assert skipped == [] and [s["kind"] for s in steps] == ["pyramid", "pipe", "helix"]
    macro = render_macro_script(log)
    assert "'Spire'" in macro and "'Elbow_Path'" in macro
    assert "App.Rotation(30.0, 0.0, 0.0)" in macro  # angle_offset used to be dropped from macros

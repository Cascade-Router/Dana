"""Universal CAD IR support for the eight Sketcher/PartDesign tools
(dana.plugins.freecad.ir's "sketch" ... "loft" kinds), and their use by the
macro exporter and the Composite Skill Compiler.

Records come from the mock engine, whose result payloads have the same shape
as the real engine's. No FreeCAD needed; tests/test_macro_parity.py runs the
same sequences through real FreeCADCmd.
"""

from __future__ import annotations

import ast
import json
from typing import Any

import pytest

from dana.platform.mock import MockFreeCADEngine
from dana.plugins.freecad import engine, ir, py_export, skill_compiler
from dana.plugins.freecad.call_log import CadCallLog

_TOOL_KINDS = {
    "create_freecad_sketch": "sketch",
    "apply_sketch_constraint": "sketch_constraint",
    "create_freecad_pad": "pad",
    "create_freecad_pocket": "pocket",
    "create_freecad_polar_pattern": "polar_pattern",
    "create_freecad_linear_pattern": "linear_pattern",
    "create_freecad_sweep": "sweep",
    "create_freecad_loft": "loft",
}


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[dict[str, Any]]:
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [{"type": "line", "start": list(c[i]), "end": list(c[(i + 1) % 4])} for i in range(4)]


class _Session:
    """Drives the mock engine and records each call the way
    dana.core.react_dispatch.dispatch_tool_call does."""

    def __init__(self) -> None:
        self.mock = MockFreeCADEngine()
        self.log = CadCallLog()
        self.log.record("create_plan", {"goal": "test"}, ok=True, result={"ok": True})

    def call(self, tool_id: str, method: str, **arguments: Any) -> dict[str, Any]:
        result = getattr(self.mock, method)(**arguments)
        if isinstance(result, str):
            result = json.loads(result)
        assert result.get("ok") is True, result
        self.log.record(tool_id, arguments, ok=True, result=result)
        return result


def _plate_session() -> tuple[_Session, dict[str, str]]:
    """Sketch -> constraint -> pad -> hole pocket -> boss pad -> ONE polar
    pattern of [boss pad, hole pocket] -> linear pattern of the hole."""
    s = _Session()
    s.call("create_freecad_sketch", "create_sketch", name="Plate", plane="XY", geometry=_rect(-30, -30, 30, 30))
    s.call(
        "apply_sketch_constraint", "apply_sketch_constraint",
        sketch_name="Plate", constraint_type="Horizontal", geometry_indices=[0],
    )
    pad = s.call("create_freecad_pad", "create_pad", sketch_name="Plate", length=4)["name"]
    s.call("create_freecad_sketch", "create_sketch", name="Hole", plane="XY",
           geometry=[{"type": "circle", "center": [15, 0], "radius": 3}])
    pocket = s.call("create_freecad_pocket", "create_pocket", sketch_name="Hole", depth=1, through_all=True)["name"]
    s.call("create_freecad_sketch", "create_sketch", name="Boss", plane="XY",
           geometry=[{"type": "circle", "center": [20, 20], "radius": 4}])
    boss = s.call("create_freecad_pad", "create_pad", sketch_name="Boss", length=6)["name"]
    polar = s.call("create_freecad_polar_pattern", "create_polar_pattern", feature_name=[boss, pocket], occurrences=4)
    linear = s.call("create_freecad_linear_pattern", "create_linear_pattern",
                    feature_name=pocket, occurrences=3, length=12, direction="Y")
    return s, {"pad": pad, "pocket": pocket, "boss": boss, "polar": polar["name"], "linear": linear["name"]}


def _sweep_loft_session() -> _Session:
    s = _Session()
    s.call("create_freecad_sketch", "create_sketch", name="Section", plane="XY",
           geometry=[{"type": "circle", "center": [0, 0], "radius": 2}])
    s.call("create_freecad_sketch", "create_sketch", name="Spine", plane="XZ",
           geometry=[{"type": "line", "start": [0, 0], "end": [0, 20]}])
    s.call("create_freecad_sweep", "create_sweep", profile_sketch="Section", path_sketch="Spine", frenet=False)
    s.call("create_freecad_sketch", "create_sketch", name="Base", plane="XY", geometry=_rect(-10, -10, 10, 10))
    s.call("create_freecad_sketch", "create_sketch", name="Top", plane="XY",
           geometry=[{"type": "circle", "center": [0, 0], "radius": 5}])
    s.call("create_freecad_loft", "create_loft", cross_section_sketches=["Base", "Top"], ruled=True)
    return s


def _steps(session: _Session) -> list[dict[str, Any]]:
    steps, skipped, _deps = skill_compiler.build_steps_with_dependencies(
        skill_compiler.slice_records_since_plan(session.log.records)
    )
    assert skipped == []
    return steps


# -- registration & step schema ------------------------------------------------------


@pytest.mark.parametrize(("tool_id", "kind"), sorted(_TOOL_KINDS.items()))
def test_every_sketch_and_feature_tool_is_a_registered_ir_kind(tool_id: str, kind: str) -> None:
    spec = ir.get_ir_kind(tool_id)
    assert spec is not None and spec.kind == kind
    assert kind in ir.PARTDESIGN_KINDS
    assert set(_TOOL_KINDS) == engine.PARTDESIGN_TOOL_IDS


def test_steps_are_generic_parametric_nodes_not_freecad_code() -> None:
    session, names = _plate_session()
    steps = _steps(session)
    assert [s["kind"] for s in steps] == [
        "sketch", "sketch_constraint", "pad", "sketch", "pocket", "sketch", "pad", "polar_pattern", "linear_pattern",
    ]
    for step in steps:
        assert "code" not in step  # FreeCAD code is generated at render time, never stored
        json.dumps(step)  # plain data
        assert ast.literal_eval(repr(step)) == step  # survives the generated skill module's repr()

    by_kind = {s["kind"]: s for s in reversed(steps)}  # first step of each kind
    assert by_kind["pad"]["length"] == 4.0
    assert by_kind["pocket"]["through_all"] is True
    assert by_kind["polar_pattern"]["features"] == [names["boss"], names["pocket"]]
    assert by_kind["polar_pattern"]["occurrences"] == 4
    # A single-string feature_name becomes a one-element list.
    assert by_kind["linear_pattern"]["features"] == [names["pocket"]]
    assert by_kind["linear_pattern"]["direction"] == "Y"
    assert by_kind["sketch_constraint"]["name"] == "Plate"


def test_sweep_and_loft_track_profile_spine_and_sections() -> None:
    steps = _steps(_sweep_loft_session())
    sweep = next(s for s in steps if s["kind"] == "sweep")
    loft = next(s for s in steps if s["kind"] == "loft")
    assert (sweep["profile_sketch"], sweep["path_sketch"], sweep["frenet"]) == ("Section", "Spine", False)
    assert (loft["sections"], loft["ruled"], loft["closed"]) == (["Base", "Top"], True, False)


@pytest.mark.parametrize("kind", ["pad", "pocket"])
@pytest.mark.parametrize("flag", [False, True])
def test_reversed_flag_maps_into_the_ir_and_the_generated_code(kind: str, flag: bool) -> None:
    tool_id = f"create_freecad_{kind}"
    size = "length" if kind == "pad" else "depth"
    rec = CadCallLog().record(
        tool_id, {"sketch_name": "S", size: 5}, ok=True,
        result={"name": kind.title(), "dimensions": {size: 5.0, "reversed_direction": flag}},
    )
    step = ir.get_ir_kind(tool_id).from_record(rec, 1)
    assert step["reversed"] is flag
    assert f"obj.Reversed = {flag!r}" in engine.partdesign_step_code(step)


def test_pocket_replays_the_engines_auto_reversed_flip() -> None:
    # The live pocket flipped itself (auto_reversed) and recorded that in
    # dimensions; the raw request said reversed_direction=False.
    rec = CadCallLog().record(
        "create_freecad_pocket", {"sketch_name": "Hole", "depth": 1, "reversed_direction": False}, ok=True,
        result={"name": "Pocket", "dimensions": {"depth": 1.0, "reversed_direction": True}, "auto_reversed": True},
    )
    assert ir.get_ir_kind("create_freecad_pocket").from_record(rec, 1)["reversed"] is True


# -- one code generator: live engine == macro == compiled skill -------------------------


def test_macro_and_skill_render_the_engines_own_code_for_every_step() -> None:
    for session in (_plate_session()[0], _sweep_loft_session()):
        records = skill_compiler.slice_records_since_plan(session.log.records)
        macro = py_export.render_macro_script(session.log)
        skill_script = ir.render_ir_script(
            _steps(session), doc_mode="session", session_path="S.FCStd", marker=engine._OK_MARKER
        )
        ast.parse(macro)
        ast.parse(skill_script)
        for rec in records:
            code = engine.partdesign_replay_code(rec.tool_id, rec.arguments, rec.result)
            assert code and code in macro and code in skill_script, rec.tool_id
        assert "import Sketcher" in macro and "import Sketcher" in skill_script
        assert "def _check_feature" in macro and "def _check_feature" in skill_script


def test_partdesign_replay_code_matches_the_live_engine_script() -> None:
    """The code the live tool executes (engine._partdesign_script) is the code
    the IR emits for the same call."""
    rec = CadCallLog().record(
        "create_freecad_pad", {"sketch_name": "Plate", "length": 4}, ok=True,
        result={"name": "Pad", "dimensions": {"length": 4.0}},
    )
    live_script, _ = engine._partdesign_script(
        "create_pad", sketch_name="Plate", length=4.0, symmetric=False, reversed_direction=False
    )
    assert engine.partdesign_replay_code(rec.tool_id, rec.arguments, rec.result) in live_script


def test_unrecoverable_record_is_skipped_not_rendered_broken() -> None:
    log = CadCallLog()
    log.record("create_freecad_pad", {"sketch_name": "S", "length": 4}, ok=True, result={"dimensions": {}})
    steps, skipped = py_export.build_replay_steps(log.records)
    assert steps == [] and "could not be replayed" in skipped[0]


# -- skill compilation: references, prefixing, parameters -------------------------------


def test_name_prefix_rewrites_single_and_list_references() -> None:
    session, names = _plate_session()
    prefixed = skill_compiler._apply_name_prefix(_steps(session), "p1")
    by_kind = {s["kind"]: s for s in prefixed}
    assert by_kind["pocket"]["sketch_name"] == "p1_Hole"
    assert by_kind["polar_pattern"]["features"] == [f"p1_{names['boss']}", f"p1_{names['pocket']}"]
    assert by_kind["linear_pattern"]["features"] == [f"p1_{names['pocket']}"]
    assert by_kind["sketch_constraint"]["name"] == "p1_Plate"

    lofted = skill_compiler._apply_name_prefix(_steps(_sweep_loft_session()), "p2")
    sweep = next(s for s in lofted if s["kind"] == "sweep")
    loft = next(s for s in lofted if s["kind"] == "loft")
    assert (sweep["profile_sketch"], sweep["path_sketch"]) == ("p2_Section", "p2_Spine")
    assert loft["sections"] == ["p2_Base", "p2_Top"]


def test_unroll_steps_rewrites_list_references_when_inlined() -> None:
    steps = _steps(_sweep_loft_session())
    unrolled = ir.unroll_steps(steps, scope="abc", reference_fields=skill_compiler._REFERENCE_FIELDS)
    loft = unrolled[-1]
    assert loft["sections"] == ["_ir_abc_Base", "_ir_abc_Top"]
    assert loft["name"] == steps[-1]["name"]  # the outward-facing result keeps its name


def test_compiled_skill_exposes_feature_dimensions_and_substitutes_them() -> None:
    session, _ = _plate_session()
    records = skill_compiler.slice_records_since_plan(session.log.records)
    out = skill_compiler.compile_call_log_to_skill(records, skill_name="plate_skill", description="plate")
    assert out["ok"] is True and out["skipped"] == []
    params = set(out["schema"]["function"]["parameters"]["properties"])
    # pad lengths, pattern counts and spacing; NOT the through-all pocket's ignored depth
    assert {"pad_2_length", "pad_6_length", "polar_pattern_7_occurrences",
            "linear_pattern_8_occurrences", "linear_pattern_8_length"} <= params
    assert not any(p.startswith("pocket_") for p in params)

    captured: dict[str, str] = {}

    def fake_run(script: str, **_kw: Any) -> dict[str, Any]:
        captured["script"] = script
        return {"ok": True, "resolved_name": "x", "bounding_box": None}

    namespace: dict[str, Any] = {}
    exec(compile(out["python_code"], "plate_skill.py", "exec"), namespace)
    original = skill_compiler._run_freecad_script
    skill_compiler._run_freecad_script = fake_run
    try:
        assert namespace["run"]({"pad_2_length": 9.5, "polar_pattern_7_occurrences": 6})["ok"] is True
    finally:
        skill_compiler._run_freecad_script = original
    script = captured["script"]
    ast.parse(script)
    assert "obj.Length = 9.5" in script
    assert "obj.Occurrences = 6" in script


@pytest.mark.parametrize("prefix", ["5e7209b0", "0abc"])
def test_digit_leading_prefix_never_reaches_freecad(prefix: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """FreeCAD renames "5e72_Box" to "_5e72_Box", after which every by-name
    lookup of "5e72_Box" fails — the prefix must start with a letter."""
    captured: dict[str, str] = {}

    def fake_run(script: str, **_kw: Any) -> dict[str, Any]:
        captured["script"] = script
        return {"ok": True, "resolved_name": "x", "bounding_box": None}

    monkeypatch.setattr(skill_compiler, "_run_freecad_script", fake_run)
    result = skill_compiler.execute_compiled_steps(_steps(_sweep_loft_session()), name_prefix=prefix)
    assert result["name_prefix"] == f"s{prefix}"
    assert f"'s{prefix}_Section'" in captured["script"]
    assert f"'{prefix}_" not in captured["script"]


def test_skill_generated_from_records_with_a_bad_parameter_fails_cleanly() -> None:
    step = ir.get_ir_kind("create_freecad_pad").from_args(sketch_name="S", length=4)
    with pytest.raises(ValueError, match="length must be positive"):
        engine.partdesign_step_code({**step, "length": -1})

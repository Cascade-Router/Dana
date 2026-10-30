"""Three-way parity for Sketcher/PartDesign sequences, in real FreeCADCmd:
the live session the engine built, the exported macro (py_export), and the
compiled skill (skill_compiler -> Universal IR) run into a fresh session
must produce the same feature tree — same feature types in the same order,
all valid, the same Body volume and the same Tip feature type.

Skipped where FreeCAD isn't installed (e.g. CI); tests/test_ir_compiler.py
covers the same paths without FreeCAD.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path
from typing import Any

import pytest

from dana.plugins.freecad import engine, py_export, skill_compiler
from dana.plugins.freecad.call_log import CadCallLog
from dana.session_context import DEFAULT_SESSION_ID, set_session_id

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        engine.detect_freecadcmd() is None,
        reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
    ),
]

_TREE_PROBE = """
import json as _json
_body = [o for o in doc.Objects if o.TypeId == "PartDesign::Body"][-1]
_feats = [o for o in _body.Group if o.TypeId.startswith("PartDesign::") and o.TypeId != "PartDesign::Body"]
_out = {
    "features": [o.TypeId for o in _feats],
    "valid": all(o.isValid() for o in _feats),
    "volume": _body.Shape.Volume,
    "tip": _body.Tip.TypeId,
}
print(%r + "_TREE " + _json.dumps(_out))
""" % engine._OK_MARKER


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(engine, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    set_session_id(f"parity-{uuid.uuid4().hex[:8]}")
    yield
    set_session_id(DEFAULT_SESSION_ID)


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[dict[str, Any]]:
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [{"type": "line", "start": list(c[i]), "end": list(c[(i + 1) % 4])} for i in range(4)]


class _Live:
    def __init__(self) -> None:
        self.log = CadCallLog()
        self.log.record("create_plan", {"goal": "parity"}, ok=True, result={"ok": True})

    def __call__(self, tool_id: str, raw: str, **arguments: Any) -> dict[str, Any]:
        result = json.loads(raw)
        assert result.get("ok") is True, result
        self.log.record(tool_id, arguments, ok=True, result=result)
        return result


def _tree(script: str) -> dict[str, Any]:
    result = engine._run_freecad_script(script + _TREE_PROBE)
    assert result["ok"], result["error"]
    match = re.search(re.escape(engine._OK_MARKER + "_TREE ") + r"(\{.*\})", result["stdout"])
    assert match, result["stdout"][-2000:]
    return json.loads(match.group(1))


def _live_tree() -> dict[str, Any]:
    return _tree(f"import FreeCAD as App\ndoc = App.openDocument({str(engine._session_document_path())!r})\n")


def _skill_tree(log: CadCallLog) -> dict[str, Any]:
    records = skill_compiler.slice_records_since_plan(log.records)
    compiled = skill_compiler.compile_call_log_to_skill(records, skill_name="parity_skill", description="parity")
    assert compiled["ok"] is True and compiled["skipped"] == [], compiled
    namespace: dict[str, Any] = {}
    exec(compile(compiled["python_code"], "parity_skill.py", "exec"), namespace)
    set_session_id(f"parity-skill-{uuid.uuid4().hex[:8]}")  # a fresh, empty session document
    # A digit-leading prefix is the case FreeCAD would otherwise silently rename.
    result = namespace["run"]({"name_prefix": "9f" + uuid.uuid4().hex[:6]})
    assert result["ok"] is True, result
    return _live_tree()


def _assert_parity(live: dict[str, Any], *others: dict[str, Any]) -> None:
    assert live["valid"] is True and live["volume"] > 0
    for other in others:
        assert other["features"] == live["features"]
        assert other["valid"] is True
        assert other["tip"] == live["tip"]
        assert math.isclose(other["volume"], live["volume"], rel_tol=1e-9)


def test_pad_pocket_and_multi_feature_patterns_round_trip() -> None:
    live = _Live()
    d = 74.25 / math.sqrt(2)
    live("create_freecad_sketch", engine.create_sketch("Plate", "XY", _rect(-30, -30, 30, 30)),
         name="Plate", plane="XY", geometry=_rect(-30, -30, 30, 30))
    live("apply_sketch_constraint", engine.apply_sketch_constraint("Plate", "Horizontal", [0]),
         sketch_name="Plate", constraint_type="Horizontal", geometry_indices=[0])
    live("create_freecad_pad", engine.create_pad("Plate", 4), sketch_name="Plate", length=4)
    motor = [{"type": "circle", "center": [d, d], "radius": 14}]
    live("create_freecad_sketch", engine.create_sketch("Motor", "XY", motor), name="Motor", plane="XY", geometry=motor)
    boss = live("create_freecad_pad", engine.create_pad("Motor", 6), sketch_name="Motor", length=6)["name"]
    shaft = [{"type": "circle", "center": [d, d], "radius": 4}]
    live("create_freecad_sketch", engine.create_sketch("Shaft", "XY", shaft), name="Shaft", plane="XY", geometry=shaft)
    hole = live("create_freecad_pocket", engine.create_pocket("Shaft", 1, through_all=True),
                sketch_name="Shaft", depth=1, through_all=True)["name"]
    live("create_freecad_polar_pattern", engine.create_polar_pattern([boss, hole], 4),
         feature_name=[boss, hole], occurrences=4)
    slot = [{"type": "circle", "center": [0, -20], "radius": 2}]
    live("create_freecad_sketch", engine.create_sketch("Slot", "XY", slot), name="Slot", plane="XY", geometry=slot)
    slot_hole = live("create_freecad_pocket", engine.create_pocket("Slot", 1, through_all=True),
                     sketch_name="Slot", depth=1, through_all=True)["name"]
    live("create_freecad_linear_pattern", engine.create_linear_pattern(slot_hole, 3, 20, direction="Y"),
         feature_name=slot_hole, occurrences=3, length=20, direction="Y")

    live_tree = _live_tree()
    assert live_tree["features"].count("PartDesign::PolarPattern") == 1
    macro_tree = _tree(py_export.render_macro_script(live.log))
    _assert_parity(live_tree, macro_tree, _skill_tree(live.log))


def test_sweep_and_loft_round_trip() -> None:
    live = _Live()
    section = [{"type": "circle", "center": [0, 0], "radius": 2}]
    spine = [{"type": "line", "start": [0, 0], "end": [0, 20]}]
    live("create_freecad_sketch", engine.create_sketch("Section", "XY", section),
         name="Section", plane="XY", geometry=section)
    live("create_freecad_sketch", engine.create_sketch("Spine", "XZ", spine), name="Spine", plane="XZ", geometry=spine)
    live("create_freecad_sweep", engine.create_sweep("Section", "Spine"), profile_sketch="Section", path_sketch="Spine")
    live("create_freecad_sketch", engine.create_sketch("Base", "XY", _rect(-10, -10, 10, 10)),
         name="Base", plane="XY", geometry=_rect(-10, -10, 10, 10))
    top = [{"type": "circle", "center": [0, 0], "radius": 5}]
    live("create_freecad_sketch", engine.create_sketch("Top", "XY", top), name="Top", plane="XY", geometry=top)
    live("modify_freecad_parameter", engine.modify_parameter("Top", "Placement", [0, 0, 30]),
         target_object="Top", parameter_name="Placement", new_value=[0, 0, 30])
    live("create_freecad_loft", engine.create_loft(["Base", "Top"]), cross_section_sketches=["Base", "Top"])

    live_tree = _live_tree()
    assert live_tree["features"] == ["PartDesign::AdditivePipe", "PartDesign::AdditiveLoft"]
    macro_tree = _tree(py_export.render_macro_script(live.log))
    _assert_parity(live_tree, macro_tree, _skill_tree(live.log))


_VOLUME_PROBE = """
import json as _json
_vols = {{}}
for _o in doc.Objects:
    for _want in {names!r}:
        if (_o.Name == _want or _o.Name.endswith("_" + _want)) and getattr(_o, "Shape", None) is not None:
            _vols[_want] = _o.Shape.Volume
print(%r + "_VOLUMES " + _json.dumps(_vols))
""" % engine._OK_MARKER


def _volumes(script: str, names: list[str]) -> dict[str, float]:
    result = engine._run_freecad_script(script + _VOLUME_PROBE.format(names=names))
    assert result["ok"], result["error"]
    match = re.search(re.escape(engine._OK_MARKER + "_VOLUMES ") + r"(\{.*\})", result["stdout"])
    assert match, result["stdout"][-2000:]
    return json.loads(match.group(1))


def test_feature_on_face_pipe_and_helix_round_trip() -> None:
    """The macro and the compiled skill used to drop create_freecad_feature_on_face
    ("not a FreeCAD geometry operation"): a drilled plate came back undrilled."""
    live = _Live()
    live("create_freecad_box", engine.create_box(60, 40, 5, name="Plate"), length=60, width=40, height=5, name="Plate")
    live("create_freecad_feature_on_face",
         engine.create_feature_on_face("Plate", "top", "circle", 10, 5, 5, "cut", radius=3, name="Drilled"),
         object_name="Plate", face="top", shape="circle", u=10, v=5, extent=5, operation="cut", radius=3, name="Drilled")
    live("create_freecad_pipe", engine.create_pipe(2, "arc", 90, name="Elbow"),
         pipe_radius=2, path_type="arc", length_or_angle=90, name="Elbow")
    live("create_freecad_helix", engine.create_helix(10, 5, 20, 1, name="Spring"),
         coil_radius=10, pitch=5, height=20, pipe_radius=1, name="Spring")
    names = ["Drilled", "Elbow", "Spring"]

    live_volumes = _volumes(
        f"import FreeCAD as App\ndoc = App.openDocument({str(engine._session_document_path())!r})\n", names
    )
    assert live_volumes["Drilled"] == pytest.approx(60 * 40 * 5 - math.pi * 9 * 5, rel=1e-6)
    macro_volumes = _volumes(py_export.render_macro_script(live.log), names)

    records = skill_compiler.slice_records_since_plan(live.log.records)
    compiled = skill_compiler.compile_call_log_to_skill(records, skill_name="shapes_parity", description="shapes")
    assert compiled["ok"] is True and compiled["skipped"] == [], compiled
    namespace: dict[str, Any] = {}
    exec(compile(compiled["python_code"], "shapes_parity.py", "exec"), namespace)
    set_session_id(f"parity-skill-{uuid.uuid4().hex[:8]}")
    skill_result = namespace["run"]({})
    assert skill_result["ok"] is True, skill_result
    skill_volumes = _volumes(
        f"import FreeCAD as App\ndoc = App.openDocument({str(engine._session_document_path())!r})\n", names
    )
    for volumes in (macro_volumes, skill_volumes):
        assert set(volumes) == set(names)
        for name in names:
            assert volumes[name] == pytest.approx(live_volumes[name], rel=1e-9), name

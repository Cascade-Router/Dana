"""Assembly tools against the real FreeCADCmd: two cubes, one assembly, a
10 mm Distance mate and a fixed kinematic joint, checked by reopening the
saved session document. Skipped when FreeCADCmd isn't installed (it isn't
in CI); the dry-run/mocked coverage lives in test_assembly_mates.py and
tests/core/test_react_dispatch.py.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest

from dana.plugins.freecad import engine
from dana.session_context import DEFAULT_SESSION_ID, set_session_id

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        engine.detect_freecadcmd() is None,
        reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
    ),
]


@pytest.fixture(autouse=True)
def _isolated_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(engine, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    set_session_id(f"assembly-{uuid.uuid4().hex[:8]}")
    yield
    set_session_id(DEFAULT_SESSION_ID)


def _ok(raw: str) -> dict[str, Any]:
    result = json.loads(raw)
    assert result.get("ok") is True, result
    return result


def _inspect(assembly: str, *parts: str) -> dict[str, Any]:
    """Reopen the saved session document and report the assembly's members
    and joints, and each part's bounding box and Dana state flags."""
    script = (
        "import FreeCAD as App, json\n"
        f"doc = App.openDocument({str(engine._session_document_path())!r})\n"
        f"asm = doc.getObject({assembly!r})\n"
        "out = {'type': asm.TypeId, 'members': [o.Name for o in asm.Group],\n"
        f"       'joints': json.loads(getattr(asm, {engine._KINEMATIC_JOINTS_PROP!r}, '') or '{{}}'), 'parts': {{}}}}\n"
        f"for n in {list(parts)!r}:\n"
        "    o = doc.getObject(n)\n"
        "    b = o.Shape.BoundBox\n"
        "    out['parts'][n] = {'bbox': [round(v, 6) for v in (b.XMin, b.YMin, b.ZMin, b.XMax, b.YMax, b.ZMax)],\n"
        "                       'anchored': bool(getattr(o, 'DanaAnchored', False)),\n"
        "                       'constrained': bool(getattr(o, 'DanaConstrained', False))}\n"
        f"print({engine._OK_MARKER!r} + '_ASM ' + json.dumps(out))\n"
    )
    result = engine._run_freecad_script(script)
    assert result["ok"], result
    match = re.search(re.escape(engine._OK_MARKER + "_ASM ") + r"(\{.*\})", result["stdout"])
    assert match, result["stdout"][-2000:]
    return json.loads(match.group(1))


def test_two_cubes_with_a_fixed_10mm_offset_joint() -> None:
    _ok(engine.create_box(20, 20, 20, name="CubeA"))
    _ok(engine.create_box(20, 20, 20, name="CubeB", placement=(100.0, 0.0, 0.0)))
    _ok(engine.create_assembly("Rig"))
    _ok(engine.add_parts_to_assembly("Rig", ["CubeA", "CubeB"]))
    _ok(engine.anchor_assembly_root("Rig", "CubeA"))
    # CubeB's bottom face 10 mm above CubeA's top face, faces opposed.
    _ok(engine.apply_assembly_constraint(
        "Rig", "CubeA", [0.0, 0.0, 1.0], "CubeB", [0.0, 0.0, -1.0], "Distance", offset=10.0,
    ))
    _ok(engine.define_kinematic_joint("Rig", child_link="CubeB", parent_link="CubeA", joint_type="fixed"))

    state = _inspect("Rig", "CubeA", "CubeB")

    assert state["type"] == "App::Part"
    assert sorted(state["members"]) == ["CubeA", "CubeB"]
    # Joints are stored keyed by child link (a link has exactly one parent).
    assert list(state["joints"]) == ["CubeB"]
    assert (state["joints"]["CubeB"]["parent"], state["joints"]["CubeB"]["type"]) == ("CubeA", "fixed")

    cube_a, cube_b = state["parts"]["CubeA"], state["parts"]["CubeB"]
    assert cube_a["bbox"] == [0, 0, 0, 20, 20, 20]  # anchored at the origin
    # Moved from x=100 onto CubeA's axis, 10 mm above its top face.
    assert cube_b["bbox"] == pytest.approx([0, 0, 30, 20, 20, 50], abs=1e-6)
    assert cube_a["anchored"] and not cube_a["constrained"]
    assert cube_b["constrained"] and not cube_b["anchored"]


def test_the_prompts_assembly_block_matches_the_document() -> None:
    """Through the real tool handlers and dispatch_tool_call: what the system
    prompt says about each part agrees with the flags saved in FreeCAD."""
    from dana.core import react_dispatch as rd
    from dana.platform import get_cad_engine
    from dana.platform.mock import MockControlPlane
    from dana.tools.schema import ToolCall

    cad = get_cad_engine()  # the real FreeCADCmd driver on this platform

    rd._set_has_plan(True, "live assembly test")

    def call(tool_id: str, **arguments: Any) -> None:
        result = rd.dispatch_tool_call(ToolCall(tool_id=tool_id, arguments=arguments), cad, MockControlPlane())
        assert result.ok, (tool_id, result.message)

    call("create_freecad_box", length=20, width=20, height=20, name="Hub")
    call("create_freecad_box", length=20, width=20, height=20, name="Spoke", placement_x=60)
    call("create_freecad_box", length=20, width=20, height=20, name="Loose", placement_y=60)
    call("create_freecad_assembly", name="Wheel")
    call("add_parts_to_assembly", assembly_name="Wheel", part_names=["Hub", "Spoke", "Loose"])
    call("anchor_assembly_root", assembly_name="Wheel", part_name="Hub")
    call(
        "apply_assembly_constraint", assembly_name="Wheel", part1_name="Hub", part1_element="Face6",
        part2_name="Spoke", part2_element="Face5", constraint_type="Distance", offset=10.0,
    )

    block = rd._format_assembly_state()
    state = _inspect("Wheel", "Hub", "Spoke", "Loose")["parts"]
    assert "Hub (anchored)" in block and state["Hub"]["anchored"]
    assert "Spoke (mated to Hub)" in block and state["Spoke"]["constrained"]
    assert "Loose (FLOATING)" in block
    assert not state["Loose"]["anchored"] and not state["Loose"]["constrained"]


def test_a_constrained_part_refuses_a_raw_placement_override() -> None:
    _ok(engine.create_box(20, 20, 20, name="Base"))
    _ok(engine.create_box(20, 20, 20, name="Lid", placement=(50.0, 0.0, 0.0)))
    _ok(engine.create_assembly("Box"))
    _ok(engine.add_parts_to_assembly("Box", ["Base", "Lid"]))
    _ok(engine.apply_assembly_constraint(
        "Box", "Base", [0.0, 0.0, 1.0], "Lid", [0.0, 0.0, -1.0], "Distance", offset=10.0,
    ))
    before = _inspect("Box", "Lid")["parts"]["Lid"]["bbox"]

    result = json.loads(engine.position_assembly_part("Lid", placement_z=500.0))

    assert result["ok"] is False
    assert _inspect("Box", "Lid")["parts"]["Lid"]["bbox"] == before

"""The system prompt's ASSEMBLIES block: per-session membership and
anchored / mated / floating status of every assembly part, recorded from
successful assembly tool calls in dispatch_tool_call. The assembly handlers
are stubbed (their FreeCAD side is covered by
tests/plugins/freecad/test_assembly_tools.py); this checks the bookkeeping.
"""

from __future__ import annotations

from typing import Any

import pytest

from dana.core import react_dispatch as rd
from dana.session_context import DEFAULT_SESSION_ID, set_session_id
from dana.tools.schema import ToolCall


@pytest.fixture(autouse=True)
def _stub_assembly_handlers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Every assembly tool succeeds without FreeCAD, unless a test sets
    ``fail`` to a tool id."""
    state: dict[str, Any] = {"fail": None}
    for tool_id in rd._ASSEMBLY_STATE_TOOL_IDS:
        def handler(args: dict[str, Any], _engine: Any, _cp: Any, _tool_id: str = tool_id) -> dict[str, Any]:
            if state["fail"] == _tool_id:
                return {"ok": False, "error": "simulated FreeCAD failure"}
            return {"ok": True, "name": args.get("name")} if _tool_id == "create_freecad_assembly" else {"ok": True}
        monkeypatch.setitem(rd.TOOL_HANDLERS, tool_id, handler)
    set_session_id("assembly-state-test")
    rd._set_has_plan(True, "test-harness plan")
    yield state
    set_session_id(DEFAULT_SESSION_ID)


def _dispatch(tool_id: str, **arguments: Any) -> rd.ToolResult:
    return rd.dispatch_tool_call(ToolCall(tool_id=tool_id, arguments=arguments), None, None)


def _build_rig() -> None:
    assert _dispatch("create_freecad_assembly", name="Rig").ok
    assert _dispatch("add_parts_to_assembly", assembly_name="Rig", part_names=["Chassis", "Arm", "Wheel"]).ok
    assert _dispatch("anchor_assembly_root", assembly_name="Rig", part_name="Chassis").ok
    assert _dispatch(
        "apply_assembly_constraint", assembly_name="Rig", part1_name="Chassis", part1_element="Face6",
        part2_name="Arm", part2_element="Face5", constraint_type="Distance", offset=10,
    ).ok
    assert _dispatch(
        "define_kinematic_joint", assembly_name="Rig", child_link="Arm", parent_link="Chassis", joint_type="revolute",
    ).ok


def test_each_part_is_listed_as_anchored_mated_or_floating() -> None:
    _build_rig()
    block = rd._format_assembly_state()
    assert "Rig: Chassis (anchored), Arm (mated to Chassis, revolute joint to Chassis), Wheel (FLOATING)" in block


def test_the_block_is_in_the_system_prompt_just_before_the_active_plan() -> None:
    _build_rig()
    prompt = rd.build_system_prompt(None, session_id="assembly-state-test")
    assert "=== ASSEMBLIES ===" in prompt
    assert prompt.index("=== ASSEMBLIES ===") < prompt.index("=== ACTIVE PLAN ===")


def test_a_failed_call_changes_nothing(_stub_assembly_handlers: dict[str, Any]) -> None:
    assert _dispatch("create_freecad_assembly", name="Rig").ok
    assert _dispatch("add_parts_to_assembly", assembly_name="Rig", part_names=["Chassis", "Arm"]).ok
    _stub_assembly_handlers["fail"] = "apply_assembly_constraint"
    assert not _dispatch(
        "apply_assembly_constraint", assembly_name="Rig", part1_name="Chassis", part1_element="Face6",
        part2_name="Arm", part2_element="Face5", constraint_type="Coincident",
    ).ok
    assert "Arm (FLOATING)" in rd._format_assembly_state()


def test_reanchoring_a_mated_part_makes_it_anchored() -> None:
    _build_rig()
    assert _dispatch("anchor_assembly_root", assembly_name="Rig", part_name="Arm").ok
    assert "Arm (anchored, revolute joint to Chassis)" in rd._format_assembly_state()


def test_state_is_per_session_and_absent_without_assemblies() -> None:
    _build_rig()
    assert rd._format_assembly_state("some-other-session") == ""
    assert "=== ASSEMBLIES ===" not in rd.build_system_prompt(None, session_id="some-other-session")

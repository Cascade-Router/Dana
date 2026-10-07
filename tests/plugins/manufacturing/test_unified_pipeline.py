"""run_full_manufacturing_pipeline: registered from manifest.json, runs the
manufacturing tools in order through dispatch_tool_call, and never starts a
print itself (dispatch_to_printer's approval prompt only exists at the ReAct
server's per-call gate, which a nested call would skip).

Every underlying tool handler is replaced with a recorder, so this runs
headless with no FreeCAD, slicer or printer.
"""

from __future__ import annotations

from typing import Any

import pytest

from dana.core import react_dispatch as rd
from dana.tools.schema import ToolCall

TOOL_ID = "run_full_manufacturing_pipeline"


class Recorder:
    """Fake handlers for every tool the pipeline could call, logging each call."""

    def __init__(
        self,
        *,
        unprintable: frozenset[str] = frozenset(),
        needs_reorienting: frozenset[str] = frozenset(),
        fail: frozenset[str] = frozenset(),
    ):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.unprintable = unprintable
        self.needs_reorienting = needs_reorienting
        self.fail = fail

    def results(self, tool_id: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_id in self.fail:
            return {"ok": False, "error": f"{tool_id} exploded"}
        if tool_id == "generate_assembly_bom":
            return {
                "ok": True,
                "path": "/ws/exports/Kit_bom.csv",
                "material": args["material"],
                "materials": sorted({args["material"], *args.get("part_materials", {}).values()}),
                "parts": [{"name": "Base"}, {"name": "Arm"}],
                "part_count": 2,
                "total_mass_g": 12.5,
                "total_cost": 0.25,
                "currency": "USD",
            }
        if tool_id == "validate_assembly_collisions":
            return {"ok": True, "collisions": [], "has_collisions": False}
        if tool_id == "export_assembly_to_urdf":
            return {"ok": True, "path": "/out/Kit_urdf/Kit.urdf"}
        if tool_id == "generate_2d_blueprint":
            return {"ok": True, "path": "/out/Kit.pdf"}
        if tool_id == "check_printability":
            name = args["object_name"]
            printable = name not in self.unprintable
            fine = printable and name not in self.needs_reorienting
            as_modelled = {"up_axis": "+Z", "rotation": None, "is_current": True, "requires_supports": False}
            flat = {"up_axis": "+Y", "rotation": {"axis": "X", "degrees": 90.0}, "is_current": False,
                    "requires_supports": False}
            return {
                "ok": True,
                "target": name,
                "printable": printable,
                "requires_supports": name in self.needs_reorienting,
                "warnings": [] if printable else ["larger than the build volume"],
                "stl_path": f"/ws/prints/{name}.stl",
                "current_orientation_printable": fine,
                "recommended_orientation": as_modelled if fine else flat,
                "remediation_hint": None if fine else f"Print {name} rotated +90° about X: no supports needed.",
            }
        if tool_id == "slice_stl_to_gcode":
            return {"ok": True, "gcode_path": args["stl_filepath"].replace(".stl", "_sliced.gcode")}
        if tool_id == "dispatch_to_printer":
            raise AssertionError("the pipeline must never start a print itself")
        raise AssertionError(f"unexpected tool {tool_id}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for tool_id in (
            "generate_assembly_bom",
            "validate_assembly_collisions",
            "export_assembly_to_urdf",
            "generate_2d_blueprint",
            "check_printability",
            "slice_stl_to_gcode",
            "dispatch_to_printer",
        ):
            monkeypatch.setitem(rd.TOOL_HANDLERS, tool_id, self._handler(tool_id))

    def _handler(self, tool_id: str):
        def handler(args: dict[str, Any], _engine: Any, _cp: Any, **_injected: Any) -> dict[str, Any]:
            self.calls.append((tool_id, dict(args)))
            return self.results(tool_id, args)

        return handler

    @property
    def order(self) -> list[str]:
        return [
            f"{tool}:{args['object_name']}" if tool == "check_printability" else
            f"{tool}:{args['stl_filepath'].rsplit('/', 1)[-1]}" if tool == "slice_stl_to_gcode" else tool
            for tool, args in self.calls
        ]


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    from dana.platform import factory

    monkeypatch.setattr(factory, "get_cad_engine", lambda: object())
    rd._set_has_plan(True, "pipeline test")


def _run(**arguments: Any):
    return rd.dispatch_tool_call(ToolCall(tool_id=TOOL_ID, arguments=arguments), object(), object())


def test_registered_from_the_manifest_as_a_gated_cad_tool() -> None:
    assert TOOL_ID in rd.TOOL_HANDLERS
    assert rd.is_mutating_tool(TOOL_ID)  # the user approves the run (it writes files)
    assert TOOL_ID not in rd.ALWAYS_PROMPT_TOOL_IDS


def test_runs_every_stage_in_order_and_stops_before_printing(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    recorder.install(monkeypatch)

    result = _run(assembly_name="Kit", material="PETG")

    assert result.ok, result.message
    assert result.payload["complete"] is True
    assert recorder.order == [
        "generate_assembly_bom",
        "validate_assembly_collisions",
        "export_assembly_to_urdf",
        "generate_2d_blueprint",
        "check_printability:Base",
        "slice_stl_to_gcode:Base.stl",
        "check_printability:Arm",
        "slice_stl_to_gcode:Arm.stl",
    ]
    assert recorder.calls[0][1] == {"assembly_name": "Kit", "material": "PETG"}
    assert result.payload["bom"]["materials"] == ["PETG"]
    assert recorder.calls[3][1] == {"object_name": "Kit", "filename": "Kit_blueprint", "scale": "auto"}
    assert recorder.calls[5][1]["printer_profile"] == "mk4_default"
    payload = result.payload
    assert payload["artifacts"] == {
        "bom_csv": "/ws/exports/Kit_bom.csv",
        "urdf": "/out/Kit_urdf/Kit.urdf",
        "blueprint_pdf": "/out/Kit.pdf",
        "gcode": ["/ws/prints/Base_sliced.gcode", "/ws/prints/Arm_sliced.gcode"],
    }
    assert [r["part"] for r in payload["ready_to_print"]] == ["Base", "Arm"]
    assert "dispatch_to_printer" in payload["next_step"]


def test_part_materials_reach_the_bom(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    recorder.install(monkeypatch)

    result = _run(assembly_name="Kit", part_materials={"Base": "Aluminum 6061"})

    assert result.ok, result.message
    assert recorder.calls[0][1] == {
        "assembly_name": "Kit",
        "material": "PLA",
        "part_materials": {"Base": "Aluminum 6061"},
    }
    assert result.payload["bom"]["materials"] == ["Aluminum 6061", "PLA"]


def test_parts_that_need_reorienting_carry_the_recommendation(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(needs_reorienting=frozenset({"Arm"}))
    recorder.install(monkeypatch)

    payload = _run(assembly_name="Kit").payload

    base, arm = payload["parts"]
    assert base["current_orientation_printable"] is True
    assert "remediation_hint" not in base and "recommended_orientation" not in base
    assert arm["current_orientation_printable"] is False
    assert arm["recommended_orientation"]["rotation"] == {"axis": "X", "degrees": 90.0}
    assert arm["remediation_hint"] == "Print Arm rotated +90° about X: no supports needed."
    # Still printable, so still sliced (as modelled); the hint is surfaced, not acted on.
    assert arm["gcode_path"] == "/ws/prints/Arm_sliced.gcode"
    assert payload["orientation_hints"] == {"Arm": arm["remediation_hint"]}
    assert "['Arm']" in payload["next_step"] and "orientation_hints" in payload["next_step"]
    assert payload["complete"] is True


def test_unprintable_part_is_not_sliced(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(unprintable=frozenset({"Arm"}))
    recorder.install(monkeypatch)

    result = _run(assembly_name="Kit")

    assert "slice_stl_to_gcode:Arm.stl" not in recorder.order
    assert "check_printability:Arm" in recorder.order
    payload = result.payload
    assert [r["part"] for r in payload["ready_to_print"]] == ["Base"]
    assert payload["not_ready_to_print"] == ["Arm"]
    assert result.ok and payload["complete"] is False  # artifacts still reach the agent
    assert "Arm" in payload["next_step"]


def test_bom_failure_stops_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(fail=frozenset({"generate_assembly_bom"}))
    recorder.install(monkeypatch)

    result = _run(assembly_name="Kit")

    assert not result.ok
    assert recorder.order == ["generate_assembly_bom"]
    assert "BOM failed" in result.message


def test_urdf_is_skipped_when_the_collision_check_fails_but_prints_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(fail=frozenset({"validate_assembly_collisions"}))
    recorder.install(monkeypatch)

    result = _run(assembly_name="Kit")

    assert "export_assembly_to_urdf" not in recorder.order
    assert recorder.order[-1] == "slice_stl_to_gcode:Arm.stl"
    payload = result.payload
    assert payload["artifacts"]["urdf"] is None
    assert len(payload["ready_to_print"]) == 2
    assert payload["complete"] is False
    assert payload["failed_steps"] == ["validate_assembly_collisions"]

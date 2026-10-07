"""run_full_manufacturing_pipeline's auto_orient: a part check_printability
says prints better in another principal pose has its print STL rotated into
that pose before slicing; nothing else (CAD part, exported STL, URDF,
drawing) changes.

The unit tests drive the pipeline with fake tool handlers, except that
check_printability's fake runs the real orientation analysis on real meshes
and writes real STL files, so the rotated STL handed to the slicer can be
measured. The live test builds the part in FreeCAD.
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

from dana.core import react_dispatch as rd
from dana.plugins.freecad import engine
from dana.plugins.freecad.printability import Facet, SolidFacet, build_report, evaluate_orientations
from dana.plugins.manufacturing import pipeline
from dana.plugins.os import file_system
from dana.session_context import DEFAULT_SESSION_ID, set_session_id
from dana.tools.schema import ToolCall

TOOL_ID = "run_full_manufacturing_pipeline"
_CLEAN = {"shape_valid": True, "closed_solid": True, "mesh_solid": True, "non_manifold": False,
          "self_intersections": False, "open_edges": 0, "components": 1}


def _box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> trimesh.Trimesh:
    lo_a, hi_a = np.array(lo, float), np.array(hi, float)
    box = trimesh.creation.box(extents=hi_a - lo_a)
    box.apply_translation((lo_a + hi_a) / 2)
    return box


def _union(*meshes: trimesh.Trimesh) -> trimesh.Trimesh:
    pytest.importorskip("manifold3d")  # trimesh's boolean backend (requirements-dev.txt)
    return trimesh.boolean.union(list(meshes))


def _solid(mesh: trimesh.Trimesh) -> list[SolidFacet]:
    tri = mesh.vertices[mesh.faces]
    return [
        SolidFacet(tuple(map(float, n)), float(a), tuple(map(float, lo)), tuple(map(float, hi)))
        for n, a, lo, hi in zip(mesh.face_normals, mesh.area_faces, tri.min(axis=1), tri.max(axis=1))
    ]


def _tee() -> trimesh.Trimesh:
    """Upright T: a 40 mm stem under a 50 mm bar whose underside overhangs."""
    return _union(_box((0, 0, 0), (10, 10, 40)), _box((-20, 0, 40), (30, 10, 50)))


def _jack() -> trimesh.Trimesh:
    """Three crossed bars: needs supports in every principal pose."""
    return _union(_box((-30, -5, -5), (30, 5, 5)), _box((-5, -30, -5), (5, 30, 5)), _box((-5, -5, -30), (5, 5, 30)))


class Shop:
    """Fake handlers for every tool the pipeline calls. check_printability is
    real analysis on ``meshes``; ``defects`` marks a part's mesh as broken."""

    def __init__(self, prints_dir: Path, meshes: dict[str, trimesh.Trimesh], defects: frozenset[str] = frozenset()):
        self.prints_dir = prints_dir
        self.meshes = meshes
        self.defects = defects
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.exported: dict[str, str] = {}  # part -> STL check_printability wrote

    def handle(self, tool_id: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_id, dict(args)))
        if tool_id == "generate_assembly_bom":
            return {"ok": True, "path": "/ws/bom.csv", "materials": ["PLA"], "parts": [{"name": n} for n in self.meshes]}
        if tool_id in ("validate_assembly_collisions", "export_assembly_to_urdf", "generate_2d_blueprint"):
            return {"ok": True, "collisions": [], "path": f"/out/{tool_id}"}
        if tool_id == "check_printability":
            name = args["object_name"]
            mesh = self.meshes[name]
            stl = self.prints_dir / f"{name}.stl"
            mesh.export(stl)
            self.exported[name] = str(stl)
            z = mesh.vertices[mesh.faces][:, :, 2]
            checks = {**_CLEAN, **({"mesh_solid": False, "open_edges": 4} if name in self.defects else {})}
            report = build_report(
                target=name,
                checks=checks,
                bbox=list(mesh.bounds.flatten()),
                facets=[Facet(float(n), float(a), float(lo), float(hi))
                        for n, a, lo, hi in zip(mesh.face_normals[:, 2], mesh.area_faces, z.min(1), z.max(1))],
                solid_facets=_solid(mesh),
            )
            return {"ok": True, **report, "stl_path": str(stl)}
        if tool_id == "slice_stl_to_gcode":
            return {"ok": True, "gcode_path": args["stl_filepath"].replace(".stl", ".gcode")}
        raise AssertionError(f"unexpected tool {tool_id}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for tool_id in ("generate_assembly_bom", "validate_assembly_collisions", "export_assembly_to_urdf",
                        "generate_2d_blueprint", "check_printability", "slice_stl_to_gcode"):
            monkeypatch.setitem(rd.TOOL_HANDLERS, tool_id, self._handler(tool_id))

    def _handler(self, tool_id: str):
        def handler(args: dict[str, Any], _engine: Any, _cp: Any, **_injected: Any) -> dict[str, Any]:
            return self.handle(tool_id, args)

        return handler

    @property
    def sliced(self) -> list[str]:
        return [args["stl_filepath"] for tool, args in self.calls if tool == "slice_stl_to_gcode"]


@pytest.fixture
def shop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from dana.platform import factory

    monkeypatch.setattr(factory, "get_cad_engine", lambda: object())
    rd._set_has_plan(True, "auto-orient test")
    prints = tmp_path / "prints"
    prints.mkdir()

    def make(meshes: dict[str, trimesh.Trimesh], defects: frozenset[str] = frozenset()) -> Shop:
        fake = Shop(prints, meshes, defects)
        fake.install(monkeypatch)
        return fake

    return make


def _run(**arguments: Any) -> dict[str, Any]:
    result = rd.dispatch_tool_call(ToolCall(tool_id=TOOL_ID, arguments=arguments), object(), object())
    assert result.ok, result.message
    return result.payload


def test_upright_tee_print_stl_is_rotated_flat_and_nothing_else_moves(shop) -> None:
    fake = shop({"Tee": _tee()})

    payload = _run(assembly_name="Kit")

    part = payload["parts"][0]
    original = fake.exported["Tee"]
    oriented = part["print_stl_path"]
    assert oriented.endswith("Tee_oriented.stl") and fake.sliced == [oriented]
    assert part["applied_orientation"]["rotation"] == {"axis": "X", "degrees": 90.0}
    assert part["applied_orientation"]["requires_supports"] is False
    # The sliced file lies flat on the bed: 10 mm tall, bottom at Z = 0, and
    # support-free as it now sits.
    printed = trimesh.load(oriented, force="mesh")
    assert printed.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert printed.bounds[1][2] == pytest.approx(10.0, abs=1e-6)
    assert evaluate_orientations(_solid(printed))["current_orientation_printable"] is True
    # The exported STL keeps the modelled pose (40 + 10 mm tall), and no other
    # tool saw anything but the assembly name.
    assert trimesh.load(original, force="mesh").bounds[1][2] == pytest.approx(50.0)
    assert [args for tool, args in fake.calls if tool == "export_assembly_to_urdf"] == [{"assembly_name": "Kit"}]
    assert payload["auto_oriented"] == ["Tee"] and payload["orientation_hints"] == {}
    assert "Auto-oriented for printing: ['Tee']" in payload["next_step"]


def test_auto_orient_off_slices_as_modelled(shop) -> None:
    fake = shop({"Tee": _tee()})

    payload = _run(assembly_name="Kit", auto_orient=False)

    part = payload["parts"][0]
    assert fake.sliced == [fake.exported["Tee"]] and "applied_orientation" not in part
    assert not Path(fake.exported["Tee"]).with_name("Tee_oriented.stl").exists()
    assert payload["auto_oriented"] == [] and "Tee" in payload["orientation_hints"]


def test_no_rotation_when_no_pose_is_better(shop) -> None:
    fake = shop({"Jack": _jack()})

    payload = _run(assembly_name="Kit")

    part = payload["parts"][0]
    assert fake.sliced == [fake.exported["Jack"]] and "applied_orientation" not in part
    assert "Jack" in payload["orientation_hints"]  # still needs supports: flagged, not hidden


def test_too_tall_part_is_sliced_lying_down(shop) -> None:
    fake = shop({"Mast": _box((0, 0, 0), (20, 20, 230))})  # taller than the MK4's 220 mm

    off = _run(assembly_name="Kit", auto_orient=False)
    assert off["parts"][0]["printable"] is False and fake.sliced == []

    on = _run(assembly_name="Kit")
    part = on["parts"][0]
    assert part["applied_orientation"]["height_mm"] == pytest.approx(20.0)
    assert fake.sliced == [part["print_stl_path"]] and part["gcode_path"]
    assert trimesh.load(part["print_stl_path"], force="mesh").extents[2] == pytest.approx(20.0)


def test_a_broken_mesh_is_never_rotated(shop) -> None:
    fake = shop({"Tee": _tee()}, defects=frozenset({"Tee"}))

    payload = _run(assembly_name="Kit")

    part = payload["parts"][0]
    assert "applied_orientation" not in part and fake.sliced == []
    assert payload["parts"][0]["remediation_hint"].startswith("Repair the mesh first")


def test_a_failed_rotation_falls_back_to_the_modelled_pose(shop, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = shop({"Tee": _tee()})

    def broken(*_args: Any) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "_orient_stl", broken)
    payload = _run(assembly_name="Kit")

    assert fake.sliced == [fake.exported["Tee"]] and "applied_orientation" not in payload["parts"][0]
    assert payload["failed_steps"] == ["auto_orient:Tee"] and payload["complete"] is False


# -- live: real FreeCAD ------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.skipif(
    engine.detect_freecadcmd() is None,
    reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
)
def test_live_l_bracket_prints_flat_while_the_urdf_keeps_it_upright(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dana.plugins.manufacturing import bom_exporter
    from dana.plugins.freecad import techdraw_export

    monkeypatch.setattr(engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(engine, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(techdraw_export, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(bom_exporter, "_EXPORT_DIR", tmp_path / "bom")
    monkeypatch.setattr(file_system, "_SANDBOX_ROOT", (tmp_path / "agent_workspace").resolve())
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    sliced: list[str] = []
    monkeypatch.setitem(
        rd.TOOL_HANDLERS,
        "slice_stl_to_gcode",
        lambda args, *_a, **_k: sliced.append(args["stl_filepath"]) or {"ok": True, "gcode_path": "/fake.gcode"},
    )
    set_session_id(f"orient-{uuid.uuid4().hex[:8]}")
    rd._set_has_plan(True, "live auto-orient")
    from dana.platform.mock import MockControlPlane
    from dana.platform.win32 import RealFreeCADEngine

    cad, control_plane = RealFreeCADEngine(), MockControlPlane()

    def call(tool_id: str, **arguments: Any) -> None:
        result = rd.dispatch_tool_call(ToolCall(tool_id=tool_id, arguments=arguments), cad, control_plane)
        assert result.ok, (tool_id, result.message)

    try:
        # Upright L: a 40 mm post with a 30 mm arm sticking out at the top.
        call("create_freecad_box", name="Post", length=10, width=10, height=40)
        call("create_freecad_box", name="Arm", length=30, width=10, height=10, placement_x=10, placement_z=30)
        call("perform_freecad_boolean", operation="union", base_object="Post", tool_object="Arm", name="Bracket")
        call("create_freecad_assembly", name="Rig")
        call("add_parts_to_assembly", assembly_name="Rig", part_names=["Bracket"])

        payload = _run(assembly_name="Rig")
    finally:
        set_session_id(DEFAULT_SESSION_ID)

    part = next(p for p in payload["parts"] if p["name"] == "Bracket")
    assert part["applied_orientation"]["requires_supports"] is False
    assert sliced == [part["print_stl_path"]]
    printed = trimesh.load(part["print_stl_path"], force="mesh")
    assert printed.bounds[0][2] == pytest.approx(0.0, abs=1e-6) and printed.extents[2] == pytest.approx(10.0, abs=1e-3)
    # check_printability's own export and the URDF's mesh keep the part upright.
    assert trimesh.load(part["stl_path"], force="mesh").extents[2] == pytest.approx(40.0, abs=1e-3)
    urdf = Path(payload["artifacts"]["urdf"])
    urdf_mesh = trimesh.load(urdf.parent / "meshes" / "Bracket.stl", force="mesh")
    assert urdf_mesh.extents[2] == pytest.approx(40.0, abs=1e-3)
    assert math.isclose(payload["parts"][0]["applied_orientation"]["height_mm"], 10.0, abs_tol=1e-3)

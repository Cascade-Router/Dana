"""URDF export of a real FreeCAD assembly, through the agent-facing tools
(dispatch_tool_call with the real engine driver): a two-part assembly with a
revolute joint is exported, and the .urdf XML and its STL meshes are checked
against the geometry that was built. Skipped when FreeCADCmd isn't installed
(it isn't in CI); the XML builder's own unit tests are in
tests/tools/test_urdf_builder.py.
"""

from __future__ import annotations

import struct
import uuid
import xml.etree.ElementTree as ET
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
    set_session_id(f"urdf-{uuid.uuid4().hex[:8]}")
    yield
    set_session_id(DEFAULT_SESSION_ID)


@pytest.fixture
def call():
    from dana.core import react_dispatch as rd
    from dana.platform import get_cad_engine
    from dana.platform.mock import MockControlPlane
    from dana.tools.schema import ToolCall

    rd._set_has_plan(True, "live URDF export test")
    cad, control_plane = get_cad_engine(), MockControlPlane()

    def _call(tool_id: str, **arguments: Any) -> dict[str, Any]:
        result = rd.dispatch_tool_call(ToolCall(tool_id=tool_id, arguments=arguments), cad, control_plane)
        assert result.ok, (tool_id, result.message)
        return result.payload

    return _call


def _stl_bounds(path: Path) -> tuple[int, list[float], list[float]]:
    """(triangle count, min xyz, max xyz) of a binary STL."""
    data = path.read_bytes()
    (count,) = struct.unpack_from("<I", data, 80)
    assert len(data) == 84 + 50 * count, "not a well-formed binary STL"
    xs: list[tuple[float, float, float]] = []
    for i in range(count):
        offset = 84 + 50 * i + 12  # skip the facet normal
        xs.extend(struct.unpack_from("<9f", data, offset)[j : j + 3] for j in (0, 3, 6))
    lo = [min(v[k] for v in xs) for k in range(3)]
    hi = [max(v[k] for v in xs) for k in range(3)]
    return count, lo, hi


def test_two_part_revolute_assembly_exports_links_joint_and_meshes(call) -> None:
    call("create_freecad_box", length=40, width=40, height=20, name="Base")
    call("create_freecad_box", length=10, width=10, height=30, name="Arm", placement_x=80)
    call("create_freecad_assembly", name="Robot")
    call("add_parts_to_assembly", assembly_name="Robot", part_names=["Base", "Arm"])
    call("anchor_assembly_root", assembly_name="Robot", part_name="Base")
    # Arm stands 5 mm above Base's top face, centred on it.
    call(
        "apply_assembly_constraint", assembly_name="Robot", part1_name="Base", part1_element="Face6",
        part2_name="Arm", part2_element="Face5", constraint_type="Distance", offset=5.0,
    )
    call(
        "define_kinematic_joint", assembly_name="Robot", child_link="Arm", parent_link="Base",
        joint_type="revolute", axis=[0, 0, 1], limits={"lower": -1.57, "upper": 1.57},
    )
    call("validate_assembly_collisions", assembly_name="Robot")  # required before export
    result = call("export_assembly_to_urdf", assembly_name="Robot")

    urdf = Path(result["path"])
    assert (result["link_count"], result["joint_count"]) == (2, 1)
    robot = ET.parse(urdf).getroot()
    assert robot.tag == "robot" and robot.get("name") == "Robot"

    links = {link.get("name"): link for link in robot.findall("link")}
    assert set(links) == {"Base", "Arm"}  # Base is the single root: no synthetic base_link
    for name, link in links.items():
        mesh = link.find("visual/geometry/mesh")
        assert mesh.get("scale") == "0.001 0.001 0.001"  # STL in mm, URDF in m
        assert link.find("collision/geometry/mesh").get("filename") == mesh.get("filename")
        assert (urdf.parent / mesh.get("filename")).is_file(), mesh.get("filename")
    # Mass from the real volume, aluminium density.
    assert float(links["Base"].find("inertial/mass").get("value")) == pytest.approx(40 * 40 * 20 * 1e-9 * 2700)
    assert float(links["Arm"].find("inertial/mass").get("value")) == pytest.approx(10 * 10 * 30 * 1e-9 * 2700)

    (joint,) = robot.findall("joint")
    assert joint.get("type") == "revolute"
    assert (joint.find("parent").get("link"), joint.find("child").get("link")) == ("Base", "Arm")
    # Arm's frame relative to Base, in metres: centred (20 - 5 mm) on the
    # 40 mm top face, 20 + 5 mm up.
    origin = [float(v) for v in joint.find("origin").get("xyz").split()]
    assert origin == pytest.approx([0.015, 0.015, 0.025], abs=1e-9)
    assert joint.find("axis").get("xyz") == "0 0 1"
    limit = joint.find("limit")
    assert (float(limit.get("lower")), float(limit.get("upper"))) == (-1.57, 1.57)

    # Each STL is in its part's own frame (the joint origin places it), so the
    # assembly pose is not applied twice.
    count, lo, hi = _stl_bounds(urdf.parent / "meshes" / "Arm.stl")
    assert count == 12
    assert lo == pytest.approx([0, 0, 0], abs=1e-4) and hi == pytest.approx([10, 10, 30], abs=1e-4)

"""Unit tests for dana.tools.urdf_builder.export_assembly_parts_to_urdf's
dynamic root-link synthesis: ROOT_LINK_NAME ("base_link") is only written
when structurally necessary (2+ parts independently defaulting to it), not
unconditionally — see that function's own docstring and this module's
git history (rover chassis stress test forensic RCA) for why. Pure XML
generation, no FreeCAD subprocess/CAD engine involved, so these are plain
pytest functions with no mock driver needed.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from dana.tools.urdf_builder import ROOT_LINK_NAME, export_assembly_parts_to_urdf


def _wheel(name: str) -> dict[str, object]:
    return {
        "name": name,
        "mesh_file": f"meshes/{name}.stl",
        "origin_xyz": [0.0, 0.0, 0.0],
        "origin_rpy": [0.0, 0.0, 0.0],
        "joint_parent": "main_body",
        "joint_type": "continuous",
        "joint_axis": [1.0, 0.0, 0.0],
        "joint_name": f"joint_{name}",
    }


def test_proper_tree_single_root_skips_synthetic_base_link(tmp_path) -> None:
    """A rover-shaped tree (1 body + 4 wheels, only the wheels declare
    joint_parent="main_body") has exactly ONE part -- main_body -- that
    defaults to ROOT_LINK_NAME. That part already IS a valid URDF root on
    its own: base_link must NOT be synthesized, main_body gets no incoming
    joint, and the counts must be exactly N links / N-1 joints (5/4 here),
    not N+1/N."""
    main_body = {
        "name": "main_body",
        "mesh_file": "meshes/main_body.stl",
        "origin_xyz": [0.0, 0.0, 0.0],
        "origin_rpy": [0.0, 0.0, 0.0],
        # No joint_parent -- defaults to ROOT_LINK_NAME, same as the real
        # rover run that surfaced this bug (main_body never got its own
        # define_kinematic_joint call).
    }
    parts = [main_body, _wheel("wheel_1"), _wheel("wheel_2"), _wheel("wheel_3"), _wheel("wheel_4")]

    result = json.loads(export_assembly_parts_to_urdf("rover_assembly", parts, str(tmp_path)))

    assert result["ok"] is True, result
    assert result["link_count"] == 5, "expected N links (5), not N+1 -- base_link should not be counted"
    assert result["joint_count"] == 4, "expected N-1 joints (4) -- main_body itself gets no incoming joint"

    urdf_path = result["path"]
    root = ET.parse(urdf_path).getroot()
    link_names = {el.get("name") for el in root.findall("link")}
    joint_names = {el.get("name") for el in root.findall("joint")}

    assert link_names == {"main_body", "wheel_1", "wheel_2", "wheel_3", "wheel_4"}, link_names
    assert ROOT_LINK_NAME not in link_names, f"{ROOT_LINK_NAME} must not be synthesized for a single natural root"
    assert len(joint_names) == 4
    for joint_el in root.findall("joint"):
        assert joint_el.get("type") == "continuous"
        assert joint_el.find("parent").get("link") == "main_body"
        assert joint_el.find("child").get("link") in {"wheel_1", "wheel_2", "wheel_3", "wheel_4"}
        assert joint_el.find("axis").get("xyz") == "1 0 0"
    # main_body has no <joint> where it's the child -- it's the tree's own root.
    children = {j.find("child").get("link") for j in root.findall("joint")}
    assert "main_body" not in children


def test_flat_star_multiple_roots_synthesizes_base_link(tmp_path) -> None:
    """Multiple independent parts that never declare a joint_parent all
    default to ROOT_LINK_NAME -- a genuine multi-root case (URDF requires
    exactly one root), so base_link MUST be synthesized, every part gets a
    fixed joint onto it, and the counts must be N+1 links / N joints,
    exactly the original flat "star" backward-compatible topology."""
    parts = [
        {
            "name": name,
            "mesh_file": f"meshes/{name}.stl",
            "origin_xyz": [float(i), 0.0, 0.0],
            "origin_rpy": [0.0, 0.0, 0.0],
        }
        for i, name in enumerate(("part_a", "part_b", "part_c"))
    ]

    result = json.loads(export_assembly_parts_to_urdf("flat_star_robot", parts, str(tmp_path)))

    assert result["ok"] is True, result
    assert result["link_count"] == 4, "expected N+1 links (4) -- base_link IS needed for 3 independent roots"
    assert result["joint_count"] == 3, "expected N joints (3) -- one fixed joint per part onto base_link"

    urdf_path = result["path"]
    root = ET.parse(urdf_path).getroot()
    link_names = {el.get("name") for el in root.findall("link")}
    assert link_names == {ROOT_LINK_NAME, "part_a", "part_b", "part_c"}

    joints = root.findall("joint")
    assert len(joints) == 3
    for joint_el in joints:
        assert joint_el.get("type") == "fixed"
        assert joint_el.find("parent").get("link") == ROOT_LINK_NAME
        assert joint_el.find("child").get("link") in {"part_a", "part_b", "part_c"}

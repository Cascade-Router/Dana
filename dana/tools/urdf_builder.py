"""URDF (Unified Robot Description Format) assembly generator.

Assembles CAD parts Dana already generated as ``.stl`` mesh artifacts
(``dana.plugins.freecad.engine.create_box``/``export_mesh_stl``, etc.) into
a single kinematic ``.urdf`` XML document — the interchange hub format
ROS2/Gazebo/Isaac Sim all consume. Pure XML text generation, no FreeCAD
subprocess or CAD engine involved, so unlike ``dana.plugins.freecad.engine``
this needs no ``get_cad_engine()`` abstraction and no mock/real driver split.

Every public function returns a JSON string (``{"ok": bool, ...}``), same
wire contract as every ``dana.plugins.freecad.engine`` function, so it slots
into ``dana.core.react_dispatch``'s handler dict the same way.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.dom import minidom

from dana.paths import DANA_WORKSPACE
from dana.session_context import session_scoped_dir

# Same base output directory dana.plugins.freecad.engine._OUTPUT_DIR writes
# create_freecad_*/export_mesh_stl artifacts to — declared as its own copy
# rather than importing engine.py's underscore-prefixed module attribute
# across modules (same precedent dana.api.cad's docstring already applies
# to py_export/techdraw_export). Placing the .urdf alongside the .stl
# meshes it references means dana.api.cad's existing artifact directory
# scan picks it up for free once ".urdf" is added to its extension allowlist.
# ``_session_dir()`` (not the bare constant) is what callers below actually
# use — see dana.session_context's own docstring for why every mesh/doc a
# chat session produces now lives under its own sessions/<session_id>/
# subdirectory instead of this flat, session-shared one.
_OUTPUT_DIR = DANA_WORKSPACE / "freecad_output"


def _session_dir() -> Path:
    return session_scoped_dir(_OUTPUT_DIR)

_JOINT_TYPES = frozenset({"fixed", "revolute", "continuous"})
_DEFAULT_JOINT_LIMIT = (-3.14159, 3.14159)


def _ok(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload})


def _error(message: str) -> str:
    return json.dumps({"ok": False, "error": str(message)})


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name or "").strip("_") or "robot"


def _xyz_tuple(value: Any, *, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if value is None:
        return default
    try:
        x, y, z = value
        return (float(x), float(y), float(z))
    except (TypeError, ValueError):
        raise ValueError(f"expected a 3-element [x, y, z] array, got {value!r}") from None


def _xyz_str(vec: tuple[float, float, float]) -> str:
    return f"{vec[0]:g} {vec[1]:g} {vec[2]:g}"


def _add_geometry(parent: ET.Element, tag: str, mesh_filename: str, origin: tuple[float, float, float]) -> None:
    node = ET.SubElement(parent, tag)
    ET.SubElement(node, "origin", xyz=_xyz_str(origin), rpy="0 0 0")
    geometry = ET.SubElement(node, "geometry")
    ET.SubElement(geometry, "mesh", filename=mesh_filename)


def _mesh_file_exists(mesh_path: str) -> bool:
    """Poll 3 candidate locations for a caller-supplied ``mesh_path`` before
    it's ever written into the URDF XML — an LLM can freely invent a
    plausible-looking ``mesh_path`` string that was never actually produced
    by ``export_mesh_stl``/``export_freecad_model``, and without this check
    that hallucinated filename would silently end up as a dangling ``<mesh
    filename="...">`` reference in the generated ``.urdf`` (broken the
    moment anything tries to load it).

    Checked in order: the path as given (absolute or already
    cwd-relative-and-correct), then its basename under THIS session's own
    output directory (``_session_dir()`` — cwd-independent, where every
    create_freecad_*/export_mesh_stl artifact for this session actually
    lands), then its basename under the process's current working
    directory (a plain relative mesh_path the caller already resolved
    against its own cwd).
    """
    candidate = Path(mesh_path)
    if candidate.exists():
        return True
    basename = candidate.name
    if (_session_dir() / basename).exists():
        return True
    return (Path.cwd() / basename).exists()


def _build_link(links_root: ET.Element, link: dict[str, Any]) -> str:
    name = str(link.get("name") or "").strip()
    if not name:
        raise ValueError("every link requires a non-empty 'name'")
    link_el = ET.SubElement(links_root, "link", name=name)
    mesh_path = link.get("mesh_path") or link.get("stl_path")
    if mesh_path:
        mesh_path = str(mesh_path)
        if not _mesh_file_exists(mesh_path):
            raise ValueError(
                f"Mesh file '{mesh_path}' does not exist. You MUST use 'export_freecad_model' "
                "to export the CAD object to an STL file before generating a URDF."
            )
        # A bare basename, never the full artifact path — matches
        # dana.api.cad._resolve_artifact's "bare filename only" contract,
        # so the frontend can fetch it as /api/cad/artifacts/{filename}/download
        # regardless of where on disk this tool's caller generated it.
        mesh_filename = Path(mesh_path).name
        origin = _xyz_tuple(link.get("origin_xyz"))
        _add_geometry(link_el, "visual", mesh_filename, origin)
        _add_geometry(link_el, "collision", mesh_filename, origin)
    return name


def generate_urdf_assembly(
    robot_name: str,
    links: list[dict[str, Any]],
    joints: list[dict[str, Any]],
) -> str:
    """Assemble ``links``/``joints`` into a URDF document and save it as
    ``<robot_name>.urdf`` under ``freecad_output/``.

    ``links`` — each ``{"name": str, "mesh_path": optional str}`` (a
    previously-generated ``.stl`` artifact path/filename to attach as that
    link's visual+collision geometry; a link may omit it for a purely
    kinematic frame with no geometry of its own).

    ``joints`` — each ``{"name": optional str, "parent": str, "child": str,
    "type": "fixed"|"revolute"|"continuous", "origin_xyz": optional [x, y,
    z], "axis": optional [x, y, z]}``. ``origin_xyz`` is the child frame's
    offset from the parent (defaults to the origin); ``axis`` is the
    rotation axis for revolute/continuous joints (defaults to +Z) and is
    omitted from fixed joints regardless of what's passed.
    """
    name = _safe_name(robot_name)
    if not links:
        return _error("generate_urdf_assembly requires at least one link")
    if not joints:
        return _error("generate_urdf_assembly requires at least one joint")

    robot_el = ET.Element("robot", name=name)
    links_seen: set[str] = set()
    try:
        for link in links:
            link_name = _build_link(robot_el, link)
            if link_name in links_seen:
                return _error(f"duplicate link name: {link_name!r}")
            links_seen.add(link_name)

        for joint in joints:
            parent = str(joint.get("parent") or "").strip()
            child = str(joint.get("child") or "").strip()
            joint_type = str(joint.get("type") or "").strip().lower()
            if not parent or not child:
                return _error("every joint requires both 'parent' and 'child' link names")
            if parent not in links_seen:
                return _error(f"joint references unknown parent link: {parent!r}")
            if child not in links_seen:
                return _error(f"joint references unknown child link: {child!r}")
            if joint_type not in _JOINT_TYPES:
                return _error(f"unknown joint type {joint_type!r} — must be fixed, revolute, or continuous")

            joint_name = str(joint.get("name") or f"{parent}_to_{child}").strip()
            joint_el = ET.SubElement(robot_el, "joint", name=joint_name, type=joint_type)
            ET.SubElement(joint_el, "parent", link=parent)
            ET.SubElement(joint_el, "child", link=child)
            origin = _xyz_tuple(joint.get("origin_xyz"))
            ET.SubElement(joint_el, "origin", xyz=_xyz_str(origin), rpy="0 0 0")
            if joint_type in ("revolute", "continuous"):
                axis = _xyz_tuple(joint.get("axis"), default=(0.0, 0.0, 1.0))
                ET.SubElement(joint_el, "axis", xyz=_xyz_str(axis))
                if joint_type == "revolute":
                    lower, upper = _DEFAULT_JOINT_LIMIT
                    ET.SubElement(
                        joint_el,
                        "limit",
                        lower=str(joint.get("limit_lower", lower)),
                        upper=str(joint.get("limit_upper", upper)),
                        effort=str(joint.get("limit_effort", 10.0)),
                        velocity=str(joint.get("limit_velocity", 1.0)),
                    )
    except ValueError as exc:
        return _error(f"generate_urdf_assembly: {exc}")

    xml_bytes = ET.tostring(robot_el, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_bytes).toprettyxml(indent="  ")

    out_path = _session_dir() / f"{name}.urdf"
    out_path.write_text(pretty_xml, encoding="utf-8")

    return _ok(
        name=name,
        type="urdf",
        path=str(out_path),
        link_count=len(links),
        joint_count=len(joints),
        movable_joint_count=sum(1 for j in joints if str(j.get("type") or "").strip().lower() != "fixed"),
    )

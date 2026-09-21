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

_JOINT_TYPES = frozenset({"fixed", "revolute", "continuous", "prismatic"})
_DEFAULT_JOINT_LIMIT = (-3.14159, 3.14159)

# The synthetic root link every export_assembly_parts_to_urdf tree is
# anchored to. Public (no leading underscore) so
# dana.plugins.freecad.engine can import the exact same string rather than
# hardcoding its own copy — a `define_kinematic_joint` parent_link of
# "base_link" and this module's own root link MUST be byte-identical or a
# joint silently fails to attach to the real root.
ROOT_LINK_NAME = "base_link"

# Aluminum. PLA (~1250) is the other common default for 3D-printed parts —
# callers pass their own `density_kg_m3` when the material actually matters.
_DEFAULT_DENSITY_KG_PER_M3 = 2700.0
_INERTIA_KEYS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")

# Every coordinate/volume/inertia value flowing into this file comes
# straight from FreeCAD, whose native unit is mm — but URDF's own spec
# mandates meters/kg/kg*m^2 (Gazebo/Isaac Sim/MuJoCo all enforce this: an
# un-converted mm-as-if-m robot is 1000x too big and, worse, has a mass
# using the true (mm-scale) volume against an SI density, so a wheel a few
# mm across gets reported as ~150,000 "kg" — real numbers seen on this
# rover's own export before this fix). Applied consistently across EVERY
# length/volume/inertia value below (origin, mesh scale, mass, inertia) —
# a partial conversion (e.g. fixing mass/inertia alone) would silently put
# the <inertial> block at a different scale than the <origin>/<mesh>
# sitting right next to it in the same link, which is worse than not
# converting at all.
_MM_TO_M = 1.0e-3
_MM3_TO_M3 = 1.0e-9
# Second moment (∫r^2 dV, r and dV both in mm) -> the same integral in m:
# length^5 total, i.e. (mm->m)^5.
_MM5_TO_M5 = _MM_TO_M**5
_MESH_SCALE_STR = f"{_MM_TO_M:g} {_MM_TO_M:g} {_MM_TO_M:g}"


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


def _mm_xyz_str(vec: tuple[float, float, float]) -> str:
    """Same as ``_xyz_str`` but for a tuple that's still in FreeCAD's native
    mm — scales to meters first. Every ``<origin xyz="...">`` below is a
    length, not a bare number, so this (never ``_xyz_str`` directly on a
    raw FreeCAD tuple) is what actually goes into the URDF.
    """
    return _xyz_str((vec[0] * _MM_TO_M, vec[1] * _MM_TO_M, vec[2] * _MM_TO_M))


# URDF's own "rpy" attribute is a FIXED-axis (extrinsic) rotation in
# RADIANS — roll about the fixed X axis, then pitch about fixed Y, then yaw
# about fixed Z (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)). Confirmed live
# (freecadcmd) that this is EXACTLY FreeCAD's own
# Rotation(yaw, pitch, roll)/.toEuler() convention (same axes, same
# composition order, just different argument order and degrees vs
# radians) — so a caller holding a FreeCAD Rotation only ever needs
# `rot.toEuler()` (yaw, pitch, roll, in degrees) reordered to
# (radians(roll), radians(pitch), radians(yaw)) for this, never a real
# re-derivation. Was hardcoded to "0 0 0" everywhere below until now —
# every origin in this file was translation-only, silently dropping any
# rotation a caller might have wanted to express.
def _rpy_str(rpy: tuple[float, float, float]) -> str:
    return f"{rpy[0]:g} {rpy[1]:g} {rpy[2]:g}"


def _add_geometry(
    parent: ET.Element,
    tag: str,
    mesh_filename: str,
    origin: tuple[float, float, float],
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """``origin`` is FreeCAD-native mm (like everywhere else in this file);
    converted to meters for the ``<origin>`` tag, with a matching
    ``scale="0.001 0.001 0.001"`` on ``<mesh>`` so the (still mm-tessellated)
    STL geometry renders at the same real-world size instead of 1000x too
    big.
    """
    node = ET.SubElement(parent, tag)
    ET.SubElement(node, "origin", xyz=_mm_xyz_str(origin), rpy=_rpy_str(rpy))
    geometry = ET.SubElement(node, "geometry")
    ET.SubElement(geometry, "mesh", filename=mesh_filename, scale=_MESH_SCALE_STR)


def _add_joint_kinematics(
    joint_el: ET.Element,
    joint_type: str,
    axis: tuple[float, float, float],
    limit_lower: float | None,
    limit_upper: float | None,
    limit_effort: float | None,
    limit_velocity: float | None,
) -> None:
    """``<axis>``/``<limit>`` for a moving joint, shared by
    ``generate_urdf_assembly`` (manual joints) and
    ``export_assembly_parts_to_urdf`` (assembly-derived joints) so the two
    generators can't drift on this.

    ``"fixed"`` gets neither (no DOF to describe). ``"revolute"``/
    ``"prismatic"`` get both ``<axis>`` and a mandatory ``<limit>`` — a
    physics engine can't simulate a bounded rotational/translational DOF
    without SOME lower/upper, unlike ``"continuous"`` (unbounded rotation
    by definition), which gets only ``<axis>`` — matching URDF's own
    convention (a ``continuous`` joint's ``<limit>`` is meaningless/absent).
    A caller-omitted limit defaults to ``_DEFAULT_JOINT_LIMIT``/10 effort/
    1 velocity — generic placeholders, not a real actuator's numbers; the
    unit is radians for a revolute joint but is just as arbitrary a
    placeholder (real URDF/SI meters, like every other length in this file
    — see ``_MM_TO_M``'s own comment) for a prismatic one. Unlike
    ``origin_xyz``, a caller-supplied ``limit_lower``/``limit_upper`` is
    used exactly as given (there's no FreeCAD-native mm value to convert
    from) — a caller wanting a prismatic joint's range to match this file's
    now-meters convention passes it in meters already.
    """
    if joint_type == "fixed":
        return
    ET.SubElement(joint_el, "axis", xyz=_xyz_str(axis))
    if joint_type in ("revolute", "prismatic"):
        lower, upper = _DEFAULT_JOINT_LIMIT
        ET.SubElement(
            joint_el,
            "limit",
            lower=str(lower if limit_lower is None else limit_lower),
            upper=str(upper if limit_upper is None else limit_upper),
            effort=str(10.0 if limit_effort is None else limit_effort),
            velocity=str(1.0 if limit_velocity is None else limit_velocity),
        )


def _add_inertial(
    parent: ET.Element,
    mass: float,
    center_of_mass: tuple[float, float, float],
    inertia: dict[str, float],
) -> None:
    """``<inertial>``: mass + the COM as the inertial frame's ``<origin>``
    (``rpy`` always "0 0 0" — the inertia tensor below is already expressed
    about axes parallel to the link frame, not rotated to principal axes)
    + the 6 independent entries of the symmetric inertia tensor.

    ``center_of_mass`` is FreeCAD-native mm, same as every other origin in
    this file — converted to meters here too.
    """
    inertial = ET.SubElement(parent, "inertial")
    ET.SubElement(inertial, "origin", xyz=_mm_xyz_str(center_of_mass), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:g}")
    ET.SubElement(inertial, "inertia", **{k: f"{inertia.get(k, 0.0):g}" for k in _INERTIA_KEYS})


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
        rpy = _xyz_tuple(link.get("origin_rpy"))
        _add_geometry(link_el, "visual", mesh_filename, origin, rpy)
        _add_geometry(link_el, "collision", mesh_filename, origin, rpy)
    return name


def generate_urdf_assembly(
    robot_name: str,
    links: list[dict[str, Any]],
    joints: list[dict[str, Any]],
) -> str:
    """Assemble ``links``/``joints`` into a URDF document and save it as
    ``<robot_name>.urdf`` under ``freecad_output/``.

    ``links`` — each ``{"name": str, "mesh_path": optional str,
    "origin_xyz": optional [x, y, z], "origin_rpy": optional [roll, pitch,
    yaw] radians}`` (a previously-generated ``.stl`` artifact path/filename
    to attach as that link's visual+collision geometry; a link may omit it
    for a purely kinematic frame with no geometry of its own).
    ``origin_rpy`` — URDF's own fixed-axis roll/pitch/yaw in RADIANS (see
    ``_rpy_str``'s own comment for the exact convention) — defaults to
    ``(0, 0, 0)``, same as before this parameter existed.

    ``joints`` — each ``{"name": optional str, "parent": str, "child": str,
    "type": "fixed"|"revolute"|"continuous"|"prismatic", "origin_xyz":
    optional [x, y, z], "origin_rpy": optional [roll, pitch, yaw] radians,
    "axis": optional [x, y, z], "limit_lower"/"limit_upper"/"limit_effort"/
    "limit_velocity": optional floats}``. ``origin_xyz``/``origin_rpy`` is
    the child frame's offset from the parent (both default to zero);
    ``axis`` is the rotation/translation axis for revolute/continuous/
    prismatic joints (defaults to +Z) and is omitted from fixed joints
    regardless of what's passed. ``limit_*`` apply to revolute/prismatic
    only (see ``_add_joint_kinematics``'s own docstring for the defaults
    used when omitted) and are ignored for continuous/fixed.
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
                return _error(
                    f"unknown joint type {joint_type!r} — must be fixed, revolute, continuous, or prismatic"
                )

            joint_name = str(joint.get("name") or f"{parent}_to_{child}").strip()
            joint_el = ET.SubElement(robot_el, "joint", name=joint_name, type=joint_type)
            ET.SubElement(joint_el, "parent", link=parent)
            ET.SubElement(joint_el, "child", link=child)
            origin = _xyz_tuple(joint.get("origin_xyz"))
            rpy = _xyz_tuple(joint.get("origin_rpy"))
            ET.SubElement(joint_el, "origin", xyz=_mm_xyz_str(origin), rpy=_rpy_str(rpy))
            axis = _xyz_tuple(joint.get("axis"), default=(0.0, 0.0, 1.0))
            _add_joint_kinematics(
                joint_el,
                joint_type,
                axis,
                joint.get("limit_lower"),
                joint.get("limit_upper"),
                joint.get("limit_effort"),
                joint.get("limit_velocity"),
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


def export_assembly_parts_to_urdf(
    robot_name: str,
    parts: list[dict[str, Any]],
    out_dir: str,
    density_kg_m3: float = _DEFAULT_DENSITY_KG_PER_M3,
) -> str:
    """Builds a URDF from an assembly's own real geometry —
    ``dana.plugins.freecad.engine.export_assembly_to_urdf``'s XML-writing
    half (this module stays FreeCAD-free per its own module docstring;
    that function does the FreeCAD-side mesh export + Placement extraction
    and calls this with the result).

    Unlike ``generate_urdf_assembly`` above (the model manually specifies
    every link/joint by hand), ``parts`` here is auto-derived from a real
    ``App::Part`` assembly's actual members: each ``{"name": str,
    "mesh_file": str — already-written, relative to out_dir (e.g.
    "meshes/Foo.stl"), "origin_xyz": [x, y, z] mm, "origin_rpy": [roll,
    pitch, yaw] radians}`` becomes one link. By default (no
    ``dana.plugins.freecad.engine.define_kinematic_joint`` call ever made
    for that part) it's fixed-jointed onto the synthetic geometry-less
    ``ROOT_LINK_NAME`` at that part's own Placement RELATIVE TO THE
    ASSEMBLY — the original flat "star" topology, preserved as the default
    for backward compatibility when MULTIPLE parts land there (independent
    islands genuinely need a shared common ancestor — URDF requires
    exactly one root). When exactly ONE part ends up there, though, that
    part already IS a valid root on its own: ``ROOT_LINK_NAME`` is skipped
    entirely and that part gets no incoming joint, rather than adding a
    redundant geometry-less parent purely for its own sake (e.g. a rover's
    ``main_body`` with 4 wheels jointed to it, but no joint ever declared
    for ``main_body`` itself — this used to report 6 links/5 joints for
    what is structurally a 5-link/4-joint tree). ``link_count``/
    ``joint_count`` in the returned payload reflect whichever shape was
    actually written.

    A part MAY instead carry ``"joint_parent"`` (another part's name, or
    ``ROOT_LINK_NAME`` — the default), ``"joint_type"`` ("fixed" default,
    else "revolute"/"continuous"/"prismatic"), ``"joint_axis"``, ``
    "joint_name"``, and ``"limit_lower"``/``"limit_upper"``/
    ``"limit_effort"``/``"limit_velocity"`` — ``dana.plugins.freecad.engine
    .define_kinematic_joint``'s persisted state, one real parent/child
    kinematic edge per part (a part can have at most one parent, same as
    any URDF link), turning this from a flat star into a genuine tree.
    ``origin_xyz``/``origin_rpy`` must already be that part's Placement
    RELATIVE TO WHICHEVER PARENT ``"joint_parent"`` names (not the
    assembly) — ``export_assembly_to_urdf``'s own FreeCAD script computes
    it that way per-part, this module never re-derives it. Every part's
    parent is validated (either ``ROOT_LINK_NAME`` or another part actually
    present in ``parts``) and the whole graph is checked for cycles before
    any XML is written — a bad reference or a cycle fails the whole export
    with a clear error rather than emitting a URDF no simulator can load.

    Each mesh path is used EXACTLY as given (already relative to
    ``out_dir``, already known to exist — the caller just wrote it) —
    unlike ``generate_urdf_assembly``'s ``_mesh_file_exists`` 3-location
    search, which exists specifically for a caller-SUPPLIED path that
    might be a hallucinated filename; there's nothing to hallucinate here,
    the mesh was written by the same call chain one step earlier.

    Each part optionally also carries ``"volume"`` (its own real ``Shape``
    volume), ``"center_of_mass"`` ([x, y, z], relative to the part's own
    local origin — same frame ``mesh_file`` was tessellated in), and
    ``"inertia"`` (``{"ixx", "ixy", "ixz", "iyy", "iyz", "izz"}``, also
    about that local origin) straight from FreeCAD's own ``Shape.Volume``/
    ``.CenterOfMass``/``.MatrixOfInertia`` (raw mm^3/mm^5, unit density —
    this function scales them to real kg / kg*m^2, see ``_MM_TO_M``'s own
    comment), and that link gets a real ``<inertial>`` block instead of
    none — a part with no volume (a pure organizational sub-group that
    slipped through, or an older caller not yet passing this data) simply
    gets no ``<inertial>``, same as before this parameter existed.
    """
    name = _safe_name(robot_name)
    if not parts:
        return _error("export_assembly_parts_to_urdf requires at least one part")

    # Pass 1: validate names/parents and build the full parent map BEFORE
    # writing any XML — a part's declared parent may be a part that appears
    # LATER in this same list (define_kinematic_joint calls, and therefore
    # this list's order, have no required topological order), so the parent
    # map can only be trusted once every part has been seen at least once.
    seen: set[str] = set()
    parent_of: dict[str, str] = {}
    for part in parts:
        part_name = str(part.get("name") or "").strip()
        mesh_file = str(part.get("mesh_file") or "").strip()
        if not part_name or not mesh_file:
            return _error("every part requires both 'name' and 'mesh_file'")
        if part_name == ROOT_LINK_NAME:
            return _error(f"part name {part_name!r} collides with the synthetic root link name")
        if part_name in seen:
            return _error(f"duplicate part name: {part_name!r}")
        seen.add(part_name)
        parent_of[part_name] = str(part.get("joint_parent") or ROOT_LINK_NAME).strip() or ROOT_LINK_NAME

    for part_name, parent in parent_of.items():
        if parent != ROOT_LINK_NAME and parent not in seen:
            return _error(f"part {part_name!r} references unknown parent link {parent!r}")
        visited = {part_name}
        current = parent
        while current != ROOT_LINK_NAME:
            if current in visited:
                return _error(f"kinematic joint cycle detected involving part {part_name!r}")
            visited.add(current)
            current = parent_of[current]

    # Pass 2: everything validated — safe to write the XML tree.
    out_path_dir = Path(out_dir)
    robot_el = ET.Element("robot", name=name)

    # Only synthesize ROOT_LINK_NAME when it's structurally necessary: two
    # or more parts independently defaulting/declaring it as their parent
    # (islands that need a shared common ancestor -- URDF requires exactly
    # one root). When exactly one part is naturally the root, it already
    # IS a valid URDF root on its own; a geometry-less base_link fixed onto
    # it is pure redundancy, not a requirement (confirmed live: a 1-body,
    # 4-wheel rover reported "6 links / 5 joints" instead of the correct
    # 5/4 — main_body never got its own define_kinematic_joint call, so it
    # defaulted to ROOT_LINK_NAME same as every other part used to, but it
    # was the ONLY one that did).
    roots = [n for n, parent in parent_of.items() if parent == ROOT_LINK_NAME]
    synthesize_root = len(roots) != 1
    if synthesize_root:
        ET.SubElement(robot_el, "link", name=ROOT_LINK_NAME)

    movable_count = 0
    for part in parts:
        part_name = str(part.get("name") or "").strip()
        mesh_file = str(part.get("mesh_file") or "").strip()

        link_el = ET.SubElement(robot_el, "link", name=part_name)
        _add_geometry(link_el, "visual", mesh_file, (0.0, 0.0, 0.0))
        _add_geometry(link_el, "collision", mesh_file, (0.0, 0.0, 0.0))

        volume = part.get("volume")
        if volume is not None and float(volume) > 0.0:
            raw_inertia = part.get("inertia") or {}
            # volume/inertia are FreeCAD-native mm^3/mm^5 (Shape.Volume /
            # Shape.MatrixOfInertia at unit density) -- scale to m^3/m^5
            # before applying density, same mm->m normalization as every
            # origin/mesh above, so mass ends up in real kg (a few-mm wheel
            # is now ~1e-4 kg, not ~1.5e5) and inertia in real kg*m^2.
            mass = density_kg_m3 * float(volume) * _MM3_TO_M3
            inertia = {
                k: density_kg_m3 * float(raw_inertia.get(k, 0.0)) * _MM5_TO_M5 for k in _INERTIA_KEYS
            }
            _add_inertial(link_el, mass, _xyz_tuple(part.get("center_of_mass")), inertia)

        parent = parent_of[part_name]
        if not synthesize_root and parent == ROOT_LINK_NAME:
            # This part IS the tree's sole natural root -- no incoming
            # joint at all, same as base_link itself never gets one.
            continue

        joint_type = str(part.get("joint_type") or "fixed").strip().lower()
        if joint_type not in _JOINT_TYPES:
            return _error(
                f"part {part_name!r}: unknown joint_type {joint_type!r} — must be fixed, revolute, "
                "continuous, or prismatic"
            )
        if joint_type != "fixed":
            movable_count += 1
        joint_name = str(part.get("joint_name") or f"{parent}_to_{part_name}").strip()

        origin = _xyz_tuple(part.get("origin_xyz"))
        rpy = _xyz_tuple(part.get("origin_rpy"))
        joint_el = ET.SubElement(robot_el, "joint", name=joint_name, type=joint_type)
        ET.SubElement(joint_el, "parent", link=parent)
        ET.SubElement(joint_el, "child", link=part_name)
        ET.SubElement(joint_el, "origin", xyz=_mm_xyz_str(origin), rpy=_rpy_str(rpy))
        axis = _xyz_tuple(part.get("joint_axis"), default=(0.0, 0.0, 1.0))
        _add_joint_kinematics(
            joint_el,
            joint_type,
            axis,
            part.get("limit_lower"),
            part.get("limit_upper"),
            part.get("limit_effort"),
            part.get("limit_velocity"),
        )

    xml_bytes = ET.tostring(robot_el, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_bytes).toprettyxml(indent="  ")

    out_path_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_path_dir / f"{name}.urdf"
    out_path.write_text(pretty_xml, encoding="utf-8")

    return _ok(
        name=name,
        type="urdf",
        path=str(out_path),
        link_count=len(parts) + (1 if synthesize_root else 0),
        joint_count=len(parts) - (0 if synthesize_root else 1),
        movable_joint_count=movable_count,
    )

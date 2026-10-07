"""One-call manufacturing pipeline for an assembly: BOM -> URDF -> 2D
blueprint -> per-part printability -> per-part slicing, ending at G-code
that is ready to print.

It deliberately stops before printing. dispatch_to_printer's human approval
is enforced by the ReAct server per tool call, before the tool runs
(``ALWAYS_PROMPT_TOOL_IDS``), not inside the printer code, so calling it from
here would start a physical print nobody approved. The summary instead lists
each part's G-code under ``ready_to_print`` for the agent to send with
separate dispatch_to_printer calls, each of which asks the user.

Each step runs an existing agent tool through ``run_tool(tool_id, args)``
(injected; see pipeline_tools.py), so every step keeps that tool's own
argument validation and name resolution.

With ``auto_orient`` (the default), a part check_printability says would
print better in another principal pose gets a rotated copy of its print STL
(``<part>_oriented.stl``, resting on Z = 0) and that copy is sliced. Only the
print file turns: the CAD part, its exported STL, the URDF and the drawing
keep the modelled pose.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

RunTool = Callable[[str, dict[str, Any]], dict[str, Any]]


def _orientation_to_apply(check: dict[str, Any]) -> dict[str, Any] | None:
    """check_printability's recommended pose, if rotating the print to it is
    worth doing: the part isn't fine as modelled, its mesh is sound (rotation
    can't fix a broken one), and the recommendation is a real rotation that
    fits the bed and is support-free, or at least strictly better than the
    modelled pose (fits where that doesn't, or overhangs less)."""
    if check.get("current_orientation_printable") is not False or not check.get("mesh_sound"):
        return None
    best = check.get("recommended_orientation") or {}
    current = next((o for o in check.get("orientations") or [] if o.get("is_current")), None)
    if not best.get("rotation") or current is None or best.get("fit") == "too_large":
        return None
    if best.get("support_free"):
        return best
    if current.get("fit") == "too_large" or best["overhang_area_mm2"] < current["overhang_area_mm2"]:
        return best
    return None


def _orient_stl(stl_path: str, rotation: dict[str, Any]) -> str:
    """Write ``stl_path`` turned by ``rotation`` ({"axis", "degrees"}, about the
    part's own axes, as check_printability reports it) and lowered so its
    bottom sits at Z = 0, next to the original as ``<stem>_oriented.stl``."""
    import trimesh

    mesh = trimesh.load(stl_path, force="mesh")
    axis = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[rotation["axis"]]
    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rotation["degrees"]), axis))
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0][2]])
    source = Path(stl_path)
    out = source.with_name(f"{source.stem}_oriented.stl")
    mesh.export(out)  # binary STL
    return str(out)


def execute_manufacturing_pipeline(
    assembly_name: str,
    run_tool: RunTool,
    *,
    material: str = "PLA",
    part_materials: dict[str, str] | None = None,
    printer_profile: str = "mk4_default",
    auto_orient: bool = True,
) -> dict[str, Any]:
    """Run every pre-print manufacturing step for ``assembly_name``.

    The BOM runs first and its failure aborts the pipeline (``ok: False``;
    it is also what lists the parts to print). After that a failed step is
    recorded and the pipeline carries on, since the URDF, the drawing and the
    prints don't depend on each other, and the result stays ``ok: True`` so
    the artifacts that were made reach the agent (dispatch_tool_call reduces
    a failed result to its error message). ``complete`` is true only when
    every step succeeded and every part has G-code. A part is sliced if
    check_printability calls it printable, or if ``auto_orient`` turned it
    into a pose that fits the bed.
    """
    assembly = (assembly_name or "").strip()
    if not assembly:
        return {"ok": False, "error": "run_full_manufacturing_pipeline requires assembly_name"}

    steps: list[dict[str, Any]] = []

    def step(tool_id: str, args: dict[str, Any], label: str | None = None) -> dict[str, Any]:
        result = run_tool(tool_id, args)
        entry = {"step": label or tool_id, "tool": tool_id, "ok": bool(result.get("ok"))}
        if not entry["ok"]:
            entry["error"] = result.get("error") or "failed without an error message"
        steps.append(entry)
        return result

    bom_args: dict[str, Any] = {"assembly_name": assembly, "material": material}
    if part_materials:
        bom_args["part_materials"] = part_materials
    bom = step("generate_assembly_bom", bom_args)
    if not bom.get("ok"):
        return {
            "ok": False,
            "error": f"run_full_manufacturing_pipeline: BOM failed, nothing else was run: {bom.get('error')}",
            "steps": steps,
        }
    part_names = [p["name"] for p in bom["parts"]]

    # export_assembly_to_urdf requires a collision check first (the dispatch
    # gate does not see calls made from here, so the order is kept by hand).
    urdf: dict[str, Any] = {}
    collisions = step("validate_assembly_collisions", {"assembly_name": assembly})
    if collisions.get("ok"):
        urdf = step("export_assembly_to_urdf", {"assembly_name": assembly})

    # Named after the assembly: the default is the source file's name, which
    # for every session object is the shared Session_Active document.
    # "auto" scale: an assembly's size isn't known up front, and 1:1 only suits mid-sized ones.
    blueprint = step(
        "generate_2d_blueprint",
        {"object_name": assembly, "filename": f"{assembly}_blueprint", "scale": "auto"},
    )

    parts: list[dict[str, Any]] = []
    for name in part_names:
        check = step("check_printability", {"object_name": name}, label=f"check_printability:{name}")
        part: dict[str, Any] = {
            "name": name,
            "printable": check.get("printable"),
            "requires_supports": check.get("requires_supports"),
            "warnings": check.get("warnings") or [],
            "stl_path": check.get("stl_path"),
            "gcode_path": None,
            "current_orientation_printable": check.get("current_orientation_printable"),
        }
        if check.get("current_orientation_printable") is False:
            part["recommended_orientation"] = check.get("recommended_orientation")
            part["remediation_hint"] = check.get("remediation_hint")
        print_stl = check.get("stl_path")
        pose = _orientation_to_apply(check) if auto_orient and check.get("ok") and print_stl else None
        if pose is not None:
            try:
                print_stl = _orient_stl(print_stl, pose["rotation"])
            except Exception as exc:  # noqa: BLE001 — fall back to the modelled pose, reported as a failed step
                steps.append({"step": f"auto_orient:{name}", "tool": "auto_orient", "ok": False, "error": str(exc)})
                print_stl, pose = check.get("stl_path"), None
            else:
                part["applied_orientation"] = {
                    key: pose[key] for key in ("up_axis", "bed_face", "rotation", "requires_supports", "height_mm")
                }
                part["print_stl_path"] = print_stl
        if check.get("ok") and print_stl and (check.get("printable") or pose is not None):
            sliced = step(
                "slice_stl_to_gcode",
                {"stl_filepath": print_stl, "printer_profile": printer_profile},
                label=f"slice_stl_to_gcode:{name}",
            )
            part["gcode_path"] = sliced.get("gcode_path") if sliced.get("ok") else None
        parts.append(part)

    ready = [{"part": p["name"], "gcode_filepath": p["gcode_path"]} for p in parts if p["gcode_path"]]
    not_ready = [p["name"] for p in parts if not p["gcode_path"]]
    failed_steps = [s["step"] for s in steps if not s["ok"]]
    complete = not failed_steps and not not_ready
    next_step = (
        "Nothing has been printed. To print, call dispatch_to_printer(printer_ip, gcode_filepath) once per "
        "entry in ready_to_print; the user approves each print."
        if ready
        else "Nothing is ready to print."
    )
    if not complete:
        next_step += (
            f" Incomplete: failed steps {failed_steps}, parts without G-code {not_ready}; "
            "see steps and parts for the errors and printability warnings."
        )
    auto_oriented = [p["name"] for p in parts if "applied_orientation" in p]
    if auto_oriented:
        next_step += (
            f" Auto-oriented for printing: {auto_oriented} (only their print STL was rotated, see "
            "applied_orientation; the CAD model, URDF and drawing keep the modelled pose)."
        )
    # Still worth a look before printing: parts printed as modelled while
    # needing supports or not fitting, and auto-oriented ones that still need
    # supports. An auto-oriented, support-free part is resolved.
    orientation_hints = {
        p["name"]: p["remediation_hint"]
        for p in parts
        if p.get("remediation_hint") and (p.get("applied_orientation") or {}).get("requires_supports") is not False
    }
    if orientation_hints:
        next_step += (
            f" Not support-free (or not fitting) as printed: {sorted(orientation_hints)}; see orientation_hints "
            "for how to reorient or fix each before printing."
        )
    return {
        "ok": True,
        "complete": complete,
        "failed_steps": failed_steps,
        "name": assembly,
        "artifacts": {
            "bom_csv": bom.get("path"),
            "urdf": urdf.get("path"),
            "blueprint_pdf": blueprint.get("path") if blueprint.get("ok") else None,
            "gcode": [r["gcode_filepath"] for r in ready],
        },
        "bom": {k: bom.get(k) for k in ("materials", "part_count", "total_mass_g", "total_cost", "currency")},
        "collisions": collisions.get("collisions") or [],
        "parts": parts,
        "orientation_hints": orientation_hints,
        "auto_oriented": auto_oriented,
        "steps": steps,
        "ready_to_print": ready,
        "not_ready_to_print": not_ready,
        "next_step": next_step,
    }

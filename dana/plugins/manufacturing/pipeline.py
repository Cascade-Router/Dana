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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

RunTool = Callable[[str, dict[str, Any]], dict[str, Any]]


def execute_manufacturing_pipeline(
    assembly_name: str,
    run_tool: RunTool,
    *,
    material: str = "PLA",
    part_materials: dict[str, str] | None = None,
    printer_profile: str = "mk4_default",
) -> dict[str, Any]:
    """Run every pre-print manufacturing step for ``assembly_name``.

    The BOM runs first and its failure aborts the pipeline (``ok: False``;
    it is also what lists the parts to print). After that a failed step is
    recorded and the pipeline carries on, since the URDF, the drawing and the
    prints don't depend on each other, and the result stays ``ok: True`` so
    the artifacts that were made reach the agent (dispatch_tool_call reduces
    a failed result to its error message). ``complete`` is true only when
    every step succeeded and every part has G-code. A part is sliced only if
    check_printability calls it printable.
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
    blueprint = step("generate_2d_blueprint", {"object_name": assembly, "filename": f"{assembly}_blueprint"})

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
        if check.get("ok") and check.get("printable") and check.get("stl_path"):
            sliced = step(
                "slice_stl_to_gcode",
                {"stl_filepath": check["stl_path"], "printer_profile": printer_profile},
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
    # The STL is sliced as modelled, so a part that needs supports (or only
    # fits rotated) in that pose is worth a look before printing.
    orientation_hints = {p["name"]: p["remediation_hint"] for p in parts if p.get("remediation_hint")}
    if orientation_hints:
        next_step += (
            f" Not support-free (or not fitting) as modelled: {sorted(orientation_hints)}. Any G-code for them "
            "was sliced as modelled; see orientation_hints for how to reorient or fix each before printing."
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
        "steps": steps,
        "ready_to_print": ready,
        "not_ready_to_print": not_ready,
        "next_step": next_step,
    }

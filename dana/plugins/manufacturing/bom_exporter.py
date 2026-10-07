"""Bill of materials for an assembly: per-part volume -> mass -> cost, written
as a CSV the agent (or a person) can read back before dispatching to the
printer/CNC pipeline.

Pure Python on purpose. Geometry comes in as a list of per-part volumes from
whichever CAD driver is active (the real FreeCAD engine's
``assembly_part_volumes``, or the mock driver's trimesh volumes), so the
mass/cost arithmetic and the CSV format are identical for both and testable
without FreeCAD.

Units: volumes arrive in mm^3 (FreeCAD's native unit), densities are g/cm^3
and prices are per kg, as listed in ``materials.json``.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from dana.paths import AGENT_WORKSPACE_DIR

_MATERIALS_PATH = Path(__file__).with_name("materials.json")

# Inside the agent's sandboxed workspace (not freecad_output/exports) so the
# agent can read the CSV back with read_file after generating it.
_EXPORT_DIR = AGENT_WORKSPACE_DIR / "exports"

CSV_HEADER = ("Part Name", "Volume (cm3)", "Mass (g)", "Material", "Cost")

_MM3_PER_CM3 = 1000.0


def _norm(name: str) -> str:
    return re.sub(r"[\s_\-]+", " ", name).strip().lower()


def load_materials() -> dict[str, Any]:
    """The parsed ``materials.json``: ``{"currency": str, "materials": {name: spec}}``."""
    return json.loads(_MATERIALS_PATH.read_text(encoding="utf-8"))


def resolve_material(name: str, library: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """``(canonical name, spec)`` for ``name`` matched case-insensitively
    against each material's key and aliases (spaces, underscores and hyphens
    are interchangeable), or ``None`` if nothing matches."""
    wanted = _norm(name or "")
    for key, spec in library["materials"].items():
        if wanted in {_norm(key), *(_norm(a) for a in spec.get("aliases") or ())}:
            return key, spec
    return None


def build_bom(assembly_name: str, parts: list[dict[str, Any]], material: str = "PLA") -> dict[str, Any]:
    """Compute mass and cost for every part in ``parts`` (``{"name",
    "volume_mm3"}`` dicts) in ``material`` and write
    ``<agent_workspace>/exports/<assembly_name>_bom.csv``.

    Returns ``{"ok": True, "path", "material", "currency", "parts": [...],
    "total_mass_g", "total_cost", ...}``, or ``{"ok": False, "error"}`` for an
    unknown material, an assembly with no solid parts, or a part whose volume
    isn't positive (an open shell or inverted solid would otherwise produce a
    silently wrong mass).
    """
    library = load_materials()
    resolved = resolve_material(material, library)
    if resolved is None:
        return {
            "ok": False,
            "error": f"generate_assembly_bom: unknown material {material!r} — choose one of "
            f"{sorted(library['materials'])}",
        }
    material_name, spec = resolved
    if not parts:
        return {
            "ok": False,
            "error": f"generate_assembly_bom: assembly {assembly_name!r} has no parts with solid geometry "
            "— add some with add_parts_to_assembly first",
        }
    bad = [p["name"] for p in parts if not float(p.get("volume_mm3") or 0.0) > 0.0]
    if bad:
        return {
            "ok": False,
            "error": f"generate_assembly_bom: part(s) {bad} have no positive solid volume (open shell or "
            "inverted solid) — fix their geometry before costing them",
        }

    density = float(spec["density_g_cm3"])
    cost_per_kg = float(spec["cost_per_kg"])
    rows = []
    total_mass_g = 0.0
    for part in parts:
        volume_cm3 = float(part["volume_mm3"]) / _MM3_PER_CM3
        mass_g = volume_cm3 * density
        total_mass_g += mass_g
        rows.append(
            {
                "name": part["name"],
                "volume_cm3": round(volume_cm3, 4),
                "mass_g": round(mass_g, 4),
                "cost": round(mass_g / 1000.0 * cost_per_kg, 4),
            }
        )

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", assembly_name).strip("_") or "assembly"
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _EXPORT_DIR / f"{safe}_bom.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow((row["name"], row["volume_cm3"], row["mass_g"], material_name, row["cost"]))

    return {
        "ok": True,
        "name": assembly_name,
        "path": str(path),
        "material": material_name,
        "process": spec.get("process"),
        "density_g_cm3": density,
        "cost_per_kg": cost_per_kg,
        "currency": library["currency"],
        "parts": rows,
        "part_count": len(rows),
        "total_mass_g": round(total_mass_g, 4),
        "total_cost": round(total_mass_g / 1000.0 * cost_per_kg, 4),
    }

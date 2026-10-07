"""Plugin entry point (manifest.json) exposing the manufacturing pipeline to
the ReAct agent as ``run_full_manufacturing_pipeline``.

Each pipeline step calls the existing agent tool's handler from
``dana.core.react_dispatch.TOOL_HANDLERS`` (plugins may import core; only the
reverse is forbidden), with the active CAD engine. Imported lazily because
react_dispatch loads this plugin while it is itself still importing.
"""

from __future__ import annotations

from typing import Any


def run_full_manufacturing_pipeline(
    assembly_name: str,
    material: str = "PLA",
    part_materials: dict[str, str] | None = None,
    printer_profile: str = "mk4_default",
) -> dict[str, Any]:
    from dana.core import react_dispatch
    from dana.platform.factory import get_cad_engine
    from dana.plugins.manufacturing.pipeline import execute_manufacturing_pipeline

    engine = get_cad_engine()

    def run_tool(tool_id: str, args: dict[str, Any]) -> dict[str, Any]:
        # None for the control plane: none of the pipeline's tools use it.
        return react_dispatch.TOOL_HANDLERS[tool_id](args, engine, None)

    return execute_manufacturing_pipeline(
        assembly_name, run_tool, material=material, part_materials=part_materials, printer_profile=printer_profile
    )

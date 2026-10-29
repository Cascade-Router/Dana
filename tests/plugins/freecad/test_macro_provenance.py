"""Exported macros carry a provenance header: session, export time, the
editable parameter table and build metrics. It is comment-only, so the
macro still parses and keeps its docstring."""

from __future__ import annotations

import ast

from dana.plugins.freecad.call_log import CadCallLog
from dana.plugins.freecad.py_export import render_macro_script


def _log() -> CadCallLog:
    log = CadCallLog()
    log.record("create_freecad_box", {}, ok=True,
               result={"name": "Plate", "dimensions": {"length": 60, "width": 40, "height": 5}})
    return log


PROVENANCE = {
    "title": "Bracket with hole",
    "session_id": "abc123",
    "exported_at": "2026-10-29 21:00 UTC",
    "parameters": [{"id": "0.length", "label": "Plate.length", "value": 60.0, "unit": "mm"}],
    "llm_calls": 12,
    "zero_token_rebuilds": 3,
    "total_usd": 0.0123,
    "tokens": 45210,
}


def test_header_lists_session_parameters_and_build_metrics() -> None:
    source = render_macro_script(_log(), provenance=PROVENANCE)
    header = source.split('"""', 1)[0]
    assert "# Session:  Bracket with hole" in header
    assert "# Exported: 2026-10-29 21:00 UTC (session abc123)" in header
    assert "Plate.length = 60.0 mm" in header
    assert "# Build: 12 LLM calls (45,210 tokens, ~$0.0123), 3 zero-token rebuilds" in header
    tree = ast.parse(source)
    assert ast.get_docstring(tree).startswith("Standalone FreeCAD macro")


def test_a_hostile_title_cannot_escape_the_comment() -> None:
    hostile = {**PROVENANCE, "title": "x\nimport os; os.remove('important')\n#"}
    source = render_macro_script(_log(), provenance=hostile)
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Import) and node.names[0].name == "os" and node.lineno < 5
                   for node in tree.body)
    assert "# Session:  x import os; os.remove('important') #" in source


def test_no_header_without_provenance() -> None:
    assert render_macro_script(_log()).startswith('"""Standalone FreeCAD macro')

"""Targeted tests for 2D Blueprint Generation:
dana.plugins.freecad.techdraw_export.generate_2d_blueprint's input
validation and dry-run behavior, plus one end-to-end dispatch_tool_call
integration check. Not a full test suite by design — dry-run mode only, no
live FreeCADCmd required (the real TechDraw/DXF/PDF pipeline was validated
manually against a live FreeCAD install during development; see the
module's own docstring for how headless PDF export actually works).
"""

from __future__ import annotations

import json

import pytest

from dana.plugins.freecad.techdraw_export import generate_2d_blueprint


@pytest.fixture(autouse=True)
def _dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")


@pytest.fixture
def existing_path(tmp_path) -> str:
    """generate_2d_blueprint checks source_path exists before consulting
    dry-run mode (matching align_objects/create_assembly_mate's own
    precedent — basic input validation isn't skipped under dry-run)."""
    fcstd = tmp_path / "Box.FCStd"
    fcstd.write_bytes(b"not a real FreeCAD document, just needs to exist")
    return str(fcstd)


def test_default_views_are_all_four_standard_projections(existing_path: str):
    result = json.loads(generate_2d_blueprint(existing_path))
    assert result["ok"] is True
    assert result["views"] == ["Front", "Top", "Right", "Isometric"]
    assert result["page_size"] == "a4"


def test_custom_view_subset_and_letter_page_size(existing_path: str):
    result = json.loads(generate_2d_blueprint(existing_path, views=["Top"], page_size="Letter"))
    assert result["ok"] is True
    assert result["views"] == ["Top"]
    assert result["page_size"] == "letter"


def test_custom_filename_is_honored(existing_path: str):
    result = json.loads(generate_2d_blueprint(existing_path, filename="MyDrawing"))
    assert result["name"] == "MyDrawing"


def test_rejects_unknown_view_name(existing_path: str):
    result = json.loads(generate_2d_blueprint(existing_path, views=["Bottom"]))
    assert result["ok"] is False
    assert "unknown view" in result["error"]


def test_rejects_unknown_page_size(existing_path: str):
    result = json.loads(generate_2d_blueprint(existing_path, page_size="Legal"))
    assert result["ok"] is False
    assert "unknown page_size" in result["error"]


def test_rejects_missing_source_file():
    result = json.loads(generate_2d_blueprint("C:/definitely/not/a/real/path.FCStd"))
    assert result["ok"] is False
    assert "source_path not found" in result["error"]


def test_empty_views_list_falls_back_to_defaults(existing_path: str):
    """An empty list is treated the same as not passing views at all —
    consistent with how the dispatch handler already normalizes an empty
    'views' argument to None before calling this function."""
    result = json.loads(generate_2d_blueprint(existing_path, views=[]))
    assert result["ok"] is True
    assert result["views"] == ["Front", "Top", "Right", "Isometric"]


# --------------------------------------------------------------------------
# End-to-end dispatch_tool_call integration
# --------------------------------------------------------------------------


def test_dispatch_tool_call_generate_2d_blueprint_end_to_end(existing_path: str):
    """generate_2d_blueprint bypasses the engine/control_plane driver
    abstraction by design (same as insert_standard_part) — engine=None/
    control_plane=None here proves dispatch never needs them for this
    tool_id, and object_name resolution via _OBJECT_PATH_REGISTRY is
    exercised for real."""
    from dana.core import react_dispatch as rd
    from dana.tools.schema import ToolCall

    rd._OBJECT_PATH_REGISTRY["BlueprintTestBox"] = existing_path

    result = rd.dispatch_tool_call(
        ToolCall(
            tool_id="generate_2d_blueprint",
            arguments={"object_name": "BlueprintTestBox", "views": ["Front", "Isometric"]},
        ),
        engine=None,
        control_plane=None,
    )
    assert result.ok is True
    assert result.payload["views"] == ["Front", "Isometric"]
    assert rd.is_mutating_tool("generate_2d_blueprint") is False


def test_dispatch_tool_call_generate_2d_blueprint_unknown_object():
    from dana.core import react_dispatch as rd
    from dana.tools.schema import ToolCall

    result = rd.dispatch_tool_call(
        ToolCall(tool_id="generate_2d_blueprint", arguments={"object_name": "NeverCreated"}),
        engine=None,
        control_plane=None,
    )
    assert result.ok is False
    assert "object_name" in result.message

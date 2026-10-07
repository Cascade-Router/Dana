"""Targeted tests for 2D Blueprint Generation:
dana.plugins.freecad.techdraw_export.generate_2d_blueprint's input
validation and dry-run behavior, plus one end-to-end dispatch_tool_call
integration check, all in dry-run mode — plus the DXF->PDF/SVG step for
real, on a DXF built with ezdxf, with FreeCADCmd itself mocked. The live
TechDraw run is in test_techdraw_export_live.py (skipped without FreeCAD).
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Any

import pytest

from dana.plugins.freecad import techdraw_export
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

    rd._object_registry()["BlueprintTestBox"] = existing_path

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


# -- the real DXF -> PDF/SVG step (no FreeCAD) -----------------------------------


def _write_part_dxf(path: Path) -> None:
    """A stand-in for a TechDraw page: a 100x50 outline with a hole, in the
    default (ACI 7) layer color TechDraw exports with."""
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(40, 60), (140, 60), (140, 110), (40, 110)], close=True)
    msp.add_circle((90, 85), 12)
    doc.saveas(path)


def _pdf_content(pdf: bytes) -> bytes:
    """All of the PDF's content streams, inflated."""
    out = b""
    for raw in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.S):
        try:
            out += zlib.decompress(raw)
        except zlib.error:
            out += raw
    return out


def test_dxf_renders_to_a_page_sized_pdf_with_visible_black_strokes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(techdraw_export, "_EXPORT_DIR", tmp_path / "exports")
    dxf = tmp_path / "page.dxf"
    _write_part_dxf(dxf)

    pdf_path = techdraw_export._render_dxf_to_pdf(str(dxf), "bracket", (297.0, 210.0))

    assert pdf_path == tmp_path / "exports" / "bracket.pdf"
    pdf = pdf_path.read_bytes()
    assert pdf.startswith(b"%PDF-") and b"%%EOF" in pdf[-64:]
    # A4 landscape in points (1 mm = 72/25.4 pt).
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
    assert box and abs(float(box[1]) - 841.89) < 1 and abs(float(box[2]) - 595.28) < 1
    content = _pdf_content(pdf)
    # Black strokes and real path segments: TechDraw's ACI-7 lines used to
    # render white-on-white (d94cb5f), a PDF with nothing visible on it.
    assert re.search(rb"(?<![\d.])0 G\b|(?<![\d.])0 0 0 RG\b", content), "no black stroke color"
    assert len(re.findall(rb" l\b", content)) >= 4, "outline segments missing"
    # Drawn 1:1 in page millimetres: the hole's centre (x = 90 mm) is at
    # 90 * 72 / 25.4 = 255.118 pt, not stretched to fill the page.
    assert b"255.11811" in content


def test_dxf_renders_to_svg_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(techdraw_export, "_EXPORT_DIR", tmp_path / "exports")
    dxf = tmp_path / "page.dxf"
    _write_part_dxf(dxf)
    svg_path = techdraw_export._render_dxf_to_svg(str(dxf), "bracket", (297.0, 210.0))
    svg = svg_path.read_text(encoding="utf-8")
    assert svg_path.suffix == ".svg" and "<path" in svg
    root = re.search(r'<svg[^>]*width="297mm" height="210mm" viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    assert root, svg[:300]
    per_mm_x, per_mm_y = float(root[1]) / 297.0, float(root[2]) / 210.0
    # 1:1 like the PDF: the 100x50 outline at (40, 60) mm keeps that size and
    # position (SVG y runs down from the top edge: 210 - 110 = 100 mm).
    outline = re.search(r'd="M ([\d.]+) ([\d.]+) l ([\d.]+) 0 l 0 -([\d.]+)', svg)
    assert outline, svg
    x, y, width, height = (float(v) for v in outline.groups())
    assert x / per_mm_x == pytest.approx(40, abs=0.01)
    assert y / per_mm_y == pytest.approx(210 - 60, abs=0.01)
    assert width / per_mm_x == pytest.approx(100, abs=0.01)
    assert height / per_mm_y == pytest.approx(50, abs=0.01)


def test_pipeline_turns_the_freecad_dxf_into_pdf_and_svg_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_path: str
) -> None:
    """generate_2d_blueprint end to end with FreeCADCmd mocked: the mock
    writes a DXF where the generated script asked TechDraw to."""
    pytest.importorskip("matplotlib")
    monkeypatch.delenv("DANA_OS_DRY_RUN")
    monkeypatch.setattr(techdraw_export, "_EXPORT_DIR", tmp_path / "exports")
    seen: dict[str, Any] = {}

    def fake_freecad(script: str, **_kw: Any) -> dict[str, Any]:
        seen["script"] = script
        seen["dxf"] = Path(re.search(r"writeDXFPage\(page, '([^']+)'\)", script)[1])
        _write_part_dxf(seen["dxf"])
        return {"ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(techdraw_export, "_run_freecad_script", fake_freecad)
    result = json.loads(
        generate_2d_blueprint(existing_path, views=["Top"], filename="plate", object_name="Plate")
    )

    assert result["ok"] is True, result
    assert Path(result["path"]) == tmp_path / "exports" / "plate.pdf"
    assert Path(result["path"]).read_bytes().startswith(b"%PDF-")
    assert Path(result["svg_path"]).is_file()
    assert "Default_Template_A4_Landscape.svg" in seen["script"]
    assert "'Top'" in seen["script"] and "'Front'" not in seen["script"]
    # The named object is resolved, never "the first unreferenced object".
    assert "resolve_object(doc, 'Plate')" in seen["script"]
    assert "not o.InList" not in seen["script"]
    assert not seen["dxf"].exists(), "the temporary DXF must be deleted"


def test_pipeline_reports_a_freecad_failure_without_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_path: str
) -> None:
    monkeypatch.delenv("DANA_OS_DRY_RUN")
    monkeypatch.setattr(techdraw_export, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(
        techdraw_export, "_run_freecad_script", lambda *_a, **_k: {"ok": False, "error": "TechDraw crashed"}
    )
    result = json.loads(generate_2d_blueprint(existing_path))
    assert result["ok"] is False and "TechDraw crashed" in result["error"]
    assert not (tmp_path / "exports").exists()

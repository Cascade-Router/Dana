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


def _write_part_dxf(path: Path, layer: str = "0") -> None:
    """A stand-in for a TechDraw page: a 100x50 outline with a hole, in the
    default (ACI 7) layer color TechDraw exports with. TechDraw puts each
    view on its own ``View<Name>`` layer; geometry on layer "0" belongs to no
    view, so it gets no dimension callouts."""
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    if layer not in doc.layers:
        doc.layers.add(layer)
    msp = doc.modelspace()
    msp.add_lwpolyline([(40, 60), (140, 60), (140, 110), (40, 110)], close=True, dxfattribs={"layer": layer})
    msp.add_circle((90, 85), 12, dxfattribs={"layer": layer})
    doc.saveas(path)


def _pdf_text(pdf: bytes) -> list[str]:
    """Each text run in the PDF's content. Matplotlib's TrueType (fonttype 42)
    runs are TJ arrays of 2-byte strings, e.g. [ (\\x001) 0.58 (\\x000) ] TJ."""
    runs = []
    for array in re.findall(rb"\[(.*?)\]\s*TJ", _pdf_content(pdf), re.S):
        chars = b"".join(re.findall(rb"\((.*?)(?<!\\)\)", array, re.S))
        runs.append(chars.replace(b"\x00", b"").decode("latin-1"))
    return runs


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
    svg_path, sheet = techdraw_export._render_dxf_to_svg(str(dxf), "bracket", (297.0, 210.0))
    svg = svg_path.read_text(encoding="utf-8")
    assert svg_path.suffix == ".svg" and "<path" in svg
    assert sheet.dimensions == [] and "<text" not in svg  # layer "0" is not a view: nothing to dimension
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


# -- dimension callouts -----------------------------------------------------------------


def _svg_texts(svg: str) -> dict[str, tuple[float, float, bool]]:
    """{label: (x, y, rotated)} for every <text> element, in viewBox units."""
    return {
        m["text"]: (float(m["x"]), float(m["y"]), "rotate(" in m["attrs"])
        for m in re.finditer(r'<text x="(?P<x>[\d.]+)" y="(?P<y>[\d.]+)"(?P<attrs>[^>]*)>(?P<text>[^<]*)</text>', svg)
    }


def test_view_gets_width_and_height_callouts_as_real_text_in_svg_and_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(techdraw_export, "_EXPORT_DIR", tmp_path / "exports")
    dxf = tmp_path / "page.dxf"
    _write_part_dxf(dxf, layer="ViewFront")  # the 100x50 outline at (40, 60) mm

    svg_path, sheet = techdraw_export._render_dxf_to_svg(str(dxf), "bracket", (297.0, 210.0))
    pdf = techdraw_export._render_dxf_to_pdf(str(dxf), "bracket", (297.0, 210.0)).read_bytes()

    assert sheet.dimensions == [{"view": "Front", "width_mm": 100.0, "height_mm": 50.0}]
    svg = svg_path.read_text(encoding="utf-8")
    per_mm = float(re.search(r'viewBox="0 0 ([\d.]+)', svg)[1]) / 297.0
    texts = _svg_texts(svg)
    assert set(texts) == {"100 mm", "50 mm"}
    # Width label centred under the outline, outside it; height label centred
    # to its left, rotated. (SVG y runs down: page y = 210 - svg y.)
    x, y, rotated = texts["100 mm"]
    assert x / per_mm == pytest.approx(90, abs=0.01) and 210 - y / per_mm < 60 and not rotated
    x, y, rotated = texts["50 mm"]
    assert x / per_mm < 40 and 210 - y / per_mm == pytest.approx(85, abs=0.01) and rotated
    assert 'fill="#000000"' in svg

    assert pdf.startswith(b"%PDF-") and b"/ToUnicode" in pdf  # searchable text
    assert sorted(_pdf_text(pdf)) == ["100 mm", "50 mm"]


def test_include_dimensions_false_renders_exactly_the_bare_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(techdraw_export, "_EXPORT_DIR", tmp_path / "exports")
    view_dxf, bare_dxf = tmp_path / "view.dxf", tmp_path / "bare.dxf"
    _write_part_dxf(view_dxf, layer="ViewFront")
    _write_part_dxf(bare_dxf)  # layer "0": no view, so the renderers add nothing

    off_svg, sheet = techdraw_export._render_dxf_to_svg(str(view_dxf), "off", (297.0, 210.0), include_dimensions=False)
    bare_svg, _ = techdraw_export._render_dxf_to_svg(str(bare_dxf), "bare", (297.0, 210.0))
    on_svg, _ = techdraw_export._render_dxf_to_svg(str(view_dxf), "on", (297.0, 210.0))
    off_pdf = techdraw_export._render_dxf_to_pdf(str(view_dxf), "off", (297.0, 210.0), include_dimensions=False)
    bare_pdf = techdraw_export._render_dxf_to_pdf(str(bare_dxf), "bare", (297.0, 210.0))

    assert sheet.dimensions == []
    assert off_svg.read_text(encoding="utf-8") == bare_svg.read_text(encoding="utf-8")
    assert _pdf_content(off_pdf.read_bytes()) == _pdf_content(bare_pdf.read_bytes())
    assert _pdf_text(off_pdf.read_bytes()) == []
    # The callouts' own lines (dimension, extension and arrow strokes) are extra paths.
    assert on_svg.read_text(encoding="utf-8").count("<path") > off_svg.read_text(encoding="utf-8").count("<path")


def test_isometric_view_is_not_dimensioned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(techdraw_export, "_EXPORT_DIR", tmp_path / "exports")
    dxf = tmp_path / "page.dxf"
    _write_part_dxf(dxf, layer="ViewIsometric")
    svg_path, sheet = techdraw_export._render_dxf_to_svg(str(dxf), "iso", (297.0, 210.0))
    assert sheet.dimensions == [] and "<text" not in svg_path.read_text(encoding="utf-8")


# -- drawing scale --------------------------------------------------------------------

_A4 = (297.0, 210.0)


def _write_views_dxf(path: Path, length: float, width: float, height: float) -> None:
    """What TechDraw writes at 1:1 for a length x width x height box: each view
    a rectangle centred on its A4 layout slot, on its own View<Name> layer
    (the isometric one approximated by its bounding rectangle)."""
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    msp = doc.modelspace()
    # True isometric projection of the box's bounding rectangle.
    iso_w, iso_h = (length + width) * 0.7071, height * 0.8165 + (length + width) * 0.4082
    for view, (w, h) in {"Front": (length, height), "Top": (length, width), "Right": (width, height),
                         "Isometric": (iso_w, iso_h)}.items():
        fx, fy = techdraw_export._VIEW_LAYOUT[view.lower()]["slot"]
        cx, cy = fx * _A4[0], fy * _A4[1]
        doc.layers.add(f"View{view}")
        msp.add_lwpolyline(
            [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)],
            close=True,
            dxfattribs={"layer": f"View{view}"},
        )
    doc.saveas(path)


def test_auto_scale_shrinks_a_large_part_to_fit_and_labels_its_true_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ezdxf import bbox

    monkeypatch.setattr(techdraw_export, "_EXPORT_DIR", tmp_path / "exports")
    dxf = tmp_path / "page.dxf"
    _write_views_dxf(dxf, 300, 200, 100)  # overflows A4 at 1:1

    assert techdraw_export._load_page(str(dxf), _A4).fits_page is False  # the default 1:1
    sheet = techdraw_export._load_page(str(dxf), _A4, scale="auto")

    # The Top view's 200 mm depth plus its callouts limits it to ~0.27, so 1:4.
    assert (sheet.scale, sheet.fits_page) == (0.25, True)
    extents = bbox.extents(sheet.doc.modelspace())  # views and callouts
    assert extents.extmin.x >= 0 and extents.extmin.y >= 0 and extents.extmax.x <= 297 and extents.extmax.y <= 210
    assert sheet.dimensions == [
        {"view": "Front", "width_mm": 300.0, "height_mm": 100.0},
        {"view": "Top", "width_mm": 300.0, "height_mm": 200.0},
        {"view": "Right", "width_mm": 200.0, "height_mm": 100.0},
    ]
    svg_path, _ = techdraw_export._render_dxf_to_svg(str(dxf), "big", _A4, scale="auto")
    labels = sorted(_svg_texts(svg_path.read_text(encoding="utf-8")))
    assert labels == ["100 mm", "200 mm", "300 mm", "SCALE 1:4"]
    pdf = techdraw_export._render_dxf_to_pdf(str(dxf), "big", _A4, scale="auto").read_bytes()
    assert "SCALE 1:4" in _pdf_text(pdf) and "300 mm" in _pdf_text(pdf)


def test_auto_scale_enlarges_a_small_part(tmp_path: Path) -> None:
    dxf = tmp_path / "page.dxf"
    _write_views_dxf(dxf, 10, 10, 10)
    sheet = techdraw_export._load_page(str(dxf), _A4, scale="auto")
    assert (sheet.scale, sheet.fits_page) == (5.0, True)
    assert sheet.dimensions[0] == {"view": "Front", "width_mm": 10.0, "height_mm": 10.0}


def test_explicit_scale_halves_the_views_but_not_the_labels(tmp_path: Path) -> None:
    dxf = tmp_path / "page.dxf"
    _write_views_dxf(dxf, 100, 50, 20)
    front_at_1 = techdraw_export._view_boxes(techdraw_export._load_page(str(dxf), _A4).doc)["Front"]
    sheet = techdraw_export._load_page(str(dxf), _A4, scale=0.5)

    x0, y0, x1, y1 = techdraw_export._view_boxes(sheet.doc)["Front"]
    assert (x1 - x0, y1 - y0) == (pytest.approx(50.0), pytest.approx(10.0))
    # Scaled about its own centre: the layout slot is unchanged.
    assert (x0 + x1) / 2 == pytest.approx((front_at_1[0] + front_at_1[2]) / 2)
    assert sheet.dimensions[0] == {"view": "Front", "width_mm": 100.0, "height_mm": 20.0}
    assert [label.text for label in sheet.labels if label.text.startswith("SCALE")] == ["SCALE 1:2"]


@pytest.mark.parametrize(("factor", "text"), [(1.0, "1:1"), (0.5, "1:2"), (0.25, "1:4"), (0.2, "1:5"), (0.1, "1:10"),
                                              (2.0, "2:1"), (5.0, "5:1"), (0.3, "1:3.33")])
def test_scale_is_written_in_iso_notation(factor: float, text: str) -> None:
    assert techdraw_export._scale_text(factor) == text


@pytest.mark.parametrize("scale", ["huge", 0, -1, 1000])
def test_rejects_a_bad_scale(existing_path: str, scale: Any) -> None:
    result = json.loads(generate_2d_blueprint(existing_path, scale=scale))
    assert result["ok"] is False and "scale must be 'auto' or a number" in result["error"]


@pytest.mark.parametrize(("value", "label"), [(100.0, "100 mm"), (50.004, "50 mm"), (12.3456, "12.35 mm"), (0.5, "0.5 mm")])
def test_dimension_labels_round_to_two_decimals(value: float, label: str) -> None:
    assert techdraw_export._format_mm(value) == label


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


@pytest.mark.parametrize("include_dimensions", [True, False])
def test_pipeline_reports_each_dimensioned_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_path: str, include_dimensions: bool
) -> None:
    pytest.importorskip("matplotlib")
    from dana.core import react_dispatch as rd
    from dana.tools.schema import ToolCall

    monkeypatch.delenv("DANA_OS_DRY_RUN")
    monkeypatch.setattr(techdraw_export, "_EXPORT_DIR", tmp_path / "exports")
    seen: dict[str, str] = {}

    def fake_freecad(script: str, **_kw: Any) -> dict[str, Any]:
        seen["script"] = script
        _write_part_dxf(Path(re.search(r"writeDXFPage\(page, '([^']+)'\)", script)[1]), layer="ViewTop")
        return {"ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(techdraw_export, "_run_freecad_script", fake_freecad)
    result = json.loads(
        generate_2d_blueprint(existing_path, views=["Top"], object_name="Plate", include_dimensions=include_dimensions)
    )

    assert result["ok"] is True, result
    # Pinned to 1:1, so the measured DXF extents are the part's real millimetres.
    assert 'view.ScaleType = "Custom"' in seen["script"] and "view.Scale = 1.0" in seen["script"]
    summary = rd.summarize_result(
        ToolCall(tool_id="generate_2d_blueprint", arguments={}), rd.ToolResult("generate_2d_blueprint", True, result, "", 0)
    )
    if include_dimensions:
        assert result["dimensions"] == [{"view": "Top", "width_mm": 100.0, "height_mm": 50.0}]
        assert "Top 100 x 50 mm" in summary
    else:
        assert result["dimensions"] == []
        assert "Dimensioned" not in summary


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

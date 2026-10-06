"""2D blueprint generation against the real FreeCADCmd + TechDraw: builds
parts in a fresh session document, runs generate_2d_blueprint, and checks
the PDF that lands in the export directory. Skipped when FreeCADCmd isn't
installed (it isn't in CI).

The TechDraw DXF is captured on its way to the renderer, so the tests can
check WHICH object was drawn and in what coordinates, not just that a file
appeared.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from dana.plugins.freecad import engine, techdraw_export
from dana.session_context import DEFAULT_SESSION_ID, set_session_id

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        engine.detect_freecadcmd() is None,
        reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
    ),
]

ezdxf = pytest.importorskip("ezdxf")
pytest.importorskip("matplotlib")


@pytest.fixture(autouse=True)
def _isolated_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(engine, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(techdraw_export, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)
    set_session_id(f"blueprint-{uuid.uuid4().hex[:8]}")
    yield
    set_session_id(DEFAULT_SESSION_ID)


@pytest.fixture
def captured_dxf(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Extents of the DXF TechDraw wrote, read before the pipeline deletes it."""
    seen: dict[str, Any] = {}
    render = techdraw_export._render_dxf_to_pdf

    def spy(dxf_path: str, name: str, page_size_mm: tuple[float, float]) -> Path:
        from ezdxf import bbox

        doc = ezdxf.readfile(dxf_path)
        extents = bbox.extents(doc.modelspace())
        seen["size"] = (extents.size.x, extents.size.y)
        seen["extmin"] = (extents.extmin.x, extents.extmin.y)
        seen["extmax"] = (extents.extmax.x, extents.extmax.y)
        return render(dxf_path, name, page_size_mm)

    monkeypatch.setattr(techdraw_export, "_render_dxf_to_pdf", spy)
    return seen


def _ok(raw: str) -> dict[str, Any]:
    result = json.loads(raw)
    assert result.get("ok") is True, result
    return result


def test_a_box_becomes_a_valid_pdf_in_the_export_directory(tmp_path: Path, captured_dxf: dict[str, Any]) -> None:
    box = _ok(engine.create_box(80, 40, 20, name="Plate"))
    result = _ok(techdraw_export.generate_2d_blueprint(box["path"], views=["Top"], filename="plate"))

    pdf = Path(result["path"])
    assert pdf == tmp_path / "exports" / "plate.pdf"
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF-") and len(data) > 1000
    assert Path(result["svg_path"]).is_file()
    # The page's DXF is in page millimetres: everything sits on an A4 sheet.
    assert captured_dxf["extmin"][0] >= -1 and captured_dxf["extmin"][1] >= -1
    assert captured_dxf["extmax"][0] <= 298 and captured_dxf["extmax"][1] <= 211


def test_the_named_object_is_drawn_not_the_first_one_in_the_document(captured_dxf: dict[str, Any]) -> None:
    """Every create_* tool writes into the one session document, so the path
    alone doesn't say which object to draw."""
    _ok(engine.create_box(10, 10, 10, name="SmallCube"))
    plate = _ok(engine.create_box(150, 60, 5, name="WidePlate"))
    _ok(techdraw_export.generate_2d_blueprint(plate["path"], views=["Top"], filename="wide", object_name="WidePlate"))

    width, height = captured_dxf["size"]
    # A Top view of the 150x60 plate is wide; the 10x10 cube's would be square
    # (and tiny). Both include the A4 template frame only if TechDraw exports
    # it, so compare the drawn geometry's aspect, not its absolute size.
    assert width > 2 * height, f"drew a {width:.1f} x {height:.1f} view — not WidePlate's top"

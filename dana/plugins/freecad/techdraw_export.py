"""2D Blueprint Generation — projects clean orthographic/isometric views of
a completed 3D object onto a standard drawing-page layout and exports a PDF,
via FreeCAD's TechDraw workbench.

No auto-dimensioning here by design (scripted TechDraw dimensioning is
brittle) — this is purely "project the geometry cleanly," matching the
directive's explicit scope.

Headless PDF export turned out to be the hard part: TechDraw's PDF/SVG page
writers (``TechDrawGui.exportPageAsPdf``/``exportPageAsSvg``) only exist in
the ``TechDrawGui`` module, which needs a live Qt ``FreeCADGui`` instance —
exactly the GUI/focus-stealing dependency this whole engine is built to
avoid. Empirically verified against a real FreeCADCmd install instead:
``TechDraw.writeDXFPage`` exports a fully-templated, multi-view page to DXF
with **no Gui import at all** — the projection/HLR math is pure C++ compute,
not a rendering concern. So the pipeline is two stateless steps: (1) the
FreeCADCmd subprocess (page + views + ``writeDXFPage``) produces a DXF, then
(2) THIS process (which has ``ezdxf``/``matplotlib`` in its own venv —
FreeCADCmd's bundled interpreter does not) renders that DXF to the final PDF.

Bypasses the ``BaseCADEngine`` platform abstraction and calls
``dana.plugins.freecad.engine``'s stateless script-runner directly — same
precedent as ``standard_parts.py``: this is FreeCAD-plugin-specific
tooling, not a cross-platform primitive.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dana.plugins.freecad.engine import (
    _EXPORT_DIR,
    _OK_MARKER,
    _error,
    _dry_run_result,
    _ok,
    _run_freecad_script,
    _safe_name,
)
from dana.platform.factory import IS_HF_SPACE
from dana.security.dry_run import is_dry_run_enabled

# Direction = viewing direction (camera -> object); XDirection = which way
# is "page-right" in 3D for that view. Slot = (x, y) as a FRACTION of the
# page's width/height — a fixed 2x2 layout (Front/Top left column, Right/
# Isometric right column) so multiple views never land on top of each
# other; FreeCAD does NOT auto-position views (every new view defaults to
# dead-center of the page, confirmed empirically), so this script always
# sets X/Y explicitly.
_VIEW_LAYOUT: dict[str, dict[str, Any]] = {
    "front": {"direction": (0.0, -1.0, 0.0), "xdirection": (1.0, 0.0, 0.0), "slot": (0.30, 0.28)},
    "top": {"direction": (0.0, 0.0, -1.0), "xdirection": (1.0, 0.0, 0.0), "slot": (0.30, 0.68)},
    "right": {"direction": (1.0, 0.0, 0.0), "xdirection": (0.0, -1.0, 0.0), "slot": (0.72, 0.28)},
    "isometric": {"direction": (1.0, -1.0, 1.0), "xdirection": (1.0, 1.0, 0.0), "slot": (0.72, 0.68)},
}
_DEFAULT_VIEWS: tuple[str, ...] = ("Front", "Top", "Right", "Isometric")

# (width_mm, height_mm) landscape — matches the physical page size FreeCAD's
# own template assigns (page.PageWidth/PageHeight), read back empirically
# rather than guessed.
_PAGE_SIZES_MM: dict[str, tuple[float, float]] = {
    "a4": (297.0, 210.0),
    "letter": (279.4, 215.9),
}
# Path components under <FreeCAD resource dir>/Mod/TechDraw/Templates/ —
# resolved via App.getResourceDir() INSIDE the subprocess script (this host
# process never imports FreeCAD itself, so it can't know that path).
_PAGE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "a4": ("Default_Template_A4_Landscape.svg",),
    "letter": ("ASME", "USLetter_Landscape.svg"),
}

_BLUEPRINT_SCRIPT = """\
import FreeCAD as App
import TechDraw
import os

doc = App.openDocument({source_path!r})
obj = next((o for o in doc.Objects if not o.InList), doc.Objects[-1])

page = doc.addObject("TechDraw::DrawPage", "BlueprintPage")
template = doc.addObject("TechDraw::DrawSVGTemplate", "BlueprintTemplate")
template.Template = os.path.join(App.getResourceDir(), "Mod", "TechDraw", "Templates", *{template_parts!r})
page.Template = template

for name, direction, xdirection, fx, fy in {view_specs!r}:
    view = doc.addObject("TechDraw::DrawViewPart", "View" + name)
    view.Source = [obj]
    view.Direction = App.Vector(*direction)
    view.XDirection = App.Vector(*xdirection)
    page.addView(view)
    view.X = fx * page.PageWidth
    view.Y = fy * page.PageHeight

doc.recompute()
TechDraw.writeDXFPage(page, {dxf_path!r})
print("{marker} path=" + {dxf_path!r})
"""


def _render_dxf_to_pdf(dxf_path: str, name: str, page_size_mm: tuple[float, float]) -> Path:
    """Renders a TechDraw-exported DXF page to a PDF, sized to the page's
    real physical dimensions — a pure Python step, no FreeCAD involved."""
    import ezdxf
    import matplotlib

    matplotlib.use("Agg")  # headless — never try to open a display/window
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing import matplotlib as ezdxf_matplotlib

    doc = ezdxf.readfile(dxf_path)
    width_mm, height_mm = page_size_mm
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    try:
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0, width_mm)
        ax.set_ylim(0, height_mm)
        ax.set_aspect("equal")
        ax.axis("off")
        Frontend(RenderContext(doc), ezdxf_matplotlib.MatplotlibBackend(ax)).draw_layout(
            doc.modelspace(), finalize=True
        )
        _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _EXPORT_DIR / f"{_safe_name(name)}.pdf"
        # A retry/re-generation with the same name must never attempt to
        # overwrite a file left over from a prior run in place — observed
        # live to trip a "ios_base::failbit set: iostream stream error"
        # further up this same pipeline (see the dxf_path fix below) when a
        # stale file already sits at the target path; removing it first
        # guarantees this write always starts from a clean, unlocked state
        # regardless of what left the old one there.
        out_path.unlink(missing_ok=True)
        fig.savefig(out_path)
    finally:
        plt.close(fig)
    return out_path


def generate_2d_blueprint(
    source_path: str,
    views: Sequence[str] | None = None,
    page_size: str = "A4",
    filename: str | None = None,
) -> str:
    """Projects orthographic (Front/Top/Right) and/or Isometric views of the
    object in ``source_path`` onto a standard drawing page and exports a
    PDF. No auto-dimensioning — clean projected geometry only.
    """
    if IS_HF_SPACE:
        # Same bypass as standard_parts.py's insert_standard_part (see this
        # module's own docstring) — never goes through get_cad_engine()'s
        # Mock/Real switch, always a real FreeCADCmd subprocess. Gated here,
        # at the shell-out itself, regardless of caller.
        return _error("generate_2d_blueprint is disabled in the hosted cloud demo — it requires the real FreeCAD engine.")
    target = Path(source_path)
    if not target.is_file():
        return _error(f"generate_2d_blueprint: source_path not found: {source_path}")

    size_key = (page_size or "A4").strip().lower()
    if size_key not in _PAGE_SIZES_MM:
        return _error(
            f"generate_2d_blueprint: unknown page_size '{page_size}' — "
            f"must be one of {', '.join(sorted(_PAGE_SIZES_MM))}"
        )

    requested = [str(v).strip() for v in (views or _DEFAULT_VIEWS) if str(v).strip()]
    if not requested:
        return _error("generate_2d_blueprint requires at least one view")
    unknown = [v for v in requested if v.lower() not in _VIEW_LAYOUT]
    if unknown:
        return _error(
            f"generate_2d_blueprint: unknown view(s) {unknown} — "
            f"must be one of {', '.join(sorted(_VIEW_LAYOUT))}"
        )

    resolved_name = filename or target.stem
    if is_dry_run_enabled():
        return _dry_run_result(
            "generate_2d_blueprint", name=resolved_name, views=requested, page_size=size_key
        )

    view_specs = [
        (
            name,
            _VIEW_LAYOUT[name.lower()]["direction"],
            _VIEW_LAYOUT[name.lower()]["xdirection"],
            _VIEW_LAYOUT[name.lower()]["slot"][0],
            _VIEW_LAYOUT[name.lower()]["slot"][1],
        )
        for name in requested
    ]

    # tempfile.mkstemp both creates AND opens the file, leaving a 0-byte
    # placeholder Python itself holds/owns the handle for — TechDraw's
    # writeDXFPage below is a C++ ofstream in a SEPARATE FreeCADCmd
    # subprocess, and asking it to open/truncate a path Python just created
    # (with Python's own restrictive mkstemp permissions, and possibly still
    # settling at the OS/filesystem level right after close() on Windows)
    # is exactly the kind of contention that surfaces as a C++ "ios_base::
    # failbit set: iostream stream error" — observed live. Deleting the
    # placeholder immediately guarantees dxf_path is a unique, guaranteed-
    # free filename with NO pre-existing file/handle for the subprocess to
    # contend with; its own ofstream creates it completely fresh.
    fd, dxf_path = tempfile.mkstemp(suffix=".dxf")
    os.close(fd)
    os.unlink(dxf_path)
    try:
        script = _BLUEPRINT_SCRIPT.format(
            source_path=str(target),
            template_parts=_PAGE_TEMPLATES[size_key],
            view_specs=view_specs,
            dxf_path=dxf_path,
            marker=_OK_MARKER,
        )
        result = _run_freecad_script(script)
        if not result["ok"]:
            return _error(f"generate_2d_blueprint failed: {result['error']}")

        try:
            pdf_path = _render_dxf_to_pdf(dxf_path, resolved_name, _PAGE_SIZES_MM[size_key])
        except Exception as exc:  # noqa: BLE001 — surface as a normal tool failure, not a crash
            return _error(f"generate_2d_blueprint: DXF->PDF conversion failed: {exc}")
    finally:
        try:
            os.unlink(dxf_path)
        except OSError:
            pass

    return _ok(name=resolved_name, views=requested, page_size=size_key, path=str(pdf_path))


__all__ = ("generate_2d_blueprint",)

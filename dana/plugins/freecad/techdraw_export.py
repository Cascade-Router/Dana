"""2D Blueprint Generation — projects clean orthographic/isometric views of
a completed 3D object onto a standard drawing-page layout and exports a PDF,
via FreeCAD's TechDraw workbench.

No TechDraw dimension objects (scripted TechDraw dimensioning is brittle).
Instead, each orthographic view gets overall width/height callouts added
after export, measured from that view's own geometry in the DXF (TechDraw
puts each view on its own ``View<Name>`` layer, at scale 1, in page
millimetres), so the numbers are the part's real size. Isometric views get
none: their extents aren't part dimensions. ``include_dimensions=False``
gives the bare projection.

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
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dana.plugins.freecad.engine import (
    _EXPORT_DIR,
    _OK_MARKER,
    _RESOLVE_OBJECT_SNIPPET,
    _object_lookup_snippet,
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

{resolve_snippet}
doc = App.openDocument({source_path!r})
{object_lookup}

page = doc.addObject("TechDraw::DrawPage", "BlueprintPage")
template = doc.addObject("TechDraw::DrawSVGTemplate", "BlueprintTemplate")
template.Template = os.path.join(App.getResourceDir(), "Mod", "TechDraw", "Templates", *{template_parts!r})
page.Template = template

for name, direction, xdirection, fx, fy in {view_specs!r}:
    view = doc.addObject("TechDraw::DrawViewPart", "View" + name)
    view.Source = [obj]
    view.Direction = App.Vector(*direction)
    view.XDirection = App.Vector(*xdirection)
    # 1:1 always: the dimension callouts read real millimetres off the DXF.
    view.ScaleType = "Custom"
    view.Scale = 1.0
    page.addView(view)
    view.X = fx * page.PageWidth
    view.Y = fy * page.PageHeight

doc.recompute()
TechDraw.writeDXFPage(page, {dxf_path!r})
print("{marker} path=" + {dxf_path!r})
"""


# Dimension callout geometry, in page millimetres.
_DIM_LAYER = "Dimensions"
_DIM_OFFSET_MM = 8.0  # geometry edge -> dimension line
_DIM_EXT_GAP_MM = 1.5  # gap between geometry and extension line
_DIM_EXT_OVERSHOOT_MM = 2.0  # extension line past the dimension line
_DIM_ARROW_MM = 2.5
_DIM_TEXT_MM = 3.5
_DIM_TEXT_GAP_MM = 1.0  # dimension line -> near edge of its label
_ORTHO_VIEWS = ("Front", "Top", "Right")

_Point = tuple[float, float]
_Segment = tuple[_Point, _Point]


@dataclass(frozen=True)
class _Label:
    x: float
    y: float
    text: str
    vertical: bool


@dataclass(frozen=True)
class _Callouts:
    sizes: list[dict[str, Any]]  # [{"view", "width_mm", "height_mm"}]
    segments: list[_Segment]
    labels: list[_Label]


def _format_mm(value: float) -> str:
    """Rounded to 2 decimals, trailing zeros dropped: 100.0 -> '100 mm'."""
    return f"{value:.2f}".rstrip("0").rstrip(".") + " mm"


def _arrow(tip: _Point, direction: _Point) -> list[_Segment]:
    """Open arrowhead at ``tip`` pointing along unit vector ``direction``."""
    dx, dy = direction
    bx, by = -dx * _DIM_ARROW_MM, -dy * _DIM_ARROW_MM
    sx, sy = -dy * _DIM_ARROW_MM * 0.3, dx * _DIM_ARROW_MM * 0.3
    return [(tip, (tip[0] + bx + sx, tip[1] + by + sy)), (tip, (tip[0] + bx - sx, tip[1] + by - sy))]


def _dimension_callouts(doc: Any) -> _Callouts:
    """Overall width (below) and height (left) callouts for every
    orthographic view present on ``doc``'s ``View<Name>`` layers."""
    from ezdxf import bbox

    msp = doc.modelspace()
    sizes: list[dict[str, Any]] = []
    segments: list[_Segment] = []
    labels: list[_Label] = []
    for view in _ORTHO_VIEWS:
        box = bbox.extents(msp.query(f'*[layer=="View{view}"]'))
        if not box.has_data:
            continue
        (x0, y0), (x1, y1) = box.extmin.vec2, box.extmax.vec2
        width, height = x1 - x0, y1 - y0
        sizes.append({"view": view, "width_mm": round(width, 2), "height_mm": round(height, 2)})

        if width > 0:
            yd = y0 - _DIM_OFFSET_MM
            segments += [
                ((x0, y0 - _DIM_EXT_GAP_MM), (x0, yd - _DIM_EXT_OVERSHOOT_MM)),
                ((x1, y0 - _DIM_EXT_GAP_MM), (x1, yd - _DIM_EXT_OVERSHOOT_MM)),
                ((x0, yd), (x1, yd)),
                *_arrow((x0, yd), (-1.0, 0.0)),
                *_arrow((x1, yd), (1.0, 0.0)),
            ]
            label_y = yd - _DIM_TEXT_GAP_MM - _DIM_TEXT_MM / 2
            labels.append(_Label((x0 + x1) / 2, label_y, _format_mm(width), vertical=False))
        if height > 0:
            xd = x0 - _DIM_OFFSET_MM
            segments += [
                ((x0 - _DIM_EXT_GAP_MM, y0), (xd - _DIM_EXT_OVERSHOOT_MM, y0)),
                ((x0 - _DIM_EXT_GAP_MM, y1), (xd - _DIM_EXT_OVERSHOOT_MM, y1)),
                ((xd, y0), (xd, y1)),
                *_arrow((xd, y0), (0.0, -1.0)),
                *_arrow((xd, y1), (0.0, 1.0)),
            ]
            label_x = xd - _DIM_TEXT_GAP_MM - _DIM_TEXT_MM / 2
            labels.append(_Label(label_x, (y0 + y1) / 2, _format_mm(height), vertical=True))
    return _Callouts(sizes, segments, labels)


def _add_callout_lines(doc: Any, callouts: _Callouts) -> None:
    """Draw the callouts' lines into ``doc``, so both renderers draw the same
    geometry (black, like the views, via ColorPolicy.BLACK). Labels are added
    per renderer as real text instead: ezdxf renders DXF TEXT as glyph
    outlines, which can't be searched or selected."""
    if _DIM_LAYER not in doc.layers:
        doc.layers.add(_DIM_LAYER)
    msp = doc.modelspace()
    for start, end in callouts.segments:
        msp.add_line(start, end, dxfattribs={"layer": _DIM_LAYER})


def _load_page(dxf_path: str, include_dimensions: bool) -> tuple[Any, _Callouts | None]:
    """The exported DXF page, with dimension lines added if requested."""
    import ezdxf

    doc = ezdxf.readfile(dxf_path)
    if not include_dimensions:
        return doc, None
    callouts = _dimension_callouts(doc)
    _add_callout_lines(doc, callouts)
    return doc, callouts


def _render_dxf_to_pdf(
    dxf_path: str, name: str, page_size_mm: tuple[float, float], include_dimensions: bool = True
) -> Path:
    """Renders a TechDraw-exported DXF page to a PDF, sized to the page's
    real physical dimensions — a pure Python step, no FreeCAD involved."""
    import matplotlib

    matplotlib.use("Agg")  # headless — never try to open a display/window
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing import matplotlib as ezdxf_matplotlib
    from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration

    doc, callouts = _load_page(dxf_path, include_dimensions)
    width_mm, height_mm = page_size_mm
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    try:
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0, width_mm)
        ax.set_ylim(0, height_mm)
        ax.set_aspect("equal")
        ax.axis("off")
        # Default ColorPolicy.COLOR keeps TechDraw's native layer color (ACI
        # 7, "white" under the dark-background convention DXF viewers
        # assume) drawn onto matplotlib's default white figure — an
        # invisible white-on-white PDF page. Confirmed live: an unpatched
        # render of a real box's Front view produced a page with zero
        # non-white pixels. Forcing black-on-white makes the page's own
        # background explicit too, rather than relying on whatever the
        # matplotlib backend defaults to.
        render_config = Configuration(
            background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.BLACK
        )
        # adjust_figure=False and the limits re-applied after drawing:
        # finalize() otherwise autoscales the axes to the drawn geometry and
        # resizes the figure to its aspect ratio, so an "A4" PDF came out the
        # shape of the part (a 100x50 outline gave a 9.6x4.8 in page) with the
        # part zoomed to fill it. TechDraw writes the DXF in page millimetres,
        # so these limits draw the page 1:1.
        backend = ezdxf_matplotlib.MatplotlibBackend(ax, adjust_figure=False)
        Frontend(RenderContext(doc), backend, config=render_config).draw_layout(doc.modelspace(), finalize=True)
        ax.set_xlim(0, width_mm)
        ax.set_ylim(0, height_mm)
        for label in callouts.labels if callouts else ():
            ax.text(
                label.x,
                label.y,
                label.text,
                fontsize=_DIM_TEXT_MM / 25.4 * 72,  # mm -> pt
                color="black",
                ha="center",
                va="center",
                rotation=90 if label.vertical else 0,
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
        # TrueType (42) instead of matplotlib's default Type 3 fonts, so the
        # dimension labels can be searched and copied out of the PDF.
        with matplotlib.rc_context({"pdf.fonttype": 42}):
            fig.savefig(out_path)
    finally:
        plt.close(fig)
    return out_path


def _svg_labels(svg_text: str, labels: list[_Label], page_size_mm: tuple[float, float]) -> str:
    """``labels`` appended to ezdxf's SVG as real ``<text>`` elements. Its
    viewBox maps the page with one uniform scale and y pointing down, so the
    page-mm point (x, y) is (x * s, (page_height - y) * s)."""
    from xml.sax.saxutils import escape

    m = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg_text)
    if not m or not labels:
        return svg_text
    scale = float(m.group(1)) / page_size_mm[0]
    elements = []
    for label in labels:
        x, y = label.x * scale, (page_size_mm[1] - label.y) * scale
        rotate = f' transform="rotate(-90 {x:.0f} {y:.0f})"' if label.vertical else ""
        elements.append(
            f'<text x="{x:.0f}" y="{y:.0f}" font-size="{_DIM_TEXT_MM * scale:.0f}" font-family="sans-serif" '
            f'fill="#000000" text-anchor="middle" dominant-baseline="central"{rotate}>{escape(label.text)}</text>'
        )
    return svg_text.replace("</svg>", '<g class="dimensions">' + "".join(elements) + "</g></svg>", 1)


def _render_dxf_to_svg(
    dxf_path: str, name: str, page_size_mm: tuple[float, float], include_dimensions: bool = True
) -> tuple[Path, list[dict[str, Any]]]:
    """Renders a TechDraw-exported DXF page to SVG — same pure-Python,
    no-FreeCAD step as ``_render_dxf_to_pdf`` above (headless SVG export via
    FreeCAD's own ``TechDrawGui.exportPageAsSvg`` needs a live Qt
    ``FreeCADGui``, exactly the dependency this module's whole PDF pipeline
    was already built to avoid — see this module's own docstring), just
    ``ezdxf``'s own ``SVGBackend`` in place of its matplotlib one. A fresh
    ``Frontend``/``RenderContext`` pass, not a reuse of ``_render_dxf_to_pdf``'s
    — ``Frontend.draw_layout`` drives exactly one backend per call, by
    ezdxf's own API shape, so each output format needs its own render pass
    over the same DXF.

    Also returns the overall size of each dimensioned view (empty without
    dimensions).
    """
    from ezdxf.addons.drawing import Frontend, RenderContext, layout
    from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration
    from ezdxf.addons.drawing.svg import SVGBackend
    from ezdxf.math import BoundingBox2d

    doc, callouts = _load_page(dxf_path, include_dimensions)
    width_mm, height_mm = page_size_mm
    backend = SVGBackend()
    # Same black-on-white forcing as _render_dxf_to_pdf — TechDraw's native
    # layer color (ACI 7, "white" under the dark-background convention DXF
    # viewers assume) would otherwise render invisible white-on-white here
    # too.
    render_config = Configuration(background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.BLACK)
    Frontend(RenderContext(doc), backend, config=render_config).draw_layout(doc.modelspace(), finalize=True)
    page = layout.Page(width_mm, height_mm, units=layout.Units.mm)
    # 1:1 like the PDF: render exactly the page rectangle (TechDraw's DXF is
    # in page millimetres) at scale 1. ezdxf's default fit_page=True instead
    # scales the drawing's own extents to fill the page, so a 10 mm part
    # came out page-sized.
    svg_text = backend.get_string(
        page,
        settings=layout.Settings(fit_page=False, scale=1.0),
        render_box=BoundingBox2d([(0.0, 0.0), (width_mm, height_mm)]),
    )
    if callouts:
        svg_text = _svg_labels(svg_text, callouts.labels, page_size_mm)

    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _EXPORT_DIR / f"{_safe_name(name)}.svg"
    # Same stale-file guard as _render_dxf_to_pdf's own out_path — see that
    # function's comment for the exact failure this prevents.
    out_path.unlink(missing_ok=True)
    out_path.write_text(svg_text, encoding="utf-8")
    return out_path, callouts.sizes if callouts else []


def generate_2d_blueprint(
    source_path: str,
    views: Sequence[str] | None = None,
    page_size: str = "A4",
    filename: str | None = None,
    object_name: str | None = None,
    include_dimensions: bool = True,
) -> str:
    """Projects orthographic (Front/Top/Right) and/or Isometric views of the
    object in ``source_path`` onto a standard drawing page and exports a PDF
    and an SVG at 1:1. With ``include_dimensions`` (the default) each
    orthographic view gets its overall width and height dimensioned in mm,
    and the result's ``dimensions`` lists them per view; ``False`` gives the
    bare projection.

    ``object_name`` picks the object inside ``source_path`` (by Name, then
    Label, then case-insensitively, like the engine's other tools). Every
    create_* tool writes into one shared session document, so without it the
    first unreferenced object was drawn, whatever was asked for. Only a
    genuinely single-object file can omit it.
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
            resolve_snippet=_RESOLVE_OBJECT_SNIPPET,
            object_lookup=_object_lookup_snippet(target_object=object_name),
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
            pdf_path = _render_dxf_to_pdf(dxf_path, resolved_name, _PAGE_SIZES_MM[size_key], include_dimensions)
        except Exception as exc:  # noqa: BLE001 — surface as a normal tool failure, not a crash
            return _error(f"generate_2d_blueprint: DXF->PDF conversion failed: {exc}")

        try:
            svg_path, dimensions = _render_dxf_to_svg(
                dxf_path, resolved_name, _PAGE_SIZES_MM[size_key], include_dimensions
            )
        except Exception as exc:  # noqa: BLE001 — surface as a normal tool failure, not a crash
            return _error(f"generate_2d_blueprint: DXF->SVG conversion failed: {exc}")
    finally:
        try:
            os.unlink(dxf_path)
        except OSError:
            pass

    return _ok(
        name=resolved_name,
        views=requested,
        page_size=size_key,
        path=str(pdf_path),
        svg_path=str(svg_path),
        dimensions=dimensions,
    )


__all__ = ("generate_2d_blueprint",)

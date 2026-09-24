#!/usr/bin/env python3
"""Synthetic ground-truth generator for the multi-view CSG vision pipeline
(``dana.plugins.vision.image_analysis.analyze_reference_design``) — builds
two parametric test parts via FreeCADCmd and exports each requested
orthographic view as its own PNG into ``agent_workspace/``, so the
multi-view ingestion path can be exercised end-to-end without a real
uploaded photo.

Renders via TechDraw + DXF, NOT a live camera/viewport screenshot: this
codebase already established (see ``dana.plugins.freecad.techdraw_export``'s
own module docstring) that headless raster/viewport export
(``Gui.ActiveDocument.ActiveView.saveImage`` and friends) needs a live Qt
``FreeCADGui`` instance, which a plain ``FreeCADCmd`` subprocess never has.
``TechDraw.writeDXFPage`` is pure C++ projection/HLR compute with no Gui
dependency, so the same two-stage pipeline applies here: FreeCADCmd builds
the geometry and writes one DXF per view, then THIS process (which has
``ezdxf``/``matplotlib`` in its own venv) rasterizes each DXF to PNG.

Test Set A (axisymmetric shaft — 2 views): a stepped cylinder built along
the X axis, so the "front" projection shows the step profile (rotational
symmetry makes a third view redundant) and the "right" projection looks
straight down the shaft's own axis, showing two genuinely concentric
circles — the classic single-view depth ambiguity the multi-view prompt
exists to resolve.

Test Set B (asymmetric L-bracket — 3 views): an L-shaped fused box pair
with a counterbored through-hole, plus a best-effort fillet on one outer
vertical edge (skipped gracefully, never a hard failure, if edge selection
doesn't find a clean match — this script's job is generating usable test
data, not exercising ``Part::Fillet`` robustness).

Usage (from repo root)::

    python scripts/generate_ortho_tests.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dana.plugins.freecad.engine import detect_freecadcmd  # noqa: E402
from dana.plugins.freecad.techdraw_export import _VIEW_LAYOUT  # noqa: E402

_OK_MARKER = "DANA_ORTHO_TEST_GEN_OK"
_WORKSPACE = _ROOT / "agent_workspace"
_TIMEOUT_S = 180
_RASTER_DPI = 200
_PAGE_WIDTH_MM = 297.0
_PAGE_HEIGHT_MM = 210.0

_SHAFT_GEOMETRY = """\
cyl_big = doc.addObject("Part::Cylinder", "CylBig")
cyl_big.Radius = 15
cyl_big.Height = 20
cyl_big.Placement = App.Placement(App.Vector(0, 0, 0), App.Rotation(App.Vector(0, 1, 0), 90))

cyl_small = doc.addObject("Part::Cylinder", "CylSmall")
cyl_small.Radius = 8
cyl_small.Height = 30
cyl_small.Placement = App.Placement(App.Vector(20, 0, 0), App.Rotation(App.Vector(0, 1, 0), 90))

doc.recompute()
shaft = doc.addObject("Part::MultiFuse", "ShaftFused")
shaft.Shapes = [cyl_big, cyl_small]
"""

_BRACKET_GEOMETRY = """\
base = doc.addObject("Part::Box", "BaseFlange")
base.Length = 60
base.Width = 40
base.Height = 10
base.Placement = App.Placement(App.Vector(0, 0, 0), App.Rotation())

wall = doc.addObject("Part::Box", "UpWall")
wall.Length = 60
wall.Width = 10
wall.Height = 40
wall.Placement = App.Placement(App.Vector(0, 0, 10), App.Rotation())

doc.recompute()
lshape = doc.addObject("Part::MultiFuse", "LShape")
lshape.Shapes = [base, wall]
doc.recompute()

through_hole = doc.addObject("Part::Cylinder", "ThroughHole")
through_hole.Radius = 6
through_hole.Height = 14
through_hole.Placement = App.Placement(App.Vector(40, 25, -2), App.Rotation())

counterbore = doc.addObject("Part::Cylinder", "Counterbore")
counterbore.Radius = 10
counterbore.Height = 6
counterbore.Placement = App.Placement(App.Vector(40, 25, 4), App.Rotation())

doc.recompute()
hole_tool = doc.addObject("Part::MultiFuse", "HoleTool")
hole_tool.Shapes = [through_hole, counterbore]
doc.recompute()

cut_result = doc.addObject("Part::Cut", "BracketCut")
cut_result.Base = lshape
cut_result.Tool = hole_tool
doc.recompute()

# Best-effort fillet on the far outer vertical edge of the base flange
# (X=60, Y=40) -- never a hard failure if no clean match is found.
fillet_edge_index = None
try:
    for i, edge in enumerate(cut_result.Shape.Edges):
        verts = edge.Vertexes
        if len(verts) != 2:
            continue
        p1, p2 = verts[0].Point, verts[1].Point
        if (
            abs(p1.x - p2.x) < 1e-6
            and abs(p1.y - p2.y) < 1e-6
            and abs(p1.x - 60) < 1e-6
            and abs(p1.y - 40) < 1e-6
        ):
            fillet_edge_index = i + 1
            break
except Exception:
    fillet_edge_index = None

bracket = None
if fillet_edge_index is not None:
    try:
        candidate = doc.addObject("Part::Fillet", "BracketFinal")
        candidate.Base = cut_result
        candidate.Edges = [(fillet_edge_index, 3.0, 3.0)]
        doc.recompute()
        if candidate.Shape.isValid():
            bracket = candidate
        else:
            doc.removeObject(candidate.Name)
    except Exception:
        try:
            doc.removeObject("BracketFinal")
        except Exception:
            pass
        bracket = None

if bracket is None:
    bracket = doc.addObject("Part::Feature", "BracketFinal")
    bracket.Shape = cut_result.Shape
"""

# {geometry_code} builds the final object under a known name in `doc`;
# {view_specs!r} is [(view_name, direction, xdirection, dxf_path), ...] —
# one TechDraw page + DXF export per entry, each a SEPARATE file (unlike
# techdraw_export.py's own multi-view-per-page layout) since the point here
# is one clean image per orthographic view, not a combined drawing sheet.
_PART_SCRIPT_TEMPLATE = """\
import FreeCAD as App
import TechDraw
import os

doc = App.newDocument("OrthoTestGen")

{geometry_code}

doc.recompute()
final_obj = doc.getObject({final_name!r})

for view_name, direction, xdirection, dxf_path in {view_specs!r}:
    page = doc.addObject("TechDraw::DrawPage", "Page_" + view_name)
    template = doc.addObject("TechDraw::DrawSVGTemplate", "Template_" + view_name)
    template.Template = os.path.join(
        App.getResourceDir(), "Mod", "TechDraw", "Templates", "Default_Template_A4_Landscape.svg"
    )
    page.Template = template

    view = doc.addObject("TechDraw::DrawViewPart", "View_" + view_name)
    view.Source = [final_obj]
    view.Direction = App.Vector(*direction)
    view.XDirection = App.Vector(*xdirection)
    view.ScaleType = "Automatic"
    try:
        view.HardHidden = True
    except Exception:
        pass
    page.addView(view)
    doc.recompute()
    view.X = page.PageWidth / 2.0
    view.Y = page.PageHeight / 2.0
    doc.recompute()

    TechDraw.writeDXFPage(page, dxf_path)
    print("{marker} view=" + view_name + " dxf=" + dxf_path)

    # Overall Width/Height callouts per view, mapped from the 3D bounding
    # box -- NOT injected as TechDraw::DrawViewAnnotation objects. Confirmed
    # live: writeDXFPage DOES include annotation text as a real DXF TEXT
    # entity with the correct declared height (verified by reading the DXF
    # back with ezdxf), but ezdxf's own matplotlib rasterizer (this script's
    # _rasterize_dxf_to_png) renders that TEXT wildly oversized regardless
    # of its declared height -- confirmed on a minimal probe (a 5.0mm-tall
    # label dwarfing a 10mm test cube rendered at accurate 1:1 scale right
    # next to it). Printing the plain numbers here and drawing them directly
    # in the matplotlib rasterization step instead sidesteps that renderer
    # bug entirely, with full control over the actual rendered font size.
    bbox = final_obj.Shape.BoundBox
    dx, dy, dz = bbox.XLength, bbox.YLength, bbox.ZLength
    if direction == (0.0, -1.0, 0.0):    # Front view: looking down -Y
        dim_w, dim_h = dx, dz
    elif direction == (0.0, 0.0, -1.0):  # Top view: looking down -Z
        dim_w, dim_h = dx, dy
    elif direction == (1.0, 0.0, 0.0):   # Right/side view: looking down +X
        dim_w, dim_h = dy, dz
    else:
        dim_w, dim_h = dx, dy
    print(
        "{marker} dims view=" + view_name
        + " width=" + "%.2f" % dim_w + " height=" + "%.2f" % dim_h
    )
"""

# Reuses dana.plugins.freecad.techdraw_export's own vetted direction/
# xdirection triples rather than re-deriving new ones -- "right" (looking
# straight down +X) doubles as the shaft's axial view here, since the shaft
# is built along X specifically so that view produces concentric circles.
_PARTS: list[dict[str, Any]] = [
    {
        "label": "Test Set A -- axisymmetric shaft",
        "geometry": _SHAFT_GEOMETRY,
        "final_name": "ShaftFused",
        "views": [
            ("front", _VIEW_LAYOUT["front"], "shaft_front.png"),
            ("side", _VIEW_LAYOUT["right"], "shaft_side.png"),
        ],
    },
    {
        "label": "Test Set B -- asymmetric L-bracket",
        "geometry": _BRACKET_GEOMETRY,
        "final_name": "BracketFinal",
        "views": [
            ("top", _VIEW_LAYOUT["top"], "bracket_top.png"),
            ("front", _VIEW_LAYOUT["front"], "bracket_front.png"),
            ("right", _VIEW_LAYOUT["right"], "bracket_right.png"),
        ],
    },
]


def _rasterize_dxf_to_png(dxf_path: Path, out_path: Path) -> None:
    import ezdxf
    import matplotlib

    matplotlib.use("Agg")  # headless -- never try to open a display/window
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import Frontend, RenderContext
    from ezdxf.addons.drawing import matplotlib as ezdxf_matplotlib
    from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration

    dxf_doc = ezdxf.readfile(str(dxf_path))
    fig = plt.figure(figsize=(_PAGE_WIDTH_MM / 25.4, _PAGE_HEIGHT_MM / 25.4), facecolor="white")
    try:
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0, _PAGE_WIDTH_MM)
        ax.set_ylim(0, _PAGE_HEIGHT_MM)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("white")
        # Without an explicit config, ezdxf's default ColorPolicy.COLOR keeps
        # TechDraw's native layer color (ACI 7 == "white" under the dark-
        # background convention DXF viewers assume) -- invisible against our
        # white page. Forcing black-on-white here since these are throwaway
        # synthetic test images, not a faithful color reproduction.
        render_config = Configuration(
            background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.BLACK
        )
        Frontend(RenderContext(dxf_doc), ezdxf_matplotlib.MatplotlibBackend(ax), config=render_config).draw_layout(
            dxf_doc.modelspace(), finalize=True
        )
        out_path.unlink(missing_ok=True)
        fig.savefig(out_path, dpi=_RASTER_DPI, facecolor="white")
    finally:
        plt.close(fig)


_DIM_LABEL_FONT_PX = 32
_DIM_LABEL_MARGIN_PX = 14


def _stamp_dimension_labels(png_path: Path, dim_w: float, dim_h: float) -> None:
    """Draws Overall Width/Height callouts directly onto the already-
    rasterized PNG via Pillow, in PIXEL space, rather than trying to place
    them inside the matplotlib/DXF data-coordinate system _rasterize_dxf_to_png
    uses for the geometry itself.

    That was tried first and abandoned: ax.set_aspect("equal") interacting
    with ezdxf's own draw_layout(finalize=True) autoscale silently
    re-adjusts the axes' data limits at savefig() time (confirmed live via
    matplotlib's own "Ignoring fixed x limits to fulfill fixed data aspect
    with adjustable data limits" warning) -- text placed in that same data
    space landed off-canvas even though the geometry itself rendered
    correctly. Pixel space has no such ambiguity: whatever the final PNG
    looks like IS the coordinate system, so this finds the actual rendered
    ink's bounding box (via inverting to find non-white pixels) and anchors
    labels relative to THAT, which works regardless of whatever internal
    scale/crop the DXF rendering step applied.

    Both labels drawn HORIZONTALLY (0deg), stacked on two lines below the
    geometry -- the height label used to be rotated 90deg and placed to the
    LEFT of the drawing, which had two confirmed-live problems: (1) EasyOCR
    struggles with rotated text (it misread the rotated height label
    specifically, while the horizontal width label on the same image read
    back clean), and (2) a left-of-geometry placement clips clean off the
    canvas edge whenever the drawing's own silhouette already starts near
    x=0 (also confirmed live, on the bracket "right" view).

    The page canvas this function receives is NOT a fixed, generously-
    margined size -- confirmed live these 5 synthetic views range from
    800x1000 to 1600x960px, because ax.set_aspect("equal")'s adjustable-
    datalim interaction with ezdxf's own autoscale (see
    _rasterize_dxf_to_png's own docstring) crops tightly around whatever
    geometry is present, per view. A first attempt at stacking two label
    lines directly below the ink bbox clipped the second line clean off the
    bottom edge on several views because there simply wasn't 2 lines' worth
    of margin below the geometry to begin with. Rather than guess at
    available margin, this pads the canvas with a guaranteed-white strip
    tall enough for both lines FIRST, then draws into that strip -- correct
    regardless of how tight the original crop is.
    """
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    img = Image.open(png_path).convert("RGB")
    ink_bbox = ImageOps.invert(img.convert("L")).getbbox()
    if ink_bbox is None:  # a blank page -- nothing to anchor labels to
        return
    left, _top, right, bottom = ink_bbox

    line_height = _DIM_LABEL_FONT_PX + _DIM_LABEL_MARGIN_PX
    pad_height = _DIM_LABEL_MARGIN_PX + 2 * line_height
    padded = Image.new("RGB", (img.width, img.height + pad_height), "white")
    padded.paste(img, (0, 0))

    draw = ImageDraw.Draw(padded)
    font = ImageFont.load_default(size=_DIM_LABEL_FONT_PX)
    center_x = (left + right) // 2

    # Bare numbers, not "W: 60.0"/"H: 50.0" -- a prefix risks EasyOCR
    # detecting it as one glued-together text blob with the number ("W:
    # 60.0"), which _repair_and_filter's exact-match dimension pattern
    # would then reject outright. Which line is width vs. height is
    # already conveyed by stacking order (width above height, matching the
    # order dims are always parsed/passed) and by the OCR view label this
    # feeds into (dana.plugins.vision.ocr_grounding), not by text on the
    # image itself.
    width_text = f"{dim_w:.1f}"
    draw.text((center_x, bottom + _DIM_LABEL_MARGIN_PX), width_text, fill="black", font=font, anchor="ma")

    height_text = f"{dim_h:.1f}"
    draw.text((center_x, bottom + _DIM_LABEL_MARGIN_PX + line_height), height_text, fill="black", font=font, anchor="ma")

    padded.save(png_path)


def _parse_dims_line(stdout: str, view_name: str) -> tuple[float, float] | None:
    """Extracts (width, height) from this view's own "{_OK_MARKER} dims
    view=<name> width=<w> height=<h>" stdout line (see _PART_SCRIPT_TEMPLATE)
    -- plain string parsing, matching the existing marker-line convention
    _run_part already uses below to find each view's dxf_path.
    """
    prefix = f"{_OK_MARKER} dims view={view_name} width="
    for line in stdout.splitlines():
        if not line.startswith(prefix):
            continue
        rest = line[len(prefix) :]
        width_str, _, height_str = rest.partition(" height=")
        try:
            return float(width_str), float(height_str)
        except ValueError:
            return None
    return None


def _run_part(freecadcmd: str, part: dict[str, Any]) -> list[str]:
    """Builds `part`'s geometry and exports every requested view to a DXF via
    ONE FreeCADCmd subprocess call, then rasterizes each DXF to its target
    PNG in agent_workspace/. Returns the list of PNG filenames actually
    written (a partial list on a partial failure, never a raised exception).
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="dana_ortho_test_"))
    dxf_paths = {name: str(tmp_dir / f"{name}.dxf") for name, _layout, _out in part["views"]}
    view_specs = [
        (name, layout["direction"], layout["xdirection"], dxf_paths[name])
        for name, layout, _out in part["views"]
    ]
    script = _PART_SCRIPT_TEMPLATE.format(
        geometry_code=part["geometry"],
        final_name=part["final_name"],
        view_specs=view_specs,
        marker=_OK_MARKER,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(script)
        script_path = tmp.name

    generated: list[str] = []
    try:
        proc = subprocess.run(
            [freecadcmd, script_path], capture_output=True, text=True, timeout=_TIMEOUT_S, check=False
        )
        print(f"[gen] {part['label']}: FreeCADCmd exit={proc.returncode}")
        if proc.stdout.strip():
            print(proc.stdout.strip())
        if proc.stderr.strip():
            print(f"[gen] stderr:\n{proc.stderr.strip()}")

        for name, _layout, out_name in part["views"]:
            dxf_path = Path(dxf_paths[name])
            marker_line = f"{_OK_MARKER} view={name} dxf="
            if marker_line not in proc.stdout or not dxf_path.is_file():
                print(f"[gen] FAILED: view '{name}' for {part['label']} did not produce a DXF.")
                continue
            out_path = _WORKSPACE / out_name
            _rasterize_dxf_to_png(dxf_path, out_path)
            dims = _parse_dims_line(proc.stdout, name)
            if dims is not None:
                _stamp_dimension_labels(out_path, *dims)
            print(f"[gen] wrote {out_path} ({out_path.stat().st_size} bytes)")
            generated.append(out_name)
    except subprocess.TimeoutExpired:
        print(f"[gen] FAILED: {part['label']} timed out after {_TIMEOUT_S}s.")
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return generated


def main() -> int:
    freecadcmd = detect_freecadcmd()
    if not freecadcmd:
        print(
            "[gen] FAIL: FreeCADCmd not found on PATH, DANA_FREECADCMD_PATH, "
            "or common install dirs."
        )
        return 1
    print(f"[gen] Using FreeCADCmd: {freecadcmd}")

    _WORKSPACE.mkdir(parents=True, exist_ok=True)

    expected = [out_name for part in _PARTS for _name, _layout, out_name in part["views"]]
    generated: list[str] = []
    for part in _PARTS:
        generated.extend(_run_part(freecadcmd, part))

    missing = [name for name in expected if name not in generated]
    print("\n" + "=" * 72)
    if missing:
        print(f"RESULT: FAIL - missing {missing}")
        return 1
    print(f"RESULT: PASS - generated {len(generated)} views in {_WORKSPACE}")
    for name in generated:
        print(f"  - agent_workspace/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

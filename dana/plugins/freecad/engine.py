"""FreeCAD native-Python CAD operator — subprocess ``FreeCADCmd``, no COM.

Unlike AutoCAD (``dana.operators.autocad_engine``), FreeCAD has no
persistent COM/RPC server to attach to on Windows — every operation here
launches a fresh ``FreeCADCmd`` process against a short, disposable Python
script that calls FreeCAD's own ``FreeCAD``/``Part`` modules directly. No
mouse/pixel actuation, same determinism guardrail as the AutoCAD engine.

All public functions return a JSON string (``{"ok": bool, ...}``) so they
drop straight into the tool broker's string-observation contract — see
``dana.tools.broker.initialize_tool_registry`` for the tool_id wiring.
"""

from __future__ import annotations

import ast
import glob
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psutil

from dana.paths import DANA_WORKSPACE
from dana.security.dry_run import is_dry_run_enabled

# One FreeCADCmd process at a time — mirrors the single foreground-owner
# discipline used for physical desktop actuators (dana.middleware.actuator_executor).
_lock = threading.Lock()
_cached_cmd_path: str | None = None  # reassigned

_ENV_OVERRIDE = "DANA_FREECADCMD_PATH"
_COMMON_INSTALL_GLOBS: tuple[str, ...] = (
    r"C:\Program Files\FreeCAD*\bin\FreeCADCmd.exe",
    r"C:\Program Files (x86)\FreeCAD*\bin\FreeCADCmd.exe",
)
_DEFAULT_TIMEOUT_S = 60.0
_WINDOW_POLL_TIMEOUT_S = 10.0
_WINDOW_POLL_INTERVAL_S = 0.75
_OK_MARKER = "DANA_FREECAD_OK"
_BBOX_MARKER = f"{_OK_MARKER}_BBOX"
_BBOX_RE = re.compile(re.escape(_BBOX_MARKER) + r" (\[.*?\])")
_OUTPUT_DIR = DANA_WORKSPACE / "freecad_output"


class FreeCADNotFoundError(RuntimeError):
    """Raised when no FreeCADCmd binary can be located."""


def _ok(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload})


def _error(message: str) -> str:
    return json.dumps({"ok": False, "error": str(message)})


def _dry_run_result(op: str, **payload: Any) -> str:
    return _ok(op=op, dry_run=True, **payload)


def _version_key(folder_name: str) -> tuple[int, ...]:
    m = re.search(r"(\d+(?:\.\d+)*)", folder_name)
    if not m:
        return (0,)
    return tuple(int(p) for p in m.group(1).split("."))


def detect_freecadcmd(*, force_refresh: bool = False) -> str | None:
    """Locate FreeCADCmd: env override > PATH > common install globs (newest wins)."""
    global _cached_cmd_path
    if _cached_cmd_path and not force_refresh:
        return _cached_cmd_path

    override = (os.environ.get(_ENV_OVERRIDE) or "").strip()
    if override and Path(override).is_file():
        _cached_cmd_path = override
        return _cached_cmd_path

    on_path = shutil.which("FreeCADCmd") or shutil.which("freecadcmd")
    if on_path:
        _cached_cmd_path = on_path
        return _cached_cmd_path

    candidates = [Path(p) for pattern in _COMMON_INSTALL_GLOBS for p in glob.glob(pattern)]
    if not candidates:
        return None
    # ".../FreeCAD 1.0/bin/FreeCADCmd.exe" -> version folder is parent.parent.
    candidates.sort(key=lambda p: _version_key(p.parent.parent.name), reverse=True)
    _cached_cmd_path = str(candidates[0])
    return _cached_cmd_path


def get_freecadcmd_path(*, force_refresh: bool = False) -> str:
    path = detect_freecadcmd(force_refresh=force_refresh)
    if not path:
        raise FreeCADNotFoundError(
            f"FreeCADCmd not found (checked {_ENV_OVERRIDE}, PATH, and "
            "C:\\Program Files\\FreeCAD*\\bin\\FreeCADCmd.exe)"
        )
    return path


def _is_freecad_gui_running() -> bool:
    """True if a FreeCAD.exe GUI process is currently running."""
    for proc in psutil.process_iter(["name"]):
        try:
            if (proc.info.get("name") or "").lower() == "freecad.exe":
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def get_freecad_gui_path(*, force_refresh: bool = False) -> str:
    """Resolve the FreeCAD GUI binary — lives next to FreeCADCmd in the same ``bin/``."""
    cmd_path = get_freecadcmd_path(force_refresh=force_refresh)
    gui_path = Path(cmd_path).with_name("FreeCAD.exe")
    if gui_path.is_file():
        return str(gui_path)
    on_path = shutil.which("FreeCAD") or shutil.which("FreeCAD.exe")
    if on_path:
        return on_path
    raise FreeCADNotFoundError(
        f"FreeCAD.exe (GUI) not found next to FreeCADCmd at {gui_path}, nor on PATH"
    )


def _find_freecad_window() -> dict[str, Any] | None:
    try:
        from dana.tools.os_control import get_active_windows

        for win in get_active_windows():
            if "freecad" in str(win.get("title") or "").lower():
                return win
    except Exception:  # noqa: BLE001
        pass
    return None


def _title_matches_file(window: dict[str, Any] | None, path: Path) -> bool:
    if not window:
        return False
    return path.stem.lower() in str(window.get("title") or "").lower()


def _send_to_secondary_monitor(hwnd: int) -> bool:
    """Move ``hwnd`` onto a second physical monitor, without ever activating it.

    Returns ``False`` (no-op, window left exactly where it was) when only
    one monitor exists — there's nowhere else to put it, and moving it
    somewhere unreachable would be worse than leaving it alone.
    """
    from dana.tools.os_control import get_secondary_monitor, move_window_no_activate

    monitor = get_secondary_monitor()
    if monitor is None:
        return False
    width = min(1280, monitor["width"])
    height = min(800, monitor["height"])
    x = monitor["left"] + 40
    y = monitor["top"] + 40
    try:
        return move_window_no_activate(hwnd, x, y, width, height)
    except Exception:  # noqa: BLE001
        return False


def _notify_cad_update_ready(path: Path, *, generated_only: bool) -> None:
    """Non-intrusive fallback when we can't (or shouldn't) focus the FreeCAD window.

    Fire-and-forget silent OS toast — never blocks the caller, never raises.
    Reuses the same helper actuator_executor already uses for task toasts,
    rather than adding a new notification dependency.
    """
    try:
        from dana.middleware.toast_notify import show_silent_toast_async

        message = (
            f"{path.name} generated. Please open to view."
            if generated_only
            else f"{path.name} is ready."
        )
        show_silent_toast_async("Dana CAD Update", message)
    except Exception:  # noqa: BLE001
        pass


_FIT_VIEW_MACRO = """\
import FreeCAD as App
import FreeCADGui as Gui

doc = App.ActiveDocument
if doc is not None:
    # A Boolean feature (Part::Cut/MultiFuse/MultiCommon) consumes its
    # Base/Tool/Shapes children into the result — only the top-level
    # feature should show, not the raw inputs it was built from.
    consumed = set()
    for obj in doc.Objects:
        base = getattr(obj, "Base", None)
        tool = getattr(obj, "Tool", None)
        if base is not None:
            consumed.add(base.Name)
        if tool is not None:
            consumed.add(tool.Name)
        for shape in getattr(obj, "Shapes", None) or []:
            consumed.add(shape.Name)
    for obj in doc.Objects:
        try:
            obj.ViewObject.Visibility = obj.Name not in consumed
        except Exception:
            pass
    try:
        Gui.activeDocument().activeView().viewAxonometric()
        Gui.SendMsgToActiveView("ViewFit")
    except Exception:
        pass
"""


def _write_fit_view_macro() -> str:
    """Write the one-shot "make objects visible + fit view" macro to a temp file.

    FreeCAD executes any ``.FCMacro``/``.py`` file passed alongside a
    document on its command line — this is the only way to get a
    just-opened, headlessly-created document to actually render its
    geometry instead of an empty viewport. ``FreeCADCmd`` never loads the
    ``Gui`` module, so objects created there have no ``ViewObject`` at all
    (no stored visibility/camera state) until the real GUI creates default
    ones on open — and those defaults aren't guaranteed visible.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".FCMacro", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(_FIT_VIEW_MACRO)
        return tmp.name


def show_in_freecad_gui(filepath: str) -> str:
    """Open ``filepath`` in the live FreeCAD GUI on the secondary monitor —
    NEVER stealing OS focus.

    Zero-focus workspace: this never calls ``set_foreground_window`` (or
    any other activation API) under any circumstances, so a fullscreen app
    or game on the primary monitor is never disturbed. Never spawns a
    second ``FreeCAD.exe`` when one is already running either — FreeCAD
    has no single-instance IPC, so a second ``Popen`` call just piles up
    an extra invisible process. Whatever FreeCAD window exists gets pushed
    onto the second monitor via ``SetWindowPos``/``SWP_NOACTIVATE`` and
    shown via ``SW_SHOWNOACTIVATE`` — visible, but never activated. Moving
    is unconditional (repositioning a background window is not
    disruptive), but if that window's title doesn't actually name
    ``filepath`` (an ALREADY-running instance showing a different or no
    document — the ambiguity a freshly spawned process doesn't have, since
    it was launched WITH this file as its argument), moving it to the
    right monitor still wouldn't show the right thing, so this falls back
    to a silent OS toast instead.
    """
    path = Path(filepath)
    if not path.is_file():
        return _error(f"show_in_freecad_gui: file not found: {filepath}")

    was_running = _is_freecad_gui_running()
    spawned = False
    if not was_running:
        try:
            gui_path = get_freecad_gui_path()
        except FreeCADNotFoundError as exc:
            return _error(str(exc))
        try:
            macro_path = _write_fit_view_macro()
            subprocess.Popen([gui_path, str(path), macro_path])  # noqa: S603
        except OSError as exc:
            return _error(f"show_in_freecad_gui: failed to launch FreeCAD GUI: {exc}")
        spawned = True

    # Poll for the window instead of trusting one fixed sleep — cold starts
    # (workbench/plugin loading) can leave the title bar generic for
    # several seconds before it updates to reflect the opened document, and
    # a fixed wait either races that or wastes time once it's already done.
    deadline = time.monotonic() + _WINDOW_POLL_TIMEOUT_S
    window = _find_freecad_window()
    while time.monotonic() < deadline:
        if spawned and window is not None:
            break
        if not spawned and _title_matches_file(window, path):
            break
        time.sleep(_WINDOW_POLL_INTERVAL_S)
        window = _find_freecad_window()
    title_matches = spawned or _title_matches_file(window, path)

    moved = False
    if window is not None:
        moved = _send_to_secondary_monitor(int(window["hwnd"]))

    if not title_matches:
        _notify_cad_update_ready(path, generated_only=True)
    elif not moved:
        _notify_cad_update_ready(path, generated_only=False)

    return _ok(
        op="show_in_freecad_gui",
        path=str(path),
        was_running=was_running,
        spawned=spawned,
        title_matched=title_matches,
        moved_to_secondary=moved,
    )


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name or "").strip("_") or "model"


def _output_path(name: str, *, ext: str) -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR / f"{_safe_name(name)}.{ext}"


def _extract_bbox(stdout: str) -> list[float] | None:
    """Parse the ``[XMin, YMin, ZMin, XMax, YMax, ZMax]`` line the parametric
    scripts print after ``saveAs``, via ``ast.literal_eval`` (safe — no
    arbitrary code execution risk from subprocess stdout, unlike ``eval``).
    """
    m = _BBOX_RE.search(stdout or "")
    if not m:
        return None
    try:
        values = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None
    if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
        return [float(v) for v in values]
    return None


def _run_freecad_script(
    script_text: str, *, timeout: float = _DEFAULT_TIMEOUT_S, require_marker: bool = True
) -> dict[str, Any]:
    """Write ``script_text`` to a temp file and execute it via FreeCADCmd.

    ``require_marker`` gates success on ``_OK_MARKER`` appearing in stdout
    (the parametric helpers below print it only after ``saveAs`` succeeds).
    ``execute_freecad_script`` passes ``require_marker=False`` since an
    arbitrary caller-supplied script defines its own notion of success.
    """
    try:
        cmd_path = get_freecadcmd_path()
    except FreeCADNotFoundError as exc:
        return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(script_text)
        script_path = tmp.name

    try:
        with _lock:
            proc = subprocess.run(
                [cmd_path, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"FreeCADCmd timed out after {timeout}s",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    ok = proc.returncode == 0 and (not require_marker or _OK_MARKER in (proc.stdout or ""))
    fail_msg = proc.stderr.strip() or proc.stdout.strip() or "FreeCADCmd reported failure"
    return {
        "ok": ok,
        "error": None if ok else fail_msg,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "bounding_box": _extract_bbox(proc.stdout) if ok else None,
    }


_BBOX_PRINT = (
    "bbox = obj.Shape.BoundBox\n"
    'print("{marker}_BBOX " + str([bbox.XMin, bbox.YMin, bbox.ZMin, '
    "bbox.XMax, bbox.YMax, bbox.ZMax]))\n"
)

# Global XYZ translation applied on top of an object's normal local
# geometry — a no-op line when placement is the origin, so every existing
# script's output is byte-for-byte unchanged for callers that never pass one.
_PLACEMENT_SNIPPET = (
    "if {placement!r} != (0.0, 0.0, 0.0):\n"
    "    _px, _py, _pz = {placement!r}\n"
    "    obj.Placement = App.Placement(App.Vector(_px, _py, _pz), App.Rotation())\n"
)

_BOX_SCRIPT = """\
import FreeCAD as App

doc = App.newDocument("DanaModel")
obj = doc.addObject("Part::Box", {name!r})
obj.Length = {length}
obj.Width = {width}
obj.Height = {height}
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""

_CYLINDER_SCRIPT = """\
import FreeCAD as App

doc = App.newDocument("DanaModel")
obj = doc.addObject("Part::Cylinder", {name!r})
obj.Radius = {radius}
obj.Height = {height}
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""

_EXTRUDE_SCRIPT = """\
import FreeCAD as App
import Part

pts = [App.Vector(float(x), float(y), 0.0) for x, y in {points!r}]
if pts[0] != pts[-1]:
    pts.append(pts[0])
wire = Part.makePolygon(pts)
face = Part.Face(wire)
solid = face.extrude(App.Vector(0.0, 0.0, {height}))

doc = App.newDocument("DanaModel")
obj = doc.addObject("Part::Feature", {name!r})
obj.Shape = solid
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""

_PYRAMID_SCRIPT = """\
import FreeCAD as App
import Part

L, W, H = {length}, {width}, {height}
base_pts = [
    App.Vector(-L / 2, -W / 2, 0.0),
    App.Vector(L / 2, -W / 2, 0.0),
    App.Vector(L / 2, W / 2, 0.0),
    App.Vector(-L / 2, W / 2, 0.0),
]
apex = App.Vector(0.0, 0.0, H)

base_face = Part.Face(Part.makePolygon(base_pts + [base_pts[0]]))
side_faces = [
    Part.Face(Part.makePolygon([base_pts[i], base_pts[(i + 1) % 4], apex, base_pts[i]]))
    for i in range(4)
]
solid = Part.Solid(Part.Shell([base_face] + side_faces))

doc = App.newDocument("DanaModel")
obj = doc.addObject("Part::Feature", {name!r})
obj.Shape = solid
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""


def _auto_show(out_path: Path) -> bool:
    """Best-effort Live Viewport Sync: open ``out_path`` in the FreeCAD GUI.

    Never raises and never fails the caller's create_* result — geometry
    generation already succeeded by the time this runs; a missing/failed
    GUI launch is a visual-convenience miss, not a tool failure.
    """
    try:
        return bool(json.loads(show_in_freecad_gui(str(out_path))).get("ok"))
    except Exception:  # noqa: BLE001
        return False


def create_box(
    length: float,
    width: float,
    height: float,
    name: str = "Box",
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Create a parametric ``Part::Box`` and save it as a ``.FCStd`` document,
    translated by ``placement`` (global X/Y/Z offset in mm) on top of its
    normal corner-at-origin position.

    Returns lean JSON (name/type/bounding_box/dimensions/path) rather than
    echoing every input verbatim — keeps context small for fast local-LLM
    ReAct turns that chain many CAD primitives in a row.
    """
    dims = {"length": float(length), "width": float(width), "height": float(height)}
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_box", name=name, type="Part::Box", dimensions=dims, placement=list(placement)
        )
    out_path = _output_path(name, ext="FCStd")
    script = _BOX_SCRIPT.format(
        name=name,
        length=dims["length"],
        width=dims["width"],
        height=dims["height"],
        placement=placement,
        out_path=str(out_path),
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_box failed: {result['error']}")
    return _ok(
        name=name,
        type="Part::Box",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(out_path),
        gui_shown=_auto_show(out_path),
    )


def create_cylinder(
    radius: float,
    height: float,
    name: str = "Cylinder",
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Create a parametric ``Part::Cylinder`` and save it as a ``.FCStd``
    document, translated by ``placement`` (global X/Y/Z offset in mm)."""
    dims = {"radius": float(radius), "height": float(height)}
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_cylinder", name=name, type="Part::Cylinder", dimensions=dims, placement=list(placement)
        )
    out_path = _output_path(name, ext="FCStd")
    script = _CYLINDER_SCRIPT.format(
        name=name,
        radius=dims["radius"],
        height=dims["height"],
        placement=placement,
        out_path=str(out_path),
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_cylinder failed: {result['error']}")
    return _ok(
        name=name,
        type="Part::Cylinder",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(out_path),
        gui_shown=_auto_show(out_path),
    )


def create_extruded_polyline(
    points_list: Sequence[Sequence[float]],
    height: float,
    name: str = "ExtrudedPolyline",
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Extrude a closed polyline profile into a solid ``Part::Feature`` and
    save it, translated by ``placement`` (global X/Y/Z offset in mm).

    Builds the profile as a ``Part.makePolygon`` wire (auto-closing it if
    the first/last points differ), faces it, and extrudes ``height`` units
    — no scratch objects to clean up, unlike AutoCAD's region-from-polyline
    detour. Works for any simple (non-self-intersecting) planar polygon,
    convex or not — ``create_star_prism`` below reuses this directly for
    its star-shaped profile rather than duplicating the FreeCAD script.
    """
    if len(points_list) < 3:
        return _error("create_extruded_polyline requires at least 3 points")
    points = [[float(p[0]), float(p[1])] for p in points_list]
    dims = {"height": float(height), "profile_points": len(points)}
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_extruded_polyline",
            name=name,
            type="Part::Feature",
            dimensions=dims,
            placement=list(placement),
        )
    out_path = _output_path(name, ext="FCStd")
    script = _EXTRUDE_SCRIPT.format(
        points=points,
        height=dims["height"],
        name=name,
        placement=placement,
        out_path=str(out_path),
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_extruded_polyline failed: {result['error']}")
    return _ok(
        name=name,
        type="Part::Feature",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(out_path),
        gui_shown=_auto_show(out_path),
    )


def create_pyramid(
    length: float,
    width: float,
    height: float,
    name: str = "Pyramid",
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Create a sharp-edged rectangular pyramid (``length`` x ``width`` base,
    apex at ``height``) as a solid ``Part::Feature`` and save it, translated
    by ``placement`` (global X/Y/Z offset in mm).

    Built from 5 explicit triangular/quad faces (one base + four sides)
    rather than a collapsed ``Part::Wedge`` — a wedge with a degenerate top
    edge is a valid solid but a fragile one (some FreeCAD versions produce
    sliver/self-intersecting geometry at the collapsed edge); building the
    shell directly from the 4 base corners + apex is unambiguous.
    """
    dims = {"length": float(length), "width": float(width), "height": float(height)}
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_pyramid", name=name, type="Part::Feature", dimensions=dims, placement=list(placement)
        )
    out_path = _output_path(name, ext="FCStd")
    script = _PYRAMID_SCRIPT.format(
        length=dims["length"],
        width=dims["width"],
        height=dims["height"],
        name=name,
        placement=placement,
        out_path=str(out_path),
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_pyramid failed: {result['error']}")
    return _ok(
        name=name,
        type="Part::Feature",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(out_path),
        gui_shown=_auto_show(out_path),
    )


def _star_polygon_vertices(points: int, outer_radius: float, inner_radius: float) -> list[list[float]]:
    """Alternating outer/inner vertices of a symmetric N-point star, first
    point straight up — pure trig, no FreeCAD dependency, so it's testable
    without a FreeCADCmd binary."""
    n = points * 2
    return [
        [
            (outer_radius if i % 2 == 0 else inner_radius) * math.cos((math.pi / points) * i - math.pi / 2),
            (outer_radius if i % 2 == 0 else inner_radius) * math.sin((math.pi / points) * i - math.pi / 2),
        ]
        for i in range(n)
    ]


def create_star_prism(
    points: int,
    outer_radius: float,
    inner_radius: float,
    height: float,
    name: str = "StarPrism",
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Extrude a sharp-edged N-point star polygon ``height`` units along Z,
    translated by ``placement`` (global X/Y/Z offset in mm).

    A star is just another closed planar polygon, so this computes its
    vertices (alternating ``outer_radius``/``inner_radius``) and hands them
    straight to ``create_extruded_polyline`` — ``Part.Face`` builds a face
    from a concave/star-shaped wire the same way it does a convex one, no
    separate extrusion script needed.
    """
    if int(points) < 3:
        return _error("create_star_prism requires at least 3 points")
    vertices = _star_polygon_vertices(int(points), float(outer_radius), float(inner_radius))
    result = json.loads(create_extruded_polyline(vertices, height, name=name, placement=placement))
    if result.get("ok"):
        result["dimensions"] = {
            "points": int(points),
            "outer_radius": float(outer_radius),
            "inner_radius": float(inner_radius),
            "height": float(height),
        }
    return json.dumps(result)


# FreeCAD Part::Cut is a Base/Tool pair; Part::MultiFuse/MultiCommon instead
# take a Shapes list — two script shapes, chosen in Python (not branched
# inside the FreeCADCmd subprocess) by which the operation actually needs.
_BOOLEAN_FEATURE_TYPE: dict[str, str] = {
    "cut": "Part::Cut",
    "union": "Part::MultiFuse",
    "intersect": "Part::MultiCommon",
}
_DEFAULT_BOOLEAN_NAME: dict[str, str] = {"cut": "Cut", "union": "Fusion", "intersect": "Common"}

_BOOLEAN_CUT_SCRIPT = """\
import FreeCAD as App

base_doc = App.openDocument({base_path!r})
tool_doc = App.openDocument({tool_path!r})
base_obj = base_doc.Objects[0]
tool_obj = tool_doc.Objects[0]

doc = App.newDocument("DanaModel")
obj = doc.addObject("Part::Cut", {name!r})
obj.Base = doc.copyObject(base_obj, False)
obj.Tool = doc.copyObject(tool_obj, False)
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""

_BOOLEAN_FUSE_COMMON_SCRIPT = """\
import FreeCAD as App

base_doc = App.openDocument({base_path!r})
tool_doc = App.openDocument({tool_path!r})
base_obj = base_doc.Objects[0]
tool_obj = tool_doc.Objects[0]

doc = App.newDocument("DanaModel")
obj = doc.addObject({feature_type!r}, {name!r})
obj.Shapes = [doc.copyObject(base_obj, False), doc.copyObject(tool_obj, False)]
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""

_EXPORT_STL_SCRIPT = """\
import FreeCAD as App
import Mesh

doc = App.openDocument({source_path!r})
Mesh.export(list(doc.Objects), {out_path!r})
print("{marker} path=" + {out_path!r})
"""


def apply_boolean(operation: str, base_path: str, tool_path: str, name: str | None = None) -> str:
    """Combine two previously-created solids with a Boolean operation.

    ``"cut"`` builds a ``Part::Cut`` (subtracts the tool from the base);
    ``"union"`` builds a ``Part::MultiFuse`` (fuses both into one solid);
    ``"intersect"`` builds a ``Part::MultiCommon`` (keeps only their
    overlapping volume). Both paths must be ``.FCStd`` documents previously
    produced by ``create_box``/``create_cylinder``/``create_extruded_polyline``
    (or a prior ``apply_boolean``) — each holds exactly one top-level
    object, which is what gets copied into the new Boolean feature.
    """
    op = (operation or "").strip().lower()
    if op not in _BOOLEAN_FEATURE_TYPE:
        return _error(f"apply_boolean: unknown operation '{operation}' — must be cut, union, or intersect")
    base = Path(base_path)
    tool = Path(tool_path)
    if not base.is_file():
        return _error(f"apply_boolean: base_path not found: {base_path}")
    if not tool.is_file():
        return _error(f"apply_boolean: tool_path not found: {tool_path}")
    feature_type = _BOOLEAN_FEATURE_TYPE[op]
    resolved_name = name or _DEFAULT_BOOLEAN_NAME[op]
    if is_dry_run_enabled():
        return _dry_run_result("apply_boolean", operation=op, name=resolved_name, type=feature_type)
    out_path = _output_path(resolved_name, ext="FCStd")
    if op == "cut":
        script = _BOOLEAN_CUT_SCRIPT.format(
            base_path=str(base),
            tool_path=str(tool),
            name=resolved_name,
            out_path=str(out_path),
            marker=_OK_MARKER,
        )
    else:
        script = _BOOLEAN_FUSE_COMMON_SCRIPT.format(
            base_path=str(base),
            tool_path=str(tool),
            feature_type=feature_type,
            name=resolved_name,
            out_path=str(out_path),
            marker=_OK_MARKER,
        )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"apply_boolean failed: {result['error']}")
    return _ok(
        name=resolved_name,
        type=feature_type,
        operation=op,
        bounding_box=result.get("bounding_box"),
        path=str(out_path),
        gui_shown=_auto_show(out_path),
    )


def export_mesh_stl(source_path: str, name: str | None = None) -> str:
    """Tessellate every object in ``source_path`` (a ``.FCStd`` document)
    into a standalone ``.stl`` mesh file — the hand-off format for
    ``gr.Model3D`` viewers that can't load native FreeCAD documents.
    """
    source = Path(source_path)
    if not source.is_file():
        return _error(f"export_mesh_stl: source_path not found: {source_path}")
    if is_dry_run_enabled():
        return _dry_run_result("export_mesh_stl", source_path=str(source))
    out_path = _output_path(name or source.stem, ext="stl")
    script = _EXPORT_STL_SCRIPT.format(
        source_path=str(source), out_path=str(out_path), marker=_OK_MARKER
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"export_mesh_stl failed: {result['error']}")
    return _ok(op="export_mesh_stl", source_path=str(source), path=str(out_path))


_MODIFY_PREAMBLE = 'import FreeCAD as App\n\ndoc = App.openDocument({in_path!r})\n'
_MODIFY_POSTAMBLE = (
    '\n\ndoc.recompute()\n'
    "doc.save()\n"
    'print("{marker} path=" + {in_path!r} + " objects=" + str(len(doc.Objects)))\n'
)


def modify_existing_document(filepath: str, modification_script: str) -> str:
    """Open an existing ``.FCStd`` document, run ``modification_script``
    against it, and save back to the SAME path — the "Modify Existing"
    revision path for iterative CAD design.

    ``modification_script`` runs with the opened document already bound to
    the local name ``doc`` (e.g. ``doc.addObject(...)``, or edit an
    existing object's parameter via ``doc.getObject("Box").Length = 20``).
    Prefer this over ``create_box``/``create_cylinder``/
    ``create_extruded_polyline`` once a project file already exists, so
    edits accumulate in one evolving document instead of scattering a new
    ``.FCStd`` per operation — those three always start a brand-new
    document, by design, since their job is "give me one clean primitive."
    """
    path = Path(filepath)
    if not path.is_file():
        return _error(f"modify_existing_document: file not found: {filepath}")
    text = (modification_script or "").strip()
    if not text:
        return _error("modify_existing_document requires a non-empty modification_script")
    if is_dry_run_enabled():
        return _dry_run_result("modify_existing_document", path=str(path))

    script = (
        _MODIFY_PREAMBLE.format(in_path=str(path))
        + text
        + _MODIFY_POSTAMBLE.format(marker=_OK_MARKER, in_path=str(path))
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"modify_existing_document failed: {result['error']}")
    return _ok(op="modify_existing_document", path=str(path), gui_shown=_auto_show(path))


def execute_freecad_script(python_script_str: str) -> str:
    """Run an arbitrary FreeCAD Python script via FreeCADCmd (escape hatch).

    Unlike the parametric helpers above, success is gated on the
    subprocess return code alone (no ``DANA_FREECAD_OK`` marker required)
    — the caller's own script defines what "success" means. Prefer the
    parametric helpers when they cover the need.
    """
    text = (python_script_str or "").strip()
    if not text:
        return _error("execute_freecad_script requires a non-empty script string")
    if is_dry_run_enabled():
        return _dry_run_result("execute_freecad_script", script=python_script_str)
    result = _run_freecad_script(text, require_marker=False)
    if not result["ok"]:
        return _error(f"execute_freecad_script failed: {result['error']}")
    return _ok(op="execute_freecad_script", stdout=result["stdout"], stderr=result["stderr"])


__all__ = (
    "FreeCADNotFoundError",
    "apply_boolean",
    "create_box",
    "create_cylinder",
    "create_extruded_polyline",
    "create_pyramid",
    "create_star_prism",
    "export_mesh_stl",
    "modify_existing_document",
    "detect_freecadcmd",
    "execute_freecad_script",
    "get_freecad_gui_path",
    "get_freecadcmd_path",
    "show_in_freecad_gui",
)

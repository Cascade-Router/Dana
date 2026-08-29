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

# Re-exported (not implemented here — neither needs a FreeCADCmd subprocess
# at all, just pure-Python mesh/XML work) purely so
# dana/plugins/freecad/manifest.json's own entries for these two tool ids
# can resolve a callable via getattr(this module, <function name>) — see
# dana.plugins.plugin_manager._load_plugin_full. The manifest declarations
# only feed introspection (plugin_registry_view/check_plugin_registry and
# the semantic tool registry); actual dispatch is still the native
# dana.core.react_dispatch.TOOL_HANDLERS entry, which refresh_plugin_tools()
# leaves authoritative by design whenever a plugin tool id collides with an
# existing native one — the exact same shadowing every create_freecad_*
# tool below already relies on.
from dana.tools.geometry_analyzer import query_geometry_properties  # noqa: F401
from dana.tools.urdf_builder import generate_urdf_assembly  # noqa: F401


def insert_standard_part(*args: Any, **kwargs: Any) -> str:
    """Re-exported for dana/plugins/freecad/manifest.json's entry-point
    resolution (same reasoning as the two plain re-exports above), but as a
    lazy-import wrapper rather than a top-level ``from ... import`` —
    ``dana.plugins.freecad.standard_parts`` itself imports several names
    FROM this module at its own top level (``_BBOX_PRINT``, ``_OK_MARKER``,
    etc.), so a top-level re-export here would be a genuine circular
    import: whichever of the two modules starts importing first, the other
    isn't finished initializing yet. Deferring the import to call time
    means both modules are already fully loaded by the time this runs.
    """
    from dana.plugins.freecad.standard_parts import insert_standard_part as _impl

    return _impl(*args, **kwargs)

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
_PLACEMENT_MARKER = f"{_OK_MARKER}_PLACEMENT"
_PLACEMENT_RE = re.compile(re.escape(_PLACEMENT_MARKER) + r" (\[.*?\])")
_SPATIAL_MARKER = f"{_OK_MARKER}_SPATIAL"
_SPATIAL_RE = re.compile(re.escape(_SPATIAL_MARKER) + r" (\[.*?\])")
# The FreeCAD-assigned Name a session-document script actually ends up with
# — NOT necessarily the requested `name` argument verbatim, since FreeCAD
# auto-suffixes ("Box" -> "Box001") on a collision with an object already in
# the shared Session_Active.FCStd document. A plain identifier line (not a
# Python literal), unlike the bbox/placement/spatial markers above.
_NAME_MARKER = f"{_OK_MARKER}_NAME"
_NAME_RE = re.compile(re.escape(_NAME_MARKER) + r" (.+)")
_OUTPUT_DIR = DANA_WORKSPACE / "freecad_output"
_EXPORT_DIR = DANA_WORKSPACE / "exports"

# The single .FCStd document create_box/create_cylinder/insert_standard_part/
# modify_parameter/apply_boolean all share for the life of this process —
# replacing the old one-object-per-file design so a multi-part chain (e.g.
# box + bolt + boolean cut) ends up as siblings in ONE document tree instead
# of scattered across separate .FCStd files. create_freecad_extrusion/
# _pyramid/_star_prism/_pipe/_sketch_extrude, batch_pattern_array,
# align_freecad_objects, create_assembly_mate, apply_edge_operation, and the
# read-only inspectors (get_bounding_box, inspect_spatial_properties) are
# UNCHANGED — they still produce/expect one-object-per-file — so an object
# built by one of those cannot currently be referenced by a session-based
# perform_freecad_boolean/modify_freecad_parameter call, and vice versa.
_SESSION_DOCUMENT_NAME = "Session_Active"


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


def _terminate_freecad_gui(*, timeout: float = 5.0) -> None:
    """Terminates every running ``FreeCAD.exe`` GUI process and waits
    (briefly) for it to actually exit before ``show_in_freecad_gui`` spawns
    its replacement.

    FreeCAD has no single-instance IPC (see ``show_in_freecad_gui``'s own
    docstring), so an already-running process can never be told a document
    changed on disk — a separate ``FreeCADCmd`` subprocess is what actually
    wrote it — or made to re-run ``_FIT_VIEW_MACRO``'s activate/isometric/
    fit-all snippet. Closing it and launching fresh is the only way to
    guarantee the next screenshot reflects current geometry. Graceful
    ``terminate()`` first, escalating to ``kill()`` only for whatever is
    still alive past ``timeout`` — a plain unsaved-document FreeCAD GUI
    closes near-instantly, so this rarely reaches the escalation path.
    """
    procs = []
    for proc in psutil.process_iter(["name"]):
        try:
            if (proc.info.get("name") or "").lower() == "freecad.exe":
                procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not procs:
        return
    for proc in procs:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass
    for proc in procs:
        try:
            if proc.is_running():
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


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
    # Force this document's own tab frontmost in the GUI before anything
    # else — dana.tools.cad_vision.capture_cad_viewport is a pure OS-level
    # PrintWindow-style screenshot with no FreeCAD scripting of its own, so
    # whatever tab is actually visually frontmost at capture time is
    # exactly what a VLM sees, regardless of what App.ActiveDocument
    # "logically" points to. Uses doc.Name rather than a hardcoded
    # "Session_Active": this ONE macro is shared by every _auto_show
    # (out_path) caller in this module, including the not-yet-migrated
    # one-off-file tools (create_pyramid, create_pipe, ...) that each open
    # their OWN differently-named document here, not just the ones sharing
    # Session_Active.FCStd.
    try:
        Gui.activateDocument(doc.Name)
    except Exception:
        pass
    # A Boolean feature (Part::Cut/MultiFuse/MultiCommon) consumes its
    # Base/Tool/Shapes children into the result, and a Part::Sweep consumes
    # its Sections/Spine profile+path — only the top-level feature should
    # show, not the raw inputs it was built from.
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
        for section in getattr(obj, "Sections", None) or []:
            consumed.add(section.Name)
        spine = getattr(obj, "Spine", None)
        if spine is not None and spine[0] is not None:
            consumed.add(spine[0].Name)
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
    """Open ``filepath`` in a FRESH FreeCAD GUI process on the secondary
    monitor — NEVER stealing OS focus.

    ALWAYS terminates any already-running ``FreeCAD.exe`` first (via
    ``_terminate_freecad_gui``) and launches a new one against ``filepath``.
    This used to reuse an already-running instance instead — but FreeCAD
    has no single-instance IPC, so that process could never be told a
    document had changed on disk (a separate ``FreeCADCmd`` subprocess is
    what actually writes it) or made to re-run ``_FIT_VIEW_MACRO``'s
    activate/isometric/fit-all snippet again. That let
    ``dana.tools.cad_vision.capture_cad_viewport``'s screenshot (a pure
    OS-level PrintWindow-style capture, no FreeCAD scripting of its own)
    show stale geometry — confirmed live: a boolean union was correctly
    computed and saved, but the already-running GUI's own screenshot still
    showed an earlier session's leftover document. Always relaunching costs
    a few seconds of latency and a brief close/reopen flash on the
    secondary monitor each time; zero-focus (``SW_SHOWNOACTIVATE``/
    ``SWP_NOACTIVATE``, never calls ``set_foreground_window`` or any other
    activation API) is otherwise unchanged, so a fullscreen app or game on
    the primary monitor is still never disturbed.
    """
    path = Path(filepath)
    if not path.is_file():
        return _error(f"show_in_freecad_gui: file not found: {filepath}")

    was_running = _is_freecad_gui_running()
    if was_running:
        _terminate_freecad_gui()

    try:
        gui_path = get_freecad_gui_path()
    except FreeCADNotFoundError as exc:
        return _error(str(exc))
    try:
        macro_path = _write_fit_view_macro()
        subprocess.Popen([gui_path, str(path), macro_path])  # noqa: S603
    except OSError as exc:
        return _error(f"show_in_freecad_gui: failed to launch FreeCAD GUI: {exc}")

    # Poll for the window instead of trusting one fixed sleep — cold starts
    # (workbench/plugin loading) can leave the title bar generic for
    # several seconds before it updates to reflect the opened document, and
    # a fixed wait either races that or wastes time once it's already done.
    deadline = time.monotonic() + _WINDOW_POLL_TIMEOUT_S
    window = _find_freecad_window()
    while time.monotonic() < deadline and window is None:
        time.sleep(_WINDOW_POLL_INTERVAL_S)
        window = _find_freecad_window()
    # Every prior instance was just terminated above, so any FreeCAD window
    # found now can only be the one just spawned for `path` — no need to
    # additionally match its title (a slow title-bar update during a cold
    # workbench-loading start would otherwise be mistaken for "wrong file").
    title_matches = window is not None

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
        spawned=True,
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


def _extract_placement(stdout: str) -> list[float] | None:
    """Parse the ``[x, y, z]`` line ``align_objects``'s script prints after
    updating ``Placement.Base`` — same ``ast.literal_eval`` safety as
    ``_extract_bbox``."""
    m = _PLACEMENT_RE.search(stdout or "")
    if not m:
        return None
    try:
        values = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None
    if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
        return [float(v) for v in values]
    return None


def _extract_spatial(stdout: str) -> list[Any] | None:
    """Parse ``inspect_spatial_properties``'s 9-element stdout line
    (``[volume, area, com_x, com_y, com_z, is_valid, face_count, edge_count,
    vertex_count]``) — a flat list rather than a dict literal, matching
    ``_extract_bbox``/``_extract_placement``'s convention, since ``.format()``
    would otherwise need every ``{``/``}`` in a dict literal escaped."""
    m = _SPATIAL_RE.search(stdout or "")
    if not m:
        return None
    try:
        values = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None
    if isinstance(values, list) and len(values) == 9:
        return values
    return None


def _extract_object_name(stdout: str) -> str | None:
    """Parse the actual FreeCAD-assigned ``Name`` a session-document script
    prints via ``_SESSION_RESULT_PRINT`` — see ``_NAME_MARKER``'s own
    comment for why this can differ from the requested ``name`` argument."""
    m = _NAME_RE.search(stdout or "")
    return m.group(1).strip() if m else None


def _run_freecad_script(
    script_text: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
    require_marker: bool = True,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write ``script_text`` to a temp file and execute it via FreeCADCmd.

    ``require_marker`` gates success on ``_OK_MARKER`` appearing in stdout
    (the parametric helpers below print it only after ``saveAs`` succeeds).
    ``execute_freecad_script`` passes ``require_marker=False`` since an
    arbitrary caller-supplied script defines its own notion of success.

    ``extra_env``, when given, is layered on top of a copy of this
    process's own environment (e.g. ``{"DANA_FREECAD_MOD_PATH": "..."}`` so
    a script's own preamble can ``sys.path.append`` an addon workbench
    directory not on FreeCADCmd's default sys.path — see
    ``standard_parts.py``'s ``insert_standard_part``/``_FASTENER_SCRIPT``
    for that exact case. Deliberately NOT ``PYTHONPATH``: FreeCADCmd.exe's
    embedded Python interpreter ignores it on Windows (confirmed live), so
    a script that needs an extra sys.path entry has to add it itself, from
    an env var of its own choosing, rather than relying on this env dict
    alone). ``None`` (the default) means "inherit this process's
    environment unchanged", identical to every other caller
    here that never passed an ``env`` at all before this parameter existed.
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

    env = {**os.environ, **extra_env} if extra_env else None

    try:
        with _lock:
            proc = subprocess.run(
                [cmd_path, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
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
        "placement": _extract_placement(proc.stdout) if ok else None,
        "resolved_name": _extract_object_name(proc.stdout) if ok else None,
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


# Multi-Stage Object Resolution — polls 3 increasingly-loose match
# strategies against ``target_name`` (exact Name, exact Label, then a
# case-insensitive Name match) so a caller can reliably reference ONE
# specific object by name, instead of the old "grab whichever object
# nothing else references" heuristic (``next(o for o in doc.Objects if
# not o.InList)``) — a heuristic that silently returned an ARBITRARY
# sibling object once create_box/create_cylinder/apply_boolean/
# modify_parameter started sharing one Session_Active.FCStd document with
# many top-level objects (get_bounding_box("Cylinder") could silently
# return the Box's bounding box instead, since the old heuristic never
# looked at the requested name at all). Lives as embedded script TEXT
# (not a plain engine.py function) because ``doc`` only exists inside the
# FreeCADCmd subprocess this module launches — see the module docstring.
_RESOLVE_OBJECT_SNIPPET = """\
def resolve_object(doc, target_name):
    obj = doc.getObject(target_name)
    if obj is not None:
        return obj
    matches = [o for o in doc.Objects if o.Label == target_name]
    if matches:
        return matches[0]
    matches = [o for o in doc.Objects if target_name.lower() == o.Name.lower()]
    if matches:
        return matches[0]
    return None
"""


def _object_lookup_snippet(
    *, obj_var: str = "obj", doc_var: str = "doc", target_object: str | None = None
) -> str:
    """Script text binding ``obj_var`` to the ``target_object``-named object in
    ``doc_var`` via ``resolve_object`` (see ``_RESOLVE_OBJECT_SNIPPET`` — must
    already be embedded earlier in the same script), raising a clear
    ``RuntimeError`` — surfaced to the LLM as ``ok: false`` by
    ``_run_freecad_script``'s existing failure path, the same way
    ``apply_boolean``/``modify_parameter`` already report an unknown object —
    rather than silently falling back to an arbitrary sibling when
    ``target_object`` doesn't resolve.

    Falls back to the legacy "first object nothing references" heuristic
    only when no ``target_object`` is given at all — still correct for the
    one-object-per-file case some internal callers use (e.g.
    ``batch_pattern_array``'s own bounding-box read, which has no name to
    give).
    """
    if target_object:
        return (
            f"{obj_var} = resolve_object({doc_var}, {target_object!r})\n"
            f"if {obj_var} is None:\n"
            f'    raise RuntimeError("Object not found: " + {target_object!r})\n'
        )
    return f"{obj_var} = next((o for o in {doc_var}.Objects if not o.InList), {doc_var}.Objects[-1])\n"


def _session_document_path() -> Path:
    """The shared ``Session_Active.FCStd`` path — see ``_SESSION_DOCUMENT_NAME``'s
    module-level comment. Same ``freecad_output/`` directory every other
    ``.FCStd``/``.stl`` artifact already lives in."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR / f"{_SESSION_DOCUMENT_NAME}.FCStd"


# Opens the session document if it already exists (a prior create_box/
# create_cylinder/insert_standard_part/apply_boolean/modify_parameter call
# started it), else starts a fresh in-memory one — `doc`/`_session_existed`
# are then in scope for the rest of the script. `{session_path!r}` is always
# a plain literal path string (never user-controlled beyond what
# ``_session_document_path`` itself returns), so this is injection-safe the
# same way every other ``!r``-formatted script value here already is.
_SESSION_OPEN_SNIPPET = """\
import os

_session_path = {session_path!r}
_session_existed = os.path.isfile(_session_path)
if _session_existed:
    doc = App.openDocument(_session_path)
else:
    doc = App.newDocument({session_doc_name!r})
"""

# Saves back to the SAME path either way: `saveAs` the very first time (the
# in-memory document created above has no path yet), a plain `save` on every
# call after — mirrors modify_parameter's existing reopen-and-overwrite
# pattern, just against a document that now holds many objects instead of one.
_SESSION_SAVE_SNIPPET = """\
if _session_existed:
    doc.save()
else:
    doc.saveAs(_session_path)
"""

# Every session-scoped creation script ends the same way: the new object's
# bounding box, its ACTUAL FreeCAD-assigned Name (see _NAME_MARKER — may
# differ from the requested `name` on a collision), and the shared path.
_SESSION_RESULT_PRINT = _BBOX_PRINT + """\
print("{marker}_NAME " + obj.Name)
print("{marker} path=" + _session_path)
"""

_BOX_SCRIPT = ("""\
import FreeCAD as App

""" + _SESSION_OPEN_SNIPPET + """\
obj = doc.addObject("Part::Box", {name!r})
obj.Length = {length}
obj.Width = {width}
obj.Height = {height}
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
""" + _SESSION_SAVE_SNIPPET + _SESSION_RESULT_PRINT)

_CYLINDER_SCRIPT = ("""\
import FreeCAD as App

""" + _SESSION_OPEN_SNIPPET + """\
obj = doc.addObject("Part::Cylinder", {name!r})
obj.Radius = {radius}
obj.Height = {height}
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
""" + _SESSION_SAVE_SNIPPET + _SESSION_RESULT_PRINT)

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

    Skipped entirely under ``DANA_HEADLESS=true``: ``show_in_freecad_gui``
    always terminates and relaunches the GUI fresh (see its own docstring
    and ``_terminate_freecad_gui``) so every screenshot reflects current
    geometry — but that means every geometry-mutating call visibly closes
    and reopens the FreeCAD window, which is exactly the "constantly
    flashing" symptom an unattended/CI run needs to avoid. Gating the
    per-tool-call vision hook in ``dana.api.server._execute_and_continue``
    alone would leave this call (the actual source of the flash) untouched,
    so it's gated here too, at the source.
    """
    if os.getenv("DANA_HEADLESS", "false").lower() == "true":
        return False
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
    """Create a parametric ``Part::Box`` inside the shared
    ``Session_Active.FCStd`` document (started fresh if this is the first
    session-scoped call), translated by ``placement`` (global X/Y/Z offset
    in mm) on top of its normal corner-at-origin position.

    Returns lean JSON (name/type/bounding_box/dimensions/path) rather than
    echoing every input verbatim — keeps context small for fast local-LLM
    ReAct turns that chain many CAD primitives in a row. ``name`` in the
    result may differ from the requested ``name`` argument if it collided
    with an object already in the session document (FreeCAD auto-suffixes).
    """
    dims = {"length": float(length), "width": float(width), "height": float(height)}
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_box", name=name, type="Part::Box", dimensions=dims, placement=list(placement)
        )
    session_path = _session_document_path()
    script = _BOX_SCRIPT.format(
        name=name,
        length=dims["length"],
        width=dims["width"],
        height=dims["height"],
        placement=placement,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_box failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or name,
        type="Part::Box",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )


def create_cylinder(
    radius: float,
    height: float,
    name: str = "Cylinder",
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Create a parametric ``Part::Cylinder`` inside the shared
    ``Session_Active.FCStd`` document, translated by ``placement`` (global
    X/Y/Z offset in mm). See ``create_box`` for the session-document/
    name-collision notes — identical here."""
    dims = {"radius": float(radius), "height": float(height)}
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_cylinder", name=name, type="Part::Cylinder", dimensions=dims, placement=list(placement)
        )
    session_path = _session_document_path()
    script = _CYLINDER_SCRIPT.format(
        name=name,
        radius=dims["radius"],
        height=dims["height"],
        placement=placement,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_cylinder failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or name,
        type="Part::Cylinder",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(session_path),
        gui_shown=_auto_show(session_path),
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


# 2D-point-per-work-plane embedding: XY keeps (x, y, 0) and extrudes along
# +Z; XZ/YZ embed the same 2 sketch coordinates into the other two axes and
# extrude along whichever axis is left over — a plain lookup, no rotation
# matrices needed since these are the 3 principal planes.
_PLANE_NORMAL: dict[str, tuple[float, float, float]] = {
    "XY": (0.0, 0.0, 1.0),
    "XZ": (0.0, 1.0, 0.0),
    "YZ": (1.0, 0.0, 0.0),
}


def _embed_2d(plane: str, x: float, y: float) -> tuple[float, float, float]:
    if plane == "XY":
        return (x, y, 0.0)
    if plane == "XZ":
        return (x, 0.0, y)
    return (0.0, x, y)  # YZ


def _sketch_edge_specs(
    segments: Sequence[dict[str, Any]], start: Sequence[float], plane: str
) -> list[tuple[str, tuple[Any, ...]]]:
    """Pure geometry prep for ``create_sketch_extrude`` — walks an ordered
    list of ``{"type": "line", "to": [x, y]}`` / ``{"type": "arc", "to":
    [x, y], "via": [x, y]}`` segments into 3D-embedded edge specs the
    FreeCAD script can build ``Part.LineSegment``/``Part.Arc`` from
    directly. No FreeCAD needed here — testable in plain Python, same style
    as ``_alignment_delta``/``_star_polygon_vertices``.
    """
    cur = _embed_2d(plane, float(start[0]), float(start[1]))
    specs: list[tuple[str, tuple[Any, ...]]] = []
    for seg in segments:
        kind = str(seg.get("type", "line")).strip().lower()
        to = _embed_2d(plane, float(seg["to"][0]), float(seg["to"][1]))
        if kind == "arc":
            via = _embed_2d(plane, float(seg["via"][0]), float(seg["via"][1]))
            specs.append(("arc", (cur, via, to)))
        else:
            specs.append(("line", (cur, to)))
        cur = to
    return specs


_SKETCH_EXTRUDE_SCRIPT = """\
import FreeCAD as App
import Part

edges = []
for kind, pts in {edge_specs!r}:
    if kind == "arc":
        p1, pm, p2 = pts
        edges.append(Part.Arc(App.Vector(*p1), App.Vector(*pm), App.Vector(*p2)).toShape())
    else:
        p1, p2 = pts
        edges.append(Part.LineSegment(App.Vector(*p1), App.Vector(*p2)).toShape())
wire = Part.Wire(edges)
nx, ny, nz = {normal!r}
solid = Part.Face(wire).extrude(App.Vector(nx * {height}, ny * {height}, nz * {height}))

doc = App.newDocument("DanaModel")
obj = doc.addObject("Part::Feature", {name!r})
obj.Shape = solid
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""


def create_sketch_extrude(
    segments: Sequence[dict[str, Any]],
    height: float,
    start: tuple[float, float] = (0.0, 0.0),
    plane: str = "XY",
    name: str = "Sketch",
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Draw a closed 2D profile from an ordered list of line/arc segments on
    a chosen work plane, then extrude it into a solid ``Part::Feature`` —
    a higher-leverage foundational primitive than ``create_extruded_polyline``
    for shapes with rounded/arc edges (slots, D-profiles, filleted 2D
    outlines) a straight-edged polyline can't express, without needing a
    full parametric ``Sketcher::SketchObject`` and its constraint solver —
    plain ``Part`` wire construction keeps this stateless and lean, matching
    every other create_* primitive here.

    Each segment is ``{"type": "line", "to": [x, y]}`` or ``{"type": "arc",
    "to": [x, y], "via": [x, y]}`` (a 3-point arc through ``via`` ending at
    ``to``). The profile starts at ``start`` and must close (the last
    segment's ``to`` should equal ``start``).
    """
    plane_u = (plane or "XY").strip().upper()
    if plane_u not in _PLANE_NORMAL:
        return _error(f"create_sketch_extrude: unknown plane '{plane}' — must be XY, XZ, or YZ")
    if not segments:
        return _error("create_sketch_extrude requires at least one segment")
    try:
        edge_specs = _sketch_edge_specs(segments, start, plane_u)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return _error(f"create_sketch_extrude: malformed segment — {exc}")

    dims = {"height": float(height), "plane": plane_u, "segment_count": len(segments)}
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_sketch_extrude", name=name, type="Part::Feature", dimensions=dims, placement=list(placement)
        )
    out_path = _output_path(name, ext="FCStd")
    script = _SKETCH_EXTRUDE_SCRIPT.format(
        edge_specs=edge_specs,
        normal=_PLANE_NORMAL[plane_u],
        height=dims["height"],
        name=name,
        placement=placement,
        out_path=str(out_path),
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_sketch_extrude failed: {result['error']}")
    return _ok(
        name=name,
        type="Part::Feature",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(out_path),
        gui_shown=_auto_show(out_path),
    )


_PATTERN_TYPES = frozenset({"linear", "grid", "circular"})


def _pattern_offsets(
    pattern_type: str,
    *,
    count_x: int = 1,
    count_y: int = 1,
    spacing_x: float = 0.0,
    spacing_y: float = 0.0,
    count: int = 1,
    radius: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """Pure arithmetic: ``(dx, dy, dz, z_rotation_deg)`` offsets for every
    copy in a linear/grid/circular pattern — no FreeCAD needed, same style
    as ``_alignment_delta``/``_star_polygon_vertices``.

    For ``"linear"``/``"grid"``, index 0 is always ``(0, 0, 0, 0)`` — the
    source object's own existing position — so ``count_x=8, count_y=8``
    produces 64 TOTAL placements in one call (the "64 tiles in one tool
    call" case ``batch_pattern_array`` exists to cover), not 64 additional
    ones. ``"circular"`` instead places all ``count`` copies on the circle
    (none necessarily coinciding with the source's original position).
    """
    pt = (pattern_type or "").strip().lower()
    if pt == "linear":
        n = max(1, int(count_x))
        return [(i * spacing_x, 0.0, 0.0, 0.0) for i in range(n)]
    if pt == "grid":
        nx, ny = max(1, int(count_x)), max(1, int(count_y))
        return [(i * spacing_x, j * spacing_y, 0.0, 0.0) for j in range(ny) for i in range(nx)]
    if pt == "circular":
        n = max(1, int(count))
        return [
            (
                radius * math.cos(2 * math.pi * i / n),
                radius * math.sin(2 * math.pi * i / n),
                0.0,
                360.0 * i / n,
            )
            for i in range(n)
        ]
    raise ValueError(f"unknown pattern_type: {pattern_type}")


_PATTERN_ARRAY_SCRIPT = """\
import FreeCAD as App

src_doc = App.openDocument({source_path!r})
base_obj = next((o for o in src_doc.Objects if not o.InList), src_doc.Objects[-1])

doc = App.newDocument("DanaModel")
copies = []
for dx, dy, dz, rot in {offsets!r}:
    c = doc.copyObject(base_obj, False)
    c.Placement = App.Placement(
        base_obj.Placement.Base + App.Vector(dx, dy, dz),
        App.Rotation(App.Vector(0, 0, 1), rot).multiply(base_obj.Placement.Rotation),
    )
    copies.append(c)

obj = doc.addObject("Part::Compound", {name!r})
obj.Links = copies
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""


def batch_pattern_array(
    source_path: str,
    pattern_type: str,
    *,
    count_x: int = 1,
    count_y: int = 1,
    spacing_x: float | None = None,
    spacing_y: float | None = None,
    count: int = 1,
    radius: float = 0.0,
    name: str = "Pattern",
) -> str:
    """Copy a previously-created object into a linear, grid, or circular
    arrangement, combined into a single ``Part::Compound`` — ONE tool call
    instead of one create_freecad_* call per copy, so a repetitive layout
    (e.g. "64 tiles" as an 8x8 grid) doesn't burn through the ReAct loop's
    per-turn iteration cap one placement at a time.

    ``spacing_x``/``spacing_y`` default to the source object's own
    bounding-box width/depth (read via ``get_bounding_box``) so adjacent
    copies sit edge-to-edge with no overlap unless a caller wants a
    deliberate gap or overlap.
    """
    src = Path(source_path)
    if not src.is_file():
        return _error(f"batch_pattern_array: source_path not found: {source_path}")
    pt = (pattern_type or "").strip().lower()
    if pt not in _PATTERN_TYPES:
        return _error(f"batch_pattern_array: unknown pattern_type '{pattern_type}' — must be linear, grid, or circular")

    sx, sy = spacing_x, spacing_y
    if pt in ("linear", "grid") and (sx is None or sy is None):
        bbox = json.loads(get_bounding_box(str(src)))
        if not bbox.get("ok"):
            return _error(f"batch_pattern_array: failed to read source bounding box: {bbox.get('error')}")
        sx = sx if sx is not None else (bbox["x_max"] - bbox["x_min"])
        sy = sy if sy is not None else (bbox["y_max"] - bbox["y_min"])

    try:
        offsets = _pattern_offsets(
            pt,
            count_x=count_x,
            count_y=count_y,
            spacing_x=float(sx or 0.0),
            spacing_y=float(sy or 0.0),
            count=count,
            radius=float(radius),
        )
    except ValueError as exc:
        return _error(f"batch_pattern_array: {exc}")

    dims = {"pattern_type": pt, "copy_count": len(offsets)}
    if is_dry_run_enabled():
        return _dry_run_result("batch_pattern_array", name=name, type="Part::Compound", dimensions=dims)

    out_path = _output_path(name, ext="FCStd")
    script = _PATTERN_ARRAY_SCRIPT.format(
        source_path=str(src), offsets=offsets, name=name, out_path=str(out_path), marker=_OK_MARKER
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"batch_pattern_array failed: {result['error']}")
    return _ok(
        name=name,
        type="Part::Compound",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        path=str(out_path),
        gui_shown=_auto_show(out_path),
    )


# FreeCAD Part::Cut is a Base/Tool pair; Part::MultiFuse/MultiCommon instead
# take a Shapes list — two script shapes, chosen in Python (not branched
# inside the FreeCADCmd subprocess) by which the operation actually needs.
_BOOLEAN_FEATURE_TYPE: dict[str, str] = {
    "cut": "Part::Cut",
    "union": "Part::MultiFuse",
    "intersect": "Part::MultiCommon",
}
_DEFAULT_BOOLEAN_NAME: dict[str, str] = {"cut": "Cut", "union": "Fusion", "intersect": "Common"}

# Both objects already live in the SAME shared session document (unlike the
# old design, which opened base_doc/tool_doc as two SEPARATE files and had
# to doc.copyObject(...) each into a third, brand-new document just to get
# them into one place for the Boolean feature) — so Base/Tool/Shapes can
# reference them DIRECTLY, looked up by Name via doc.getObject.
_BOOLEAN_CUT_SCRIPT = """\
import FreeCAD as App

_session_path = {session_path!r}
doc = App.openDocument(_session_path)
base_obj = doc.getObject({base_object!r})
tool_obj = doc.getObject({tool_object!r})
if base_obj is None:
    raise RuntimeError("no object named " + {base_object!r} + " in the session document")
if tool_obj is None:
    raise RuntimeError("no object named " + {tool_object!r} + " in the session document")

obj = doc.addObject("Part::Cut", {name!r})
obj.Base = base_obj
obj.Tool = tool_obj
obj.Base.Visibility = False
obj.Tool.Visibility = False
doc.recompute()
doc.save()
""" + _SESSION_RESULT_PRINT

_BOOLEAN_FUSE_COMMON_SCRIPT = """\
import FreeCAD as App

_session_path = {session_path!r}
doc = App.openDocument(_session_path)
base_obj = doc.getObject({base_object!r})
tool_obj = doc.getObject({tool_object!r})
if base_obj is None:
    raise RuntimeError("no object named " + {base_object!r} + " in the session document")
if tool_obj is None:
    raise RuntimeError("no object named " + {tool_object!r} + " in the session document")

obj = doc.addObject({feature_type!r}, {name!r})
obj.Shapes = [base_obj, tool_obj]
for _shape_obj in obj.Shapes:
    _shape_obj.Visibility = False
doc.recompute()
doc.save()
""" + _SESSION_RESULT_PRINT

_EXPORT_STL_SCRIPT = """\
import FreeCAD as App
import Mesh

""" + _RESOLVE_OBJECT_SNIPPET + """\
doc = App.openDocument({source_path!r})
{lookup}Mesh.export({export_targets}, {out_path!r})
print("{marker} path=" + {out_path!r})
"""


def apply_boolean(operation: str, base_object: str, tool_object: str, name: str | None = None) -> str:
    """Combine two objects already in the shared ``Session_Active.FCStd``
    document with a Boolean operation, looked up by NAME — not path, since
    every session-scoped creation tool (``create_box``/``create_cylinder``/
    ``insert_standard_part``) now shares that one document, so a path alone
    can no longer tell two objects apart the way it could when each lived in
    its own file.

    ``"cut"`` builds a ``Part::Cut`` (subtracts the tool from the base);
    ``"union"`` builds a ``Part::MultiFuse`` (fuses both into one solid);
    ``"intersect"`` builds a ``Part::MultiCommon`` (keeps only their
    overlapping volume). ``base_object``/``tool_object`` must already exist
    in the session document — built by a session-scoped creation tool, or a
    prior ``apply_boolean`` call's own result name.
    """
    op = (operation or "").strip().lower()
    if op not in _BOOLEAN_FEATURE_TYPE:
        return _error(f"apply_boolean: unknown operation '{operation}' — must be cut, union, or intersect")
    feature_type = _BOOLEAN_FEATURE_TYPE[op]
    resolved_name = name or _DEFAULT_BOOLEAN_NAME[op]
    if is_dry_run_enabled():
        return _dry_run_result("apply_boolean", operation=op, name=resolved_name, type=feature_type)
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error(
            "apply_boolean: no session document yet — create objects with create_box/"
            "create_cylinder/insert_standard_part first"
        )
    if op == "cut":
        script = _BOOLEAN_CUT_SCRIPT.format(
            session_path=str(session_path),
            base_object=base_object,
            tool_object=tool_object,
            name=resolved_name,
            marker=_OK_MARKER,
        )
    else:
        script = _BOOLEAN_FUSE_COMMON_SCRIPT.format(
            session_path=str(session_path),
            base_object=base_object,
            tool_object=tool_object,
            feature_type=feature_type,
            name=resolved_name,
            marker=_OK_MARKER,
        )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"apply_boolean failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or resolved_name,
        type=feature_type,
        operation=op,
        bounding_box=result.get("bounding_box"),
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )


_EDGE_FEATURE_TYPE: dict[str, str] = {"fillet": "Part::Fillet", "chamfer": "Part::Chamfer"}
_DEFAULT_EDGE_NAME: dict[str, str] = {"fillet": "Fillet", "chamfer": "Chamfer"}

# Targets every edge of the copied object — a global fillet/chamfer.
_EDGE_OP_WHOLE_SCRIPT = """\
import FreeCAD as App

""" + _RESOLVE_OBJECT_SNIPPET + """\
base_doc = App.openDocument({target_path!r})
{lookup}
doc = App.newDocument("DanaModel")
copied = doc.copyObject(base_obj, False)
target_indices = list(range(1, len(copied.Shape.Edges) + 1))
if not target_indices:
    raise RuntimeError("target object has no edges")

obj = doc.addObject({feature_type!r}, {name!r})
obj.Base = copied
obj.Edges = [(i, {value}, {value}) for i in target_indices]
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""

# No raycasting against the tessellated STL used for display — the nearest
# BRep face (by CenterOfMass) to the clicked-point centroid is found directly
# against FreeCAD's own exact geometry, then only that face's bounding edges
# are targeted.
_EDGE_OP_FACE_SCRIPT = """\
import FreeCAD as App

""" + _RESOLVE_OBJECT_SNIPPET + """\
base_doc = App.openDocument({target_path!r})
{lookup}
doc = App.newDocument("DanaModel")
copied = doc.copyObject(base_obj, False)

target_point = App.Vector({cx}, {cy}, {cz})
faces = copied.Shape.Faces
if not faces:
    raise RuntimeError("target object has no faces")
nearest_face = min(faces, key=lambda f: f.CenterOfMass.distanceToPoint(target_point))
target_indices = [
    i for i, edge in enumerate(copied.Shape.Edges, start=1)
    if any(edge.isSame(face_edge) for face_edge in nearest_face.Edges)
]
if not target_indices:
    raise RuntimeError("no edges found on the nearest face")

obj = doc.addObject({feature_type!r}, {name!r})
obj.Base = copied
obj.Edges = [(i, {value}, {value}) for i in target_indices]
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""


def apply_edge_operation(
    operation: str,
    target_path: str,
    value: float,
    face_centroid: tuple[float, float, float] | None = None,
    name: str | None = None,
    target_object: str | None = None,
) -> str:
    """Round (``"fillet"``) or bevel (``"chamfer"``) the edges of a
    previously-created solid.

    Without ``face_centroid``, every edge of the object gets the operation
    (a global fillet/chamfer, ``value`` mm). With ``face_centroid`` —
    typically the active canvas selection's clicked-face centroid — only
    the edges bounding the face nearest that point are targeted, found
    against FreeCAD's exact BRep geometry (no raycasting against the
    tessellated display mesh).

    ``target_object``, when given, is resolved via Multi-Stage Object
    Resolution — see ``get_bounding_box``'s matching note.
    """
    op = (operation or "").strip().lower()
    if op not in _EDGE_FEATURE_TYPE:
        return _error(f"apply_edge_operation: unknown operation '{operation}' — must be fillet or chamfer")
    target = Path(target_path)
    if not target.is_file():
        return _error(f"apply_edge_operation: target_path not found: {target_path}")
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return _error(f"apply_edge_operation: value must be a number, got {value!r}")
    if value_f <= 0:
        return _error("apply_edge_operation: value must be a positive number")

    feature_type = _EDGE_FEATURE_TYPE[op]
    resolved_name = name or _DEFAULT_EDGE_NAME[op]
    face_targeted = face_centroid is not None
    if is_dry_run_enabled():
        return _dry_run_result(
            "apply_edge_operation",
            operation=op,
            name=resolved_name,
            type=feature_type,
            face_targeted=face_targeted,
        )
    out_path = _output_path(resolved_name, ext="FCStd")
    lookup = _object_lookup_snippet(obj_var="base_obj", doc_var="base_doc", target_object=target_object)
    if face_targeted:
        cx, cy, cz = (float(face_centroid[0]), float(face_centroid[1]), float(face_centroid[2]))
        script = _EDGE_OP_FACE_SCRIPT.format(
            target_path=str(target),
            feature_type=feature_type,
            name=resolved_name,
            value=value_f,
            cx=cx,
            cy=cy,
            cz=cz,
            out_path=str(out_path),
            marker=_OK_MARKER,
            lookup=lookup,
        )
    else:
        script = _EDGE_OP_WHOLE_SCRIPT.format(
            target_path=str(target),
            feature_type=feature_type,
            name=resolved_name,
            value=value_f,
            out_path=str(out_path),
            marker=_OK_MARKER,
            lookup=lookup,
        )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"apply_edge_operation failed: {result['error']}")
    return _ok(
        name=resolved_name,
        type=feature_type,
        operation=op,
        face_targeted=face_targeted,
        bounding_box=result.get("bounding_box"),
        path=str(out_path),
        gui_shown=_auto_show(out_path),
    )


_MODIFY_PARAMETER_SCRIPT = """\
import FreeCAD as App

doc = App.openDocument({session_path!r})
obj = doc.getObject({target_object!r})
if obj is None:
    raise RuntimeError("no object named " + {target_object!r} + " in the session document")
setattr(obj, {parameter_name!r}, {new_value})
doc.recompute()
doc.save()
""" + _BBOX_PRINT + """\
print("{marker} path=" + {session_path!r})
"""

# "Placement"/"Placement.Base" is a 3D translation, not a bare settable
# number — setattr(obj, "Placement", 5.0) would fail outright. Replace the
# whole Placement with a new Vector base while keeping the object's
# existing Rotation, so a move never silently discards prior orientation.
_VECTOR_PARAMETER_NAMES = frozenset({"placement", "placement.base"})

_MODIFY_PARAMETER_VECTOR_SCRIPT = """\
import FreeCAD as App

doc = App.openDocument({session_path!r})
obj = doc.getObject({target_object!r})
if obj is None:
    raise RuntimeError("no object named " + {target_object!r} + " in the session document")
obj.Placement = App.Placement(App.Vector({x}, {y}, {z}), obj.Placement.Rotation)
doc.recompute()
doc.save()
""" + _BBOX_PRINT + """\
print("{marker} path=" + {session_path!r})
"""


def modify_parameter(
    target_object: str, parameter_name: str, new_value: float | Sequence[float]
) -> str:
    """Change a single dimensional property (e.g. ``"Height"``, ``"Radius"``)
    on an object already in the shared ``Session_Active.FCStd`` document, by
    NAME — in place, so the object's parametric history/name are preserved
    across the edit.

    ``parameter_name`` of ``"Placement"`` or ``"Placement.Base"`` is special:
    it moves the object, so ``new_value`` must be a 3-number ``[x, y, z]``
    vector (mm) instead of a single float — applied as a new
    ``FreeCAD.Vector`` on ``Placement.Base`` while the object's current
    ``Placement.Rotation`` is preserved.
    """
    param = (parameter_name or "").strip()
    if not param:
        return _error("modify_parameter requires a non-empty parameter_name")
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error(
            "modify_parameter: no session document yet — create objects with create_box/"
            "create_cylinder/insert_standard_part first"
        )

    if param.lower() in _VECTOR_PARAMETER_NAMES:
        try:
            x, y, z = (float(component) for component in new_value)
        except (TypeError, ValueError):
            return _error(
                f"modify_parameter: {param} new_value must be a 3-number [x, y, z] vector, got {new_value!r}"
            )
        if is_dry_run_enabled():
            return _dry_run_result(
                "modify_parameter", path=str(session_path), parameter_name=param, new_value=[x, y, z]
            )
        script = _MODIFY_PARAMETER_VECTOR_SCRIPT.format(
            session_path=str(session_path),
            target_object=target_object,
            x=x,
            y=y,
            z=z,
            marker=_OK_MARKER,
        )
        result_value: float | list[float] = [x, y, z]
    else:
        try:
            value_f = float(new_value)
        except (TypeError, ValueError):
            return _error(f"modify_parameter: new_value must be a number, got {new_value!r}")
        if is_dry_run_enabled():
            return _dry_run_result(
                "modify_parameter", path=str(session_path), parameter_name=param, new_value=value_f
            )
        script = _MODIFY_PARAMETER_SCRIPT.format(
            session_path=str(session_path),
            target_object=target_object,
            parameter_name=param,
            new_value=value_f,
            marker=_OK_MARKER,
        )
        result_value = value_f

    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"modify_parameter failed: {result['error']}")
    return _ok(
        name=target_object,
        path=str(session_path),
        parameter_name=param,
        new_value=result_value,
        bounding_box=result.get("bounding_box"),
        gui_shown=_auto_show(session_path),
    )


_GET_BOUNDING_BOX_SCRIPT = """\
import FreeCAD as App

""" + _RESOLVE_OBJECT_SNIPPET + """\
doc = App.openDocument({target_path!r})
{lookup}""" + _BBOX_PRINT + """\
print("{marker} path=" + {target_path!r})
"""


def get_bounding_box(target_path: str, target_object: str | None = None) -> str:
    """Read-only: the physical bounding box of a previously-created object,
    in mm. Never saves — a query, not a mutation, so it never needs the
    HITL approval gate the create_*/apply_* mutators do.

    ``target_object``, when given, is resolved via Multi-Stage Object
    Resolution (exact Name, then Label, then case-insensitive Name) rather
    than the legacy "first object nothing references" heuristic — required
    once ``target_path`` can point at a shared multi-object session
    document rather than a dedicated one-object-per-file document.
    """
    target = Path(target_path)
    if not target.is_file():
        return _error(f"get_bounding_box: target_path not found: {target_path}")
    if is_dry_run_enabled():
        return _dry_run_result(
            "get_bounding_box", path=str(target),
            x_min=0.0, y_min=0.0, z_min=0.0, x_max=0.0, y_max=0.0, z_max=0.0,
        )
    script = _GET_BOUNDING_BOX_SCRIPT.format(
        target_path=str(target),
        marker=_OK_MARKER,
        lookup=_object_lookup_snippet(target_object=target_object),
    )
    result = _run_freecad_script(script, require_marker=True)
    if not result["ok"]:
        return _error(f"get_bounding_box failed: {result['error']}")
    bbox = result.get("bounding_box") or [0.0] * 6
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    return _ok(
        path=str(target),
        x_min=x_min,
        y_min=y_min,
        z_min=z_min,
        x_max=x_max,
        y_max=y_max,
        z_max=z_max,
    )


_INSPECT_SPATIAL_SCRIPT = """\
import FreeCAD as App

""" + _RESOLVE_OBJECT_SNIPPET + """\
doc = App.openDocument({target_path!r})
{lookup}shape = obj.Shape
com = shape.CenterOfMass
print("{marker}_SPATIAL " + str([
    shape.Volume, shape.Area, com.x, com.y, com.z,
    shape.isValid(), len(shape.Faces), len(shape.Edges), len(shape.Vertexes),
]))
""" + _BBOX_PRINT + """\
print("{marker} path=" + {target_path!r})
"""


def inspect_spatial_properties(target_path: str, target_object: str | None = None) -> str:
    """Read-only: richer topology introspection than ``get_bounding_box`` —
    solid volume, surface area, center of mass, validity, and face/edge/
    vertex counts for a previously-created object. Never saves, so — like
    ``get_bounding_box`` — it never needs the HITL approval gate the
    create_*/apply_* mutators do.

    Lets a caller (the LLM, mid-ReAct-loop) "look before it leaps": check
    edge/face count and validity before a risky fillet/chamfer/boolean
    rather than only discovering geometric infeasibility after the fact.

    ``target_object``, when given, is resolved via Multi-Stage Object
    Resolution — see ``get_bounding_box``'s matching note.
    """
    target = Path(target_path)
    if not target.is_file():
        return _error(f"inspect_spatial_properties: target_path not found: {target_path}")
    if is_dry_run_enabled():
        return _dry_run_result(
            "inspect_spatial_properties",
            path=str(target),
            volume=0.0,
            area=0.0,
            center_of_mass=[0.0, 0.0, 0.0],
            is_valid=True,
            face_count=0,
            edge_count=0,
            vertex_count=0,
        )
    script = _INSPECT_SPATIAL_SCRIPT.format(
        target_path=str(target),
        marker=_OK_MARKER,
        lookup=_object_lookup_snippet(target_object=target_object),
    )
    result = _run_freecad_script(script, require_marker=True)
    if not result["ok"]:
        return _error(f"inspect_spatial_properties failed: {result['error']}")
    spatial = _extract_spatial(result["stdout"]) or [0.0, 0.0, 0.0, 0.0, 0.0, True, 0, 0, 0]
    volume, area, cx, cy, cz, is_valid, face_count, edge_count, vertex_count = spatial
    return _ok(
        path=str(target),
        volume=float(volume),
        area=float(area),
        center_of_mass=[float(cx), float(cy), float(cz)],
        is_valid=bool(is_valid),
        face_count=int(face_count),
        edge_count=int(edge_count),
        vertex_count=int(vertex_count),
        bounding_box=result.get("bounding_box"),
    )


_ALIGNMENT_TYPES = frozenset({"top_center", "bottom_center", "flush_left", "flush_right"})

_ALIGN_APPLY_SCRIPT = """\
import FreeCAD as App

""" + _RESOLVE_OBJECT_SNIPPET + """\
doc = App.openDocument({source_path!r})
{lookup}obj.Placement.Base = obj.Placement.Base + App.Vector({dx}, {dy}, {dz})
doc.recompute()
doc.save()
print("{marker}_PLACEMENT " + str([obj.Placement.Base.x, obj.Placement.Base.y, obj.Placement.Base.z]))
""" + _BBOX_PRINT + """\
print("{marker} path=" + {source_path!r})
"""


def _alignment_delta(
    alignment_type: str, source_bbox: dict[str, Any], target_bbox: dict[str, Any]
) -> tuple[float, float, float]:
    """Pure-Python XYZ delta for each ``alignment_type`` — plain arithmetic
    on two ``get_bounding_box``-shaped dicts, no FreeCAD needed, so this is
    independently unit-testable.

    ``top_center``/``bottom_center`` stack the source directly above/below
    the target (a mirror pair — the directive spells out ``top_center``'s
    formula explicitly; ``bottom_center`` is its natural counterpart, source
    hanging below rather than resting above). ``flush_left``/``flush_right``
    instead make the source's -X/+X face coincide with the target's (a flush
    seam, not a stack). Every axis not being explicitly aligned is centered
    on the target rather than left at an arbitrary offset — matching "snap
    to the bounding box" rather than "move part-way and hope."
    """
    sbb, tbb = source_bbox, target_bbox
    scx = (sbb["x_min"] + sbb["x_max"]) / 2.0
    scy = (sbb["y_min"] + sbb["y_max"]) / 2.0
    scz = (sbb["z_min"] + sbb["z_max"]) / 2.0
    tcx = (tbb["x_min"] + tbb["x_max"]) / 2.0
    tcy = (tbb["y_min"] + tbb["y_max"]) / 2.0
    tcz = (tbb["z_min"] + tbb["z_max"]) / 2.0

    if alignment_type == "top_center":
        return (tcx - scx, tcy - scy, tbb["z_max"] - sbb["z_min"])
    if alignment_type == "bottom_center":
        return (tcx - scx, tcy - scy, tbb["z_min"] - sbb["z_max"])
    if alignment_type == "flush_left":
        return (tbb["x_min"] - sbb["x_min"], tcy - scy, tcz - scz)
    if alignment_type == "flush_right":
        return (tbb["x_max"] - sbb["x_max"], tcy - scy, tcz - scz)
    raise ValueError(f"unknown alignment_type: {alignment_type}")


def align_objects(
    source_path: str,
    target_path: str,
    alignment_type: str,
    source_object: str | None = None,
    target_object: str | None = None,
) -> str:
    """Snap ``source_path``'s object directly to ``target_path``'s
    bounding box (``alignment_type`` one of ``top_center``/``bottom_center``/
    ``flush_left``/``flush_right``) by translating the source object's
    ``Placement.Base`` in place — same document/path, like
    ``modify_parameter``, since this moves the existing object rather than
    creating a new feature. Reads both bounding boxes via
    ``get_bounding_box`` (plain Python delta math, no FreeCAD needed for
    that part) before touching FreeCAD at all.

    ``source_object``/``target_object``, when given, are resolved via
    Multi-Stage Object Resolution rather than the legacy heuristic —
    required once ``source_path``/``target_path`` can be the SAME shared
    session document (two distinct objects can no longer be told apart by
    path alone in that case).
    """
    align = (alignment_type or "").strip().lower()
    if align not in _ALIGNMENT_TYPES:
        return _error(
            f"align_objects: unknown alignment_type '{alignment_type}' — "
            f"must be one of {', '.join(sorted(_ALIGNMENT_TYPES))}"
        )
    source = Path(source_path)
    target = Path(target_path)
    if not source.is_file():
        return _error(f"align_objects: source_path not found: {source_path}")
    if not target.is_file():
        return _error(f"align_objects: target_path not found: {target_path}")

    source_bbox = json.loads(get_bounding_box(str(source), target_object=source_object))
    if not source_bbox.get("ok"):
        return _error(f"align_objects: failed to read source bounding box: {source_bbox.get('error')}")
    target_bbox = json.loads(get_bounding_box(str(target), target_object=target_object))
    if not target_bbox.get("ok"):
        return _error(f"align_objects: failed to read target bounding box: {target_bbox.get('error')}")

    dx, dy, dz = _alignment_delta(align, source_bbox, target_bbox)

    if is_dry_run_enabled():
        return _dry_run_result("align_objects", alignment_type=align, path=str(source), delta=[dx, dy, dz])

    script = _ALIGN_APPLY_SCRIPT.format(
        source_path=str(source),
        dx=dx,
        dy=dy,
        dz=dz,
        marker=_OK_MARKER,
        lookup=_object_lookup_snippet(target_object=source_object),
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"align_objects failed: {result['error']}")
    return _ok(
        name=source.stem,
        path=str(source),
        alignment_type=align,
        placement=result.get("placement"),
        bounding_box=result.get("bounding_box"),
        gui_shown=_auto_show(source),
    )


_MATE_TYPES = frozenset({"concentric", "coincident_planar", "offset_axial"})


def _mate_delta(
    mate_type: str, params: dict[str, Any], fixed_bbox: dict[str, Any], moving_bbox: dict[str, Any]
) -> tuple[float, float, float]:
    """Pure-Python XYZ delta for each ``mate_type`` — same bbox-center
    arithmetic style as ``_alignment_delta``, generalized with caller-
    supplied numeric params so two primitives can be positioned as a true
    (if approximate) kinematic pair rather than only bbox-snapped.

    Approximates a real CAD assembly mate's constraint solve with bounding-
    box-center math rather than genuine face/axis-normal detection or a
    persistent LCS/constraint-solver object — exact for axis-aligned
    primitives (create_freecad_*'s own output), and it keeps every mate a
    single stateless file-in/file-out translation like ``align_objects``,
    with no separate assembly-constraint state to keep in sync.

    ``"concentric"``: center the MOVING object's XY footprint on the FIXED
    object's (their vertical central axes coincide), at an optional
    ``z_offset`` — e.g. a shaft (moving) mated concentric inside a bearing
    bore (fixed).
    ``"coincident_planar"``: make the moving object's bottom face coincide
    with the fixed object's top face (a flat mating plane), at an optional
    in-plane ``offset_x``/``offset_y`` — e.g. a plate resting flush on a boss.
    ``"offset_axial"``: center the moving object's XY footprint on the
    fixed object's, standing off ``distance`` mm along Z from the fixed
    object's top (or, with ``from_face="bottom"``, bottom) face — e.g. a
    shaft protruding a fixed distance above a motor's pilot boss.
    """
    fbb, mbb = fixed_bbox, moving_bbox
    fcx = (fbb["x_min"] + fbb["x_max"]) / 2.0
    fcy = (fbb["y_min"] + fbb["y_max"]) / 2.0
    mcx = (mbb["x_min"] + mbb["x_max"]) / 2.0
    mcy = (mbb["y_min"] + mbb["y_max"]) / 2.0
    dx, dy = fcx - mcx, fcy - mcy

    if mate_type == "concentric":
        return (dx, dy, float(params.get("z_offset", 0.0)))
    if mate_type == "coincident_planar":
        offset_x = float(params.get("offset_x", 0.0))
        offset_y = float(params.get("offset_y", 0.0))
        dz = fbb["z_max"] - mbb["z_min"]
        return (dx + offset_x, dy + offset_y, dz)
    if mate_type == "offset_axial":
        distance = float(params.get("distance", 0.0))
        from_face = str(params.get("from_face", "top")).strip().lower()
        if from_face == "bottom":
            dz = (fbb["z_min"] - distance) - mbb["z_max"]
        else:
            dz = (fbb["z_max"] + distance) - mbb["z_min"]
        return (dx, dy, dz)
    raise ValueError(f"unknown mate_type: {mate_type}")


def create_assembly_mate(
    fixed_path: str,
    moving_path: str,
    mate_type: str,
    mate_params: dict[str, Any] | None = None,
    fixed_object: str | None = None,
    moving_object: str | None = None,
) -> str:
    """Position ``moving_path``'s object relative to ``fixed_path``'s
    object as a named kinematic mate (``mate_type`` one of ``concentric``/
    ``coincident_planar``/``offset_axial``), translating the MOVING
    object's ``Placement.Base`` in place — same document/path, like
    ``align_objects``/``modify_parameter``, since this moves an existing
    object rather than creating a new feature. Reads both bounding boxes
    via ``get_bounding_box`` (plain Python delta math via ``_mate_delta``,
    no FreeCAD needed for that part) before touching FreeCAD at all, then
    reuses ``align_objects``'s own apply script verbatim — a mate and an
    alignment are the same FreeCAD operation (translate + save), they only
    differ in how the delta gets computed.

    ``fixed_object``/``moving_object``, when given, are resolved via
    Multi-Stage Object Resolution — see ``align_objects``'s matching note.
    """
    mt = (mate_type or "").strip().lower()
    if mt not in _MATE_TYPES:
        return _error(
            f"create_assembly_mate: unknown mate_type '{mate_type}' — must be one of {', '.join(sorted(_MATE_TYPES))}"
        )
    fixed = Path(fixed_path)
    moving = Path(moving_path)
    if not fixed.is_file():
        return _error(f"create_assembly_mate: fixed_path not found: {fixed_path}")
    if not moving.is_file():
        return _error(f"create_assembly_mate: moving_path not found: {moving_path}")

    params = dict(mate_params or {})
    fixed_bbox = json.loads(get_bounding_box(str(fixed), target_object=fixed_object))
    if not fixed_bbox.get("ok"):
        return _error(f"create_assembly_mate: failed to read fixed object's bounding box: {fixed_bbox.get('error')}")
    moving_bbox = json.loads(get_bounding_box(str(moving), target_object=moving_object))
    if not moving_bbox.get("ok"):
        return _error(f"create_assembly_mate: failed to read moving object's bounding box: {moving_bbox.get('error')}")

    try:
        dx, dy, dz = _mate_delta(mt, params, fixed_bbox, moving_bbox)
    except ValueError as exc:
        return _error(f"create_assembly_mate: {exc}")

    if is_dry_run_enabled():
        return _dry_run_result("create_assembly_mate", mate_type=mt, path=str(moving), delta=[dx, dy, dz])

    script = _ALIGN_APPLY_SCRIPT.format(
        source_path=str(moving),
        dx=dx,
        dy=dy,
        dz=dz,
        marker=_OK_MARKER,
        lookup=_object_lookup_snippet(target_object=moving_object),
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_assembly_mate failed: {result['error']}")
    return _ok(
        name=moving.stem,
        path=str(moving),
        mate_type=mt,
        fixed_object=str(fixed),
        placement=result.get("placement"),
        bounding_box=result.get("bounding_box"),
        gui_shown=_auto_show(moving),
    )


# Part::Sweep's own bend radius isn't exposed by create_freecad_pipe's schema
# (only the pipe's cross-section radius and the sweep angle are) — this is a
# conventional-enough elbow curvature default for that narrower schema.
_PIPE_ARC_BEND_RADIUS_MULTIPLIER = 3.0
_PIPE_ARC_MIN_BEND_RADIUS = 20.0

# A circle profile in the XY plane already faces +Z, which is exactly right
# for sweeping straight up along a Part::Line path — no reorientation needed.
_PIPE_STRAIGHT_SCRIPT = """\
import FreeCAD as App

doc = App.newDocument("DanaModel")

profile = doc.addObject("Part::Circle", "Profile")
profile.Radius = {pipe_radius}

path = doc.addObject("Part::Line", "Path")
path.X1, path.Y1, path.Z1 = 0.0, 0.0, 0.0
path.X2, path.Y2, path.Z2 = 0.0, 0.0, {length_or_angle}

doc.recompute()

obj = doc.addObject("Part::Sweep", {name!r})
obj.Sections = [profile]
obj.Spine = (path, [])
obj.Solid = True
obj.Frenet = False
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""

# The arc path (a Part::Circle restricted to Angle1..Angle2) starts at angle
# 0 -> point (arc_radius, 0, 0), tangent +Y there — rotating the profile
# circle 90 deg about X turns its default +Z-facing normal to face +Y,
# perpendicular to that tangent, so the sweep starts from a valid cross-
# section regardless of how far Angle2 extends. Frenet=True lets Part::Sweep
# transport/reorient that cross-section along the rest of the curving path.
_PIPE_ARC_SCRIPT = """\
import FreeCAD as App

doc = App.newDocument("DanaModel")

profile = doc.addObject("Part::Circle", "Profile")
profile.Radius = {pipe_radius}
profile.Placement = App.Placement(App.Vector({arc_radius}, 0.0, 0.0), App.Rotation(App.Vector(1, 0, 0), 90))

path = doc.addObject("Part::Circle", "Path")
path.Radius = {arc_radius}
path.Angle1 = 0.0
path.Angle2 = {length_or_angle}

doc.recompute()

obj = doc.addObject("Part::Sweep", {name!r})
obj.Sections = [profile]
obj.Spine = (path, [])
obj.Solid = True
obj.Frenet = True
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""


def create_pipe(
    pipe_radius: float,
    path_type: str,
    length_or_angle: float,
    name: str = "Pipe",
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Sweep a circular profile (``pipe_radius`` mm) into a tubular
    ``Part::Sweep`` solid and save it.

    ``path_type="straight"`` sweeps ``length_or_angle`` mm along a straight
    line (a plain cylindrical pipe). ``path_type="arc"`` sweeps
    ``length_or_angle`` degrees along a circular arc (a curved elbow) with
    a default bend radius (see ``_PIPE_ARC_BEND_RADIUS_MULTIPLIER`` — the
    schema has no separate bend-radius parameter of its own).
    """
    pt = (path_type or "").strip().lower()
    if pt not in ("straight", "arc"):
        return _error(f"create_pipe: unknown path_type '{path_type}' — must be straight or arc")
    try:
        radius_f = float(pipe_radius)
        value_f = float(length_or_angle)
    except (TypeError, ValueError):
        return _error("create_pipe: pipe_radius and length_or_angle must be numbers")
    if radius_f <= 0:
        return _error("create_pipe: pipe_radius must be a positive number")
    if value_f <= 0:
        return _error("create_pipe: length_or_angle must be a positive number")
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))

    dims = {"pipe_radius": radius_f, "path_type": pt, "length_or_angle": value_f}
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_pipe", name=name, type="Part::Sweep", dimensions=dims, placement=list(placement)
        )

    out_path = _output_path(name, ext="FCStd")
    if pt == "straight":
        script = _PIPE_STRAIGHT_SCRIPT.format(
            pipe_radius=radius_f,
            length_or_angle=value_f,
            name=name,
            placement=placement,
            out_path=str(out_path),
            marker=_OK_MARKER,
        )
    else:
        arc_radius = max(radius_f * _PIPE_ARC_BEND_RADIUS_MULTIPLIER, _PIPE_ARC_MIN_BEND_RADIUS)
        script = _PIPE_ARC_SCRIPT.format(
            pipe_radius=radius_f,
            arc_radius=arc_radius,
            length_or_angle=value_f,
            name=name,
            placement=placement,
            out_path=str(out_path),
            marker=_OK_MARKER,
        )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_pipe failed: {result['error']}")
    return _ok(
        name=name,
        type="Part::Sweep",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(out_path),
        gui_shown=_auto_show(out_path),
    )


def export_mesh_stl(source_path: str, name: str | None = None, target_object: str | None = None) -> str:
    """Tessellate ``source_path`` (a ``.FCStd`` document) into a standalone
    ``.stl`` mesh file — the hand-off format for ``gr.Model3D`` viewers
    that can't load native FreeCAD documents.

    ``target_object``, when given, tessellates ONLY that resolved object
    (Multi-Stage Object Resolution — see ``get_bounding_box``'s matching
    note) rather than every object in the document — required once
    ``source_path`` can be a shared multi-object session document, where
    exporting ``list(doc.Objects)`` would silently bundle in unrelated
    sibling objects (and, after a Boolean, its already-consumed Base/Tool
    inputs) alongside the one the caller actually meant. Without it, every
    object in the document is exported — unchanged legacy behavior for
    callers that don't have a specific object name to give.
    """
    source = Path(source_path)
    if not source.is_file():
        return _error(f"export_mesh_stl: source_path not found: {source_path}")
    if is_dry_run_enabled():
        return _dry_run_result("export_mesh_stl", source_path=str(source))
    out_path = _output_path(name or source.stem, ext="stl")
    script = _EXPORT_STL_SCRIPT.format(
        source_path=str(source),
        out_path=str(out_path),
        marker=_OK_MARKER,
        lookup=_object_lookup_snippet(target_object=target_object) if target_object else "",
        export_targets="[obj]" if target_object else "list(doc.Objects)",
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"export_mesh_stl failed: {result['error']}")
    return _ok(op="export_mesh_stl", source_path=str(source), path=str(out_path))


# Distinct from export_mesh_stl above (an internal single-document ->
# temp-STL hop for the 3D viewer's preview mesh) — this is the user-facing,
# possibly-multi-object manufacturing/handoff export, saved to a permanent
# named file under _EXPORT_DIR rather than a throwaway viewer temp path.
_EXPORT_FORMAT_EXT: dict[str, str] = {"stl": "stl", "step": "step"}

_EXPORT_MODEL_STL_SCRIPT = """\
import FreeCAD as App
import Mesh

""" + _RESOLVE_OBJECT_SNIPPET + """\
objects = []
for p, n in {target_specs!r}:
    d = App.openDocument(p)
    if n:
        o = resolve_object(d, n)
        if o is None:
            raise RuntimeError("Object not found: " + n)
    else:
        o = next((x for x in d.Objects if not x.InList), d.Objects[-1])
    objects.append(o)

Mesh.export(objects, {out_path!r})
print("{marker} path=" + {out_path!r})
"""

_EXPORT_MODEL_STEP_SCRIPT = """\
import FreeCAD as App
import Part

""" + _RESOLVE_OBJECT_SNIPPET + """\
objects = []
for p, n in {target_specs!r}:
    d = App.openDocument(p)
    if n:
        o = resolve_object(d, n)
        if o is None:
            raise RuntimeError("Object not found: " + n)
    else:
        o = next((x for x in d.Objects if not x.InList), d.Objects[-1])
    objects.append(o)

Part.export(objects, {out_path!r})
print("{marker} path=" + {out_path!r})
"""


def export_model(
    target_paths: list[str], format: str, filename: str, target_objects: list[str] | None = None
) -> str:
    """Export one or more previously-created objects together into a single
    named ``.stl`` (3D printing) or ``.step`` (external CAD interchange)
    file under ``_EXPORT_DIR`` — only each requested object is exported,
    not every helper object a Boolean/Sweep/Fillet result's document (or a
    shared multi-object session document) happens to also contain.

    ``target_objects``, when given, must be the same length as
    ``target_paths`` — the object at each index is resolved via
    Multi-Stage Object Resolution (see ``get_bounding_box``'s matching
    note) within that index's document. A ``None``/missing entry (or
    omitting ``target_objects`` entirely) falls back to the legacy "first
    object nothing references" heuristic for that path.
    """
    fmt = (format or "").strip().lower()
    if fmt not in _EXPORT_FORMAT_EXT:
        return _error(f"export_model: unknown format '{format}' — must be stl or step")
    paths = [Path(p) for p in (target_paths or [])]
    if not paths:
        return _error("export_model requires at least one target path")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        return _error(f"export_model: target path(s) not found: {missing}")
    names = list(target_objects or [])

    ext = _EXPORT_FORMAT_EXT[fmt]
    safe_name = _safe_name(filename or "export")
    if is_dry_run_enabled():
        out_path = _EXPORT_DIR / f"{safe_name}.{ext}"
        return _dry_run_result("export_model", format=fmt, path=str(out_path), target_count=len(paths))

    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _EXPORT_DIR / f"{safe_name}.{ext}"
    template = _EXPORT_MODEL_STL_SCRIPT if fmt == "stl" else _EXPORT_MODEL_STEP_SCRIPT
    target_specs = [
        (str(p), (names[i].strip() if i < len(names) and names[i] else None))
        for i, p in enumerate(paths)
    ]
    script = template.format(target_specs=target_specs, out_path=str(out_path), marker=_OK_MARKER)
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"export_model failed: {result['error']}")
    return _ok(format=fmt, path=str(out_path), target_count=len(paths))


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
    "align_objects",
    "apply_boolean",
    "apply_edge_operation",
    "batch_pattern_array",
    "create_assembly_mate",
    "create_box",
    "create_cylinder",
    "create_extruded_polyline",
    "create_pipe",
    "create_pyramid",
    "create_sketch_extrude",
    "create_star_prism",
    "export_mesh_stl",
    "export_model",
    "get_bounding_box",
    "inspect_spatial_properties",
    "modify_existing_document",
    "modify_parameter",
    "detect_freecadcmd",
    "execute_freecad_script",
    "get_freecad_gui_path",
    "get_freecadcmd_path",
    "show_in_freecad_gui",
)

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

import atexit

import glob

import json

import math

import os

import re

import shutil

import subprocess

import sys

import tempfile

import threading

import time

import uuid

from collections.abc import Sequence

from pathlib import Path

from typing import Any, Literal

from dana.plugins.freecad import ir

import psutil

from dana.paths import DANA_WORKSPACE

from dana.security.dry_run import is_dry_run_enabled

from dana.session_context import session_scoped_dir

from dana.tools.geometry_analyzer import query_geometry_properties  # noqa: F401

from dana.tools.urdf_builder import ROOT_LINK_NAME, generate_urdf_assembly  # noqa: F401

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

_lock = threading.Lock()

_cached_cmd_path: str | None = None  # reassigned

_ENV_OVERRIDE = "DANA_FREECADCMD_PATH"

if sys.platform == "win32":
    _COMMON_INSTALL_GLOBS: tuple[str, ...] = (
        r"C:\Program Files\FreeCAD*\bin\FreeCADCmd.exe",
        r"C:\Program Files (x86)\FreeCAD*\bin\FreeCADCmd.exe",
    )
elif sys.platform == "darwin":
    # The official freecad.org macOS build is a plain .app bundle dragged
    # into /Applications — never on PATH by default, unlike a Homebrew
    # install (which lands on PATH on its own and is already caught by the
    # shutil.which() check in detect_freecadcmd, so it never needs this
    # fallback at all).
    _COMMON_INSTALL_GLOBS = (
        "/Applications/FreeCAD*.app/Contents/Resources/bin/FreeCADCmd",
        "/Applications/FreeCAD*.app/Contents/MacOS/FreeCADCmd",
        os.path.expanduser("~/Applications/FreeCAD*.app/Contents/Resources/bin/FreeCADCmd"),
    )
else:
    # Linux desktop installs outside a package manager (the official
    # freecad.org AppImage extracted to a fixed prefix, or a manual /opt
    # install). An apt/dnf-managed install (incl. this project's own
    # HF Space packages.txt `freecad` package — see dana.platform.factory's
    # IS_HF_SPACE branch) already lands freecadcmd on PATH and is caught by
    # shutil.which() above, so it never reaches this fallback either.
    _COMMON_INSTALL_GLOBS = (
        "/opt/freecad*/bin/freecadcmd",
        "/opt/FreeCAD*/bin/freecadcmd",
        "/usr/lib/freecad*/bin/freecadcmd",
        os.path.expanduser("~/.local/opt/freecad*/bin/freecadcmd"),
    )

_DEFAULT_TIMEOUT_S = 60.0

_WINDOW_POLL_TIMEOUT_S = 10.0

_WINDOW_POLL_INTERVAL_S = 0.75

_OK_MARKER = "DANA_FREECAD_OK"

_BBOX_MARKER = f"{_OK_MARKER}_BBOX"

_BBOX_RE = re.compile(re.escape(_BBOX_MARKER) + r" (\[.*?\])")

_PLACEMENT_MARKER = f"{_OK_MARKER}_PLACEMENT"

_PLACEMENT_RE = re.compile(re.escape(_PLACEMENT_MARKER) + r" (\[.*?\])")

_VOLUME_MARKER = f"{_OK_MARKER}_VOLUME"

_VOLUME_RE = re.compile(re.escape(_VOLUME_MARKER) + r" ([0-9eE+\-.]+)")

_SPATIAL_MARKER = f"{_OK_MARKER}_SPATIAL"

_SPATIAL_RE = re.compile(re.escape(_SPATIAL_MARKER) + r" (\[.*?\])")

_COLLISIONS_MARKER = f"{_OK_MARKER}_COLLISIONS"

_COLLISIONS_RE = re.compile(re.escape(_COLLISIONS_MARKER) + r" (\[.*\])")

_TOPOLOGY_MARKER = f"{_OK_MARKER}_TOPOLOGY"

_TOPOLOGY_RE = re.compile(re.escape(_TOPOLOGY_MARKER) + r" (\[.*\])")

_RESOLVED_UV_MARKER = f"{_OK_MARKER}_RESOLVED_UV"

_RESOLVED_UV_RE = re.compile(re.escape(_RESOLVED_UV_MARKER) + r" (\{.*\})")

_CHECKED_MARKER = f"{_OK_MARKER}_CHECKED"

_CHECKED_RE = re.compile(re.escape(_CHECKED_MARKER) + r" ([0-9]+)")

_NAME_MARKER = f"{_OK_MARKER}_NAME"

_NAME_RE = re.compile(re.escape(_NAME_MARKER) + r" (.+)")

_SCRIPT_EXCEPTION_MARKER = f"{_OK_MARKER}_SCRIPT_EXCEPTION"

_FREECAD_INTERNAL_EXCEPTION_BANNER = "Exception while processing file"

_OUTPUT_DIR = DANA_WORKSPACE / "freecad_output"

_EXPORT_DIR = DANA_WORKSPACE / "exports"

_SESSION_DOCUMENT_NAME = "Session_Active"

_KINEMATIC_JOINTS_PROP = "DanaKinematicJoints"

_KINEMATIC_JOINT_TYPES = frozenset({"fixed", "revolute", "continuous", "prismatic"})

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

# _terminate_freecad_gui (show_in_freecad_gui's own always-relaunch fix)
# kills this process outright on every geometry-mutating tool call — from
# FreeCAD's own crash-recovery heuristic's point of view, that's
# indistinguishable from an actual crash, so without this it pops a
# "Recover Document" dialog on nearly every relaunch. This is a
# persistent, on-disk preference (User parameter:BaseApp/Preferences/
# Document), not per-process state, so setting it here also takes effect
# for every later launch, not just this one; unconditional (runs even when
# doc is None) since the dialog itself would otherwise still fire before
# any document-dependent code below ever got a chance to run.
#
# The explicit App.saveParameter() is NOT optional: confirmed live (via
# FreeCADCmd, mimicking psutil.Process.terminate()'s abrupt
# TerminateProcess() on Windows with a hard os._exit()) that SetBool alone
# only changes the value in memory — a process that dies without a clean
# shutdown never flushes it, so the NEXT launch would read the OLD value
# from disk and show the dialog anyway, forever, under this always-kill
# relaunch pattern. saveParameter() forces an immediate, synchronous write
# that survives even an abrupt kill moments later.
try:
    App.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool("SaveAutoRecovery", False)
    App.saveParameter()
except Exception:
    pass

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

    Absolute headless lock: under ``DANA_HEADLESS=true`` this returns
    immediately, before even checking whether ``filepath`` exists — a
    backstop for any caller that reaches this function directly (e.g.
    ``dana.api.cad.open_desktop``) rather than through ``_auto_show``
    (which already has its own ``DANA_HEADLESS`` check), so no code path
    can ever launch the GUI in headless mode.
    """
    if os.getenv("DANA_HEADLESS", "false").lower() == "true":
        # A bare `False`/bool return would violate this function's own
        # `-> str` JSON-string contract — every caller does
        # `json.loads(show_in_freecad_gui(...))` (see _auto_show and
        # dana.api.cad.open_desktop), which raises TypeError on a non-str/
        # bytes argument. `_ok(...)` keeps this parseable exactly like
        # every other return path below, with a flag a caller can check
        # instead of just falling through to a confusing crash.
        return _ok(op="show_in_freecad_gui", path=filepath, spawned=False, skipped="headless_mode")

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

def _session_dir() -> Path:
    """``freecad_output/sessions/<current session_id>/`` — every one-off
    file (``create_pyramid``, ``create_pipe``, ...) AND the shared
    ``Session_Active.FCStd`` document now live under here instead of the
    flat top-level ``freecad_output/``, so two different chat sessions can
    never collide on an object sharing the same name (previously a real
    bug: two sessions each creating a "Box" silently clobbered each other's
    file). See ``dana.session_context`` for how the current session_id is
    resolved.
    """
    return session_scoped_dir(_OUTPUT_DIR)

def _export_dir() -> Path:
    """``exports/sessions/<current session_id>/`` — same reasoning as
    ``_session_dir`` above, for ``export_model``'s named STL/STEP output."""
    return session_scoped_dir(_EXPORT_DIR)

def _output_path(name: str, *, ext: str) -> Path:
    return _session_dir() / f"{_safe_name(name)}.{ext}"

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

def _extract_volume(stdout: str) -> float | None:
    """Parse the ``Shape.Volume`` float the Universal CAD IR template prints
    right after its BoundBox line (Fix #4 — Deterministic Post-Conditions).
    ``float()`` directly (not ``ast.literal_eval``) since this is a bare
    number, not a Python list literal like the bbox/placement lines."""
    m = _VOLUME_RE.search(stdout or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
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

def _extract_collisions(stdout: str) -> list[dict[str, Any]] | None:
    """Parse ``validate_assembly_collisions``'s list-of-dicts result line —
    same ``ast.literal_eval`` safety as ``_extract_bbox``/``_extract_spatial``
    (the printed dicts only ever contain plain strings/floats, so this is
    valid Python literal syntax, not just valid JSON)."""
    m = _COLLISIONS_RE.search(stdout or "")
    if not m:
        return None
    try:
        values = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None
    return values if isinstance(values, list) else None

def _extract_topology(stdout: str) -> list[dict[str, Any]] | None:
    """Parse ``query_topology``'s per-face JSON array — ``json.loads`` (not
    ``ast.literal_eval`` like ``_extract_collisions``) since the script
    itself prints via ``json.dumps``, and a curved face's ``normal`` is
    JSON ``null`` rather than a Python literal ``None``."""
    m = _TOPOLOGY_RE.search(stdout or "")
    if not m:
        return None
    try:
        values = json.loads(m.group(1))
    except (ValueError, TypeError):
        return None
    return values if isinstance(values, list) else None

def _extract_checked_count(stdout: str) -> int | None:
    m = _CHECKED_RE.search(stdout or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def _extract_object_name(stdout: str) -> str | None:
    """Parse the actual FreeCAD-assigned ``Name`` a session-document script
    prints via ``_SESSION_RESULT_PRINT`` — see ``_NAME_MARKER``'s own
    comment for why this can differ from the requested ``name`` argument."""
    m = _NAME_RE.search(stdout or "")
    return m.group(1).strip() if m else None

_FREECAD_SUBPROCESS_HOME: str | None = None

def _freecad_subprocess_home() -> str:
    """A scratch ``HOME``/``XDG_*_HOME`` for the FreeCADCmd subprocess on
    non-Windows hosts — created once per process (cached, like
    ``_cached_cmd_path`` above), not once per call, so this never churns a
    fresh directory (and FreeCAD's own first-run preference-parsing
    overhead) on every tool invocation.

    FreeCADCmd writes its first-run config (``~/.FreeCAD``,
    ``~/.local/share/FreeCAD``, ``~/.config/FreeCAD``) the instant it
    starts. A container's real ``$HOME`` is frequently unset, root-owned,
    or mounted read-only (Hugging Face Spaces, CI) — under
    ``dana.platform.factory``'s ``IS_HF_SPACE`` branch this engine is now
    reachable from exactly that kind of container — and FreeCADCmd crashes
    on startup in that case, before it ever reaches the script this module
    generated. Registered for cleanup at interpreter exit since it's pure
    scratch state, never anything a caller needs to read back.
    """
    global _FREECAD_SUBPROCESS_HOME
    if _FREECAD_SUBPROCESS_HOME is None:
        home = tempfile.mkdtemp(prefix="dana_freecad_home_")
        for sub in ("config", "data", "cache"):
            os.makedirs(os.path.join(home, sub), exist_ok=True)
        atexit.register(shutil.rmtree, home, ignore_errors=True)
        _FREECAD_SUBPROCESS_HOME = home
    return _FREECAD_SUBPROCESS_HOME

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
    here that never passed an ``env`` at all before this parameter existed
    — but only on Windows; every non-Windows caller always gets an explicit
    ``HOME``/``XDG_*_HOME`` override (see ``_freecad_subprocess_home``),
    since that platform is the one where a container's real ``$HOME`` can't
    be trusted to be writable.
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

    if sys.platform == "win32":
        env = {**os.environ, **extra_env} if extra_env else None
    else:
        freecad_home = _freecad_subprocess_home()
        env = {
            **os.environ,
            **(extra_env or {}),
            "HOME": freecad_home,
            "XDG_CONFIG_HOME": os.path.join(freecad_home, "config"),
            "XDG_DATA_HOME": os.path.join(freecad_home, "data"),
            "XDG_CACHE_HOME": os.path.join(freecad_home, "cache"),
        }

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

    ok = (
        proc.returncode == 0
        and (not require_marker or _OK_MARKER in (proc.stdout or ""))
        and _SCRIPT_EXCEPTION_MARKER not in (proc.stdout or "")
        and _FREECAD_INTERNAL_EXCEPTION_BANNER not in (proc.stderr or "")
    )
    fail_msg = proc.stderr.strip() or proc.stdout.strip() or "FreeCADCmd reported failure"
    return {
        "ok": ok,
        "error": None if ok else fail_msg,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "bounding_box": _extract_bbox(proc.stdout) if ok else None,
        "placement": _extract_placement(proc.stdout) if ok else None,
        "resolved_name": _extract_object_name(proc.stdout) if ok else None,
        # Fix #4 — Deterministic Post-Conditions. None for any script that
        # doesn't go through the Universal CAD IR's marker block (every
        # still-legacy per-tool f-string script here never prints a VOLUME
        # line) — callers already treat a None bounding_box/placement the
        # same lenient way.
        "volume": _extract_volume(proc.stdout) if ok else None,
    }

_BBOX_PRINT = (
    "bbox = obj.Shape.BoundBox\n"
    'print("{marker}_BBOX " + str([bbox.XMin, bbox.YMin, bbox.ZMin, '
    "bbox.XMax, bbox.YMax, bbox.ZMax]))\n"
)

_PLACEMENT_SNIPPET = (
    "if {placement!r} != (0.0, 0.0, 0.0):\n"
    "    _px, _py, _pz = {placement!r}\n"
    "    obj.Placement = App.Placement(App.Vector(_px, _py, _pz), App.Rotation())\n"
)

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
    genuinely one-object-per-file callers that remain (e.g. a
    ``get_bounding_box`` query against ``align_freecad_objects``'/
    ``create_freecad_pipe``'s own dedicated output files, which have no
    other object to disambiguate against). ``batch_pattern_array`` always
    supplies a name now (Document Lifecycle Unification moved it onto the
    shared, multi-object session document, where a name is required to get
    the RIGHT object) and never reaches this fallback.
    """
    if target_object:
        return (
            f"{obj_var} = resolve_object({doc_var}, {target_object!r})\n"
            f"if {obj_var} is None:\n"
            f'    raise RuntimeError("Object not found: " + {target_object!r})\n'
        )
    return f"{obj_var} = next((o for o in {doc_var}.Objects if not o.InList), {doc_var}.Objects[-1])\n"

def _session_document_path() -> Path:
    """THIS chat session's own ``Session_Active.FCStd`` path — see
    ``_SESSION_DOCUMENT_NAME``'s module-level comment. Under ``_session_dir()``
    (``freecad_output/sessions/<session_id>/``), not the flat top-level
    ``freecad_output/`` every other artifact used to share — see
    ``_session_dir``'s own docstring for why."""
    return _session_dir() / f"{_SESSION_DOCUMENT_NAME}.FCStd"

_SESSION_OPEN_SNIPPET = """\
import os

_session_path = {session_path!r}
_session_existed = os.path.isfile(_session_path)
if _session_existed:
    doc = App.openDocument(_session_path)
else:
    doc = App.newDocument({session_doc_name!r})
"""

_SESSION_SAVE_SNIPPET = """\
if _session_existed:
    doc.save()
else:
    doc.saveAs(_session_path)
"""

_SESSION_RESULT_PRINT = _BBOX_PRINT + """\
print("{marker}_NAME " + obj.Name)
print("{marker} path=" + _session_path)
"""

_POLYGON_SCRIPT = ("""\
import FreeCAD as App
import Part

""" + _SESSION_OPEN_SNIPPET + """\
pts = [App.Vector(float(x), float(y), 0.0) for x, y in {points!r}]
if pts[0] != pts[-1]:
    pts.append(pts[0])
wire = Part.makePolygon(pts)
face = Part.Face(wire)
solid = face.extrude(App.Vector(0.0, 0.0, {height}))

obj = doc.addObject("Part::Feature", {name!r})
obj.Shape = solid
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
""" + _SESSION_SAVE_SNIPPET + _SESSION_RESULT_PRINT)

_EXTRUDE_SCRIPT = ("""\
import FreeCAD as App
import Part

""" + _SESSION_OPEN_SNIPPET + """\
pts = [App.Vector(float(x), float(y), 0.0) for x, y in {points!r}]
if pts[0] != pts[-1]:
    pts.append(pts[0])
wire = Part.makePolygon(pts)
face = Part.Face(wire)
solid = face.extrude(App.Vector(0.0, 0.0, {height}))

obj = doc.addObject("Part::Feature", {name!r})
obj.Shape = solid
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
""" + _SESSION_SAVE_SNIPPET + _SESSION_RESULT_PRINT)

_PYRAMID_SCRIPT = ("""\
import FreeCAD as App
import Part

""" + _SESSION_OPEN_SNIPPET + """\
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

obj = doc.addObject("Part::Feature", {name!r})
obj.Shape = solid
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
""" + _SESSION_SAVE_SNIPPET + _SESSION_RESULT_PRINT)

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

def _execute_ir_tool(
    tool_id: str, *, doc_mode: Literal["session", "standalone"] = "session", **args: Any
) -> tuple[dict[str, Any], list[dict[str, Any]], Path | None]:
    """THE execution pipeline — the ONE place any IR-migrated tool_id's
    real-time engine.py wrapper hands off to FreeCAD. Replaces
    ``_run_ir_session_step`` (which only handled a single pre-built atomic
    step dict): this takes the tool's OWN native keyword arguments directly,
    looks ``tool_id`` up in the ONE shared registry
    (``ir.get_ir_kind``/``ir.get_composite_ir`` — at most one of the two
    ever returns non-None for a given tool_id, since both read the same
    ``ir._IR_REGISTRY``), unrolls via ``ir.unroll_composite`` if and only if
    it's composite, and renders + executes the result through EXACTLY ONE
    ``ir.render_ir_script``/``_run_freecad_script`` call regardless of step
    count — one atomic kind, a whole composite, doesn't matter, the calling
    wrapper never branches on which.

    There is no legacy-script fallback branch here (nor an
    ``is_ir_migrated`` check) — every caller in this module is ONLY ever
    reachable for a tool_id that IS registered; an unregistered tool_id is a
    programming error (a wrapper calling this before its kind/composite is
    registered), not a user-reachable condition, so it raises ``KeyError``
    rather than silently degrading to a script this function doesn't know
    how to build.

    Deliberately does NOT build the caller's ``_ok(...)``/``_error(...)``
    payload itself — each tool's own result shape differs (dimensions vs.
    operation vs. parameter_name, ...) and genericizing that away isn't
    this helper's job; every wrapper still owns its own
    ``if not result["ok"]: return _error(...)`` / ``return _ok(...)`` tail.
    Returns ``(the raw _run_freecad_script result, the resolved+indexed
    steps list, the session path — None when doc_mode="standalone", since
    there is no single shared path in that mode)`` so a wrapper can still
    read fields off its own last step (e.g. ``steps[-1]["feature_type"]``)
    when building that payload.

    ``doc_mode="standalone"`` is for a kind that owns its own document
    lifecycle entirely (see ``ir.render_ir_script``'s docstring) — the
    caller must pass ``out_path`` among ``args`` so the step itself knows
    where to ``saveAs`` its result; this function derives
    ``final_path_expr`` from the step's own index by the fixed
    ``f"_out_path_{index}"`` convention every "standalone" kind's template
    block commits to. No currently-registered kind uses this mode — the
    "pattern" kind that originally motivated it was migrated onto the
    ordinary ``doc_mode="session"`` path (Document Lifecycle Unification;
    see ``batch_pattern_array``'s own docstring) once "one object, one
    file" turned out to make its own result unreachable, by name, to any
    later session tool call. Kept as real, working infrastructure for a
    future tool that genuinely needs its own isolated document, not dead
    code — nothing about it is unique to "pattern".

    Fix #4 — Deterministic Post-Conditions: on success, the returned dict
    also carries a ``"geometry"`` field — ``{"length", "width", "height",
    "volume"}`` derived from the resulting object's own real ``BoundBox``
    and ``Shape.Volume`` (read by FreeCAD itself, never estimated ahead of
    time) — alongside the existing ``"bounding_box"``/``"resolved_name"``
    fields every wrapper below already reads off this same dict. This is
    what lets a wrapper's own ``_ok(...)`` payload tell the LLM "you asked
    for a horizontal cylinder, but the object it just built is 10mm long
    and 60mm tall" without a screenshot: the model can compare its own
    intended dimensions against ``geometry`` in the SAME turn's tool result.
    """
    atomic = ir.get_ir_kind(tool_id)
    composite = ir.get_composite_ir(tool_id)
    if atomic is not None:
        steps = [atomic.from_args(**args, index=1)]
    elif composite is not None:
        steps = ir.unroll_composite(composite, args)
    else:
        raise KeyError(f"{tool_id!r} is not registered in the Universal CAD IR")

    if doc_mode == "session":
        session_path = _session_document_path()
        script = ir.render_ir_script(
            steps, doc_mode="session", session_path=str(session_path),
            final_var=steps[-1]["var"], marker=_OK_MARKER,
        )
    else:
        session_path = None
        script = ir.render_ir_script(
            steps, doc_mode="standalone", final_var=steps[-1]["var"], marker=_OK_MARKER,
            final_path_expr=f"_out_path_{steps[-1]['index']}",
        )
    result = _run_freecad_script(script)
    bbox = result.get("bounding_box")
    if result["ok"] and bbox:
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        result["geometry"] = {
            "length": x_max - x_min,
            "width": y_max - y_min,
            "height": z_max - z_min,
            "volume": result.get("volume"),
        }
    return result, steps, session_path

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

    Migrated to the Universal CAD IR (``dana.plugins.freecad.ir``) via the
    shared ``_execute_ir_tool`` pipeline — no per-tool branching on whether
    this tool_id is "migrated"; every wrapper in this module goes through
    the same call. ``_BOX_SCRIPT`` (the old bespoke f-string template) is
    correspondingly retired, not kept as a parallel dead path.
    """
    dims = {"length": float(length), "width": float(width), "height": float(height)}
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_box", name=name, type="Part::Box", dimensions=dims, placement=list(placement)
        )
    result, steps, session_path = _execute_ir_tool(
        "create_freecad_box", name=name, length=dims["length"], width=dims["width"], height=dims["height"],
        placement=placement,
    )
    if not result["ok"]:
        return _error(f"create_box failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or name,
        type="Part::Box",
        bounding_box=result.get("bounding_box"),
        geometry=result.get("geometry"),
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
    result, steps, session_path = _execute_ir_tool(
        "create_freecad_cylinder", name=name, radius=dims["radius"], height=dims["height"], placement=placement,
    )
    if not result["ok"]:
        return _error(f"create_cylinder failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or name,
        type="Part::Cylinder",
        bounding_box=result.get("bounding_box"),
        geometry=result.get("geometry"),
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
    session_path = _session_document_path()
    script = _EXTRUDE_SCRIPT.format(
        points=points,
        height=dims["height"],
        name=name,
        placement=placement,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_extruded_polyline failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or name,
        type="Part::Feature",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(session_path),
        gui_shown=_auto_show(session_path),
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
    session_path = _session_document_path()
    script = _PYRAMID_SCRIPT.format(
        length=dims["length"],
        width=dims["width"],
        height=dims["height"],
        name=name,
        placement=placement,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_pyramid failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or name,
        type="Part::Feature",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(session_path),
        gui_shown=_auto_show(session_path),
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

def _regular_polygon_vertices(sides: int, radius: float) -> list[list[float]]:
    """Evenly-spaced vertices of a regular N-gon inscribed in ``radius``,
    first vertex straight up — same convention as ``_star_polygon_vertices``
    (a regular polygon IS that star's ``inner_radius == outer_radius`` case,
    geometrically, but computed directly here rather than routed through
    the star helper with a duplicated radius argument)."""
    return [
        [
            radius * math.cos((2 * math.pi / sides) * i - math.pi / 2),
            radius * math.sin((2 * math.pi / sides) * i - math.pi / 2),
        ]
        for i in range(sides)
    ]

def create_polygon(
    sides: int,
    radius: float,
    height: float,
    name: str = "Polygon",
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Create an extruded regular N-gon (``Part::Feature``) inside the
    shared ``Session_Active.FCStd`` document, translated by ``placement``
    (global X/Y/Z offset in mm). See ``create_box`` for the session-
    document/name-collision notes — identical here.

    Exists so a regular polygon (hexagon, pentagon, octagon, ...) no longer
    needs ``create_star_prism``'s ``inner_radius == outer_radius``
    degenerate-star trick to get built. ``create_star_prism`` (via
    ``create_extruded_polyline``'s ``_EXTRUDE_SCRIPT``) used to route through
    its own standalone-file document instead of the shared session one — see
    ``apply_boolean``'s own docstring, which listed ``create_box``/
    ``create_cylinder``/``insert_standard_part`` as the only session-scoped
    creators — but ``_EXTRUDE_SCRIPT`` was converted to the same
    ``_SESSION_OPEN_SNIPPET``/``_SESSION_SAVE_SNIPPET`` shape this tool
    already used, so both now compose correctly with a later boolean/modify
    call by name. This reuses ``create_box``/``create_cylinder``'s
    session-scoped script shape (``_POLYGON_SCRIPT``) too, just for its own
    simpler N-gon geometry rather than the star's alternating-radius trick.
    """
    if int(sides) < 3:
        return _error("create_polygon requires at least 3 sides")
    dims = {"sides": int(sides), "radius": float(radius), "height": float(height)}
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_polygon", name=name, type="Part::Feature", dimensions=dims, placement=list(placement)
        )
    vertices = _regular_polygon_vertices(dims["sides"], dims["radius"])
    session_path = _session_document_path()
    script = _POLYGON_SCRIPT.format(
        points=vertices,
        height=dims["height"],
        name=name,
        placement=placement,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_polygon failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or name,
        type="Part::Feature",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

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

_SKETCH_EXTRUDE_SCRIPT = ("""\
import FreeCAD as App
import Part

""" + _SESSION_OPEN_SNIPPET + """\
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

obj = doc.addObject("Part::Feature", {name!r})
obj.Shape = solid
""" + _PLACEMENT_SNIPPET + """\
doc.recompute()
""" + _SESSION_SAVE_SNIPPET + _SESSION_RESULT_PRINT)

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
    session_path = _session_document_path()
    script = _SKETCH_EXTRUDE_SCRIPT.format(
        edge_specs=edge_specs,
        normal=_PLANE_NORMAL[plane_u],
        height=dims["height"],
        name=name,
        placement=placement,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_sketch_extrude failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or name,
        type="Part::Feature",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

_ASSEMBLY_RESULT_PRINT = """\
print("{marker}_NAME " + obj.Name)
print("{marker} path=" + _session_path)
"""

_CREATE_ASSEMBLY_SCRIPT = ("""\
import FreeCAD as App

""" + _SESSION_OPEN_SNIPPET + """\
obj = doc.addObject("App::Part", {name!r})
doc.recompute()
""" + _SESSION_SAVE_SNIPPET + _ASSEMBLY_RESULT_PRINT)

def create_assembly(name: str) -> str:
    """Create a real ``App::Part`` assembly container in the shared
    ``Session_Active.FCStd`` document — a plain organizational grouping
    object with no geometry of its own, used to gather independent
    ``PartDesign::Body`` instances (or other top-level objects) into one
    positioned sub-assembly via ``add_parts_to_assembly``/
    ``position_assembly_part``.
    """
    resolved_name = (name or "").strip()
    if not resolved_name:
        return _error("create_assembly requires a non-empty name")
    if is_dry_run_enabled():
        return _dry_run_result("create_assembly", name=resolved_name, type="App::Part")
    session_path = _session_document_path()
    script = _CREATE_ASSEMBLY_SCRIPT.format(
        name=resolved_name,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_assembly failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or resolved_name,
        type="App::Part",
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

_ADD_PARTS_TO_ASSEMBLY_SCRIPT = ("""\
import FreeCAD as App

""" + _RESOLVE_OBJECT_SNIPPET + _SESSION_OPEN_SNIPPET + """\
assembly = resolve_object(doc, {assembly_name!r})
if assembly is None:
    raise RuntimeError("Object not found: " + {assembly_name!r})
_part_names = {part_names!r}
for _p_name in _part_names:
    _p = resolve_object(doc, _p_name)
    if _p is None:
        raise RuntimeError("Object not found: " + _p_name)
    if _p not in assembly.Group:
        assembly.addObject(_p)
doc.recompute()

obj = assembly
""" + _SESSION_SAVE_SNIPPET + _ASSEMBLY_RESULT_PRINT)

def add_parts_to_assembly(assembly_name: str, part_names: Sequence[str]) -> str:
    """Move the named parts (typically ``PartDesign::Body`` instances left
    behind by ``create_pad``/``create_pocket``/``create_sweep``/
    ``create_loft``) into a previously-created ``create_assembly``
    container, resolved by NAME — same by-name story as ``apply_boolean``/
    ``modify_parameter``. Uses the real ``App::Part.addObject`` API (the
    same grouping mechanism ``PartDesign::Body.addObject`` already uses for
    sketches — see ``_PARTDESIGN_BODY_SNIPPET``), so a part already in the
    assembly is silently left alone rather than re-added.
    """
    assembly = (assembly_name or "").strip()
    if not assembly:
        return _error("add_parts_to_assembly requires assembly_name")
    names = [str(p).strip() for p in part_names if str(p).strip()]
    if not names:
        return _error("add_parts_to_assembly requires a non-empty part_names list")

    dims = {"part_names": names}
    if is_dry_run_enabled():
        return _dry_run_result("add_parts_to_assembly", name=assembly, type="App::Part", dimensions=dims)
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error(
            "add_parts_to_assembly: no session document yet — create an assembly with "
            "create_freecad_assembly first"
        )
    script = _ADD_PARTS_TO_ASSEMBLY_SCRIPT.format(
        assembly_name=assembly,
        part_names=names,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"add_parts_to_assembly failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or assembly,
        type="App::Part",
        dimensions=dims,
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

_POSITION_ASSEMBLY_PART_SCRIPT = ("""\
import FreeCAD as App

""" + _RESOLVE_OBJECT_SNIPPET + _SESSION_OPEN_SNIPPET + """\
{lookup}if getattr(obj, "DanaAnchored", False):
    raise RuntimeError(
        "'" + obj.Name + "' is anchored (anchor_assembly_root) and cannot be moved/rotated by "
        "position_assembly_part -- it is this assembly's fixed reference frame."
    )
if getattr(obj, "DanaConstrained", False):
    raise RuntimeError(
        "'" + obj.Name + "' is locked by an active assembly constraint (apply_assembly_constraint) "
        "and cannot be moved/rotated by position_assembly_part -- re-call apply_assembly_constraint "
        "against the correct face/edge instead of overriding its Placement directly."
    )
obj.Placement = App.Placement(App.Vector({x}, {y}, {z}), App.Rotation({yaw}, {pitch}, {roll}))
doc.recompute()
""" + _SESSION_SAVE_SNIPPET + _ASSEMBLY_RESULT_PRINT)

def position_assembly_part(
    part_name: str,
    placement_x: float = 0.0,
    placement_y: float = 0.0,
    placement_z: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> str:
    """Move and/or orient a previously-created part (typically a
    ``PartDesign::Body`` inside a ``create_assembly`` container, but works
    on any named object) by replacing its whole ``Placement`` — resolved by
    NAME, same by-name story as ``apply_boolean``/``modify_parameter``.
    ``(placement_x, placement_y, placement_z)`` is the new position in mm;
    ``(yaw, pitch, roll)`` is a fresh Euler rotation in DEGREES (FreeCAD's
    own ``App.Rotation(yaw, pitch, roll)`` convention — see
    ``modify_parameter``'s matching 6-element-vector docstring for the same
    confirmed-live convention), REPLACING any prior rotation rather than
    composing with it.

    Refuses outright (never silently no-ops) if ``part_name`` was anchored via
    ``anchor_assembly_root``, or if it is currently locked by an active
    ``apply_assembly_constraint`` mate (``DanaConstrained``) — re-call
    ``apply_assembly_constraint`` instead of overriding a mated part's
    ``Placement`` directly.
    """
    target = (part_name or "").strip()
    if not target:
        return _error("position_assembly_part requires part_name")
    try:
        x, y, z = float(placement_x), float(placement_y), float(placement_z)
        yaw_f, pitch_f, roll_f = float(yaw), float(pitch), float(roll)
    except (TypeError, ValueError):
        return _error("position_assembly_part: placement_x/y/z and yaw/pitch/roll must all be numbers")

    dims = {"placement": [x, y, z], "yaw": yaw_f, "pitch": pitch_f, "roll": roll_f}
    if is_dry_run_enabled():
        return _dry_run_result("position_assembly_part", name=target, dimensions=dims)
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error("position_assembly_part: no session document yet — create a part first")
    script = _POSITION_ASSEMBLY_PART_SCRIPT.format(
        lookup=_object_lookup_snippet(target_object=target),
        x=x,
        y=y,
        z=z,
        yaw=yaw_f,
        pitch=pitch_f,
        roll=roll_f,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"position_assembly_part failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or target,
        dimensions=dims,
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

_ASSEMBLY_CONSTRAINT_ELEMENT_HELPER = """\
def _resolve_face_by_world_normal(obj, normal, obj_label):
    # Semantic Normal Targeting: an alternative to a literal 'FaceN' index
    # reference for part1_element/part2_element -- a [nx, ny, nz] world-space
    # direction, resolved here to whichever of obj's own PLANAR faces has a
    # world-space normal aligned with it (dot >= 0.9, same threshold
    # convention _resolve_world_fractions_uv/_face_axis_world_hints already
    # use elsewhere in this module). Exists because an LLM asked to re-derive
    # "the opposite lateral face" from a numbered Face index across several
    # ReAct iterations was confirmed live to lose that mapping and mate
    # wheels to two DIFFERENT, non-parallel faces instead of a matching
    # opposite pair -- a world-space normal is a fixed target independent of
    # any particular OpenCASCADE face ordering, so it can't drift between
    # calls the way a remembered index can. Only ever matches PLANAR faces
    # (a curved face's normal isn't a single constant direction, same
    # reasoning as _require_planar_face's own restriction below) -- a
    # cylindrical part's lateral surface is never a valid match, use an
    # explicit 'FaceN' reference (or 'Concentric') for that instead.
    if len(normal) != 3:
        raise RuntimeError(
            "world-space normal reference on '" + obj_label + "' must have exactly 3 numbers "
            "[nx, ny, nz], got " + repr(list(normal))
        )
    vec = App.Vector(float(normal[0]), float(normal[1]), float(normal[2]))
    if vec.Length < 1e-9:
        raise RuntimeError(
            "world-space normal reference on '" + obj_label + "' is a zero-length vector -- "
            "must point in some direction, e.g. [0, -1, 0]."
        )
    vec.normalize()
    shape = getattr(obj, "Shape", None)
    if shape is None or shape.isNull():
        raise RuntimeError("'" + obj_label + "' has no usable geometry (empty Shape).")
    best_face, best_dot, checked = None, -2.0, 0
    for f in shape.Faces:
        if f.Surface.TypeId != "Part::GeomPlane":
            continue
        checked += 1
        pt = f.CenterOfMass
        u, v = f.Surface.parameter(pt)
        n = f.normalAt(u, v)
        dot = n.x * vec.x + n.y * vec.y + n.z * vec.z
        if dot > best_dot:
            best_face, best_dot = f, dot
    if best_face is None or best_dot < 0.9:
        raise RuntimeError(
            "no planar face on '" + obj_label + "' has a world-space normal aligned with "
            + repr(list(normal)) + " (best match dot=" + str(round(best_dot, 3)) + " across "
            + str(checked) + " planar face(s), need >= 0.9) -- call query_topology to inspect "
            "this part's real face normals, or pass an explicit 'FaceN' reference instead."
        )
    return best_face


def _resolve_constraint_element(obj, element_name, obj_label):
    if isinstance(element_name, (list, tuple)):
        return _resolve_face_by_world_normal(obj, element_name, obj_label)
    if element_name.startswith("Face"):
        kind, plural, prefix_len = "Faces", "faces", 4
    elif element_name.startswith("Edge"):
        kind, plural, prefix_len = "Edges", "edges", 4
    else:
        raise RuntimeError(
            "'" + element_name + "' is not a recognized element reference on '" + obj_label
            + "' -- must be e.g. 'Face1' or 'Edge3', or a [nx, ny, nz] world-space normal vector."
        )
    try:
        index = int(element_name[prefix_len:])
    except ValueError:
        raise RuntimeError(
            "'" + element_name + "' is not a valid reference on '" + obj_label
            + "' -- expected a number after 'Face'/'Edge', e.g. 'Face1'."
        )
    shape = getattr(obj, "Shape", None)
    if shape is None or shape.isNull():
        raise RuntimeError("'" + obj_label + "' has no usable geometry (empty Shape).")
    items = getattr(shape, kind)
    if index < 1 or index > len(items):
        available = ", ".join(kind[:-1] + str(i + 1) for i in range(len(items)))
        raise RuntimeError(
            "'" + element_name + "' does not exist on '" + obj_label + "'. Available " + plural
            + " are: " + (available if available else "none") + "."
        )
    return items[index - 1]


def _element_reference(element):
    # Returns (point, direction, has_axis, axis_point, axis_dir). direction
    # is the face normal (any surface type, via Surface.parameter) or edge
    # direction (straight Line) / axis (Circle); has_axis is True ONLY for
    # a cylindrical face or circular edge -- the only elements Concentric
    # can legally use. Confirmed live (freecadcmd probe) that
    # Surface.parameter + normalAt, Surface.Axis/Center (cylinder), and
    # Curve.Axis/Center (circle) are all real, headlessly-usable APIs on
    # this exact FreeCAD build -- not assumed from general knowledge.
    if element.ShapeType == "Face":
        point = element.CenterOfMass
        u, v = element.Surface.parameter(point)
        direction = element.normalAt(u, v)
        if element.Surface.TypeId == "Part::GeomCylinder":
            return point, direction, True, element.Surface.Center, element.Surface.Axis
        return point, direction, False, None, None
    if element.ShapeType == "Edge":
        point = element.CenterOfMass
        curve_type = element.Curve.TypeId
        if curve_type == "Part::GeomLine":
            v0, v1 = element.Vertexes[0].Point, element.Vertexes[-1].Point
            direction = v1 - v0
            if direction.Length < 1e-9:
                raise RuntimeError("edge has zero length -- cannot determine a direction")
            direction.normalize()
            return point, direction, False, None, None
        if curve_type == "Part::GeomCircle":
            return point, element.Curve.Axis, True, element.Curve.Center, element.Curve.Axis
        raise RuntimeError(
            "edge curve type '" + curve_type + "' is not supported -- only straight edges and "
            "circles/arcs are"
        )
    raise RuntimeError("unsupported element shape type: " + element.ShapeType)


def _require_planar_face(obj, element_name, obj_label, el):
    if el.ShapeType != "Face" or el.Surface.TypeId == "Part::GeomPlane":
        return
    planar = [
        "Face" + str(i + 1)
        for i, f in enumerate(obj.Shape.Faces)
        if f.Surface.TypeId == "Part::GeomPlane"
    ]
    raise RuntimeError(
        "'" + element_name + "' on '" + obj_label + "' is a curved face (" + el.Surface.TypeId
        + ") -- its normal is only well-defined at one arbitrary point on the surface, not a "
        "single orientation for the whole face, so it cannot be used here. Use 'Concentric' "
        "instead for a cylindrical face, or pick one of this part's FLAT faces: "
        + (", ".join(planar) if planar else "none available") + "."
    )


def _face_alignment_delta(face, element_name, obj_label, uv_tensor, moving_part, moving_part_label):
    # Continuous Parametric UV Placement (replaces the old discrete
    # semantic_alignment quadrant system): shifts a Coincident/Distance
    # target from a planar face's CENTER to any point on it, expressed as a
    # normalized (u, v) tensor in [0, 1] x [0, 1] -- never a raw coordinate
    # -- or, worse, a world-space -- value) so the caller can't hallucinate
    # a point that falls outside the face's own bounds. Resolved against
    # that face's OWN trimmed (u, v) parameter range -- verified live
    # (freecadcmd probe against this exact FreeCAD build) that
    # Face.ParameterRange returns the FACE's actual bounded rectangle, not
    # the underlying infinite plane, and Face.valueAt(u, v) lands exactly
    # on that face's real surface.
    #
    # An Edge reference has no 2D parametric footprint to place a point
    # within (only a single reference point/direction, see
    # _element_reference) -- uv_tensor is meaningless there, so this still
    # short-circuits to a zero delta for the DEFAULT (0.5, 0.5) center
    # value (the caller didn't ask for anything beyond the plain point
    # match _element_reference already computed), but raises outright for
    # any other explicit uv_tensor against an Edge -- same "reject clearly,
    # never silently ignore" convention the old semantic_alignment used.
    if face.ShapeType != "Face":
        if tuple(uv_tensor) == (0.5, 0.5):
            return App.Vector(0, 0, 0)
        raise RuntimeError("uv_tensor requires part1_element to be a Face, not an Edge")
    #
    # Geometric Fit Guard: the inward padding margin is NOT an arbitrary
    # percentage of the face's own extent (a fixed 10% overlapped moving
    # parts whose footprint happened to be larger than that slice of a
    # small face) -- it's moving_part's own real physical half-extent
    # (max(BoundBox.XLength, YLength, ZLength) / 2), so the padding always
    # matches the actual part being placed, on any size face. If the face
    # is too small to keep that padding on BOTH axes (parametric space
    # available < 2x the part's own half-extent, i.e. the part physically
    # cannot fit inside this face at all, corner or center), this rejects
    # the call outright rather than silently overlapping it with a
    # sibling or hanging part off the face's edge. Runs UNCONDITIONALLY for
    # any Face target now, including the default center point -- unlike
    # the retired semantic_alignment="center" token, which bypassed this
    # guard entirely.
    u_min, u_max, v_min, v_max = face.ParameterRange
    u_available = u_max - u_min
    v_available = v_max - v_min
    _moving_bbox = moving_part.Shape.BoundBox
    padding = max(_moving_bbox.XLength, _moving_bbox.YLength, _moving_bbox.ZLength) / 2.0
    if u_available < 2 * padding or v_available < 2 * padding:
        raise RuntimeError(
            "Face '" + element_name + "' on '" + obj_label + "' is physically too small to "
            "accommodate '" + moving_part_label + "'s dimensions. You must resize the parts or "
            "choose a larger face."
        )
    u_lo, u_hi = u_min + padding, u_max - padding
    v_lo, v_hi = v_min + padding, v_max - padding
    _u_frac, _v_frac = uv_tensor
    u = u_lo + _u_frac * (u_hi - u_lo)
    v = v_lo + _v_frac * (v_hi - v_lo)
    target_point = face.valueAt(u, v)
    return target_point - face.CenterOfMass


_CARDINAL_WORLD_AXES = (
    ("+X", (1.0, 0.0, 0.0)), ("-X", (-1.0, 0.0, 0.0)),
    ("+Y", (0.0, 1.0, 0.0)), ("-Y", (0.0, -1.0, 0.0)),
    ("+Z", (0.0, 0.0, 1.0)), ("-Z", (0.0, 0.0, -1.0)),
)


def _face_axis_world_hints(face):
    # Always-on version of world_fractions's own alignment check, surfaced
    # in EVERY Coincident/Distance uv_tensor result (not just when
    # world_fractions is explicitly requested) -- so a caller who guessed uv_tensor wrong
    # (assuming u is world X when it's actually world Z on THIS face) sees
    # the real mapping immediately in that same call's own result, in time
    # to correct the next sibling call, instead of only finding out via a
    # separate query_topology call it may never think to make. Returns
    # (u_hint, v_hint), each a cardinal label like "+X" or None if that
    # axis isn't planar/doesn't align closely (>= 0.9) with any single
    # cardinal world direction.
    if face.ShapeType != "Face" or face.Surface.TypeId != "Part::GeomPlane":
        return (None, None)
    u_min, u_max, v_min, v_max = face.ParameterRange
    origin = face.valueAt(u_min, v_min)
    u_delta = face.valueAt(u_max, v_min) - origin
    v_delta = face.valueAt(u_min, v_max) - origin
    u_delta.normalize()
    v_delta.normalize()

    def _best_cardinal(delta):
        best_label, best_dot = None, 0.9
        for label, axis in _CARDINAL_WORLD_AXES:
            dot = delta.x * axis[0] + delta.y * axis[1] + delta.z * axis[2]
            if dot > best_dot:
                best_label, best_dot = label, dot
        return best_label

    return (_best_cardinal(u_delta), _best_cardinal(v_delta))


_CARDINAL_AXIS_VECTORS = {{"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}}


def _resolve_world_fractions_uv(face, world_fractions):
    # world_fractions lets a caller say "place this at fraction F along
    # world axis A [, and fraction G along world axis B]" instead of
    # guessing whether a given world axis is this face's u or v --
    # OpenCASCADE's (u, v) parameterization for a given planar face is
    # implementation-defined (verified live: two faces with the same
    # real-world orientation on the same box can map u/v to different
    # world axes depending on the underlying Geom_Plane's own Position/
    # rotation), so a fixed "u = length, v = height" assumption is unsound
    # in general -- this resolves it per-call from the face's own real
    # geometry instead, the same u_direction_vector/v_direction_vector
    # computation query_topology exposes for a caller to do this by hand.
    # A planar face has exactly 2 degrees of freedom (u, v), so up to two
    # DIFFERENT world axes can be specified -- each is matched to whichever
    # parametric axis it actually aligns with, so both u AND v can be
    # driven by world-space intent (e.g. corner placement expressed via two
    # entries, X and Y both set), not just one at a time.
    if world_fractions is None:
        return None
    if face.ShapeType != "Face":
        raise RuntimeError("world_fractions requires part1_element to be a Face, not an Edge")
    if face.Surface.TypeId != "Part::GeomPlane":
        raise RuntimeError(
            "world_fractions is only supported for a planar face (got '" + face.Surface.TypeId
            + "') -- a curved face's (u, v) axes have no single constant world-space direction. "
            "Use 'Concentric' for a cylindrical face, or pick a flat face."
        )
    u_min, u_max, v_min, v_max = face.ParameterRange
    origin = face.valueAt(u_min, v_min)
    u_delta = face.valueAt(u_max, v_min) - origin
    v_delta = face.valueAt(u_min, v_max) - origin
    u_delta.normalize()
    v_delta.normalize()

    u_frac, v_frac = 0.5, 0.5
    u_claimed_by, v_claimed_by = None, None
    for axis_name, frac in world_fractions.items():
        axis_vec = _CARDINAL_AXIS_VECTORS[axis_name]
        dot_u = u_delta.x * axis_vec[0] + u_delta.y * axis_vec[1] + u_delta.z * axis_vec[2]
        dot_v = v_delta.x * axis_vec[0] + v_delta.y * axis_vec[1] + v_delta.z * axis_vec[2]
        if max(abs(dot_u), abs(dot_v)) < 0.9:
            raise RuntimeError(
                "world_fractions: world axis '" + axis_name + "' does not align with either "
                "parametric axis of this face (best alignment: "
                + str(round(max(abs(dot_u), abs(dot_v)), 3)) + ", need >= 0.9) -- this face is not "
                "oriented so that world axis maps cleanly to u or v. Call query_topology on this "
                "part first to inspect its real u_direction_vector/v_direction_vector."
            )
        if abs(dot_u) >= abs(dot_v):
            if u_claimed_by is not None:
                raise RuntimeError(
                    "world_fractions: both '" + u_claimed_by + "' and '" + axis_name
                    + "' resolve to this face's SAME parametric axis (u) -- they conflict. Pick "
                    "world axes that map to this face's two DIFFERENT parametric axes (see "
                    "query_topology's u_direction_vector/v_direction_vector)."
                )
            u_claimed_by = axis_name
            u_frac = frac if dot_u >= 0 else (1.0 - frac)
        else:
            if v_claimed_by is not None:
                raise RuntimeError(
                    "world_fractions: both '" + v_claimed_by + "' and '" + axis_name
                    + "' resolve to this face's SAME parametric axis (v) -- they conflict. Pick "
                    "world axes that map to this face's two DIFFERENT parametric axes (see "
                    "query_topology's u_direction_vector/v_direction_vector)."
                )
            v_claimed_by = axis_name
            v_frac = frac if dot_v >= 0 else (1.0 - frac)
    return (u_frac, v_frac)
"""

_APPLY_ASSEMBLY_CONSTRAINT_SCRIPT = ("""\
import FreeCAD as App
import json
import math

""" + _RESOLVE_OBJECT_SNIPPET + _SESSION_OPEN_SNIPPET + _ASSEMBLY_CONSTRAINT_ELEMENT_HELPER + """\
_assembly = resolve_object(doc, {assembly_name!r})
if _assembly is None:
    raise RuntimeError("Object not found: " + {assembly_name!r})
part1 = resolve_object(doc, {part1_name!r})
if part1 is None:
    raise RuntimeError("Object not found: " + {part1_name!r})
part2 = resolve_object(doc, {part2_name!r})
if part2 is None:
    raise RuntimeError("Object not found: " + {part2_name!r})
_assembly_members = getattr(_assembly, "Group", [])
for _p, _p_name in ((part1, {part1_name!r}), (part2, {part2_name!r})):
    if _p not in _assembly_members:
        raise RuntimeError(
            "'" + _p_name + "' is not in assembly '" + {assembly_name!r}
            + "' -- call add_parts_to_assembly first."
        )
if getattr(part2, "DanaAnchored", False):
    raise RuntimeError(
        "'" + {part2_name!r} + "' is anchored (anchor_assembly_root) and cannot be moved/rotated "
        "as part2 -- it is this assembly's fixed reference frame. Pass it as part1 instead (it "
        "never moves), or anchor a different part."
    )

el1 = _resolve_constraint_element(part1, {part1_element!r}, {part1_name!r})
el2 = _resolve_constraint_element(part2, {part2_element!r}, {part2_name!r})
# Everything below this point uses these display-only LABEL strings (never
# the raw part1_element/part2_element above, which may be a [nx, ny, nz]
# list when Semantic Normal Targeting was used) for error-message text --
# see apply_assembly_constraint's own _element_label helper for why: a raw
# list concatenated with "+" against a str literal raises TypeError, and
# these error paths ARE reachable for a normal-vector reference (e.g. the
# Geometric Fit Guard/Collision Guard below), unlike _resolve_constraint_element
# itself which needs the real (str or list) type to dispatch correctly.
part1_element_label = {part1_element_label!r}
part2_element_label = {part2_element_label!r}
point1, dir1, has_axis1, axis_point1, axis_dir1 = _element_reference(el1)
point2, dir2, has_axis2, axis_point2, axis_dir2 = _element_reference(el2)

constraint_type = {constraint_type!r}
offset = {offset!r}
uv_tensor = {uv_tensor!r}
_world_fractions = {world_fractions!r}


def _rotate_part2_about(pivot, rotation):
    xf = (
        App.Placement(pivot, App.Rotation())
        * App.Placement(App.Vector(0, 0, 0), rotation)
        * App.Placement(pivot, App.Rotation()).inverse()
    )
    part2.Placement = xf * part2.Placement


def _translate_part2(delta):
    part2.Placement = App.Placement(delta, App.Rotation()) * part2.Placement


if constraint_type == "Concentric":
    if not has_axis1 or not has_axis2:
        bad_element = part1_element_label if not has_axis1 else part2_element_label
        bad_owner = {part1_name!r} if not has_axis1 else {part2_name!r}
        raise RuntimeError(
            "'Concentric' requires both elements to be circular/cylindrical -- '" + bad_element
            + "' on '" + bad_owner + "' is not."
        )
    _rotate_part2_about(axis_point2, App.Rotation(axis_dir2, axis_dir1))
    _translate_part2(axis_point1 - axis_point2)

elif constraint_type == "Parallel":
    _require_planar_face(part1, part1_element_label, {part1_name!r}, el1)
    _require_planar_face(part2, part2_element_label, {part2_name!r}, el2)
    _rotate_part2_about(point2, App.Rotation(dir2, dir1))

elif constraint_type == "Perpendicular":
    _require_planar_face(part1, part1_element_label, {part1_name!r}, el1)
    _require_planar_face(part2, part2_element_label, {part2_name!r}, el2)
    current_angle = dir1.getAngle(dir2)
    axis = dir1.cross(dir2)
    if axis.Length < 1e-9:
        # dir1/dir2 already parallel/antiparallel -- any axis perpendicular
        # to dir1 works; pick one deterministically rather than fail.
        arbitrary = App.Vector(1, 0, 0) if abs(dir1.x) < 0.9 else App.Vector(0, 1, 0)
        axis = dir1.cross(arbitrary)
    axis.normalize()
    delta_angle_deg = math.degrees((math.pi / 2.0) - current_angle)
    _rotate_part2_about(point2, App.Rotation(axis, delta_angle_deg))

elif constraint_type in ("Coincident", "Distance"):
    if el1.ShapeType == "Face" and el2.ShapeType == "Face":
        # Standard face-mating convention: normals point at each other, so
        # part2's normal must end up ANTI-parallel to part1's.
        _require_planar_face(part1, part1_element_label, {part1_name!r}, el1)
        _require_planar_face(part2, part2_element_label, {part2_name!r}, el2)
        _rotate_part2_about(point2, App.Rotation(dir2, dir1.negative()))
    if constraint_type == "Distance":
        _require_planar_face(part1, part1_element_label, {part1_name!r}, el1)
    # Any other element-type combination (edge-edge, face-edge) skips the
    # rotation step entirely -- there is no unambiguous "correct" relative
    # orientation to infer, so only the reference points are aligned.

    _axis_resolved_uv = _resolve_world_fractions_uv(el1, _world_fractions)
    if _axis_resolved_uv is not None:
        uv_tensor = _axis_resolved_uv
    _u_hint, _v_hint = _face_axis_world_hints(el1)
    print("{marker}_RESOLVED_UV " + json.dumps(
        {{"uv_tensor": list(uv_tensor), "u_world_axis": _u_hint, "v_world_axis": _v_hint}}
    ))

    target_point = (point1 + dir1 * offset) if constraint_type == "Distance" else point1
    target_point = target_point + _face_alignment_delta(
        el1, part1_element_label, {part1_name!r}, uv_tensor, part2, {part2_name!r}
    )

    _translate_part2(target_point - point2)

    # Collision Guard -- the ONLY double-booking check now (Continuous
    # Parametric UV Placement retires the old Semantic Occupancy Registry:
    # a discrete 5-token enum could be exact-matched as a dict key,
    # ("top_left" == "top_left"), but a continuous uv_tensor float pair
    # can't be, so there is no equivalent pre-mutation slot-claim check
    # possible anymore -- this purely geometric, post-mutation check is the
    # only defense against two parts landing on top of each other.
    # Compared AFTER the translation so both sides are the same quantity
    # (each part's own Placement.Base) -- comparing a sibling's
    # Placement.Base against target_point (a raw point on part1's face, a
    # different reference frame) let mismatched wheels slip through
    # undetected.
    for _sibling in _assembly_members:
        if _sibling is part1 or _sibling is part2:
            continue
        _sib_pl = getattr(_sibling, "Placement", None)
        if _sib_pl is not None and (_sib_pl.Base - part2.Placement.Base).Length < 1e-6:
            raise RuntimeError(
                "'" + {part2_name!r} + "' would land at the exact same point as '"
                + _sibling.Name + "' -- both mated to '" + part1_element_label + "' on '"
                + {part1_name!r} + "'. Pass a distinct uv_tensor (e.g. [0.0, 0.0] vs "
                "[1.0, 1.0]) so they land at different points on that face."
            )

else:
    raise RuntimeError("unknown constraint_type: " + constraint_type)

if not hasattr(part2, "DanaConstrained"):
    part2.addProperty(
        "App::PropertyBool", "DanaConstrained", "Dana",
        "Set by apply_assembly_constraint -- this part's Placement was just set by a "
        "face/edge mate, so position_assembly_part, modify_freecad_parameter's Placement "
        "branch, and align_freecad_objects all refuse to move or rotate it directly. "
        "Re-call apply_assembly_constraint (a different element/uv_tensor/constraint_type "
        "is fine) to reposition it instead -- that is the only path that keeps this flag "
        "set and the part correctly mated."
    )
part2.DanaConstrained = True
doc.recompute()

obj = part2
""" + _SESSION_SAVE_SNIPPET + _ASSEMBLY_RESULT_PRINT)

_ASSEMBLY_CONSTRAINT_TYPES = frozenset({"Coincident", "Concentric", "Parallel", "Distance", "Perpendicular"})


def _validate_constraint_element(value: Any, param_name: str) -> str | list[float]:
    """``part1_element``/``part2_element`` accepts either a literal
    ``"Face1"``/``"Edge3"`` reference (returned as-is) or a Semantic Normal
    Target — a ``[nx, ny, nz]`` world-space direction (returned as a plain
    ``list[float]``) — resolved inside the generated FreeCAD script
    (``_resolve_face_by_world_normal``) to whichever of that object's own
    PLANAR faces has a world-space normal aligned with it, instead of
    requiring the caller to already know that face's own OpenCASCADE index.

    Added because a numbered ``FaceN`` reference was confirmed live (rover-
    assembly stress test) to NOT survive multiple ReAct iterations reliably:
    an LLM asked to mate a second pair of wheels to "the opposite lateral
    face" from a first pair, several tool calls later, mated them to a
    DIFFERENT, non-parallel face instead (``Face2`` vs ``Face4`` on the same
    box) — a plain indexing mistake with no geometric error to catch it,
    since each individual ``apply_assembly_constraint`` call was itself
    perfectly valid. A world-space normal has no such drift: ``[0, -1, 0]``
    means the same real-world direction on every call, independent of
    which numbered face happens to be there.
    """
    def _is_numeric_triple(seq: Any) -> bool:
        return (
            isinstance(seq, (list, tuple))
            and len(seq) == 3
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in seq)
        )

    if isinstance(value, str):
        text = value.strip()
        # Every tool argument is declared "type": "string" in tools.json
        # (dana.tools.schema has no union-type/anyOf support today, and
        # adding one would touch the pydantic model builder, the OpenAI
        # schema generator, AND schema_minify.py for every tool, not just
        # this one) -- so a Semantic Normal Target arrives as a bracketed
        # JSON array STRING, e.g. '[0, -1, 0]', same as every other caller
        # sends a plain 'Face1' string. Only a leading '[' attempts the
        # JSON parse -- a real "Face1"/"Edge3" reference never starts with
        # one, so this can't misfire on the existing common case.
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if _is_numeric_triple(parsed):
                return [float(x) for x in parsed]
            raise ValueError(
                f"{param_name} looks like a world-space normal vector but isn't valid JSON for "
                f"exactly 3 numbers, e.g. '[0, -1, 0]' — got {value!r}"
            )
        return text
    if _is_numeric_triple(value):
        return [float(x) for x in value]
    raise ValueError(
        f"{param_name} must be a 'FaceN'/'EdgeN' string reference or a '[nx, ny, nz]' JSON "
        f"world-space normal vector string (3 numbers), got {value!r}"
    )


def _element_label(elem: str | list[float]) -> str:
    """Always-a-string display form of a validated element reference, for
    every place ``apply_assembly_constraint`` needs to embed it in an error
    message or the ``dimensions`` result payload — the raw value itself may
    be a ``list`` (see ``_validate_constraint_element``), and Python can't
    ``+``-concatenate a ``list`` onto a ``str``.
    """
    return f"normal{elem!r}" if isinstance(elem, list) else elem


def apply_assembly_constraint(
    assembly_name: str,
    part1_name: str,
    part1_element: str | Sequence[float],
    part2_name: str,
    part2_element: str | Sequence[float],
    constraint_type: str,
    offset: float = 0.0,
    uv_tensor: Sequence[float] | None = None,
    world_fractions: dict[str, float] | None = None,
) -> str:
    """Move ``part2_name`` (by replacing its whole ``Placement``, same
    "REPLACES rather than composes" convention as ``position_assembly_part``)
    so that ``part2_element`` satisfies ``constraint_type`` against
    ``part1_name``'s ``part1_element`` — both elements resolved and
    VALIDATED against their object's real ``Shape.Faces``/``Shape.Edges``
    (a reference to a face/edge that doesn't exist is rejected with the
    exact available list, never silently ignored), both parts required to
    already be members of ``assembly_name`` (``add_parts_to_assembly``
    first).

    ``part1_element``/``part2_element`` each accept EITHER a literal
    ``"Face1"``/``"Edge3"`` OpenCASCADE index reference, OR a Semantic Normal
    Target — a bracketed JSON ``"[nx, ny, nz]"`` world-space direction
    STRING (e.g. ``"[0, -1, 0]"`` — still a plain ``str`` argument, same as
    every ``"FaceN"`` reference; only a leading ``[`` is treated as JSON),
    resolved to whichever of that object's own PLANAR faces has a
    world-space normal aligned with it (dot product >= 0.9 against every
    candidate face's real ``normalAt``, same threshold convention
    ``world_fractions`` below already uses) — REJECTED outright, with the
    best alignment score and how many planar faces were even checked, if no
    face matches closely enough. Prefer this over a bare index whenever the
    caller's actual intent is a world-space direction rather than "whichever
    face happened to be numbered N" — confirmed live that an LLM re-deriving
    "the opposite lateral face" from a remembered index across several
    ReAct iterations can mate a second pair of parts to the WRONG (non-
    parallel) face with no geometric error to catch it, since each
    individual call is itself perfectly valid; a world-space normal targets
    the same real direction on every call regardless of numbering. Only ever
    matches a PLANAR face (same restriction as ``uv_tensor``/
    ``world_fractions`` below) — use an explicit ``"FaceN"`` reference (or
    ``"Concentric"``) for a cylindrical face.

    This computes a ONE-SHOT geometric Placement from each element's real
    BRep geometry — a genuine upgrade over ``position_assembly_part``'s
    typed-in XYZ guessing — it is NOT a live FreeCAD constraint object (see
    this function's own module-level comment for exactly why the native
    Assembly workbench's real joints aren't reachable from this headless
    FreeCADCmd execution model, confirmed live against this install, not
    assumed). If the referenced geometry changes later, re-call this to
    re-align; nothing here auto-re-solves.

    A successful call marks ``part2_name`` with a persistent ``DanaConstrained``
    property. From then on, ``position_assembly_part``, ``modify_freecad_parameter``'s
    ``Placement``/``Placement.Base`` branch, and ``align_freecad_objects`` all refuse
    outright to touch that part's ``Placement`` directly — a raw Euclidean override
    would silently pull it back out of the mate this call just computed. The only way
    to reposition an already-constrained part is to call this function again (any
    element/uv_tensor/constraint_type is fine, including against a different face
    entirely); doing so simply re-sets ``DanaConstrained`` and moves the part to the
    newly-computed placement. Refuses outright (same as ``position_assembly_part``) if
    ``part2_name`` was anchored via ``anchor_assembly_root``.

    ``constraint_type``:

    - ``"Coincident"``: aligns the two elements' reference points. If BOTH
      elements are faces, ALSO orients part2 so its face normal points
      opposite part1's (the standard "faces pushed together" mating
      convention) — any other element-type pairing (edge-edge, face-edge)
      only aligns the points, since there is no single unambiguous relative
      orientation to infer for those.
    - ``"Concentric"``: requires BOTH elements to be circular (an Edge whose
      Curve is a circle/arc) or cylindrical (a Face whose Surface is a
      cylinder) — rejected otherwise. Aligns both axes' directions, then
      their center points.
    - ``"Parallel"``: rotates part2 so its element's reference direction
      (face normal, or edge direction/axis) is parallel to part1's. No
      translation.
    - ``"Perpendicular"``: rotates part2 by the minimal angle that makes its
      reference direction exactly 90 degrees from part1's. No translation.
    - ``"Distance"``: same as ``"Coincident"`` (including its face-normal
      orientation, when both elements are faces), except part2's reference
      point ends up ``offset`` mm from part1's along part1's own reference
      direction, instead of exactly coincident.

    ``uv_tensor`` (Coincident/Distance only, ``part1_element`` must be a
    Face — REJECTED outright, not silently ignored, if given for any other
    ``constraint_type`` or for an Edge reference, same "reject clearly"
    convention as an unknown enum value or a nonexistent Face index
    elsewhere in this function): an optional ``[u, v]`` pair, each value in
    ``[0.0, 1.0]`` inclusive — a normalized, CONTINUOUS parametric
    coordinate, NEVER a raw world-space value, so the caller cannot
    hallucinate a point that falls outside the face's real bounds. Defaults
    to ``[0.5, 0.5]`` (dead center) when omitted. Linearly interpolated
    across that face's own trimmed ``ParameterRange`` (verified live: this
    returns the FACE's actual bounded rectangle, not the underlying
    infinite plane), after first shrinking it inward by the Geometric Fit
    Guard's padding below — ``uv_tensor=[0.0, 0.0]``/``[1.0, 0.0]``/
    ``[0.0, 1.0]``/``[1.0, 1.0]`` land at the four corners of that padded
    rectangle, ``[0.5, 0.5]`` at its center, and any value in between at a
    continuous blend of the two axes. This replaces the old fixed 5-token
    ``semantic_alignment`` enum (``"center"``/``"top_left"``/...) with a
    strictly more general coordinate space — every former token is just one
    particular ``uv_tensor`` value now (``"top_left"`` was ``[0.0, 1.0]``,
    ``"bottom_right"`` was ``[1.0, 0.0]``, etc.). This is what makes a
    symmetric N-part layout (e.g. four wheels at four corners of a chassis)
    expressible with zero raw-coordinate math: mate each wheel to the SAME
    face with a different ``uv_tensor``, e.g. ``[0.0, 0.0]``/``[1.0, 0.0]``/
    ``[0.0, 1.0]``/``[1.0, 1.0]``.

    Geometric Fit Guard: the inward padding pulling the requested ``uv``
    point off ``part1_element``'s true edge is ``part2``'s OWN real
    physical half-extent (``max(part2's BoundBox X/Y/Z Length) / 2``) —
    dynamic per call, never a fixed percentage of the face — so a small
    part gets pulled in only a little and a large part correctly gets
    pulled in further, on any size face. Runs UNCONDITIONALLY now,
    including for the default center point — unlike the retired
    ``semantic_alignment="center"`` token, which bypassed this guard
    entirely. If ``part1_element`` isn't physically large enough to hold
    that padding on BOTH parametric axes (i.e. ``part2`` cannot fit on this
    face at all, not even centered), this is REJECTED outright with an
    explicit "too small to accommodate" error rather than silently
    overlapping ``part2`` with a sibling or letting it hang off the face's
    real edge.

    Double-booking is now caught purely geometrically (the Collision Guard
    further below), not by a separate token registry: a continuous
    ``uv_tensor`` float pair can't be exact-matched as a dict key the way
    the old 5-token enum could, so the ``DanaOccupiedAlignments`` slot
    registry is retired outright — if the computed placement lands within
    ``1e-6`` mm of another sibling already in this assembly, the call is
    REJECTED with the same "pick a different point" guidance a double-
    booked slot used to give, just detected after computing the real point
    instead of before.

    Verified live (freecadcmd, this exact FreeCAD build) against a real
    box+cylinder: ``Coincident`` between two planar faces produces an EXACT
    point match and exactly-antiparallel normals, with the moved part's
    body correctly extending away from the shared face; ``Concentric``
    between two cylindrical faces correctly aligns both axes and centers.

    Known scope limit, confirmed by that same testing, not a hypothetical:
    ``"Coincident"``/``"Parallel"``/``"Perpendicular"``/``"Distance"`` use a
    face's ``CenterOfMass`` + ``normalAt`` as its reference point/direction —
    well-defined for a flat (planar) face, but a FULL cylindrical/conical
    face's centroid sits ON its own axis rather than on the surface itself,
    making the "normal at that point" geometrically ambiguous. Use
    ``"Concentric"`` for a cylindrical face or circular edge instead — it
    uses ``Surface.Axis``/``Surface.Center`` (or ``Curve.Axis``/``Curve.Center``
    for a circular edge), which stays well-defined for exactly this case.

    ``world_fractions`` (Coincident/Distance only, same Face-only restriction
    as ``uv_tensor``, and mutually exclusive with it — passing both is
    REJECTED outright): the deterministic alternative to guessing whether
    ``u`` or ``v`` corresponds to a particular world-space direction. A
    planar face's ``(u, v)`` parameterization is set by OpenCASCADE's
    underlying ``Geom_Plane``, which is NOT guaranteed to line up with any
    particular world axis or with any other face's own ``(u, v)`` — verified
    live: two side faces of the same box can map ``u``/``v`` to different
    world axes from each other, so a caller-side assumption like "u is
    always length, v is always height" is unsound in general, and choosing
    the wrong one silently distributes parts along the WRONG world axis
    (e.g. spreading wheels vertically instead of front-to-back) even though
    every individual call succeeds and looks reasonable in isolation.
    ``world_fractions`` is a ``{"X"/"Y"/"Z": fraction}`` dict with ONE or TWO
    entries (a planar face has exactly two degrees of freedom, ``u`` and
    ``v``, so at most two independent world axes can be pinned at once —
    e.g. ``{"X": 0.1}`` for one axis, or ``{"X": 0.0, "Y": 0.0}`` for a
    corner expressed in world terms), each fraction in ``[0.0, 1.0]``, same
    convention as one axis of ``uv_tensor``. This call resolves, from the
    face's OWN real geometry, whichever of ``u``/``v`` actually points along
    (or against) each requested world axis, applies that axis's fraction to
    it (flipped if the parametric axis runs opposite the world axis), and
    centers any axis not covered by a parametric match at ``0.5`` — so
    ``world_fractions={"X": 0.1, "Z": 0.9}`` reliably means "10% along world
    X, 90% along world Z on this face", regardless of how that face's
    ``u``/``v`` happen to be oriented. REJECTED outright if a requested axis
    doesn't align with either parametric axis closely enough, or if two
    requested axes resolve to the SAME parametric axis (they'd conflict) —
    call ``query_topology`` first to inspect the real mapping in either
    case. Prefer this over ``uv_tensor`` whenever the caller's actual intent
    is phrased in terms of world axes (e.g. "distribute along the chassis
    length" / "spread out longitudinally" / "place at this corner") rather
    than a face-local ``(u, v)`` guess.
    """
    assembly = (assembly_name or "").strip()
    p1_name = (part1_name or "").strip()
    p2_name = (part2_name or "").strip()
    ctype = (constraint_type or "").strip()
    try:
        p1_elem = _validate_constraint_element(part1_element, "part1_element")
        p2_elem = _validate_constraint_element(part2_element, "part2_element")
    except ValueError as exc:
        return _error(f"apply_assembly_constraint: {exc}")
    missing = [
        n
        for n, v in (
            ("assembly_name", assembly),
            ("part1_name", p1_name),
            ("part1_element", p1_elem),
            ("part2_name", p2_name),
            ("part2_element", p2_elem),
            ("constraint_type", ctype),
        )
        if not v
    ]
    if missing:
        return _error(f"apply_assembly_constraint requires {', '.join(missing)}")
    if ctype not in _ASSEMBLY_CONSTRAINT_TYPES:
        return _error(
            f"apply_assembly_constraint: constraint_type must be one of "
            f"{sorted(_ASSEMBLY_CONSTRAINT_TYPES)}, got {constraint_type!r}"
        )
    if p1_name == p2_name:
        return _error("apply_assembly_constraint: part1_name and part2_name must be two different parts")
    try:
        offset_f = float(offset)
    except (TypeError, ValueError):
        return _error("apply_assembly_constraint: offset must be a number")

    uv_explicit = uv_tensor is not None
    if not uv_explicit:
        uv: tuple[float, float] = (0.5, 0.5)
    else:
        valid_shape = (
            isinstance(uv_tensor, (list, tuple))
            and len(uv_tensor) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in uv_tensor)
        )
        if not valid_shape:
            return _error(
                f"apply_assembly_constraint: uv_tensor must be a [u, v] array of two numbers, "
                f"got {uv_tensor!r}"
            )
        u_val, v_val = float(uv_tensor[0]), float(uv_tensor[1])
        if not (0.0 <= u_val <= 1.0 and 0.0 <= v_val <= 1.0):
            return _error(
                f"apply_assembly_constraint: uv_tensor values must each be between 0.0 and 1.0 "
                f"inclusive, got {uv_tensor!r}"
            )
        uv = (u_val, v_val)
    if uv_explicit and ctype not in ("Coincident", "Distance"):
        return _error(
            f"apply_assembly_constraint: uv_tensor is only meaningful for 'Coincident'/"
            f"'Distance', not {ctype!r} — omit it for this constraint_type"
        )

    axis_explicit = world_fractions is not None
    world_fractions_norm: dict[str, float] | None = None
    if axis_explicit:
        if uv_explicit:
            return _error(
                "apply_assembly_constraint: world_fractions and uv_tensor are mutually "
                "exclusive — pass only one"
            )
        if not isinstance(world_fractions, dict) or not (1 <= len(world_fractions) <= 2):
            return _error(
                f"apply_assembly_constraint: world_fractions must be a dict with 1 or 2 entries "
                f"from {{'X', 'Y', 'Z'}}, got {world_fractions!r}"
            )
        world_fractions_norm = {}
        for axis_name, frac in world_fractions.items():
            axis_key = str(axis_name).strip().upper()
            if axis_key not in ("X", "Y", "Z"):
                return _error(
                    f"apply_assembly_constraint: world_fractions keys must be 'X'/'Y'/'Z', "
                    f"got {axis_name!r}"
                )
            if axis_key in world_fractions_norm:
                return _error(
                    f"apply_assembly_constraint: world_fractions has duplicate axis {axis_key!r}"
                )
            try:
                frac_f = float(frac)
            except (TypeError, ValueError):
                return _error(f"apply_assembly_constraint: world_fractions[{axis_name!r}] must be a number")
            if not (0.0 <= frac_f <= 1.0):
                return _error(
                    f"apply_assembly_constraint: world_fractions[{axis_name!r}] must be between "
                    f"0.0 and 1.0 inclusive, got {frac!r}"
                )
            world_fractions_norm[axis_key] = frac_f
        if ctype not in ("Coincident", "Distance"):
            return _error(
                f"apply_assembly_constraint: world_fractions is only meaningful for "
                f"'Coincident'/'Distance', not {ctype!r} — omit it for this constraint_type"
            )

    dims = {
        "part1": f"{p1_name}.{_element_label(p1_elem)}",
        "part2": f"{p2_name}.{_element_label(p2_elem)}",
        "constraint_type": ctype,
        "offset": offset_f,
        "uv_tensor": list(uv),
    }
    if axis_explicit:
        dims["world_fractions"] = world_fractions_norm
    if is_dry_run_enabled():
        return _dry_run_result("apply_assembly_constraint", name=p2_name, dimensions=dims)
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error(
            "apply_assembly_constraint: no session document yet — create objects with create_box/"
            "create_cylinder first"
        )
    script = _APPLY_ASSEMBLY_CONSTRAINT_SCRIPT.format(
        assembly_name=assembly,
        part1_name=p1_name,
        part1_element=p1_elem,
        part1_element_label=_element_label(p1_elem),
        part2_name=p2_name,
        part2_element=p2_elem,
        part2_element_label=_element_label(p2_elem),
        constraint_type=ctype,
        offset=offset_f,
        uv_tensor=uv,
        world_fractions=world_fractions_norm,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"apply_assembly_constraint failed: {result['error']}")
    resolved_uv_match = _RESOLVED_UV_RE.search(result["stdout"] or "")
    if resolved_uv_match:
        resolved = json.loads(resolved_uv_match.group(1))
        dims["uv_tensor"] = resolved["uv_tensor"]
        if resolved.get("u_world_axis") or resolved.get("v_world_axis"):
            dims["axis_alignment"] = (
                f"NOTE: on this face, u points along world {resolved.get('u_world_axis') or '(no single axis)'}"
                f" and v points along world {resolved.get('v_world_axis') or '(no single axis)'}"
                " -- if uv_tensor was guessed assuming a different axis, the placement above may not "
                "be where you intended; re-check against this, or use world_fractions instead."
            )
    return _ok(
        name=result.get("resolved_name") or p2_name,
        dimensions=dims,
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

_ANCHOR_ASSEMBLY_ROOT_SCRIPT = ("""\
import FreeCAD as App

""" + _RESOLVE_OBJECT_SNIPPET + _SESSION_OPEN_SNIPPET + """\
_assembly = resolve_object(doc, {assembly_name!r})
if _assembly is None:
    raise RuntimeError("Object not found: " + {assembly_name!r})
part = resolve_object(doc, {part_name!r})
if part is None:
    raise RuntimeError("Object not found: " + {part_name!r})
if part not in getattr(_assembly, "Group", []):
    raise RuntimeError(
        "'" + {part_name!r} + "' is not in assembly '" + {assembly_name!r}
        + "' -- call add_parts_to_assembly first."
    )

part.Placement = App.Placement(App.Vector(0, 0, 0), App.Rotation(0, 0, 0))
if not hasattr(part, "DanaAnchored"):
    part.addProperty(
        "App::PropertyBool", "DanaAnchored", "Dana",
        "Set by anchor_assembly_root -- Placement is locked to the assembly's fixed "
        "reference frame; every other placement-mutating tool (position_assembly_part, "
        "modify_freecad_parameter's Placement branch, apply_assembly_constraint's part2) "
        "refuses to move or rotate this object until a fresh anchor_assembly_root call."
    )
part.DanaAnchored = True
doc.recompute()

obj = part
""" + _SESSION_SAVE_SNIPPET + _ASSEMBLY_RESULT_PRINT)

def anchor_assembly_root(assembly_name: str, part_name: str) -> str:
    """Deterministically pins ``part_name`` — typically the chassis/main
    body, the rigid reference every other part in ``assembly_name`` gets
    positioned against — to the assembly's origin: resets its
    ``Placement`` to IDENTITY (position ``(0, 0, 0)``, rotation
    ``(0, 0, 0)``) and marks it with a persistent ``DanaAnchored`` custom
    property.

    There is no live FreeCAD Assembly-workbench solver reachable from this
    headless FreeCADCmd execution model to enforce a real "Fixed" joint
    constraint (see ``apply_assembly_constraint``'s own docstring for
    exactly why) — ``DanaAnchored`` is the deterministic Python-side
    substitute: every other placement-mutating tool here
    (``position_assembly_part``, ``modify_freecad_parameter``'s
    ``Placement``/``Placement.Base`` branch, ``apply_assembly_constraint``
    when this part is passed as ``part2``) checks it and REFUSES to move or
    rotate the object outright, so once anchored, nothing can silently tilt
    it again except another ``anchor_assembly_root`` call (which just
    re-applies the same identity reset). It may still be used as
    ``apply_assembly_constraint``'s ``part1`` (the fixed reference), since
    that side never moves anyway.

    ``part_name`` must already be a member of ``assembly_name`` (call
    ``add_parts_to_assembly`` first) — same "must already belong" contract
    as ``apply_assembly_constraint``'s part1_name/part2_name.
    """
    assembly = (assembly_name or "").strip()
    part = (part_name or "").strip()
    missing = [n for n, v in (("assembly_name", assembly), ("part_name", part)) if not v]
    if missing:
        return _error(f"anchor_assembly_root requires {', '.join(missing)}")
    dims = {"placement": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}
    if is_dry_run_enabled():
        return _dry_run_result("anchor_assembly_root", name=part, dimensions=dims)
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error(
            "anchor_assembly_root: no session document yet — create objects with create_box/"
            "create_cylinder first"
        )
    script = _ANCHOR_ASSEMBLY_ROOT_SCRIPT.format(
        assembly_name=assembly,
        part_name=part,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"anchor_assembly_root failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or part,
        dimensions=dims,
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

_DEFINE_KINEMATIC_JOINT_SCRIPT = ("""\
import FreeCAD as App
import json

""" + _RESOLVE_OBJECT_SNIPPET + _SESSION_OPEN_SNIPPET + """\
assembly = resolve_object(doc, {assembly_name!r})
if assembly is None:
    raise RuntimeError("Object not found: " + {assembly_name!r})

child = resolve_object(doc, {child_link!r})
if child is None:
    raise RuntimeError("Object not found: " + {child_link!r})
if child not in assembly.Group:
    raise RuntimeError(
        {child_link!r} + " is not a member of assembly " + assembly.Name
        + " -- add it first with add_parts_to_assembly"
    )

# Kinematic Axis Guard: for a joint type where `axis` is physically
# meaningful (everything except "fixed"), reject it outright if `child`
# is a simple rotationally-symmetric part (exactly one cylindrical face --
# a plain cylinder/wheel, not a compound shape with multiple curved faces,
# which is left unchecked rather than guessed at) whose REAL geometric
# axis doesn't line up with the requested `axis`. This is a pure geometry
# check independent of how `child` got positioned (which face it was
# mated to, what uv_tensor was used, how the plan worded the task) --
# closing the exact failure mode a text-only prompt rule couldn't: a wheel
# mated flush to the chassis's TOP/BOTTOM face is a perfectly valid planar
# Coincident mate (nothing about it is geometrically wrong), so it sails
# through apply_assembly_constraint, but declaring "axis=[1,0,0]" for a
# part whose real symmetry axis is Z would build a URDF joint that makes
# the wheel tumble in place instead of roll -- caught HERE, at the one
# point the model's actual intent (the joint axis) and the part's actual
# geometry are both in hand at once.
_jtype_for_axis_check = {joint_type!r}
if _jtype_for_axis_check != "fixed":
    _requested_axis = App.Vector(*{axis!r})
    if _requested_axis.Length > 1e-9:
        _requested_axis.normalize()
        _cyl_faces = [f for f in child.Shape.Faces if f.Surface.TypeId == "Part::GeomCylinder"]
        if len(_cyl_faces) == 1:
            _physical_axis = _cyl_faces[0].Surface.Axis
            if _physical_axis.Length > 1e-9:
                _physical_axis = App.Vector(_physical_axis.x, _physical_axis.y, _physical_axis.z)
                _physical_axis.normalize()
                # abs() because a joint axis anti-parallel to the part's axis
                # (e.g. requested [-1,0,0] against a physical +X symmetry
                # axis) is still the same rotational axis -- only genuine
                # off-axis mismatches (roughly orthogonal or skewed) should
                # reject. 0.9 ~= 26 degrees of tolerance: comfortably below
                # a real mismatch (this exact bug measured dot=0.0, axle on
                # Z vs requested X) but past ordinary floating-point noise
                # from composed rotations.
                _alignment = abs(_requested_axis.dot(_physical_axis))
                if _alignment < 0.9:
                    raise RuntimeError(
                        "define_kinematic_joint: '" + child.Name + "' is a cylindrical part whose real "
                        "rotational axis is " + str(tuple(round(c, 3) for c in _physical_axis)) + ", but the "
                        "requested joint axis " + str({axis!r}) + " is not aligned with it (alignment=" +
                        str(round(_alignment, 3)) + ", need >= 0.9). A '" + _jtype_for_axis_check + "' joint "
                        "about a mismatched axis builds a wheel that tumbles/wobbles instead of rolling. "
                        "Re-mate '" + child.Name + "' with apply_assembly_constraint so its FLAT end-cap face "
                        "is Coincident with a chassis face whose outward normal matches your intended rolling "
                        "axis (see Rule 16), then retry this call with `axis` set to that same direction."
                    )

_parent_link = {parent_link!r}
if _parent_link != {root_link!r}:
    _parent_obj = resolve_object(doc, _parent_link)
    if _parent_obj is None:
        raise RuntimeError("Object not found: " + _parent_link)
    if _parent_obj not in assembly.Group:
        raise RuntimeError(_parent_link + " is not a member of assembly " + assembly.Name)
    if _parent_obj.Name == child.Name:
        raise RuntimeError("a link cannot be its own parent: " + child.Name)
    _parent_link = _parent_obj.Name

if not hasattr(assembly, {joints_prop!r}):
    assembly.addProperty(
        "App::PropertyString", {joints_prop!r}, "Dana",
        "JSON-encoded URDF kinematic joint overrides, keyed by child link Name"
    )
    setattr(assembly, {joints_prop!r}, "{{}}")

_joints = json.loads(getattr(assembly, {joints_prop!r}) or "{{}}")
_joints[child.Name] = {{
    "parent": _parent_link,
    "type": {joint_type!r},
    "axis": {axis!r},
    "joint_name": {joint_name!r},
    "limit_lower": {limit_lower!r},
    "limit_upper": {limit_upper!r},
    "limit_effort": {limit_effort!r},
    "limit_velocity": {limit_velocity!r},
}}
setattr(assembly, {joints_prop!r}, json.dumps(_joints))
doc.recompute()

obj = assembly
""" + _SESSION_SAVE_SNIPPET + _ASSEMBLY_RESULT_PRINT)

def define_kinematic_joint(
    assembly_name: str,
    child_link: str,
    parent_link: str = ROOT_LINK_NAME,
    joint_type: str = "fixed",
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    joint_name: str | None = None,
    limit_lower: float | None = None,
    limit_upper: float | None = None,
    limit_effort: float | None = None,
    limit_velocity: float | None = None,
) -> str:
    """Declares a real parent/child kinematic joint between two members of
    an existing ``create_assembly`` container — what ``export_assembly_to_urdf``
    was missing to emit anything besides a flat "every part fixed to one
    synthetic base_link" star topology.

    Persisted as a JSON-encoded custom property directly on the assembly's
    own ``App::Part`` object (``DanaKinematicJoints``) inside the shared
    session document — the ONLY place this can live, since every tool call
    here is its own fresh FreeCADCmd subprocess with no Python state kept
    in memory between calls (see this module's own docstring), so a LATER
    ``export_assembly_to_urdf`` call — a completely separate process — can
    still read back what this call wrote.

    ``child_link``/``parent_link`` must both already be real members of
    ``assembly_name`` (added via ``add_parts_to_assembly``, validated
    against the assembly's actual ``Group`` — same story as
    ``apply_assembly_constraint``'s part1_name/part2_name) — EXCEPT
    ``parent_link`` may also be the literal string ``"base_link"`` (the
    default), the synthetic root ``export_assembly_to_urdf`` always emits,
    meaning "attach directly to the world" rather than to another real
    part. Calling this again for the same ``child_link`` REPLACES its
    joint definition (keyed by child — a link can only ever have one
    parent in a valid kinematic tree), so redefining a joint (change its
    type, move it under a different parent) is just calling this again,
    no separate update/delete tool needed.

    Kinematic Axis Guard: for any ``joint_type`` other than ``"fixed"``, if
    ``child_link`` is a simple rotationally-symmetric part (exactly one
    cylindrical face — a plain wheel-shaped cylinder), ``axis`` is
    rejected unless it's actually aligned (or anti-aligned) with that
    part's REAL geometric symmetry axis, regardless of which face it was
    mated to or how the plan worded the positioning step — confirmed live
    (rover chassis stress test) that a wheel mated flush to a chassis's
    TOP/BOTTOM face is a perfectly valid planar Coincident mate, so
    ``apply_assembly_constraint`` has no reason to reject it, yet
    declaring a joint axis that doesn't match the part's actual symmetry
    axis silently builds a URDF robot whose "wheel" tumbles in place
    instead of rolling. A part with zero or more than one cylindrical face
    (e.g. after a boolean cut, or a non-cylindrical link) is left
    unchecked rather than guessed at.

    ``joint_type``: ``"fixed"`` (no ``axis``/``limits`` — the same rigid
    joint every part got before this tool existed), ``"revolute"``/
    ``"prismatic"`` (rotate/slide along ``axis``; omitted ``limit_lower``/
    ``limit_upper``/``limit_effort``/``limit_velocity`` fall back to
    generic placeholders at export time — see
    ``dana.tools.urdf_builder._add_joint_kinematics``'s own docstring), or
    ``"continuous"`` (unlimited rotation about ``axis``, no limit).

    Cycle/dangling-parent validation happens at ``export_assembly_to_urdf``
    time, not here — joints are typically defined in an arbitrary order
    while a whole tree is still being built, so any single definition call
    can't yet know whether the OVERALL tree it will end up part of is
    valid.

    ``joint_name`` (optional) is resolved to a concrete name RIGHT HERE —
    ``"<parent_link>_to_<child_link>"`` when omitted — and echoed back
    as ``dimensions["joint_name"]``, rather than left ``None`` for
    ``export_assembly_to_urdf``'s own export-time default to fill in
    later: ``dana.core.react_dispatch``'s caller registers this exact name
    into the session's object registry/topology graph the instant this
    call succeeds, so a model that immediately claims credit for it (e.g.
    via ``mark_task_completed``) is checking against a name that's already
    real, not one that won't exist until a much later export call decides
    it.
    """
    assembly = (assembly_name or "").strip()
    if not assembly:
        return _error("define_kinematic_joint requires assembly_name")
    child = (child_link or "").strip()
    if not child:
        return _error("define_kinematic_joint requires child_link")
    parent = (parent_link or ROOT_LINK_NAME).strip() or ROOT_LINK_NAME
    if child == parent:
        return _error("define_kinematic_joint: child_link and parent_link cannot be the same part")
    jtype = (joint_type or "fixed").strip().lower()
    if jtype not in _KINEMATIC_JOINT_TYPES:
        return _error(
            f"define_kinematic_joint: unknown joint_type {jtype!r} — must be one of "
            f"{sorted(_KINEMATIC_JOINT_TYPES)}"
        )
    try:
        axis_vec = [float(v) for v in axis]
    except (TypeError, ValueError):
        return _error("define_kinematic_joint: axis must be 3 numbers")
    if len(axis_vec) != 3:
        return _error("define_kinematic_joint: axis must have exactly 3 elements [x, y, z]")
    try:
        lower = None if limit_lower is None else float(limit_lower)
        upper = None if limit_upper is None else float(limit_upper)
        effort = None if limit_effort is None else float(limit_effort)
        velocity = None if limit_velocity is None else float(limit_velocity)
    except (TypeError, ValueError):
        return _error("define_kinematic_joint: limit_lower/upper/effort/velocity must all be numbers if given")

    # Resolved HERE, not left as None for urdf_builder's own export-time
    # default to fill in later: mark_task_completed's Evidence-Based Gate
    # (_object_registry()) needs ONE concrete, stable name to register the
    # instant this call succeeds — a caller (or the LLM) checking
    # "does the joint I just created exist" can't be told to wait until a
    # LATER export_assembly_to_urdf call decides what it's actually named.
    resolved_joint_name = (joint_name or "").strip() or f"{parent}_to_{child}"
    dims = {
        "parent_link": parent,
        "child_link": child,
        "joint_type": jtype,
        "axis": axis_vec,
        "joint_name": resolved_joint_name,
    }
    if is_dry_run_enabled():
        return _dry_run_result("define_kinematic_joint", name=assembly, dimensions=dims)
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error(
            "define_kinematic_joint: no session document yet — create an assembly with "
            "create_freecad_assembly first"
        )
    script = _DEFINE_KINEMATIC_JOINT_SCRIPT.format(
        assembly_name=assembly,
        child_link=child,
        parent_link=parent,
        root_link=ROOT_LINK_NAME,
        joint_type=jtype,
        axis=axis_vec,
        joint_name=resolved_joint_name,
        limit_lower=lower,
        limit_upper=upper,
        limit_effort=effort,
        limit_velocity=velocity,
        joints_prop=_KINEMATIC_JOINTS_PROP,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"define_kinematic_joint failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or assembly,
        dimensions=dims,
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

_EXPORT_ASSEMBLY_URDF_SCRIPT = ("""\
import FreeCAD as App
import Part
import MeshPart
import json
import math
import os

""" + _RESOLVE_OBJECT_SNIPPET + _SESSION_OPEN_SNIPPET + """\
assembly = resolve_object(doc, {assembly_name!r})
if assembly is None:
    raise RuntimeError("Object not found: " + {assembly_name!r})

# Consumed-Operand Filter: a Boolean feature (Part::Cut/MultiFuse/
# MultiCommon) keeps its Base/Tool/Shapes inputs as REAL, live parametric
# links -- FreeCAD's own scope-consistency rules require them to stay in
# the same GeoFeatureGroup as the feature that references them (removing
# them from the assembly breaks the recompute graph outright -- confirmed
# live: doing so collapsed the WHOLE Group to empty instead of cleanly
# dropping just that one member). They are therefore still real Group
# members here, not independent parts any more -- excluding them from
# export keeps a source solid that's now purely a dependency of its own
# boolean result from generating a second, phantom, perfectly-overlapping
# <link> alongside it.
_consumed_operands = set()
for _obj in doc.Objects:
    for _attr in ("Base", "Tool"):
        _ref = getattr(_obj, _attr, None)
        if _ref is not None and hasattr(_ref, "Name"):
            _consumed_operands.add(_ref.Name)
    for _ref in (getattr(_obj, "Shapes", None) or []):
        if hasattr(_ref, "Name"):
            _consumed_operands.add(_ref.Name)

_members = [m for m in (getattr(assembly, "Group", []) or []) if m.Name not in _consumed_operands]
if not _members:
    raise RuntimeError("assembly '" + {assembly_name!r} + "' has no parts — add some with add_parts_to_assembly first")

_meshes_dir = {meshes_dir!r}
os.makedirs(_meshes_dir, exist_ok=True)

_root_link = {root_link!r}
_joint_defs = json.loads(getattr(assembly, {joints_prop!r}, "{{}}") or "{{}}")

_parts = []
for _m in _members:
    _shape = getattr(_m, "Shape", None)
    if _shape is None or _shape.isNull():
        continue  # a pure organizational sub-group with no geometry of its own -- nothing to export
    _local_shape = _shape.copy()
    _local_shape.transformShape(_m.Placement.inverse().toMatrix())
    _mesh = MeshPart.meshFromShape(Shape=_local_shape, LinearDeflection=0.1, AngularDeflection=0.1)
    _mesh_path = os.path.join(_meshes_dir, _m.Name + ".stl")
    _mesh.write(_mesh_path)

    _jdef = _joint_defs.get(_m.Name, {{}})
    _joint_parent = _jdef.get("parent") or _root_link
    if _joint_parent != _root_link:
        _parent_obj = resolve_object(doc, _joint_parent)
        if _parent_obj is None:
            raise RuntimeError(
                "'" + _m.Name + "' has a kinematic joint referencing unknown parent '" + _joint_parent + "'"
            )
        _rel = _parent_obj.Placement.inverse().multiply(_m.Placement)
    else:
        _rel = assembly.Placement.inverse().multiply(_m.Placement)
    _yaw, _pitch, _roll = _rel.Rotation.toEuler()
    # Both read off `_local_shape` (Placement already un-baked above), NOT
    # `_shape` — Part.Shape.CenterOfMass/.MatrixOfInertia are otherwise
    # computed in the document's GLOBAL frame, and requirement was "relative
    # to the part's own local origin" so URDF joint transforms compose
    # correctly instead of double-counting the Placement a second time (the
    # exact bug this whole script's module comment already documents for the
    # mesh). Live-verified (freecadcmd, a translated+rotated box): OCC's
    # MatrixOfInertia is ALWAYS computed about the shape's own center of mass
    # (translation-invariant) and its A11.. entries map directly to the
    # standard ixx/iyy/izz/ixy/ixz/iyz tensor with no extra sign flip needed.
    #
    # A boolean-chain result (Part::Cut/MultiFuse/MultiCommon) reports its
    # own .Shape as a Part::Compound wrapping the real solid -- CenterOfMass/
    # MatrixOfInertia aren't defined on a bare Compound in this FreeCAD
    # build (confirmed live: "'Part.Compound' object has no attribute
    # 'CenterOfMass'", every export of a boolean-derived member failing
    # outright) -- same limitation this module's other mass-property script
    # already unwraps (see its own "shape.ShapeType == 'Compound' and
    # shape.Solids" guard elsewhere in this file); mirrored here rather than
    # shared, since each is its own independently-rendered FreeCADCmd script
    # string, not shared Python. Only the first solid is used when compound
    # -- a genuinely multi-solid compound (several disjoint bodies grouped
    # together) would need per-solid mass aggregation this doesn't attempt,
    # same single-solid assumption the existing unwrap already accepts.
    # Volume is read off the same unwrapped shape too, so mass/CoM/inertia/
    # volume all agree on which solid they describe instead of volume
    # silently including sub-shapes CenterOfMass/MatrixOfInertia ignore.
    _mass_shape = _local_shape
    if _mass_shape.ShapeType == 'Compound' and _mass_shape.Solids:
        _mass_shape = _mass_shape.Solids[0]
    _com = _mass_shape.CenterOfMass
    _moi = _mass_shape.MatrixOfInertia
    _parts.append(
        {{
            "name": _m.Name,
            "mesh_file": "meshes/" + _m.Name + ".stl",
            "origin_xyz": [_rel.Base.x, _rel.Base.y, _rel.Base.z],
            "origin_rpy": [math.radians(_roll), math.radians(_pitch), math.radians(_yaw)],
            "volume": _mass_shape.Volume,
            "center_of_mass": [_com.x, _com.y, _com.z],
            "inertia": {{
                "ixx": _moi.A11, "ixy": _moi.A12, "ixz": _moi.A13,
                "iyy": _moi.A22, "iyz": _moi.A23, "izz": _moi.A33,
            }},
            "joint_parent": _joint_parent,
            "joint_type": _jdef.get("type") or "fixed",
            "joint_axis": _jdef.get("axis") or [0.0, 0.0, 1.0],
            "joint_name": _jdef.get("joint_name"),
            "limit_lower": _jdef.get("limit_lower"),
            "limit_upper": _jdef.get("limit_upper"),
            "limit_effort": _jdef.get("limit_effort"),
            "limit_velocity": _jdef.get("limit_velocity"),
        }}
    )

if not _parts:
    raise RuntimeError("assembly '" + {assembly_name!r} + "' has no parts with real geometry to export")

# Parts Manifest File (robustness fix): written to disk in ADDITION to the
# stdout marker print below, and read back as the AUTHORITATIVE source by
# export_assembly_to_urdf's own Python wrapper. Live-confirmed (rover
# chassis stress test #3, dana_runtime.log): a run where this script's own
# _run_freecad_script() call correctly reported ok=True (returncode 0, no
# exception banner in stderr) still came back with the "{marker}_URDF_PARTS"
# line missing from captured stdout -- FreeCADCmd's subprocess stdout can
# apparently drop/truncate output right at process exit on this platform
# (the exact "exit code is not proof stdout is complete" class of issue
# this module's own _SCRIPT_EXCEPTION_MARKER comment already documents for
# exceptions specifically; this is the same unreliability applying to a
# plain, successful print instead). A file write is not subject to
# subprocess stdout capture at all, so it can't be dropped the same way.
_manifest_path = os.path.join(os.path.dirname(_meshes_dir), "_parts_manifest.json")
with open(_manifest_path, "w", encoding="utf-8") as _f:
    json.dump({{"parts": _parts, "name": assembly.Name}}, _f)

print("{marker}_URDF_PARTS " + json.dumps(_parts))
print("{marker}_NAME " + assembly.Name)
""")

_VALIDATE_ASSEMBLY_COLLISIONS_SCRIPT = ("""\
import FreeCAD as App

""" + _RESOLVE_OBJECT_SNIPPET + _SESSION_OPEN_SNIPPET + """\
assembly = resolve_object(doc, {assembly_name!r})
if assembly is None:
    raise RuntimeError("Object not found: " + {assembly_name!r})

# Consumed-Operand Filter: same reasoning as export_assembly_to_urdf's own
# copy of this filter (kept independent rather than shared, since each is
# its own separately-rendered FreeCADCmd script string, not shared Python)
# -- a Boolean feature's Base/Tool/Shapes inputs stay real Group members
# (FreeCAD's scope-consistency rules require it), but they no longer
# represent independent geometry, so a box and the exact cut derived from
# it (or the tool that cut it) would otherwise always "collide" with each
# other by definition.
_consumed_operands = set()
for _obj in doc.Objects:
    for _attr in ("Base", "Tool"):
        _ref = getattr(_obj, _attr, None)
        if _ref is not None and hasattr(_ref, "Name"):
            _consumed_operands.add(_ref.Name)
    for _ref in (getattr(_obj, "Shapes", None) or []):
        if hasattr(_ref, "Name"):
            _consumed_operands.add(_ref.Name)

_members = [
    m for m in getattr(assembly, "Group", [])
    if getattr(m, "Shape", None) is not None and not m.Shape.isNull()
    and m.Name not in _consumed_operands
]

_collisions = []
_epsilon = {epsilon!r}
for _i in range(len(_members)):
    for _j in range(_i + 1, len(_members)):
        _a, _b = _members[_i], _members[_j]
        try:
            _overlap_volume = _a.Shape.common(_b.Shape).Volume
        except Exception:
            # Non-solid/degenerate geometry can't be intersected -- skipped
            # rather than failing the whole audit over one bad pair; a
            # genuinely broken shape shows up via inspect_spatial_properties
            # instead, which is what that tool exists for.
            continue
        if _overlap_volume > _epsilon:
            _collisions.append(dict(part_a=_a.Name, part_b=_b.Name, overlap_volume=_overlap_volume))

print("{marker}_COLLISIONS " + str(_collisions))
print("{marker}_CHECKED " + str(len(_members)))
print("{marker} ok")
""")

_COLLISION_VOLUME_EPSILON = 1e-6

def validate_assembly_collisions(assembly_name: str) -> str:
    """Volumetric Validation Gate (Phase 2 of the layout safeguard, the
    downstream complement to ``apply_assembly_constraint``'s Collision
    Guard): a whole-assembly, TRUE solid-intersection audit —
    every member pair's real ``Shape.common(...).Volume`` via FreeCAD's
    native OCCT boolean intersection, not a bounding-box overlap (see
    ``analyze_bounding_box_collisions`` for that cheaper, pairwise,
    box-only check) — so a wheel whose bounding box merely brushes the
    chassis's but whose actual curved geometry doesn't touch it reports NO
    collision, while a wheel whose axle genuinely clips through the
    chassis body (mounted to the wrong face, or positioned before the
    chassis's real dimensions were known) is caught even though nothing
    about the MATING CONSTRAINT itself was invalid — uv_tensor and
    the Geometric Fit Guard only ever reason about ONE face's 2D footprint,
    never the full 3D solid, so a legal-looking Coincident mate can still
    produce a real interpenetration this function is the one thing left to
    catch.

    Intentionally NOT real-time coordinate feedback wired into every
    ``apply_assembly_constraint``/``position_assembly_part`` call — a
    volumetric check is comparatively expensive (a real boolean op per
    pair) and, more importantly, gives no actionable DIRECTION to correct
    from (unlike the Fit Guard's "too small, resize or pick another face"),
    so running it after every single placement call would just trade one
    failure mode (silent bad geometry) for another (the LLM guessing
    coordinates in a loop with no better signal than "still overlapping").
    Call this ONCE, explicitly, as a final QA pass — after every part in
    ``assembly_name`` has been positioned, before ``export_assembly_to_urdf``
    — the same "measure before you trust it" discipline
    ``inspect_spatial_properties``/``analyze_bounding_box_collisions``
    already enforce elsewhere in this module, just scoped to the WHOLE
    assembly at once instead of one pair.

    Read-only (never saves, never needs HITL approval — see
    ``analyze_bounding_box_collisions``'s matching note): returns
    ``collisions`` (a list of ``{"part_a", "part_b", "overlap_volume"}``
    dicts, empty if none found) and ``has_collisions``/``checked_members``
    alongside it. A pair whose actual boolean intersection can't be
    computed (degenerate/non-solid geometry) is silently skipped rather
    than failing the whole audit — a broken shape shows up via
    ``inspect_spatial_properties`` instead.
    """
    assembly = (assembly_name or "").strip()
    if not assembly:
        return _error("validate_assembly_collisions requires assembly_name")
    if is_dry_run_enabled():
        return _dry_run_result(
            "validate_assembly_collisions", name=assembly, collisions=[], has_collisions=False
        )
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error(
            "validate_assembly_collisions: no session document yet — create objects with create_box/"
            "create_cylinder first"
        )
    script = _VALIDATE_ASSEMBLY_COLLISIONS_SCRIPT.format(
        assembly_name=assembly,
        epsilon=_COLLISION_VOLUME_EPSILON,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script, require_marker=True)
    if not result["ok"]:
        return _error(f"validate_assembly_collisions failed: {result['error']}")
    collisions = _extract_collisions(result["stdout"]) or []
    checked = _extract_checked_count(result["stdout"]) or 0
    return _ok(
        name=assembly,
        collisions=collisions,
        has_collisions=bool(collisions),
        checked_members=checked,
        path=str(session_path),
    )

def export_assembly_to_urdf(
    assembly_name: str,
    export_directory: str | None = None,
    density_kg_m3: float | None = None,
) -> str:
    """Export ``assembly_name`` (a real ``create_freecad_assembly``
    ``App::Part`` container) into a ``.urdf`` robot description — the
    CAD-to-robotics bridge: every member becomes a URDF ``<link>`` (its
    real geometry, exported as its own ``.stl`` under ``meshes/``) jointed
    onto whatever parent ``define_kinematic_joint`` last declared for it
    (or the synthetic ``base_link`` root by default, for a member no one
    ever called ``define_kinematic_joint`` on), at that member's real
    ``Placement`` RELATIVE TO THAT PARENT (not the document's absolute
    coordinates) — see this function's own module-level comment for the
    live-verified mesh/Placement double-transform bug this specifically
    avoids.

    Each link also gets a ``<collision>`` (the same mesh as ``<visual>``)
    and an ``<inertial>`` block — mass/center-of-mass/inertia tensor
    computed from the part's own real ``Shape`` (volume × ``density_kg_m3``,
    default aluminum — see ``dana.tools.urdf_builder``'s own module
    constant), not guessed, so the export is usable directly in a physics
    simulator instead of only a viewer.

    Without any ``define_kinematic_joint`` calls this is still the
    original flat "star" topology (every part fixed directly to
    ``base_link``) — assembly group membership alone carries no
    information about which part should be whose parent in a real joint
    hierarchy, so nothing here guesses one. Call ``define_kinematic_joint``
    beforehand (once per part that needs a real parent/moving joint) to
    build an actual kinematic tree instead — this function just reads back
    whatever was declared, per part, at export time.

    ``export_directory`` defaults to a ``<assembly_name>_urdf`` folder
    under this session's own output directory. The ``.urdf`` file and its
    ``meshes/`` subdirectory are both written there.
    """
    assembly = (assembly_name or "").strip()
    if not assembly:
        return _error("export_assembly_to_urdf requires assembly_name")

    safe = _safe_name(assembly)
    out_dir = Path(export_directory) if (export_directory or "").strip() else (_export_dir() / f"{safe}_urdf")
    if is_dry_run_enabled():
        return _dry_run_result("export_assembly_to_urdf", name=assembly, path=str(out_dir / f"{safe}.urdf"))
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error(
            "export_assembly_to_urdf: no session document yet — create an assembly with "
            "create_freecad_assembly first"
        )

    meshes_dir = out_dir / "meshes"
    script = _EXPORT_ASSEMBLY_URDF_SCRIPT.format(
        assembly_name=assembly,
        meshes_dir=str(meshes_dir).replace("\\", "/"),
        root_link=ROOT_LINK_NAME,
        joints_prop=_KINEMATIC_JOINTS_PROP,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"export_assembly_to_urdf failed: {result['error']}")

    # Parts Manifest File is the AUTHORITATIVE source (see the script's own
    # comment above the write) -- a file write can't be dropped by
    # subprocess stdout capture the way a print can. The stdout marker line
    # is kept only as a fallback for a manifest file that somehow didn't
    # get written (e.g. a permissions error on out_dir) despite the script
    # otherwise reporting success.
    manifest_path = meshes_dir.parent / "_parts_manifest.json"
    parts: list[dict[str, Any]] | None = None
    if manifest_path.is_file():
        try:
            parts = json.loads(manifest_path.read_text(encoding="utf-8")).get("parts")
        except (OSError, json.JSONDecodeError):
            parts = None

    if parts is None:
        stdout = result.get("stdout") or ""
        prefix = f"{_OK_MARKER}_URDF_PARTS "
        parts_line = next((line for line in stdout.splitlines() if line.startswith(prefix)), None)
        if parts_line is None:
            # Diagnostic tail instead of a bare one-liner -- live-confirmed
            # (stress test #3) the model burned 3 search_tool_catalog calls
            # plus 2 blind retries guessing why this failed, with nothing in
            # the error to go on. stdout/stderr tails let it (or a human)
            # actually see what FreeCAD did instead of guessing again.
            tail = (stdout or result.get("stderr") or "").strip()[-500:]
            return _error(
                "export_assembly_to_urdf: FreeCAD script succeeded but neither the parts manifest "
                f"file ({manifest_path}) nor the stdout marker was found. This is a tool bug, not a "
                f"usage error -- retrying with different arguments will not help. Last {len(tail)} "
                f"chars of FreeCAD output: {tail!r}"
            )
        try:
            parts = json.loads(parts_line[len(prefix):])
        except json.JSONDecodeError as exc:
            return _error(f"export_assembly_to_urdf: could not parse parts manifest: {exc}")

    from dana.tools.urdf_builder import export_assembly_parts_to_urdf

    density_kwargs = {"density_kg_m3": density_kg_m3} if density_kg_m3 is not None else {}
    urdf_result = json.loads(export_assembly_parts_to_urdf(assembly, parts, str(out_dir), **density_kwargs))
    if not urdf_result.get("ok"):
        return _error(f"export_assembly_to_urdf: {urdf_result.get('error')}")
    return _ok(
        name=urdf_result.get("name") or safe,
        type="urdf",
        path=urdf_result.get("path"),
        link_count=urdf_result.get("link_count"),
        joint_count=urdf_result.get("joint_count"),
        meshes_dir=str(meshes_dir),
    )

_PATTERN_TYPES = ir._PATTERN_TYPES

_pattern_offsets = ir._pattern_offsets

def batch_pattern_array(
    source_path: str,
    pattern_type: str,
    *,
    source_object: str | None = None,
    count_x: int = 1,
    count_y: int = 1,
    spacing_x: float | None = None,
    spacing_y: float | None = None,
    count: int = 1,
    radius: float = 0.0,
    name: str = "Pattern",
) -> str:
    """Copy an object already in the shared ``Session_Active.FCStd``
    document into a linear, grid, or circular arrangement, combined into a
    single ``Part::Compound`` left in that SAME document — ONE tool call
    instead of one create_freecad_* call per copy, so a repetitive layout
    (e.g. "64 tiles" as an 8x8 grid) doesn't burn through the ReAct loop's
    per-turn iteration cap one placement at a time.

    ``source_path`` is accepted ONLY for call-site/ABC compatibility with
    ``dana.platform.base.BaseCADEngine.batch_pattern_array`` and its
    headless ``dana.platform.mock`` sibling — that mock driver genuinely
    still needs a real mesh FILE path (its own one-object-per-file
    simulation has no session-document concept at all). THIS real FreeCAD
    engine no longer opens it, or needs it to be a real path at all: since
    Document Lifecycle Unification (below), every by-name lookup here goes
    straight to the shared session document instead.

    ``spacing_x``/``spacing_y`` default to the source object's own
    bounding-box width/depth (read via ``get_bounding_box`` against the
    session document) so adjacent copies sit edge-to-edge with no overlap
    unless a caller wants a deliberate gap or overlap.

    ``source_object`` resolves by NAME (Multi-Stage Object Resolution —
    Name/Label/case-insensitive, same as every other by-name lookup in this
    module) against the shared session document — same session-document/
    name-collision story as ``apply_boolean``/``apply_edge_operation``. Must
    already exist there, built by a session-scoped creation tool or a prior
    session-scoped call's own result name.

    Document Lifecycle Unification: migrated OFF ``doc_mode="standalone"``
    (which built the array in a brand-new, SEPARATE ``.FCStd`` file — the
    array object was then unreachable, by name, to any LATER session tool
    call, e.g. ``perform_freecad_boolean``/``export_freecad_model``, both of
    which resolve their own object arguments against ``Session_Active.FCStd``
    only, producing an "object not located" failure every time a pattern's
    result fed into anything downstream) onto the shared ``_execute_ir_tool``
    pipeline's ordinary ``doc_mode="session"`` path — the SAME document
    every other creation tool in this module already builds into, exactly
    like ``create_box``/``apply_boolean``/``apply_edge_operation``.
    """
    del source_path  # see docstring — kept for call-site/ABC compatibility only
    source_name = (source_object or "").strip()
    if not source_name:
        return _error("batch_pattern_array requires source_object")
    pt = (pattern_type or "").strip().lower()
    if pt not in _PATTERN_TYPES:
        return _error(f"batch_pattern_array: unknown pattern_type '{pattern_type}' — must be linear, grid, or circular")
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error(
            "batch_pattern_array: no session document yet — create objects with create_box/"
            "create_cylinder/insert_standard_part first"
        )

    sx, sy = spacing_x, spacing_y
    if pt in ("linear", "grid") and (sx is None or sy is None):
        bbox = json.loads(get_bounding_box(str(session_path), target_object=source_name))
        if not bbox.get("ok"):
            return _error(f"batch_pattern_array: failed to read source bounding box: {bbox.get('error')}")
        sx = sx if sx is not None else (bbox["x_max"] - bbox["x_min"])
        sy = sy if sy is not None else (bbox["y_max"] - bbox["y_min"])

    dims = {"pattern_type": pt}
    if is_dry_run_enabled():
        return _dry_run_result("batch_pattern_array", name=name, type="Part::Compound", dimensions=dims)

    try:
        result, steps, session_path = _execute_ir_tool(
            "batch_pattern_array",
            doc_mode="session",
            name=name,
            source_object=source_name,
            pattern_type=pt,
            count_x=count_x,
            count_y=count_y,
            spacing_x=float(sx or 0.0),
            spacing_y=float(sy or 0.0),
            count=count,
            radius=float(radius),
        )
    except ValueError as exc:
        return _error(f"batch_pattern_array: {exc}")
    if not result["ok"]:
        return _error(f"batch_pattern_array failed: {result['error']}")
    dims["copy_count"] = len(steps[-1]["offsets"])
    return _ok(
        name=result.get("resolved_name") or name,
        type="Part::Compound",
        bounding_box=result.get("bounding_box"),
        geometry=result.get("geometry"),
        dimensions=dims,
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

_BOOLEAN_FEATURE_TYPE: dict[str, str] = {
    "cut": "Part::Cut",
    "union": "Part::MultiFuse",
    "intersect": "Part::MultiCommon",
}

_DEFAULT_BOOLEAN_NAME: dict[str, str] = {"cut": "Cut", "union": "Fusion", "intersect": "Common"}

def apply_boolean(
    operation: str,
    base_object: str = "",
    tool_object: str = "",
    name: str | None = None,
    objects: list[str] | None = None,
) -> str:
    """Combine two-or-more objects already in the shared ``Session_Active.FCStd``
    document with a Boolean operation, looked up by NAME — not path, since
    every session-scoped creation tool (``create_box``/``create_cylinder``/
    ``insert_standard_part``) now shares that one document, so a path alone
    can no longer tell two objects apart the way it could when each lived in
    its own file.

    ``"cut"`` builds a ``Part::Cut`` (subtracts the tool from the base) and
    only ever takes exactly two names, via ``base_object``/``tool_object``.
    ``"union"``/``"intersect"`` build an N-ary ``Part::MultiFuse``/
    ``Part::MultiCommon`` (fuse everything into one solid / keep only the
    shared overlap) and accept ``objects`` (2+ names) instead — the caller
    (``dana.core.react_dispatch._tool_perform_freecad_boolean``) always uses
    this form for non-cut operations, even for exactly 2 objects. Every name
    across ``base_object``/``tool_object``/``objects`` must already exist in
    the session document — built by a session-scoped creation tool, or a
    prior ``apply_boolean`` call's own result name. Forwarded verbatim to
    ``ir._boolean_from_args``, which already merges all three fields (and
    whose Jinja2 template already renders the N-ary ``MultiFuse``/
    ``MultiCommon`` case) — this wrapper's own signature was the one piece
    of that pipeline never updated to accept ``objects`` at all, so any
    non-cut call reaching here always raised ``TypeError: apply_boolean()
    got an unexpected keyword argument 'objects'`` before this fix.
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
    result, steps, session_path = _execute_ir_tool(
        "perform_freecad_boolean", name=resolved_name, operation=op, feature_type=feature_type,
        base_object=base_object, tool_object=tool_object, objects=objects,
    )
    if not result["ok"]:
        return _error(f"apply_boolean failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or resolved_name,
        type=feature_type,
        operation=op,
        bounding_box=result.get("bounding_box"),
        geometry=result.get("geometry"),
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

_face_axes = ir._face_axes

def create_feature_on_face(
    object_name: str,
    face: str,
    shape: str,
    u: float,
    v: float,
    extent: float,
    operation: str,
    radius: float | None = None,
    width: float | None = None,
    length: float | None = None,
    name: str | None = None,
) -> str:
    """Adds or cuts a circle/rectangle feature directly on a named face
    (top/bottom/front/back/left/right) of an existing object in the shared
    session document, using local 2D (``u``, ``v``) coordinates on that
    face instead of a 3D ``App.Placement``/``App.Rotation`` the caller would
    otherwise have to compute by hand — this is the whole point: FreeCAD
    resolves the face's actual world-space position/orientation from the
    object's REAL geometry (``_face_axes`` + the target's own
    ``Shape.BoundBox``), not a value guessed ahead of time.

    ``operation="cut"`` subtracts the shape into ``object_name`` (a hole/
    pocket/slot); ``operation="add"`` unions it onto the surface (a boss/
    tab). Per the topology rule this composition implies: ``object_name``
    is CONSUMED by the boolean step exactly like any other
    ``perform_freecad_boolean`` call — only the returned result name is
    valid for a later call, never ``object_name`` again.

    Restricted to the 6 axis-aligned bounding-box faces of a genuinely
    box-like (prismatic) object — the generated script itself verifies the
    resolved face point actually lies on a flat face of the real geometry
    (not just the bbox) and fails clearly rather than silently cutting/
    adding nothing on a curved surface (e.g. the side of a cylinder).

    Migrated to the Universal CAD IR's Hierarchical/Composite node via the
    same shared ``_execute_ir_tool`` pipeline every other tool_id in this
    module goes through — the profile-build and boolean steps render and
    execute as ONE FreeCAD script (one document open/save, two
    ``doc.recompute()``s) instead of the previous two separate
    ``_run_freecad_script`` round trips bridged by a string object-name
    handoff (``result.get("resolved_name")``).
    """
    dims = {
        "face": face,
        "shape": (shape or "").strip().lower(),
        "u": float(u) if isinstance(u, (int, float)) else u,
        "v": float(v) if isinstance(v, (int, float)) else v,
        "extent": extent,
        "operation": (operation or "").strip().lower(),
    }
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_feature_on_face",
            name=name or "Feature",
            type="Part::Cut" if dims["operation"] == "cut" else "Part::MultiFuse",
            dimensions=dims,
        )

    try:
        result, steps, session_path = _execute_ir_tool(
            "create_freecad_feature_on_face",
            object_name=object_name, face=face, shape=shape, u=u, v=v, extent=extent, operation=operation,
            radius=radius, width=width, length=length, name=name,
        )
    except ValueError as exc:
        return _error(f"create_feature_on_face: {exc}")

    dims["extent"] = steps[0]["extent"]
    dims["u"] = steps[0]["u"]
    dims["v"] = steps[0]["v"]
    if not result["ok"]:
        return _error(f"create_feature_on_face failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or steps[-1]["name"],
        type=steps[-1]["feature_type"],
        operation=steps[-1]["operation"],
        bounding_box=result.get("bounding_box"),
        geometry=result.get("geometry"),
        dimensions=dims,
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

_EDGE_FEATURE_TYPE: dict[str, str] = {"fillet": "Part::Fillet", "chamfer": "Part::Chamfer"}

_DEFAULT_EDGE_NAME: dict[str, str] = {"fillet": "Fillet", "chamfer": "Chamfer"}

def apply_edge_operation(
    operation: str,
    target_object: str,
    value: float,
    face_centroid: tuple[float, float, float] | None = None,
    name: str | None = None,
) -> str:
    """Round (``"fillet"``) or bevel (``"chamfer"``) the edges of an object
    already in the shared ``Session_Active.FCStd`` document, looked up by
    NAME — same session-document/name-collision story as ``apply_boolean``/
    ``modify_parameter``.

    Without ``face_centroid``, every edge of the object gets the operation
    (a global fillet/chamfer, ``value`` mm). With ``face_centroid`` —
    typically the active canvas selection's clicked-face centroid — only
    the edges bounding the face nearest that point are targeted, found
    against FreeCAD's exact BRep geometry (no raycasting against the
    tessellated display mesh).

    Migrated to the Universal CAD IR's "edge_operation" kind via the shared
    ``_execute_ir_tool`` pipeline (``doc_mode="session"``, the default) —
    ``target_object`` may be a ``Part::Compound`` left behind by an earlier
    boolean chain; the generated script unwraps to the first real solid and
    heals stray coplanar seams (``removeSplitter``) before any face/edge
    matching, so filleting the result of a multi-cut part works the same as
    filleting a single primitive. ``_EDGE_OP_WHOLE_SCRIPT``/
    ``_EDGE_OP_FACE_SCRIPT`` (the old one-object-one-file f-string templates)
    are retired, not kept as a parallel dead path.
    """
    op = (operation or "").strip().lower()
    if op not in _EDGE_FEATURE_TYPE:
        return _error(f"apply_edge_operation: unknown operation '{operation}' — must be fillet or chamfer")
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
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error(
            "apply_edge_operation: no session document yet — create objects with create_box/"
            "create_cylinder/insert_standard_part first"
        )
    centroid = (
        (float(face_centroid[0]), float(face_centroid[1]), float(face_centroid[2])) if face_targeted else None
    )
    result, steps, session_path = _execute_ir_tool(
        "perform_freecad_edge_operation", name=resolved_name, feature_type=feature_type,
        target_object=target_object, value=value_f, centroid=centroid,
    )
    if not result["ok"]:
        return _error(f"apply_edge_operation failed: {result['error']}")
    return _ok(
        name=result.get("resolved_name") or resolved_name,
        type=feature_type,
        operation=op,
        face_targeted=face_targeted,
        bounding_box=result.get("bounding_box"),
        geometry=result.get("geometry"),
        path=str(session_path),
        gui_shown=_auto_show(session_path),
    )

_VECTOR_PARAMETER_NAMES = frozenset({"placement", "placement.base"})

def modify_parameter(
    target_object: str,
    parameter_name: str,
    new_value: float | Sequence[float],
    yaw: float | None = None,
    pitch: float | None = None,
    roll: float | None = None,
) -> str:
    """Change a single dimensional property (e.g. ``"Height"``, ``"Radius"``)
    on an object already in the shared ``Session_Active.FCStd`` document, by
    NAME — in place, so the object's parametric history/name are preserved
    across the edit.

    ``parameter_name`` of ``"Placement"`` or ``"Placement.Base"`` is special:
    it moves (and optionally rotates) the object, so ``new_value`` must be a
    3-number ``[x, y, z]`` (mm) vector instead of a single float — moves
    ``Placement.Base`` to that point. Rotation is expressed ONLY via the
    separate ``yaw``/``pitch``/``roll`` parameters (Euler angles in
    DEGREES) — never packed into ``new_value`` (no raw quaternion/matrix/
    6-number array can reach this function):

    - All three omitted (``None``): PRESERVES the object's current
      ``Placement.Rotation`` (a translate never silently discards prior
      orientation).
    - Any one given: REPLACES the whole rotation with a fresh
      ``FreeCAD.Rotation(yaw, pitch, roll)`` (omitted axes default to
      ``0.0``) — the exact Euler convention FreeCAD's own Placement dialog
      uses (confirmed live: ``Rotation(90, 0, 0).toEuler() ==
      (90.0, 0.0, 0.0)`` — the constructor takes degrees directly, no
      radian conversion needed or wanted). Needed for kinematic assemblies
      (URDF joints, assembly mates) where a linked part must be moved AND
      oriented in one call, not just translated.

    Refuses outright (never silently no-ops) if ``target_object`` was
    anchored via ``anchor_assembly_root`` — see that function's own
    docstring — or, for the ``Placement``/``Placement.Base`` branch, if it
    is currently locked by an active ``apply_assembly_constraint`` mate
    (``DanaConstrained``); re-call ``apply_assembly_constraint`` instead of
    overriding a mated part's ``Placement`` directly.
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
            components = [float(component) for component in new_value]
        except (TypeError, ValueError):
            return _error(
                f"modify_parameter: {param} new_value must be a 3-number [x, y, z] vector, "
                f"got {new_value!r}"
            )
        if len(components) != 3:
            return _error(
                f"modify_parameter: {param} new_value must have exactly 3 elements [x, y, z] "
                f"(mm) — pass rotation via the separate yaw/pitch/roll parameters instead of "
                f"packing it into this vector, got {len(components)}"
            )
        x, y, z = components
        if yaw is None and pitch is None and roll is None:
            rotation: tuple[float, float, float] | None = None
            result_value: float | list[float] = [x, y, z]
        else:
            try:
                yaw_f = float(yaw) if yaw is not None else 0.0
                pitch_f = float(pitch) if pitch is not None else 0.0
                roll_f = float(roll) if roll is not None else 0.0
            except (TypeError, ValueError):
                return _error("modify_parameter: yaw/pitch/roll must all be numbers, got "
                              f"yaw={yaw!r} pitch={pitch!r} roll={roll!r}")
            rotation = (yaw_f, pitch_f, roll_f)
            result_value = [x, y, z, yaw_f, pitch_f, roll_f]
        if is_dry_run_enabled():
            return _dry_run_result(
                "modify_parameter", path=str(session_path), parameter_name=param, new_value=result_value
            )
        result, steps, session_path = _execute_ir_tool(
            "modify_placement", target_object=target_object, x=x, y=y, z=z, rotation=rotation,
        )
        if not result["ok"]:
            return _error(f"modify_parameter failed: {result['error']}")
        return _ok(
            name=target_object,
            path=str(session_path),
            parameter_name=param,
            new_value=result_value,
            bounding_box=result.get("bounding_box"),
            geometry=result.get("geometry"),
            gui_shown=_auto_show(session_path),
        )

    # Scalar case — migrated to the Universal CAD IR's "modify_parameter" kind.
    try:
        value_f = float(new_value)
    except (TypeError, ValueError):
        return _error(f"modify_parameter: new_value must be a number, got {new_value!r}")
    if is_dry_run_enabled():
        return _dry_run_result(
            "modify_parameter", path=str(session_path), parameter_name=param, new_value=value_f
        )
    result, steps, session_path = _execute_ir_tool(
        "modify_freecad_parameter", target_object=target_object, parameter_name=param, new_value=value_f,
    )
    if not result["ok"]:
        return _error(f"modify_parameter failed: {result['error']}")
    return _ok(
        name=target_object,
        path=str(session_path),
        parameter_name=param,
        new_value=value_f,
        bounding_box=result.get("bounding_box"),
        geometry=result.get("geometry"),
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
# A boolean-chain result can be a Part.Compound wrapping the real solid —
# CenterOfMass (and the other mass properties below) aren't defined on a
# bare Compound, so unwrap to the real solid first.
if shape.ShapeType == 'Compound' and shape.Solids:
    shape = shape.Solids[0]
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

_QUERY_TOPOLOGY_SCRIPT = ("""\
import FreeCAD as App
import json

""" + _RESOLVE_OBJECT_SNIPPET + _SESSION_OPEN_SNIPPET + """\
obj = resolve_object(doc, {part_name!r})
if obj is None:
    raise RuntimeError("Object not found: " + {part_name!r})
shape = getattr(obj, "Shape", None)
if shape is None or shape.isNull():
    raise RuntimeError("'" + {part_name!r} + "' has no usable geometry (empty Shape).")
# A boolean-chain result can be a Part.Compound wrapping the real solid --
# same unwrap inspect_spatial_properties already does before touching
# mass-property-dependent fields.
if shape.ShapeType == "Compound" and shape.Solids:
    shape = shape.Solids[0]

_faces = []
for _i, _face in enumerate(shape.Faces):
    # Curved-Face Normal Guard: a face's normal is only a single,
    # well-defined vector when the face is PLANAR -- the exact same
    # "Part::GeomPlane" check _require_planar_face uses elsewhere in this
    # module (verified live against this FreeCAD build). For any curved
    # face (cylindrical, conical, toroidal, B-Spline, ...), normalAt(u, v)
    # only returns the normal AT one arbitrary sampled point, not "the"
    # normal of the whole face -- reporting that back as if it were
    # face-wide would let the caller silently reason about the wrong
    # direction. `normal` is `None` (serializes to JSON `null`) with an
    # explicit `normal_warning` instead.
    _is_planar = _face.Surface.TypeId == "Part::GeomPlane"
    if _is_planar:
        _n = _face.normalAt(0, 0)
        _normal = [_n.x, _n.y, _n.z]
        _normal_warning = None
        # UV Axis Grounding: exposes the SAME (u, v) parametric quantities
        # apply_assembly_constraint's own Fit Guard already computes
        # internally (_face_alignment_delta's u_available/v_available,
        # engine.py) so a caller can pre-check "is this face big enough,
        # and which of u/v is the long axis" BEFORE ever calling
        # apply_assembly_constraint, instead of discovering it via a
        # rejected call. u_min/u_max/v_min/v_max are the face's own
        # TRIMMED parametric bounds (Face.ParameterRange -- verified live,
        # same fact _face_alignment_delta's own comment already documents:
        # this is the face's real bounded rectangle, not the underlying
        # infinite plane), and physically scaled in mm for a planar face
        # (Geom_Plane's (u, v) parametrization uses unit-length basis
        # vectors, so a 1-unit change in u/v is a literal 1mm move along
        # the plane's own XDirection/YDirection -- the same assumption
        # _face_alignment_delta's own fit-guard comparison already relies
        # on). u_direction_vector/v_direction_vector are those basis
        # vectors in WORLD space -- normal-agnostic (correct for an
        # X-normal, Y-normal, Z-normal, or arbitrarily rotated face alike)
        # -- letting the caller compare against a chassis's own elongation
        # axis (from get_bounding_box/inspect_spatial_properties) to
        # deterministically pick u vs v for a longitudinal layout, rather
        # than guessing.
        #
        # Derived via face.valueAt(u, v) finite-differencing across the
        # face's own ParameterRange corners, NOT via any Part.Plane
        # attribute -- two prior attempts at a direct accessor
        # (Surface.Position.Rotation.multVec(...), then Surface.XAxis/
        # Surface.YAxis) both either went untested or failed live
        # ("'Part.Plane' object has no attribute 'XAxis'", confirmed via a
        # real query_topology call against this exact FreeCAD build).
        # face.valueAt is already proven live in this exact codebase
        # (_face_alignment_delta uses it the same way), so this reuses
        # ONLY already-verified API surface instead of guessing a third
        # attribute name. Exact, not approximate, for a planar face:
        # Geom_Plane's (u, v) parametrization is affine (valueAt is linear
        # in u and v), so the vector between valueAt at the two ends of
        # EITHER axis, holding the other fixed, is that axis's constant
        # direction everywhere on the face -- no epsilon-step numerical
        # error the way this same technique would have on a curved
        # surface (never done here; gated to _is_planar).
        u_min, u_max, v_min, v_max = _face.ParameterRange
        u_span_physical_mm = u_max - u_min
        v_span_physical_mm = v_max - v_min
        _origin_pt = _face.valueAt(u_min, v_min)
        _u_delta = _face.valueAt(u_max, v_min) - _origin_pt
        _v_delta = _face.valueAt(u_min, v_max) - _origin_pt
        _u_delta.normalize()
        _v_delta.normalize()
        u_direction_vector = [_u_delta.x, _u_delta.y, _u_delta.z]
        v_direction_vector = [_v_delta.x, _v_delta.y, _v_delta.z]
    else:
        _normal = None
        _normal_warning = (
            "face is not planar (" + _face.Surface.TypeId + ") -- a single normal vector is not "
            "well-defined across a curved surface; use a Concentric-style axis/center reference "
            "(apply_assembly_constraint's 'Concentric') instead of this face's normal"
        )
        # Same non-planar exclusion as normal/normal_warning above: a
        # curved face's own (u, v) parametrization is frequently angular
        # (e.g. a cylinder's u is a radians sweep, not a linear mm span)
        # and has no single constant in-plane direction -- reporting a
        # fabricated "physical mm span"/"direction vector" here would be
        # actively misleading, not just incomplete, so all four are None.
        u_span_physical_mm = None
        v_span_physical_mm = None
        u_direction_vector = None
        v_direction_vector = None
    _com = _face.CenterOfMass
    _faces.append({{
        "face_index": "Face" + str(_i + 1),
        "area": _face.Area,
        "is_planar": _is_planar,
        "surface_type": _face.Surface.TypeId,
        "normal": _normal,
        "normal_warning": _normal_warning,
        "centroid": [_com.x, _com.y, _com.z],
        "u_span_physical_mm": u_span_physical_mm,
        "v_span_physical_mm": v_span_physical_mm,
        "u_direction_vector": u_direction_vector,
        "v_direction_vector": v_direction_vector,
    }})

print("{marker}_TOPOLOGY " + json.dumps(_faces))
print("{marker}_NAME " + obj.Name)
print("{marker} path=" + _session_path)
""")

def query_topology(part_name: str) -> str:
    """Read-only: per-face topology of ``part_name`` (a previously-created
    object in the shared session document) -- for every face on its
    ``Shape``: ``face_index`` ("Face3", 1-based, matching FreeCAD's own
    element numbering used elsewhere in this module e.g.
    ``apply_assembly_constraint``'s ``part1_element``/``part2_element``),
    ``area``, ``is_planar``, ``surface_type``, ``centroid``, ``normal``
    (``None`` for a curved face -- see ``normal_warning`` on that face),
    and — for a planar face only, ``None`` for a curved one, same reasoning
    as ``normal`` -- ``u_span_physical_mm``/``v_span_physical_mm`` (that
    face's own real, trimmed parametric extent along each axis, in mm —
    the SAME quantity ``apply_assembly_constraint``'s Fit Guard already
    enforces internally, exposed here so a caller can pre-check "is this
    face big enough" before that call, rather than discovering it via a
    rejection) and ``u_direction_vector``/``v_direction_vector`` (those two
    axes' real WORLD-space directions, normal-agnostic — correct whether
    the face's normal points along X, Y, Z, or anywhere else). Comparing
    these two direction vectors against a chassis's own elongation axis
    (from ``get_bounding_box``/``inspect_spatial_properties``) is what lets
    a caller deterministically choose ``apply_assembly_constraint``'s
    ``uv_tensor`` axis (``u`` vs ``v``) for a longitudinal layout instead
    of guessing.
    Never saves, so — like ``get_bounding_box``/``inspect_spatial_properties``
    — it never needs the HITL approval gate the create_*/apply_* mutators do.

    Intended as a "look before you leap" query ahead of
    ``apply_assembly_constraint``: lets the caller mathematically deduce
    opposite faces (matching areas, anti-parallel normals), the largest
    flat face (a candidate mounting surface), or top/bottom (by centroid Z)
    from real BRep data instead of guessing a Face index blind.

    Curved-face limitation (confirmed live against this exact FreeCAD
    build, the same finding ``_require_planar_face`` already encodes for
    ``apply_assembly_constraint``): a face's normal is only a single,
    well-defined vector when the face is PLANAR. For a cylindrical/
    conical/toroidal/B-Spline face, ``normal`` is ``None`` with an
    explicit ``normal_warning`` explaining why, rather than a normal
    sampled at one arbitrary point pretending to represent the whole face.

    Face indices reflect THIS call's snapshot of ``part_name``'s CURRENT
    ``Shape`` only -- OCC's own face numbering is not guaranteed stable
    across a later boolean/fillet/pattern operation on this part, so a
    stale index from an earlier call should not be trusted without
    re-querying.
    """
    part = (part_name or "").strip()
    if not part:
        return _error("query_topology requires part_name")
    if is_dry_run_enabled():
        return _dry_run_result("query_topology", part_name=part, faces=[])
    session_path = _session_document_path()
    if not session_path.is_file():
        return _error(
            "query_topology: no session document yet — create objects with create_box/"
            "create_cylinder first"
        )
    script = _QUERY_TOPOLOGY_SCRIPT.format(
        part_name=part,
        session_path=str(session_path),
        session_doc_name=_SESSION_DOCUMENT_NAME,
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"query_topology failed: {result['error']}")
    faces = _extract_topology(result["stdout"])
    if faces is None:
        return _error("query_topology: failed to parse topology output")
    # Deliberately "part_name", never "name": react_dispatch.dispatch_tool_call's
    # generic success path registers ANY payload with both "name" and "path"
    # keys into the object registry/topology DAG as if this call had just
    # PRODUCED a new object (see get_bounding_box/inspect_spatial_properties's
    # own matching omission there) -- a pure read-only query must not trip that,
    # or a plain query_topology call would spuriously re-touch this part's DAG
    # node on every single call.
    return _ok(
        part_name=result.get("resolved_name") or part,
        path=str(session_path),
        face_count=len(faces),
        faces=faces,
    )

_ALIGNMENT_TYPES = frozenset({"top_center", "bottom_center", "flush_left", "flush_right"})

_ALIGN_APPLY_SCRIPT = """\
import FreeCAD as App

""" + _RESOLVE_OBJECT_SNIPPET + """\
doc = App.openDocument({source_path!r})
{lookup}if getattr(obj, "DanaAnchored", False):
    raise RuntimeError(
        "'" + obj.Name + "' is anchored (anchor_assembly_root) and cannot be moved by "
        "align_freecad_objects/create_assembly_mate -- it is this assembly's fixed reference frame."
    )
if getattr(obj, "DanaConstrained", False):
    raise RuntimeError(
        "'" + obj.Name + "' is locked by an active assembly constraint (apply_assembly_constraint) "
        "and cannot be moved by align_freecad_objects/create_assembly_mate -- re-call "
        "apply_assembly_constraint against the correct face/edge instead of overriding its "
        "Placement directly."
    )
obj.Placement.Base = obj.Placement.Base + App.Vector({dx}, {dy}, {dz})
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

    Refuses outright (never silently no-ops) if the source object was
    anchored via ``anchor_assembly_root`` — same ``DanaAnchored`` guard as
    ``position_assembly_part``/``modify_parameter``/
    ``apply_assembly_constraint``, enforced here via ``_ALIGN_APPLY_SCRIPT``
    (shared with ``create_assembly_mate``, which gets the same guard for
    free).
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

    Refuses outright if the moving object was anchored via
    ``anchor_assembly_root`` — same ``DanaAnchored`` guard as
    ``align_objects``, since both share ``_ALIGN_APPLY_SCRIPT`` verbatim.
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

_PIPE_ARC_BEND_RADIUS_MULTIPLIER = 3.0

_PIPE_ARC_MIN_BEND_RADIUS = 20.0

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

_HELIX_SCRIPT = """\
import FreeCAD as App

doc = App.newDocument("DanaModel")

path = doc.addObject("Part::Helix", "Path")
path.Pitch = {pitch}
path.Height = {height}
path.Radius = {coil_radius}
path.Angle = 0.0

profile = doc.addObject("Part::Circle", "Profile")
profile.Radius = {pipe_radius}
profile.Placement = App.Placement(App.Vector({coil_radius}, 0.0, 0.0), App.Rotation(App.Vector(1, 0, 0), 90))

doc.recompute()

obj = doc.addObject("Part::Sweep", {name!r})
obj.Sections = [profile]
obj.Spine = (path, [])
obj.Solid = True
obj.Frenet = True
doc.recompute()
obj.Placement = App.Placement(App.Vector({px}, {py}, {pz}), App.Rotation({angle_offset}, 0.0, 0.0))
doc.recompute()
doc.saveAs({out_path!r})
""" + _BBOX_PRINT + """\
print("{marker} path=" + {out_path!r})
"""

def create_helix(
    coil_radius: float,
    pitch: float,
    height: float,
    pipe_radius: float,
    name: str = "Helix",
    placement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angle_offset: float = 0.0,
) -> str:
    """Sweeps a circular cross-section (``pipe_radius`` mm) along a
    cylindrical ``Part::Helix`` path (``coil_radius``/``pitch``/``height``
    mm) into a solid ``Part::Sweep`` coil — wound coils, springs, helical
    stands. ``Frenet = True`` (same as ``create_pipe``'s own arc case)
    reorients the profile to the path's own computed frame at every point,
    so the profile's initial placement only needs to sit at the helix's
    actual start point (``coil_radius``, 0, 0) with a non-degenerate
    orientation, not a perfectly-tangent one.

    Same own-document-per-call precedent as ``create_pipe`` — not yet
    migrated to the shared session document (see the FreeCAD session
    document migration note: create_box/cylinder/insert_standard_part/
    apply_boolean/modify_parameter share ONE Session_Active.FCStd; this
    tool doesn't yet).

    ``angle_offset`` degrees is a rigid rotation about the GLOBAL Z axis
    applied to the finished coil — NOT FreeCAD's own Part::Helix ``Angle``
    property (which tapers a helix into a cone; left at 0.0/cylindrical
    here, unrelated to this parameter) — so several coils can be fanned
    evenly around one shared hub without overlapping.
    """
    try:
        coil_radius_f = float(coil_radius)
        pitch_f = float(pitch)
        height_f = float(height)
        pipe_radius_f = float(pipe_radius)
        angle_offset_f = float(angle_offset)
    except (TypeError, ValueError):
        return _error("create_helix: coil_radius/pitch/height/pipe_radius/angle_offset must be numbers")
    if coil_radius_f <= 0 or pitch_f <= 0 or height_f <= 0 or pipe_radius_f <= 0:
        return _error("create_helix: coil_radius, pitch, height, and pipe_radius must all be positive numbers")
    if pipe_radius_f >= coil_radius_f:
        return _error("create_helix: pipe_radius must be smaller than coil_radius")
    placement = (float(placement[0]), float(placement[1]), float(placement[2]))

    dims = {
        "coil_radius": coil_radius_f,
        "pitch": pitch_f,
        "height": height_f,
        "pipe_radius": pipe_radius_f,
        "angle_offset": angle_offset_f,
    }
    if is_dry_run_enabled():
        return _dry_run_result(
            "create_helix", name=name, type="Part::Sweep", dimensions=dims, placement=list(placement)
        )

    out_path = _output_path(name, ext="FCStd")
    script = _HELIX_SCRIPT.format(
        coil_radius=coil_radius_f,
        pitch=pitch_f,
        height=height_f,
        pipe_radius=pipe_radius_f,
        name=name,
        px=placement[0],
        py=placement[1],
        pz=placement[2],
        angle_offset=angle_offset_f,
        out_path=str(out_path),
        marker=_OK_MARKER,
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"create_helix failed: {result['error']}")
    return _ok(
        name=name,
        type="Part::Sweep",
        bounding_box=result.get("bounding_box"),
        dimensions=dims,
        placement=list(placement),
        path=str(out_path),
        gui_shown=_auto_show(out_path),
    )

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
    ``.glb`` (GLTF Binary) mesh file — the hand-off format for the viewer,
    matching ``dana.platform.mock``'s own ``_mesh_output_path`` default
    (name kept as ``export_mesh_stl`` for every existing caller/tool_id;
    only the actual output format changed, same "legacy name, new
    behavior" precedent that function's own docstring already documents).

    FreeCAD's ``Mesh`` module has no glTF writer, so this drives FreeCADCmd
    to tessellate a throwaway intermediate ``.stl`` first, then converts
    that to ``.glb`` in THIS process via ``trimesh`` (already a hard
    dependency — see ``dana.platform.mock``'s identical use). Both the
    intermediate ``.stl`` and the ``.glb`` conversion's own output are
    written under distinctive ``__tmp_``/``__glbtmp_`` per-call names
    (``dana.api.cad._is_throwaway_temp_file`` already filters these out of
    every artifact listing) and the final ``.glb`` is produced by one
    atomic ``Path.replace()`` — Torn-Write: a concurrent reader (the
    download route, or the auto-mesh-export hook's own immediate re-read)
    can only ever observe either the complete previous file at this name
    or the complete new one, never a partially-written one.

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

    resolved_name = _safe_name(name or source.stem)
    unique = uuid.uuid4().hex[:8]
    tmp_stl_path = _output_path(f"{resolved_name}__tmp_{unique}", ext="stl")
    script = _EXPORT_STL_SCRIPT.format(
        source_path=str(source),
        out_path=str(tmp_stl_path),
        marker=_OK_MARKER,
        lookup=_object_lookup_snippet(target_object=target_object) if target_object else "",
        export_targets="[obj]" if target_object else "list(doc.Objects)",
    )
    result = _run_freecad_script(script)
    if not result["ok"]:
        return _error(f"export_mesh_stl failed: {result['error']}")

    glb_path = _output_path(resolved_name, ext="glb")
    tmp_glb_path = _output_path(f"{resolved_name}__glbtmp_{unique}", ext="glb")
    try:
        import trimesh

        mesh = trimesh.load(str(tmp_stl_path), force="mesh")
        mesh.export(str(tmp_glb_path))
        tmp_glb_path.replace(glb_path)
    except Exception as exc:  # noqa: BLE001 — surface as a normal tool failure, not a crash
        return _error(f"export_mesh_stl: STL->GLB conversion failed: {exc}")
    finally:
        tmp_stl_path.unlink(missing_ok=True)
        tmp_glb_path.unlink(missing_ok=True)  # no-op once replace() above succeeds; cleans up on failure

    return _ok(op="export_mesh_stl", source_path=str(source), path=str(glb_path))

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
        out_path = _export_dir() / f"{safe_name}.{ext}"
        return _dry_run_result("export_model", format=fmt, path=str(out_path), target_count=len(paths))

    out_path = _export_dir() / f"{safe_name}.{ext}"
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
    "create_helix",
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


# --- restored from HEAD (excluded refactor did not touch these) ---

_COMMON_INSTALL_GLOBS: tuple[str, ...] = (
    r"C:\Program Files\FreeCAD*\bin\FreeCADCmd.exe",
    r"C:\Program Files (x86)\FreeCAD*\bin\FreeCADCmd.exe",
)

_EXPORT_STL_SCRIPT = """\
import FreeCAD as App
import Mesh

""" + _RESOLVE_OBJECT_SNIPPET + """\
doc = App.openDocument({source_path!r})
{lookup}Mesh.export({export_targets}, {out_path!r})
print("{marker} path=" + {out_path!r})
"""

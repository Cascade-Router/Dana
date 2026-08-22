"""Shared ReAct tool-dispatch core — UI-agnostic.

Originally extracted from the now-deleted Gradio ``dana.ui.unified_app``
(replaced by the headless ``dana.api.server`` FastAPI/WebSocket server + the
Tauri/React frontend) so the dispatch pipeline (intent parsing, tool
registry, driver/plugin introspection) lives in exactly one place, reusable
by any future frontend. Every tool dispatch still goes through
``dana.platform.get_control_plane()`` / ``get_cad_engine()`` — never Win32 or
FreeCADCmd directly.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from dana.core.context_manager import prune_message_history
from dana.core.model_provider import ModelProvider, tool_calling_provider
from dana.core.openai_tool_bridge import _MAX_REASONABLE_RETRY_AFTER_SEC
from dana.core.skill_loader import delete_skill, load_user_skills, read_skill_source, save_skill
from dana.core.tool_retrieval import narrow_tool_ids_by_query
from dana.paths import CAPTURES_DIR
from dana.platform import factory as platform_factory
from dana.platform import get_cad_engine, get_control_plane
from dana.plugins.freecad.call_log import CadCallLog
from dana.plugins.freecad.error_digest import digest_error
from dana.plugins.memory.core_memory import format_core_memory_for_prompt, write_core_memory
from dana.plugins.os.background_services import list_background_services as _fs_list_background_services
from dana.plugins.os.background_services import start_background_service as _fs_start_background_service
from dana.plugins.os.background_services import stop_background_service as _fs_stop_background_service
from dana.plugins.os.desktop_vision import analyze_desktop_screen as _os_analyze_desktop_screen
from dana.plugins.os.file_system import edit_file as _fs_edit_file
from dana.plugins.os.file_system import list_directory as _fs_list_directory
from dana.plugins.os.file_system import read_file as _fs_read_file
from dana.plugins.os.file_system import search_files as _fs_search_files
from dana.plugins.os.file_system import write_file as _fs_write_file
from dana.plugins.os.process_manager import execute_terminal_command as _fs_execute_terminal_command
from dana.plugins.os.process_manager import run_python_script as _fs_run_python_script
from dana.plugins.planning.task_board import create_plan as _tb_create_plan
from dana.plugins.planning.task_board import get_active_plan as _tb_get_active_plan
from dana.plugins.planning.task_board import mark_task_completed as _tb_mark_task_completed
from dana.plugins.plugin_manager import discover_plugin_dirs, load_all_plugins
from dana.plugins.vision.image_analysis import analyze_workspace_image as _vision_analyze_workspace_image
from dana.plugins.web.research import read_webpage as _web_read_webpage
from dana.plugins.web.research import search_web as _web_search_web
from dana.security.dry_run import is_dry_run_enabled
from dana.tools.cad_vision import analyze_cad_blueprint, capture_cad_viewport
from dana.tools.schema import (
    ToolCall,
    ToolSpec,
    load_tool_registry,
    openai_tools_schema,
    to_openai_function_schema,
)
from dana.tools.schema_minify import minify_tool_schemas


class ToolResult:
    """Dispatch outcome for one ``ToolCall`` — mirrors the broker's shape
    without importing ``dana.tools.broker`` (heavier than callers need)."""

    __slots__ = ("tool_id", "ok", "payload", "message", "duration_ms")

    def __init__(self, tool_id: str, ok: bool, payload: dict[str, Any], message: str, duration_ms: int):
        self.tool_id = tool_id
        self.ok = ok
        self.payload = payload
        self.message = message
        self.duration_ms = duration_ms


# ---------------------------------------------------------------------------
# Tool registry — every handler receives (arguments, cad_engine, control_plane)
# and returns a plain result dict. This is where "dispatch through the
# abstract drivers, not Win32/FreeCADCmd directly" is actually enforced.
# ---------------------------------------------------------------------------


def _extract_placement(args: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(args.get("placement_x", 0.0) or 0.0),
        float(args.get("placement_y", 0.0) or 0.0),
        float(args.get("placement_z", 0.0) or 0.0),
    )


def _tool_create_box(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    return engine.create_box(
        float(args.get("length", 40)),
        float(args.get("width", 25)),
        float(args.get("height", 15)),
        name=str(args.get("name") or "Box"),
        placement=_extract_placement(args),
    )


def _tool_create_cylinder(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    return engine.create_cylinder(
        float(args.get("radius", 10)),
        float(args.get("height", 30)),
        name=str(args.get("name") or "Cylinder"),
        placement=_extract_placement(args),
    )


def _tool_resync_workspace(_args: dict[str, Any], _engine: Any, cp: Any) -> dict[str, Any]:
    return cp.resync_workspace()


def _tool_get_active_display(_args: dict[str, Any], _engine: Any, cp: Any) -> dict[str, Any]:
    return cp.get_active_display()


def _tool_prevent_focus_steal(_args: dict[str, Any], _engine: Any, cp: Any) -> dict[str, Any]:
    return cp.prevent_focus_steal()


def _tool_system_state(_args: dict[str, Any], engine: Any, cp: Any) -> dict[str, Any]:
    return {"ok": True, **driver_state(engine, cp)}


def _tool_check_plugin_registry(_args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    return {"ok": True, **plugin_registry_view()}


def _tool_load_capability(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    """Autonomous semantic routing: unlocks a capability domain's tools —
    for as long as it (or one of its own tools) keeps getting used; see P1's
    per-session decay below. This handler is stateless — it only reports
    which tools a domain unlocks; dana.api.server is what actually stamps
    the domain into session["capability_unlocked_at_turn"] once
    dispatch_tool_call reports success (see server.py's
    _execute_and_continue/_touch_capability_domains), since this module has
    no session object of its own to mutate. Every later turn's tool schema
    is built from active_plugins UNION whatever hasn't yet decayed out of
    that dict (dana.api.server._effective_capabilities), so the very next
    turn already sees whatever this call unlocked.
    """
    domain = str(args.get("domain") or "").strip()
    # "freecad" defaults to the trimmed essential set (see
    # _FREECAD_ESSENTIAL_TOOL_IDS) rather than all 24 FreeCAD tools — the
    # full set is heavy enough (~30KB of schemas) to blow a free-tier cloud
    # model's tokens-per-minute budget on the very next turn. This ONLY
    # affects the agent's own autonomous load_capability calls: the
    # frontend's explicit CAD-plugin activation goes through
    # dana.api.server's active_plugins union instead, never through this
    # function, so a real CAD-panel session is unaffected. "freecad_full"
    # is the explicit escalation domain — the agent can call this again
    # with domain="freecad_full" any time a task genuinely needs the
    # heavier tools (patterns, assembly mates, blueprints, standard parts,
    # engineering-standard lookups, camera control).
    resolved_domain = "freecad_essential" if domain == "freecad" else domain
    unlocked = _CAPABILITY_TOOL_IDS.get(resolved_domain)
    if unlocked is None:
        return {
            "ok": False,
            "error": f"Unknown capability domain {domain!r}. Available: {sorted(_CAPABILITY_TOOL_IDS)}",
        }
    tools = sorted(unlocked)
    message = f"Loaded '{resolved_domain}' — newly available tools: {', '.join(tools)}."
    if resolved_domain == "freecad_essential":
        message += (
            " This is the essential FreeCAD set (basic shapes, one boolean op, one edge op, "
            "parameter tweaks, bounding-box checks, export). If this task needs a heavier tool "
            "(patterns, assembly mates, blueprints, standard parts, engineering-standard lookup, "
            "camera control), call load_capability again with domain='freecad_full'."
        )
    return {
        "ok": True,
        "domain": resolved_domain,
        "unlocked_tools": tools,
        "message": message,
    }


def _tool_update_core_memory(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    """Writes ``section``/``content`` into the on-disk Core Memory file (see
    dana.plugins.memory.core_memory) — the agent's own persistent notes on
    user preferences, an active project's constraints, or a learned
    workflow, surviving a server restart. Always available (in
    _CORE_TOOL_IDS below), never HITL-gated: unlike write_file, this can't
    reach an arbitrary path or overwrite anything outside its own one
    fixed, low-risk JSON file.
    """
    return write_core_memory(str(args.get("section") or ""), str(args.get("content") or ""))


# Task Planner / Executive Function — create_plan/mark_task_completed
# (dana.plugins.planning.task_board). Always core (see _CORE_TOOL_IDS
# below): a long-horizon goal can span any plugin/capability combination,
# so the agent must always be able to lay out and update its own plan
# regardless of what else is currently active. NEVER mutating (its tools.json
# entry declares "read_only": true) — this only ever updates this module's own
# in-memory scratchpad, never a file, a process, or anything the user
# would need to review/approve before it happens.
def _tool_create_plan(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    raw_tasks = args.get("tasks")
    tasks = [str(t) for t in raw_tasks] if isinstance(raw_tasks, list) else []
    return _tb_create_plan(str(args.get("objective") or ""), tasks)


def _tool_mark_task_completed(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    try:
        task_id = int(args.get("task_id"))
    except (TypeError, ValueError):
        return {"ok": False, "error": f"task_id must be an integer, got: {args.get('task_id')!r}"}

    raw_next_task_id = args.get("next_task_id")
    next_task_id: int | None = None
    if raw_next_task_id is not None:
        try:
            next_task_id = int(raw_next_task_id)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"next_task_id must be an integer, got: {raw_next_task_id!r}"}

    return _tb_mark_task_completed(task_id, next_task_id)


# Autonomous Skill Acquisition — save_new_skill (dana.core.skill_loader).
# Always core (see _CORE_TOOL_IDS below): the agent must always be able to
# teach itself a new tool, regardless of which plugin/capability is active.
# ALWAYS mutating (no "read_only": true in its tools.json entry) — writes new Python source to
# disk AND immediately expands what the agent can subsequently execute
# with no further approval, so the user must see and approve the exact
# source before it's ever saved.
def _tool_save_new_skill(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    """Writes ``skill_name``/``python_code``/``schema`` to
    ``skills/<skill_name>.py`` (dana.core.skill_loader.save_skill) and
    immediately hot-reloads the "user_skills" capability domain
    (``refresh_user_skills``) so the new tool is dispatchable in the very
    next ReAct turn — no server restart needed.
    """
    skill_name = str(args.get("skill_name") or "").strip()
    python_code = str(args.get("python_code") or "")
    schema = args.get("schema")
    if not isinstance(schema, dict):
        return {"ok": False, "error": "save_new_skill requires 'schema' to be an object"}
    if skill_name in TOOL_HANDLERS and skill_name not in _USER_SKILL_TOOL_IDS:
        return {
            "ok": False,
            "error": f"'{skill_name}' collides with an existing built-in tool id — choose a different skill_name",
        }

    result = save_skill(skill_name, python_code, schema)
    if not result.get("ok"):
        return result

    refresh_result = refresh_user_skills()
    if skill_name not in _USER_SKILL_TOOL_IDS:
        skipped = refresh_result.get("skipped", [])
        reason = next(
            (s["reason"] for s in skipped if s["file"] == f"{skill_name}.py"), "unknown validation failure"
        )
        return {"ok": False, "error": f"skill file written to disk but failed to load: {reason}"}

    return {
        "ok": True,
        "skill_name": skill_name,
        "message": f"Skill '{skill_name}' saved to skills/{skill_name}.py and is now available as a tool.",
    }


# Always core (see _CORE_TOOL_IDS below), NEVER mutating ("read_only": true
# in its tools.json entry) — dispatches immediately with no HITL approval. This is
# the agent's own debugging tool: see _SKILL_DEBUGGING_SECTION in the
# system prompt, and dispatch_tool_call's traceback attachment above —
# together they let the agent locate and fix its own buggy skill code
# without any human walking it through the source line by line.
def _tool_read_skill_source(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    skill_name = str(args.get("skill_name") or "").strip()
    source = read_skill_source(skill_name)
    if source is None:
        return {"ok": False, "error": f"no skill named '{skill_name}' found (or its file could not be read)"}
    return {"ok": True, "skill_name": skill_name, "code": source}


# Always core (see _CORE_TOOL_IDS below) and always mutating (no
# "read_only": true in its tools.json entry): the agent must always be able to remove a skill
# it (or a previous session) taught itself, but only with the same explicit
# human approval saving one in the first place required.
def _tool_delete_skill(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    """Deletes ``skills/<skill_name>.py`` (dana.core.skill_loader.delete_skill)
    and immediately hot-reloads the "user_skills" capability domain
    (``refresh_user_skills``) so the removed tool disappears from the very
    next ReAct turn's schema — no server restart needed.
    """
    skill_name = str(args.get("skill_name") or "").strip()
    result = delete_skill(skill_name)
    if not result.get("ok"):
        return result

    refresh_user_skills()
    if not result.get("deleted"):
        return {
            "ok": True,
            "skill_name": skill_name,
            "message": f"No skill named '{skill_name}' was found — nothing to delete.",
        }
    return {"ok": True, "skill_name": skill_name, "message": f"Skill '{skill_name}' deleted and unloaded."}


# os_tools domain — sandboxed real filesystem access (dana.plugins.os.
# file_system) plus time-boxed script execution (dana.plugins.os.
# process_manager), gated exactly like any other capability domain:
# reachable only once "os_tools" is in the session's effective capabilities
# (the LLM is simply never offered these schemas until then). write_file
# and run_python_script also declare no "read_only": true, so they
# additionally suspend for HITL approval — list_directory/read_file are
# read-only and dispatch immediately.
#
# run_python_script (not "execute_python_script") deliberately avoids
# colliding with dana/tools/actuators.py's own, DIFFERENT
# "execute_python_script" tool (execution_jail-scoped, background-job
# capable, used by dana.core.agent_loop's separate broker) — reusing that
# id would have silently overwritten its tools.json schema entry.


def _tool_list_directory(
    args: dict[str, Any], _engine: Any, _cp: Any, *, allowed_mounts: list[str] | None = None
) -> dict[str, Any]:
    return _fs_list_directory(str(args.get("path") or "."), allowed_mounts)


def _tool_read_file(
    args: dict[str, Any], _engine: Any, _cp: Any, *, allowed_mounts: list[str] | None = None
) -> dict[str, Any]:
    return _fs_read_file(str(args.get("path") or ""), allowed_mounts)


def _tool_write_file(
    args: dict[str, Any], _engine: Any, _cp: Any, *, allowed_mounts: list[str] | None = None
) -> dict[str, Any]:
    return _fs_write_file(str(args.get("path") or ""), str(args.get("content") or ""), allowed_mounts)


def _tool_edit_file(
    args: dict[str, Any], _engine: Any, _cp: Any, *, allowed_mounts: list[str] | None = None
) -> dict[str, Any]:
    return _fs_edit_file(
        str(args.get("path") or ""),
        str(args.get("search_block") or ""),
        str(args.get("replace_block") or ""),
        allowed_mounts,
    )


def _tool_search_files(
    args: dict[str, Any], _engine: Any, _cp: Any, *, allowed_mounts: list[str] | None = None
) -> dict[str, Any]:
    return _fs_search_files(str(args.get("directory_path") or "."), str(args.get("query") or ""), allowed_mounts)


def _tool_run_python_script(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    raw_args = args.get("args")
    script_args = [str(a) for a in raw_args] if isinstance(raw_args, list) else None
    return _fs_run_python_script(str(args.get("script_path") or ""), script_args)


def _tool_execute_terminal_command(
    args: dict[str, Any], _engine: Any, _cp: Any, *, allowed_mounts: list[str] | None = None
) -> dict[str, Any]:
    return _fs_execute_terminal_command(
        str(args.get("command") or ""), str(args.get("working_dir") or "."), allowed_mounts
    )


def _tool_start_background_service(
    args: dict[str, Any], _engine: Any, _cp: Any, *, allowed_mounts: list[str] | None = None
) -> dict[str, Any]:
    return _fs_start_background_service(
        str(args.get("command") or ""),
        str(args.get("alias") or ""),
        str(args.get("working_dir") or "."),
        allowed_mounts,
    )


def _tool_stop_background_service(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    return _fs_stop_background_service(str(args.get("alias") or ""))


def _tool_list_background_services(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    return _fs_list_background_services()


# web_tools domain — keyless DuckDuckGo search + httpx/BeautifulSoup page
# extraction (dana.plugins.web.research), gated exactly like os_tools:
# reachable only once "web_tools" is in the session's effective
# capabilities. Both tools are read-only reconnaissance ("read_only": true
# in their tools.json entries), dispatching immediately with no HITL approval.


def _tool_search_web(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    raw_max_results = args.get("max_results")
    try:
        max_results = int(raw_max_results) if raw_max_results is not None else 5
    except (TypeError, ValueError):
        max_results = 5
    return _web_search_web(str(args.get("query") or ""), max_results)


def _tool_read_webpage(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    return _web_read_webpage(str(args.get("url") or ""))


# vision_tools domain — VLM analysis of a sandboxed image file
# (dana.plugins.vision.image_analysis), gated exactly like os_tools/
# web_tools. Read-only inspection ("read_only": true in tools.json). The one
# handler in this codebase that needs the session's BYOK api_keys on the
# ordinary (non-suspended) dispatch path — see _TOOLS_NEEDING_API_KEYS
# near dispatch_tool_call below.
def _tool_analyze_workspace_image(
    args: dict[str, Any], _engine: Any, _cp: Any, *, api_keys: dict[str, str] | None = None
) -> dict[str, Any]:
    return _vision_analyze_workspace_image(
        str(args.get("file_path") or ""), str(args.get("query") or ""), api_keys=api_keys
    )


# Desktop Omni-Vision (dana.plugins.os.desktop_vision) — also needs the
# session's BYOK api_keys (see _TOOLS_NEEDING_API_KEYS near
# dispatch_tool_call below), same reason analyze_workspace_image does.
# UNLIKE analyze_workspace_image, this one declares no "read_only": true: capturing
# the user's actual desktop (not a sandboxed artifact this agent already
# wrote) is a high-privacy action regardless of what ends up on screen, so
# it must always pause for explicit human approval first — see
# dana.plugins.os.desktop_vision's module docstring.
def _tool_analyze_desktop_screen(
    args: dict[str, Any], _engine: Any, _cp: Any, *, api_keys: dict[str, str] | None = None
) -> dict[str, Any]:
    return _os_analyze_desktop_screen(str(args.get("query") or ""), api_keys=api_keys)


def _tool_query_engineering_standard(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    from dana.plugins.freecad.engineering_standards import query_engineering_standard

    return query_engineering_standard(str(args.get("query") or ""))


# take_canvas_screenshot needs an external actor (the R3F canvas, live in
# the Tauri frontend) to actually produce anything — dana.api.server
# intercepts this tool_id BEFORE dispatch_tool_call ever runs (see
# is_visual_inspection_tool below), suspending the ReAct loop exactly like a
# mutating tool suspends for HITL approval, then resolves it once the
# frontend's screenshot arrives over the websocket. This handler only
# exists so TOOL_HANDLERS/tools.json schema resolution recognizes the
# tool_id at all; reaching it via a direct dispatch_tool_call (bypassing
# that websocket round-trip) is always a caller error, so it fails loudly
# and specifically rather than silently no-op'ing.
VISUAL_INSPECTION_TOOLS: frozenset[str] = frozenset({"take_canvas_screenshot"})


def is_visual_inspection_tool(tool_id: str) -> bool:
    return tool_id in VISUAL_INSPECTION_TOOLS


def _tool_take_canvas_screenshot(_args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "error": (
            "take_canvas_screenshot requires the interactive WebSocket round-trip — "
            "dana.api.server intercepts and resolves it asynchronously against the "
            "live R3F canvas; it cannot be dispatched synchronously."
        ),
    }


def _tool_execute_vision_analysis(_args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    capture = capture_cad_viewport(save_copy=True)
    if not capture.get("ok"):
        return {"ok": False, "error": capture.get("error") or "capture failed"}
    if not capture.get("window_found"):
        return {"ok": False, "error": "no AutoCAD/FreeCAD window found on screen"}

    try:
        analysis = json.loads(analyze_cad_blueprint(capture["path"]))
    except (json.JSONDecodeError, ValueError):
        return {"ok": False, "error": "vision analysis returned non-JSON"}
    if not analysis.get("ok"):
        return {"ok": False, "error": analysis.get("error") or "vision analysis failed"}

    return {**analysis, "image_url": "/api/vision/last_cad_viewport.png"}


_LAST_CANVAS_SCREENSHOT_PATH = CAPTURES_DIR / "last_canvas_screenshot.png"


def build_visual_inspection_result(
    image_b64: str | None, *, error: str | None = None, api_key: str | None = None
) -> dict[str, Any]:
    """The pure core of the take_canvas_screenshot round-trip: given the
    base64 PNG the R3F frontend sent back (or an ``error`` it reported
    instead of a screenshot), decode+save it under ``CAPTURES_DIR`` and run
    it through the same VLM analysis ``execute_vision_analysis`` already
    uses, producing one consistent ``{"ok": bool, ...}`` payload either way.

    Split out from the websocket glue in ``dana.api.server`` specifically
    so this logic is unit-testable without a live socket — the frontend
    round-trip itself is just "send a request, receive base64 back", never
    worth testing beyond that plumbing.

    A captured screenshot with a failed/unavailable VLM analysis is still
    ``"ok": True`` — the tool's job is letting the LLM SEE the canvas; a
    missing text summary is a lesser, separately-noted limitation, not a
    tool failure (same "convenience miss, not a failure" philosophy
    ``dana.plugins.freecad.engine._auto_show`` already uses for GUI-open).

    ``api_key`` is the calling session's BYOK OpenAI key (``dana.api.server``'s
    ``session["api_keys"]["openai"]``), passed straight through to
    ``analyze_cad_blueprint``. ``execute_vision_analysis`` (the synchronous
    tool-dispatch VLM path, not this suspend/resume one) is intentionally
    NOT wired to a session key here — it goes through the uniform
    ``TOOL_HANDLERS`` dispatch signature shared by every tool, which this
    change doesn't touch, so it still falls back to ``OPENAI_API_KEY`` only.
    """
    if not image_b64:
        return {"ok": False, "error": error or "frontend returned no screenshot data"}
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (ValueError, binascii.Error) as exc:
        return {"ok": False, "error": f"take_canvas_screenshot: invalid base64 image data ({exc})"}
    if not raw:
        return {"ok": False, "error": "take_canvas_screenshot: decoded image data is empty"}

    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    _LAST_CANVAS_SCREENSHOT_PATH.write_bytes(raw)
    image_url = "/api/vision/last_canvas_screenshot.png"

    try:
        analysis = json.loads(analyze_cad_blueprint(image_b64, api_key=api_key))
    except (json.JSONDecodeError, ValueError):
        analysis = {"ok": False, "error": "vision analysis returned non-JSON"}
    if not analysis.get("ok"):
        return {
            "ok": True,
            "image_url": image_url,
            "summary": None,
            "note": f"screenshot captured but vision analysis unavailable: {analysis.get('error')}",
        }
    return {**analysis, "ok": True, "image_url": image_url}


def _tool_manipulate_camera(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    position = args.get("position")
    target = args.get("target")
    if not (isinstance(position, (list, tuple)) and len(position) == 3):
        return {"ok": False, "error": "manipulate_camera requires a 3-element 'position'"}
    if not (isinstance(target, (list, tuple)) and len(target) == 3):
        return {"ok": False, "error": "manipulate_camera requires a 3-element 'target'"}
    return {"ok": True, "position": [float(v) for v in position], "target": [float(v) for v in target]}


# Half-width (mm) of the square footprint synthesized when "extrude this/
# here" gives us only a clicked point+normal, not real profile geometry —
# a raycast hit alone can never recover a face's true boundary.
_EXTRUSION_DEFAULT_HALF_WIDTH = 10.0


def _tool_create_freecad_extrusion(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    height = float(args.get("height", args.get("distance", 25)))
    profile_points = args.get("profile_points")

    if not profile_points:
        target_position = args.get("target_position")
        target_normal = args.get("target_normal")
        if not target_position or not target_normal:
            return {
                "ok": False,
                "error": (
                    "create_freecad_extrusion needs either explicit 2D profile points or a "
                    "selected face — click a face in the viewer and say 'extrude this' so its "
                    "position can anchor the profile"
                ),
            }
        # The underlying engine only extrudes straight up along Z (see
        # dana.plugins.freecad.engine.create_extruded_polyline) — it has no
        # arbitrary extrusion-axis support, so a face whose normal isn't
        # close to Z can't be extruded meaningfully here yet.
        if abs(float(target_normal[2])) < 0.5:
            return {
                "ok": False,
                "error": (
                    "the selected face's normal isn't close enough to the Z axis for a "
                    "straight-up extrusion — FreeCAD extrusion here only extrudes along Z, "
                    "so a steep side face can't be extruded correctly yet"
                ),
            }
        x, y = float(target_position[0]), float(target_position[1])
        half = _EXTRUSION_DEFAULT_HALF_WIDTH
        profile_points = [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]]

    return engine.create_extrusion(profile_points, height, name=str(args.get("name") or "Extrusion"))


def _tool_create_freecad_pyramid(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    return engine.create_pyramid(
        float(args.get("length", 40)),
        float(args.get("width", 40)),
        float(args.get("height", 60)),
        name=str(args.get("name") or "Pyramid"),
        placement=_extract_placement(args),
    )


def _tool_create_freecad_star_prism(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    points = int(args.get("points", 5))
    if points < 3:
        return {"ok": False, "error": "create_freecad_star_prism requires at least 3 points"}
    return engine.create_star_prism(
        points,
        float(args.get("outer_radius", 50)),
        float(args.get("inner_radius", 20)),
        float(args.get("height", 10)),
        name=str(args.get("name") or "StarPrism"),
        placement=_extract_placement(args),
    )


# Maps a FreeCAD object's LLM-visible label (e.g. "Box") to the on-disk
# .FCStd/.stl path it was last saved to — every create_*/perform_freecad_boolean
# call spawns a brand-new document/subprocess (see dana.plugins.freecad.engine's
# module docstring), so there's no persistent ActiveDocument to fetch objects
# from by name across calls; this is that continuity, entirely dispatch-side so
# neither CAD engine driver needs to know about LLM-facing object names.
_OBJECT_PATH_REGISTRY: dict[str, str] = {}

_BOOLEAN_OPERATIONS = frozenset({"cut", "union", "intersect"})


def _tool_perform_freecad_boolean(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    operation = str(args.get("operation") or "").strip().lower()
    if operation not in _BOOLEAN_OPERATIONS:
        return {
            "ok": False,
            "error": "perform_freecad_boolean requires operation to be one of cut, union, intersect",
        }
    base_name = str(args.get("base_object") or "").strip()
    tool_name = str(args.get("tool_object") or "").strip()
    if not base_name or not tool_name:
        return {"ok": False, "error": "perform_freecad_boolean requires base_object and tool_object"}
    base_path = _OBJECT_PATH_REGISTRY.get(base_name)
    tool_path = _OBJECT_PATH_REGISTRY.get(tool_name)
    if not base_path:
        return {
            "ok": False,
            "error": f"unknown base_object '{base_name}' — create it first with a create_freecad_* tool",
        }
    if not tool_path:
        return {
            "ok": False,
            "error": f"unknown tool_object '{tool_name}' — create it first with a create_freecad_* tool",
        }
    return engine.apply_boolean(operation, base_path, tool_path, name=str(args.get("name") or "").strip() or None)


_EDGE_OPERATIONS = frozenset({"fillet", "chamfer"})


def _tool_perform_freecad_edge_operation(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    operation = str(args.get("operation") or "").strip().lower()
    if operation not in _EDGE_OPERATIONS:
        return {
            "ok": False,
            "error": "perform_freecad_edge_operation requires operation to be one of fillet, chamfer",
        }
    target_name = str(args.get("target_object") or "").strip()
    if not target_name:
        return {"ok": False, "error": "perform_freecad_edge_operation requires target_object"}
    target_path = _OBJECT_PATH_REGISTRY.get(target_name)
    if not target_path:
        return {
            "ok": False,
            "error": f"unknown target_object '{target_name}' — create it first with a create_freecad_* tool",
        }
    try:
        value = float(args.get("value"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "perform_freecad_edge_operation requires a numeric value"}

    # face_centroid is never LLM-supplied (not in this tool's tools.json
    # schema) — it's injected by _finalize_call_arguments from the active
    # canvas selection before dispatch ever sees this call.
    face_centroid = args.get("face_centroid")
    if isinstance(face_centroid, (list, tuple)) and len(face_centroid) == 3:
        face_centroid = tuple(float(v) for v in face_centroid)
    else:
        face_centroid = None

    return engine.apply_edge_operation(
        operation, target_path, value, face_centroid=face_centroid, name=str(args.get("name") or "").strip() or None
    )


def _tool_modify_freecad_parameter(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    target_name = str(args.get("target_object") or "").strip()
    if not target_name:
        return {"ok": False, "error": "modify_freecad_parameter requires target_object"}
    target_path = _OBJECT_PATH_REGISTRY.get(target_name)
    if not target_path:
        return {
            "ok": False,
            "error": f"unknown target_object '{target_name}' — create it first with a create_freecad_* tool",
        }
    parameter_name = str(args.get("parameter_name") or "").strip()
    if not parameter_name:
        return {"ok": False, "error": "modify_freecad_parameter requires parameter_name"}
    try:
        new_value = float(args.get("new_value"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "modify_freecad_parameter requires a numeric new_value"}
    return engine.modify_parameter(target_path, parameter_name, new_value)


def _tool_get_freecad_bounding_box(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    target_name = str(args.get("target_object") or "").strip()
    if not target_name:
        return {"ok": False, "error": "get_freecad_bounding_box requires target_object"}
    target_path = _OBJECT_PATH_REGISTRY.get(target_name)
    if not target_path:
        return {
            "ok": False,
            "error": f"unknown target_object '{target_name}' — create it first with a create_freecad_* tool",
        }
    return engine.get_bounding_box(target_path)


def _tool_inspect_spatial_properties(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    target_name = str(args.get("target_object") or "").strip()
    if not target_name:
        return {"ok": False, "error": "inspect_spatial_properties requires target_object"}
    target_path = _OBJECT_PATH_REGISTRY.get(target_name)
    if not target_path:
        return {
            "ok": False,
            "error": f"unknown target_object '{target_name}' — create it first with a create_freecad_* tool",
        }
    return engine.inspect_spatial_properties(target_path)


def _bbox_overlap(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """AABB overlap test + overlap-region volume — pure arithmetic on two
    ``get_bounding_box``-shaped dicts, no FreeCAD needed (same style as
    ``dana.plugins.freecad.engine._alignment_delta``). Deliberately lives
    here rather than as a new engine call: two existing ``get_bounding_box``
    reads are all the raw data this needs."""
    ox_min, ox_max = max(a["x_min"], b["x_min"]), min(a["x_max"], b["x_max"])
    oy_min, oy_max = max(a["y_min"], b["y_min"]), min(a["y_max"], b["y_max"])
    oz_min, oz_max = max(a["z_min"], b["z_min"]), min(a["z_max"], b["z_max"])
    if not (ox_min < ox_max and oy_min < oy_max and oz_min < oz_max):
        return {"collision": False, "overlap_bbox": None, "overlap_volume": 0.0}
    return {
        "collision": True,
        "overlap_bbox": {
            "x_min": ox_min, "y_min": oy_min, "z_min": oz_min,
            "x_max": ox_max, "y_max": oy_max, "z_max": oz_max,
        },
        "overlap_volume": (ox_max - ox_min) * (oy_max - oy_min) * (oz_max - oz_min),
    }


def _tool_analyze_bounding_box_collisions(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    name_a = str(args.get("object_a") or "").strip()
    name_b = str(args.get("object_b") or "").strip()
    if not name_a or not name_b:
        return {"ok": False, "error": "analyze_bounding_box_collisions requires object_a and object_b"}
    path_a = _OBJECT_PATH_REGISTRY.get(name_a)
    if not path_a:
        return {"ok": False, "error": f"unknown object_a '{name_a}' — create it first with a create_freecad_* tool"}
    path_b = _OBJECT_PATH_REGISTRY.get(name_b)
    if not path_b:
        return {"ok": False, "error": f"unknown object_b '{name_b}' — create it first with a create_freecad_* tool"}

    bbox_a = engine.get_bounding_box(path_a)
    if not bbox_a.get("ok"):
        return {"ok": False, "error": f"failed to read {name_a}'s bounding box: {bbox_a.get('error')}"}
    bbox_b = engine.get_bounding_box(path_b)
    if not bbox_b.get("ok"):
        return {"ok": False, "error": f"failed to read {name_b}'s bounding box: {bbox_b.get('error')}"}
    return {"ok": True, "object_a": name_a, "object_b": name_b, **_bbox_overlap(bbox_a, bbox_b)}


_PIPE_PATH_TYPES = frozenset({"straight", "arc"})


def _tool_create_freecad_pipe(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    path_type = str(args.get("path_type") or "").strip().lower()
    if path_type not in _PIPE_PATH_TYPES:
        return {"ok": False, "error": "create_freecad_pipe requires path_type to be one of straight, arc"}
    try:
        pipe_radius = float(args.get("pipe_radius"))
        length_or_angle = float(args.get("length_or_angle"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "create_freecad_pipe requires numeric pipe_radius and length_or_angle"}
    return engine.create_pipe(
        pipe_radius,
        path_type,
        length_or_angle,
        name=str(args.get("name") or "Pipe"),
        placement=_extract_placement(args),
    )


_ALIGNMENT_TYPES = frozenset({"top_center", "bottom_center", "flush_left", "flush_right"})


def _tool_align_freecad_objects(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    source_name = str(args.get("source_object") or "").strip()
    if not source_name:
        return {"ok": False, "error": "align_freecad_objects requires source_object"}
    target_name = str(args.get("target_object") or "").strip()
    if not target_name:
        return {"ok": False, "error": "align_freecad_objects requires target_object"}
    alignment_type = str(args.get("alignment_type") or "").strip().lower()
    if alignment_type not in _ALIGNMENT_TYPES:
        return {
            "ok": False,
            "error": "align_freecad_objects requires alignment_type to be one of "
            "top_center, bottom_center, flush_left, flush_right",
        }

    source_path = _OBJECT_PATH_REGISTRY.get(source_name)
    if not source_path:
        return {
            "ok": False,
            "error": f"unknown source_object '{source_name}' — create it first with a create_freecad_* tool",
        }
    target_path = _OBJECT_PATH_REGISTRY.get(target_name)
    if not target_path:
        return {
            "ok": False,
            "error": f"unknown target_object '{target_name}' — create it first with a create_freecad_* tool",
        }
    return engine.align_objects(source_path, target_path, alignment_type)


_MATE_TYPES = frozenset({"concentric", "coincident_planar", "offset_axial"})


def _tool_create_assembly_mate(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    fixed_name = str(args.get("fixed_obj") or "").strip()
    if not fixed_name:
        return {"ok": False, "error": "create_assembly_mate requires fixed_obj"}
    moving_name = str(args.get("moving_obj") or "").strip()
    if not moving_name:
        return {"ok": False, "error": "create_assembly_mate requires moving_obj"}
    mate_type = str(args.get("mate_type") or "").strip().lower()
    if mate_type not in _MATE_TYPES:
        return {
            "ok": False,
            "error": "create_assembly_mate requires mate_type to be one of "
            "concentric, coincident_planar, offset_axial",
        }

    fixed_path = _OBJECT_PATH_REGISTRY.get(fixed_name)
    if not fixed_path:
        return {"ok": False, "error": f"unknown fixed_obj '{fixed_name}' — create it first with a create_freecad_* tool"}
    moving_path = _OBJECT_PATH_REGISTRY.get(moving_name)
    if not moving_path:
        return {
            "ok": False,
            "error": f"unknown moving_obj '{moving_name}' — create it first with a create_freecad_* tool",
        }

    mate_params = args.get("mate_params")
    if not isinstance(mate_params, dict):
        mate_params = {}
    return engine.create_assembly_mate(fixed_path, moving_path, mate_type, mate_params)


_EXPORT_FORMATS = frozenset({"stl", "step"})


def _tool_export_freecad_model(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    export_format = str(args.get("format") or "").strip().lower()
    if export_format not in _EXPORT_FORMATS:
        return {"ok": False, "error": "export_freecad_model requires format to be one of stl, step"}
    target_names = args.get("target_objects")
    if not isinstance(target_names, list) or not target_names:
        return {"ok": False, "error": "export_freecad_model requires a non-empty target_objects list"}
    filename = str(args.get("filename") or "").strip()
    if not filename:
        return {"ok": False, "error": "export_freecad_model requires filename"}

    target_paths = []
    for raw_name in target_names:
        name = str(raw_name).strip()
        path = _OBJECT_PATH_REGISTRY.get(name)
        if not path:
            return {
                "ok": False,
                "error": f"unknown target object '{name}' — create it first with a create_freecad_* tool",
            }
        target_paths.append(path)

    return engine.export_model(target_paths, export_format, filename)


# Mirrors techdraw_export._DEFAULT_VIEWS — kept as a plain literal here
# (display text only, for describe_tool_call's HITL-preview fallback) rather
# than importing techdraw_export at module scope for one constant.
_DEFAULT_BLUEPRINT_VIEWS: tuple[str, ...] = ("Front", "Top", "Right", "Isometric")


def _tool_generate_2d_blueprint(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    """Bypasses the engine/control_plane driver abstraction entirely — same
    precedent as insert_standard_part/query_engineering_standard:
    dana.plugins.freecad.techdraw_export calls dana.plugins.freecad.engine's
    stateless script-runner directly (see its module docstring for why)."""
    from dana.plugins.freecad.techdraw_export import generate_2d_blueprint

    object_name = str(args.get("object_name") or "").strip()
    if not object_name:
        return {"ok": False, "error": "generate_2d_blueprint requires object_name"}
    source_path = _OBJECT_PATH_REGISTRY.get(object_name)
    if not source_path:
        return {
            "ok": False,
            "error": f"unknown object_name '{object_name}' — create it first with a create_freecad_* tool",
        }
    views = args.get("views")
    return json.loads(
        generate_2d_blueprint(
            source_path,
            views=views if isinstance(views, list) and views else None,
            page_size=str(args.get("page_size") or "A4"),
            filename=str(args.get("filename") or "").strip() or None,
        )
    )


def _tool_create_freecad_sketch_extrude(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    segments = args.get("segments")
    if not isinstance(segments, list) or not segments:
        return {"ok": False, "error": "create_freecad_sketch_extrude requires a non-empty segments list"}
    try:
        height = float(args.get("height"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "create_freecad_sketch_extrude requires a numeric height"}
    start = args.get("start") or [0.0, 0.0]
    if not (isinstance(start, (list, tuple)) and len(start) == 2):
        return {"ok": False, "error": "create_freecad_sketch_extrude requires a 2-element 'start' if given"}
    return engine.create_sketch_extrude(
        segments,
        height,
        start=(float(start[0]), float(start[1])),
        plane=str(args.get("plane") or "XY"),
        name=str(args.get("name") or "Sketch"),
        placement=_extract_placement(args),
    )


_PATTERN_TYPES = frozenset({"linear", "grid", "circular"})


def _tool_batch_pattern_array(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    source_name = str(args.get("source_object") or "").strip()
    if not source_name:
        return {"ok": False, "error": "batch_pattern_array requires source_object"}
    source_path = _OBJECT_PATH_REGISTRY.get(source_name)
    if not source_path:
        return {
            "ok": False,
            "error": f"unknown source_object '{source_name}' — create it first with a create_freecad_* tool",
        }
    pattern_type = str(args.get("pattern_type") or "").strip().lower()
    if pattern_type not in _PATTERN_TYPES:
        return {"ok": False, "error": "batch_pattern_array requires pattern_type to be one of linear, grid, circular"}
    return engine.batch_pattern_array(
        source_path,
        pattern_type,
        count_x=int(args.get("count_x", 1)),
        count_y=int(args.get("count_y", 1)),
        spacing_x=args.get("spacing_x"),
        spacing_y=args.get("spacing_y"),
        count=int(args.get("count", 1)),
        radius=float(args.get("radius", 0.0)),
        name=str(args.get("name") or f"{source_name}Pattern"),
    )


def _tool_insert_standard_part(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    """Bypasses the engine/control_plane driver abstraction entirely —
    dana.plugins.freecad.standard_parts calls dana.plugins.freecad.engine's
    stateless script-runner directly (see standard_parts.py's module
    docstring for why), so this handler's job is just argument plumbing,
    same as _tool_query_engineering_standard's engine-free lookup below.
    """
    from dana.plugins.freecad.standard_parts import insert_standard_part

    part_type = str(args.get("part_type") or "").strip()
    if not part_type:
        return {"ok": False, "error": "insert_standard_part requires part_type"}
    return json.loads(
        insert_standard_part(
            part_type,
            specification=str(args.get("specification") or ""),
            name=str(args.get("name") or "").strip() or None,
            placement=_extract_placement(args),
        )
    )


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any], Any, Any], dict[str, Any]]] = {
    "create_freecad_box": _tool_create_box,
    "create_freecad_cylinder": _tool_create_cylinder,
    "create_freecad_pyramid": _tool_create_freecad_pyramid,
    "create_freecad_star_prism": _tool_create_freecad_star_prism,
    "create_freecad_extrusion": _tool_create_freecad_extrusion,
    "perform_freecad_boolean": _tool_perform_freecad_boolean,
    "perform_freecad_edge_operation": _tool_perform_freecad_edge_operation,
    "modify_freecad_parameter": _tool_modify_freecad_parameter,
    "get_freecad_bounding_box": _tool_get_freecad_bounding_box,
    "inspect_spatial_properties": _tool_inspect_spatial_properties,
    "analyze_bounding_box_collisions": _tool_analyze_bounding_box_collisions,
    "create_freecad_pipe": _tool_create_freecad_pipe,
    "align_freecad_objects": _tool_align_freecad_objects,
    "create_assembly_mate": _tool_create_assembly_mate,
    "export_freecad_model": _tool_export_freecad_model,
    "generate_2d_blueprint": _tool_generate_2d_blueprint,
    "create_freecad_sketch_extrude": _tool_create_freecad_sketch_extrude,
    "batch_pattern_array": _tool_batch_pattern_array,
    "insert_standard_part": _tool_insert_standard_part,
    "resync_workspace": _tool_resync_workspace,
    "get_active_display": _tool_get_active_display,
    "prevent_focus_steal": _tool_prevent_focus_steal,
    "system_state": _tool_system_state,
    "check_plugin_registry": _tool_check_plugin_registry,
    "load_capability": _tool_load_capability,
    "update_core_memory": _tool_update_core_memory,
    "create_plan": _tool_create_plan,
    "mark_task_completed": _tool_mark_task_completed,
    "save_new_skill": _tool_save_new_skill,
    "read_skill_source": _tool_read_skill_source,
    "delete_skill": _tool_delete_skill,
    "list_directory": _tool_list_directory,
    "read_file": _tool_read_file,
    "write_file": _tool_write_file,
    "edit_file": _tool_edit_file,
    "search_files": _tool_search_files,
    "run_python_script": _tool_run_python_script,
    "execute_terminal_command": _tool_execute_terminal_command,
    "start_background_service": _tool_start_background_service,
    "stop_background_service": _tool_stop_background_service,
    "list_background_services": _tool_list_background_services,
    "search_web": _tool_search_web,
    "read_webpage": _tool_read_webpage,
    "analyze_workspace_image": _tool_analyze_workspace_image,
    "analyze_desktop_screen": _tool_analyze_desktop_screen,
    "query_engineering_standard": _tool_query_engineering_standard,
    "take_canvas_screenshot": _tool_take_canvas_screenshot,
    "execute_vision_analysis": _tool_execute_vision_analysis,
    "manipulate_camera": _tool_manipulate_camera,
}

# Snapshot of every hard-wired, native tool id — taken BEFORE any
# refresh_user_skills()/refresh_plugin_tools() call ever runs (both happen
# at the bottom of this module, at import time). This is the authoritative
# "never let a dynamically-loaded tool overwrite this" set:
# dana/plugins/freecad/manifest.json is a real, pre-existing example — a
# LEGACY manifest from before this dispatch table existed, declaring its
# own create_box/create_cylinder/... under the SAME tool ids as the
# hardcoded handlers above but with a different calling convention. It was
# harmless before refresh_plugin_tools() existed (dana.plugins.plugin_manager's
# discovery was introspection-only — see plugin_registry_view), but without
# this guard, refresh_plugin_tools() would silently clobber the correct,
# tested handlers with that legacy manifest's incompatible ones the moment
# generic plugin dispatch was wired up.
_NATIVE_TOOL_IDS: frozenset[str] = frozenset(TOOL_HANDLERS)

# Tool ids currently backed by a loaded user skill (dana.core.skill_loader)
# — mutated in place by refresh_user_skills() below as skills are
# added/removed. Checked unconditionally FIRST in is_mutating_tool, ahead
# of any tools.json/manifest read_only lookup, since a skill is real,
# unsandboxed Python running in-process regardless of what it declares.
_USER_SKILL_TOOL_IDS: set[str] = set()

# Generic manifest.json plugin dispatch (dana.plugins.plugin_manager) — the
# counterpart to _USER_SKILL_TOOL_IDS/_USER_SKILL_SCHEMAS above, but for
# file-based (not agent-authored) plugins. Populated by refresh_plugin_tools()
# below, called once at import time (bottom of this module): a new plugin
# folder needs ZERO edits here to become dispatchable, capability-routed,
# and schema-visible — see refresh_plugin_tools's own docstring.
_PLUGIN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {}
_PLUGIN_MUTATING_TOOL_IDS: set[str] = set()


def _native_tool_spec(tool_id: str) -> ToolSpec | None:
    """``tool_id``'s own registered ``ToolSpec`` (tools.json-backed) — the
    fail-closed source of truth ``is_mutating_tool`` defers to for every
    tool that isn't a user skill or a manifest.json plugin tool. Lazily
    loads/reuses the SAME registry cache ``_llm_tools_schema_cached``
    already maintains, so this never re-parses tools.json on every dispatch.
    """
    global _llm_tool_registry_cache
    if _llm_tool_registry_cache is None:
        _llm_tool_registry_cache = load_tool_registry()
    return _llm_tool_registry_cache.get(tool_id)


def is_mutating_tool(tool_id: str) -> bool:
    """Fail-closed HITL gate — replaces a former hardcoded ``MUTATING_TOOLS``
    allow-list (a native tool a developer forgot to add there dispatched
    with NO human approval, silently). There is no allow-list anymore: HITL
    requirement is derived directly from each tool's own registered schema,
    and anything that doesn't explicitly clear itself is gated by default.

    - Every loaded user skill is unconditionally mutation-gated (see
      dana.core.skill_loader's module docstring — real, unsandboxed Python
      running in-process; HITL approval is the actual safety boundary, not
      sandboxing the interpreter) regardless of anything it declares.
    - A manifest.json plugin tool is gated unless its own manifest
      explicitly opted out (``"read_only": true`` — see
      ``refresh_plugin_tools``/``_PLUGIN_MUTATING_TOOL_IDS``, already
      fail-closed the same direction as this function).
    - Every other REGISTERED tool (tools.json, native or not) is gated
      unless ITS OWN ``ToolSpec.read_only`` is explicitly ``True`` — a new
      native handler nobody annotated defaults to requiring approval, not
      to silently dispatching.
    - A tool_id that isn't registered ANYWHERE (not in ``TOOL_HANDLERS`` at
      all — e.g. a deleted skill, or a hallucinated/typo'd id) is NOT
      gated: ``dispatch_tool_call`` rejects it outright as an unknown
      tool_id before anything can run, so there is nothing here that
      needs — or benefits from — a human approval prompt.
    """
    if tool_id in _USER_SKILL_TOOL_IDS:
        return True
    if tool_id in _PLUGIN_TOOL_SCHEMAS:
        return tool_id in _PLUGIN_MUTATING_TOOL_IDS
    if tool_id not in TOOL_HANDLERS:
        return False
    spec = _native_tool_spec(tool_id)
    if spec is None:
        return True  # dispatchable, but with NO registered schema at all -- fail closed
    return not spec.read_only


def describe_tool_call(call: ToolCall) -> str:
    if call.tool_id == "create_freecad_box":
        length = call.arguments.get("length", 40)
        width = call.arguments.get("width", 25)
        height = call.arguments.get("height", 15)
        return f"Create a {length}x{width}x{height}mm box in FreeCAD."
    if call.tool_id == "create_freecad_cylinder":
        radius = call.arguments.get("radius", 10)
        height = call.arguments.get("height", 30)
        return f"Create a cylinder (radius {radius}mm, height {height}mm) in FreeCAD."
    if call.tool_id == "create_freecad_extrusion":
        height = call.arguments.get("height", 25)
        if call.arguments.get("profile_points"):
            return f"Extrude the given 2D profile by {height}mm in FreeCAD."
        return (
            f"Extrude a default {2 * _EXTRUSION_DEFAULT_HALF_WIDTH:g}x"
            f"{2 * _EXTRUSION_DEFAULT_HALF_WIDTH:g}mm footprint at the selected point by {height}mm "
            "(approximate — exact face bounds aren't available from a single click)."
        )
    if call.tool_id == "create_freecad_pyramid":
        length = call.arguments.get("length", 40)
        width = call.arguments.get("width", 40)
        height = call.arguments.get("height", 60)
        return f"Create a sharp-edged {length}x{width}mm base pyramid, height {height}mm, in FreeCAD."
    if call.tool_id == "create_freecad_star_prism":
        points = call.arguments.get("points", 5)
        outer = call.arguments.get("outer_radius", 50)
        inner = call.arguments.get("inner_radius", 20)
        height = call.arguments.get("height", 10)
        return (
            f"Create a sharp-edged {points}-point star prism (outer radius {outer}mm, "
            f"inner radius {inner}mm, thickness {height}mm) in FreeCAD."
        )
    if call.tool_id == "perform_freecad_boolean":
        operation = str(call.arguments.get("operation") or "?")
        base = call.arguments.get("base_object", "?")
        tool = call.arguments.get("tool_object", "?")
        verbs = {"cut": f"Cut `{tool}` out of `{base}`", "union": f"Fuse `{base}` and `{tool}` together",
                 "intersect": f"Keep only the overlap of `{base}` and `{tool}`"}
        return f"{verbs.get(operation, f'Combine {base} and {tool}')} in FreeCAD."
    if call.tool_id == "perform_freecad_edge_operation":
        operation = str(call.arguments.get("operation") or "?")
        target = call.arguments.get("target_object", "?")
        value = call.arguments.get("value", "?")
        verb = "Fillet" if operation == "fillet" else "Chamfer"
        scope = "the selected face's edges of" if call.arguments.get("face_centroid") else "every edge of"
        return f"{verb} {scope} `{target}` by {value}mm in FreeCAD."
    if call.tool_id == "modify_freecad_parameter":
        target = call.arguments.get("target_object", "?")
        param = call.arguments.get("parameter_name", "?")
        value = call.arguments.get("new_value", "?")
        return f"Set `{target}`.{param} = {value}mm in FreeCAD."
    if call.tool_id == "get_freecad_bounding_box":
        target = call.arguments.get("target_object", "?")
        return f"Read the bounding box of `{target}` in FreeCAD."
    if call.tool_id == "inspect_spatial_properties":
        target = call.arguments.get("target_object", "?")
        return f"Inspect volume/area/validity/topology of `{target}` in FreeCAD."
    if call.tool_id == "analyze_bounding_box_collisions":
        a = call.arguments.get("object_a", "?")
        b = call.arguments.get("object_b", "?")
        return f"Check whether `{a}` and `{b}`'s bounding boxes overlap."
    if call.tool_id == "create_freecad_pipe":
        radius = call.arguments.get("pipe_radius", "?")
        path_type = call.arguments.get("path_type", "?")
        value = call.arguments.get("length_or_angle", "?")
        if path_type == "arc":
            return f"Create a curved pipe (radius {radius}mm) bending {value} degrees in FreeCAD."
        return f"Create a straight pipe (radius {radius}mm, length {value}mm) in FreeCAD."
    if call.tool_id == "align_freecad_objects":
        source = call.arguments.get("source_object", "?")
        target = call.arguments.get("target_object", "?")
        alignment = call.arguments.get("alignment_type", "?")
        return f"Snap `{source}` to `{target}` ({alignment}) in FreeCAD."
    if call.tool_id == "create_assembly_mate":
        fixed = call.arguments.get("fixed_obj", "?")
        moving = call.arguments.get("moving_obj", "?")
        mate = call.arguments.get("mate_type", "?")
        return f"Mate `{moving}` to `{fixed}` ({mate}) in FreeCAD."
    if call.tool_id == "export_freecad_model":
        targets = call.arguments.get("target_objects", [])
        fmt = str(call.arguments.get("format") or "?").upper()
        filename = call.arguments.get("filename", "?")
        joined = ", ".join(str(t) for t in targets) if isinstance(targets, list) else str(targets)
        return f"Export {joined} as {fmt} named `{filename}`."
    if call.tool_id == "generate_2d_blueprint":
        target = call.arguments.get("object_name", "?")
        views = call.arguments.get("views") or list(_DEFAULT_BLUEPRINT_VIEWS)
        joined = ", ".join(str(v) for v in views)
        return f"Generate a 2D blueprint PDF of `{target}` ({joined})."
    if call.tool_id == "create_freecad_sketch_extrude":
        height = call.arguments.get("height", "?")
        plane = str(call.arguments.get("plane") or "XY").upper()
        segment_count = len(call.arguments.get("segments") or [])
        return f"Sketch a {segment_count}-segment profile on the {plane} plane and extrude it {height}mm in FreeCAD."
    if call.tool_id == "batch_pattern_array":
        source = call.arguments.get("source_object", "?")
        pattern = str(call.arguments.get("pattern_type") or "?")
        if pattern == "grid":
            count = int(call.arguments.get("count_x", 1)) * int(call.arguments.get("count_y", 1))
        elif pattern == "circular":
            count = call.arguments.get("count", "?")
        else:
            count = call.arguments.get("count_x", "?")
        return f"Create a {pattern} pattern of `{source}` ({count} total copies) in FreeCAD."
    if call.tool_id == "insert_standard_part":
        part_type = call.arguments.get("part_type", "?")
        spec = call.arguments.get("specification")
        suffix = f" ({spec})" if spec else ""
        return f"Insert a standard `{part_type}`{suffix} in FreeCAD."
    if call.tool_id == "resync_workspace":
        return "Reposition managed FreeCAD windows onto their target monitor."
    if call.tool_id == "prevent_focus_steal":
        return "Read the foreground window without changing OS focus."
    if call.tool_id == "query_engineering_standard":
        return f"Look up the engineering standard dimensions for '{call.arguments.get('query', '?')}'."
    if call.tool_id == "take_canvas_screenshot":
        return "Capture and visually inspect the current 3D viewport."
    if call.tool_id == "analyze_desktop_screen":
        query = call.arguments.get("query", "?")
        return f"Capture your ACTUAL DESKTOP screen (primary monitor) and ask a vision model: \"{query}\"."
    if call.tool_id == "save_new_skill":
        name = call.arguments.get("skill_name", "?")
        return f"Save and permanently load a new skill named `{name}` (writes skills/{name}.py)."
    if call.tool_id == "delete_skill":
        name = call.arguments.get("skill_name", "?")
        return f"Delete the skill `{name}` (removes skills/{name}.py and unloads it)."
    if call.tool_id == "edit_file":
        path = call.arguments.get("path", "?")
        return f"Apply a surgical search-and-replace edit to `{path}`."
    if call.tool_id == "execute_terminal_command":
        command = call.arguments.get("command", "?")
        working_dir = call.arguments.get("working_dir") or "."
        return f"Run shell command `{command}` in `{working_dir}`."
    if call.tool_id == "start_background_service":
        command = call.arguments.get("command", "?")
        alias = call.arguments.get("alias", "?")
        working_dir = call.arguments.get("working_dir") or "."
        return f"Start background service `{alias}` (`{command}` in `{working_dir}`) — runs until stopped."
    if call.tool_id == "stop_background_service":
        alias = call.arguments.get("alias", "?")
        return f"Stop the background service `{alias}` (kills its entire process tree)."
    return f"Run `{call.tool_id}`."


# Preset camera poses for "look at it from the top/front/side" phrasing —
# distances are tuned for the ~40-100mm primitives create_freecad_* produces.
# The LLM proposes a preset NAME (see manipulate_camera's tools.json schema);
# this is where that name still gets turned into an actual position/target,
# exactly as the old regex parser did.
_CAMERA_PRESETS: dict[str, tuple[float, float, float]] = {
    "top": (0, 200, 0.001),
    "front": (0, 0, 200),
    "side": (200, 0, 0),
    "iso": (120, 120, 120),
}

# Cheap deterministic backstop, not a parser: local 7B tool-calling models
# don't reliably copy multi-decimal floats verbatim into JSON arguments, so
# if the LLM omits target_position/target_normal but the user clearly meant
# the active selection ("this"/"here"/...), inject it ourselves rather than
# silently losing the anchor point.
_SELECTION_REFERENCE_PATTERN = re.compile(r"\b(this|here|that spot|selected)\b", re.I)

# Tools exposed to the LLM for function-calling — a subset of tools.json's
# full registry (which also serves dana.tools.broker's regex/alias router
# for the legacy desktop agent; that router is untouched by this module).
#
# Split into a always-on "core" set (general-assistant capabilities — status
# introspection, visual inspection — that make sense with no plugin active
# at all) and a per-plugin set unlocked only while that plugin is active in
# the frontend (dana.api.server's session["active_plugins"], populated from
# the "update_context" websocket message). This is capability routing: a
# plain-chat session's LLM turn never even sees FreeCAD's 24 tools, so it
# can't hallucinate calling one, and doesn't burn tokens on their schemas.
_CORE_TOOL_IDS = frozenset(
    {
        "take_canvas_screenshot",  # "core capabilities like visual inspection stay global" — even
        "system_state",  # though the canvas it screenshots is the CAD plugin's own viewport,
        "check_plugin_registry",  # keeping this global matches the task's explicit instruction.
        "load_capability",  # autonomous semantic routing — see below; always available so the
        # agent can retrieve a domain it wasn't handed, without needing a frontend plugin for it.
        "update_core_memory",  # persistent Core Memory — see dana.plugins.memory.core_memory;
        # always available so the agent can save a durable note in ANY session, plugin or not.
        "save_new_skill",  # Autonomous Skill Acquisition — see dana.core.skill_loader;
        # always available so the agent can permanently teach itself a new tool in ANY session.
        "delete_skill",  # same reasoning — always available so a previously-taught skill can
        # be removed in ANY session, not just one with "user_skills" active.
        "read_skill_source",  # read-only debugging companion to save_new_skill/delete_skill —
        # see _SKILL_DEBUGGING_SECTION in the system prompt above.
        "create_plan",  # Task Planner / Executive Function — see dana.plugins.planning.task_board;
        # always available so a long-horizon goal can be broken down in ANY session, plugin or not.
        "mark_task_completed",  # same reasoning — updating progress on the current plan must
        # never be gated behind a specific plugin being active.
    }
)

_FREECAD_TOOL_IDS = frozenset(
    {
        "create_freecad_box",
        "create_freecad_cylinder",
        "create_freecad_extrusion",
        "create_freecad_pyramid",
        "create_freecad_star_prism",
        "perform_freecad_boolean",
        "perform_freecad_edge_operation",
        "modify_freecad_parameter",
        "get_freecad_bounding_box",
        "inspect_spatial_properties",
        "analyze_bounding_box_collisions",
        "create_freecad_pipe",
        "align_freecad_objects",
        "create_assembly_mate",
        "export_freecad_model",
        "generate_2d_blueprint",
        "create_freecad_sketch_extrude",
        "batch_pattern_array",
        "insert_standard_part",
        "manipulate_camera",
        "resync_workspace",
        "get_active_display",
        "prevent_focus_steal",
        "query_engineering_standard",
    }
)

# Trimmed default for the AGENT'S OWN autonomous `load_capability` call
# (never for the frontend's explicit CAD-plugin activation: dana.api.server's
# "cad"/"freecad" plugin-id mapping is untouched, so a real CAD-panel
# session still activates domain "freecad" directly -> the FULL
# _FREECAD_TOOL_IDS above, exactly as before this change). Root cause
# of the parse-0/parse-1 failure this addresses: sending all 24 FreeCAD tool
# schemas (~30KB / ~6.8-6.9k tokens serialized, confirmed by direct probe
# against Groq) as `tools=` on the VERY NEXT turn after load_capability
# reliably blows a free/on-demand-tier Groq model's tokens-per-minute
# ceiling (observed live: HTTP 429 "Rate limit reached ... tokens per
# minute (TPM): Limit 8000, Used 6895, Requested 6842" on
# openai/gpt-oss-120b) — a 400 "too many tools" was the original
# hypothesis, but the actual failure is TPM exhaustion; cutting the
# schema payload is still the correct fix either way. These 7 cover the
# overwhelming majority of "make me a basic part" requests (box/cylinder,
# one boolean op, one edge op, a parameter tweak, a bounding-box check,
# export) without the model ever seeing the heavier/rarer schemas
# (patterns, assembly mates, blueprints, standard-parts lookup, camera
# control). The agent can still reach the full set any time a task
# genuinely needs it by calling load_capability again with
# domain="freecad_full".
_FREECAD_ESSENTIAL_TOOL_IDS = frozenset(
    {
        "create_freecad_box",
        "create_freecad_cylinder",
        "perform_freecad_boolean",
        "perform_freecad_edge_operation",
        "modify_freecad_parameter",
        "get_freecad_bounding_box",
        "export_freecad_model",
    }
)

# Real, sandboxed filesystem access (dana.plugins.os.file_system) — every
# path is confined to AGENT_WORKSPACE_DIR (dana.paths), never the whole
# host filesystem. write_file/edit_file declare no "read_only": true;
# search_files does — recursive grep is read-only reconnaissance (same
# policy as list_directory/read_file), meant to be dispatched freely so the
# model can locate something BEFORE spending a HITL-gated edit_file/
# write_file call on it. analyze_desktop_screen
# (dana.plugins.os.desktop_vision — Desktop Omni-Vision) lives in this same
# domain despite touching no filesystem path at all: it's grouped with the
# OS-level capabilities, not vision_tools, since it's a real-desktop
# action, not sandboxed-artifact inspection. Also declares no "read_only":
# true (privacy gate, not a file mutation). execute_terminal_command
# (dana.plugins.os.process_manager — Generalized Terminal Execution) is the
# HIGHEST-RISK tool in this whole registry (a raw shell=True string, not a
# fixed argv list the way run_python_script is) — also un-annotated,
# non-negotiably. start_background_service/stop_background_service
# (dana.plugins.os.background_services — Background Process Management) are
# the non-blocking counterpart to execute_terminal_command, for a command
# that's never expected to exit on its own (a dev server, a watcher) —
# both ALSO declare no "read_only": true (same shell=True/tree-kill stakes);
# list_background_services is the one read-only member of that trio (just
# reports which aliases are still running) and DOES declare "read_only":
# true, same policy as search_files above.
_OS_TOOLS_TOOL_IDS = frozenset(
    {
        "list_directory",
        "read_file",
        "write_file",
        "edit_file",
        "search_files",
        "run_python_script",
        "execute_terminal_command",
        "start_background_service",
        "stop_background_service",
        "list_background_services",
        "analyze_desktop_screen",
    }
)

# Real, keyless web research (dana.plugins.web.research) — both tools are
# read-only reconnaissance (search + page-text extraction), both declare
# "read_only": true in tools.json.
_WEB_TOOLS_TOOL_IDS = frozenset({"search_web", "read_webpage"})

# VLM analysis of a sandboxed image file — read-only inspection, declares
# "read_only": true in tools.json.
_VISION_TOOLS_TOOL_IDS = frozenset({"analyze_workspace_image"})

# Capability domain name -> the tool ids it unlocks on top of _CORE_TOOL_IDS.
# Two independent things can add a name to a session's active set (merged in
# dana.api.server._effective_capabilities before it ever reaches this
# module, so react_dispatch itself doesn't care which):
#   - the FRONTEND, via "update_context" (session["active_plugins"]) — keyed
#     by the CANONICAL domain name ("freecad", matching dana/plugins/freecad/*;
#     dana.api.server normalizes the frontend's own UI plugin id "cad" to
#     this first).
#   - the AGENT ITSELF, via the load_capability tool (session
#     ["capability_unlocked_at_turn"], subject to P1's per-session decay) —
#     autonomous semantic routing: the model can retrieve a domain (e.g.
#     "os_tools") it wasn't handed by the UI.
_CAPABILITY_TOOL_IDS: dict[str, frozenset[str]] = {
    "freecad": _FREECAD_TOOL_IDS,
    # "freecad_essential"/"freecad_full" exist as directly-activatable domain
    # names too (not just as _tool_load_capability's internal redirect
    # target below) so _tool_ids_for_plugins/_effective_capabilities can
    # resolve whichever one ends up stamped into session state without any
    # special-casing outside this dict.
    "freecad_essential": _FREECAD_ESSENTIAL_TOOL_IDS,
    "freecad_full": _FREECAD_TOOL_IDS,
    "os_tools": _OS_TOOLS_TOOL_IDS,
    "web_tools": _WEB_TOOLS_TOOL_IDS,
    "vision_tools": _VISION_TOOLS_TOOL_IDS,
    # Autonomous Skill Acquisition (dana.core.skill_loader) — unlike every
    # other entry here, this one is NOT a fixed frozenset: it's rebuilt by
    # refresh_user_skills() below every time a skill is saved/removed, so
    # a session with "user_skills" active always sees whatever's currently
    # on disk. Starts empty; refresh_user_skills() runs once at import
    # time (bottom of this module) to pick up any skills already saved
    # from a previous run, so this is never stale even before the first
    # save_new_skill call of a fresh process.
    "user_skills": frozenset(),
}


def domains_for_tool_id(tool_id: str) -> frozenset[str]:
    """Which capability domain(s) in ``_CAPABILITY_TOOL_IDS`` unlock
    ``tool_id`` beyond the always-available ``_CORE_TOOL_IDS`` — empty for a
    core-only or unknown tool_id.

    Used by ``dana.api.server``'s per-session capability decay (P1 of the
    local-agent rescue plan): every time a tool actually dispatches, the
    session refreshes the "last used" turn for whichever domain(s) unlock
    it, so a domain the agent is ACTIVELY using never expires mid-task —
    only one it loaded once (via ``load_capability``) and then stopped
    touching decays away, shrinking the tool schema back down.
    """
    if tool_id in _CORE_TOOL_IDS:
        return frozenset()
    return frozenset(domain for domain, ids in _CAPABILITY_TOOL_IDS.items() if tool_id in ids)


# Every tool any capability combination could ever offer — kept as one
# frozenset for backward compatibility: existing callers/tests that don't
# pass active_plugins at all (None) get this full, pre-capability-routing
# set, so nothing that isn't capability-aware yet silently loses tool access.
# Reassigned (not just computed once) by refresh_user_skills() below so a
# loaded skill is included here too.
_LLM_TOOL_IDS = _CORE_TOOL_IDS.union(*_CAPABILITY_TOOL_IDS.values())

# tool_id -> that skill's full OpenAI tool schema dict, for every currently
# loaded user skill — merged into _llm_tools_schema's output below for
# whichever of this turn's active tool_ids happen to be skills. Schemas
# don't come from dana/tools/tools.json (this is content _llm_tools_schema_cached's
# tools.json-backed cache has no notion of), so they're merged in a SEPARATE,
# uncached step every call — cheap: a plain dict filter over however many
# skills are currently loaded, never a re-parse of tools.json.
_USER_SKILL_SCHEMAS: dict[str, dict[str, Any]] = {}


def _wrap_skill_handler(
    run_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any], Any, Any], dict[str, Any]]:
    """Adapts a skill's 1-argument ``run(args)`` entrypoint to
    ``TOOL_HANDLERS``'s uniform ``(arguments, engine, control_plane)``
    signature ``dispatch_tool_call`` calls every handler with —
    engine/control_plane are simply unused: user skills are general-purpose
    Python (data transforms, OS tasks), never CAD engine operations.
    """

    def _handler(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
        return run_fn(args)

    return _handler


def refresh_user_skills() -> dict[str, Any]:
    """Re-scans ``AGENT_WORKSPACE_DIR/skills/`` (dana.core.skill_loader)
    and hot-reloads the "user_skills" capability domain IN PLACE: drops
    any tool_id whose skill file disappeared, (re)registers every
    currently-loaded skill into ``TOOL_HANDLERS``, and rebuilds
    ``_CAPABILITY_TOOL_IDS["user_skills"]`` — so the very next ReAct turn's
    tool schema reflects the change, no server restart needed.

    Called once at import time (bottom of this module) to pick up whatever
    skills already exist on disk from a previous run, and again by the
    ``save_new_skill`` tool every time the agent saves a new one.

    Clears both ``@lru_cache``s below it (``_tool_ids_for_plugins``,
    ``_llm_tools_schema_cached``) — they cache purely on their INPUT
    argument value, with no way to know ``_CAPABILITY_TOOL_IDS["user_skills"]``
    (global state they read internally) just changed, so a stale cached
    result for the exact same ``active_plugins`` frozenset would otherwise
    keep being served for the rest of the process's lifetime.
    """
    global _LLM_TOOL_IDS
    result = load_user_skills()
    fresh_skills: dict[str, dict[str, Any]] = result.get("skills", {})

    for stale_id in _USER_SKILL_TOOL_IDS - fresh_skills.keys():
        TOOL_HANDLERS.pop(stale_id, None)
        _USER_SKILL_SCHEMAS.pop(stale_id, None)

    _USER_SKILL_TOOL_IDS.clear()
    for tool_id, entry in fresh_skills.items():
        TOOL_HANDLERS[tool_id] = _wrap_skill_handler(entry["handler"])
        _USER_SKILL_SCHEMAS[tool_id] = entry["schema"]
        _USER_SKILL_TOOL_IDS.add(tool_id)

    _CAPABILITY_TOOL_IDS["user_skills"] = frozenset(_USER_SKILL_TOOL_IDS)
    _LLM_TOOL_IDS = _CORE_TOOL_IDS.union(*_CAPABILITY_TOOL_IDS.values())
    _tool_ids_for_plugins.cache_clear()
    _llm_tools_schema_cached.cache_clear()
    return result


def list_user_skills() -> dict[str, dict[str, Any]]:
    """The CURRENTLY loaded "user_skills" registry — tool_id -> full OpenAI
    schema — for external consumers (dana.api.skills's ``GET /api/skills``)
    that need to see exactly what's dispatchable right now, without
    reaching into this module's private ``_USER_SKILL_SCHEMAS`` directly.
    A plain copy, not a live view — safe for a caller to iterate even while
    a concurrent ``refresh_user_skills()`` might be mutating the original.
    """
    return dict(_USER_SKILL_SCHEMAS)


def _wrap_plugin_handler(
    fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any], Any, Any], dict[str, Any]]:
    """Adapts a manifest.json plugin's 1-argument ``fn(args)`` entrypoint to
    ``TOOL_HANDLERS``'s uniform ``(arguments, engine, control_plane)``
    signature — identical reasoning to ``_wrap_skill_handler`` above: a
    generic plugin (a codebase search, a shell wrapper, ...) has no business
    touching the CAD engine or control plane.
    """

    def _handler(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
        return fn(args)

    return _handler


def refresh_plugin_tools() -> dict[str, Any]:
    """Re-scans ``dana/plugins/*/manifest.json`` (``dana.plugins.
    plugin_manager.load_all_plugins_grouped``) and (re)registers every
    declared plugin's tools into ``TOOL_HANDLERS``, ``_CAPABILITY_TOOL_IDS``
    (one capability domain per plugin's manifest ``"domain"`` field),
    ``_PLUGIN_MUTATING_TOOL_IDS`` (any tool whose manifest didn't explicitly
    set ``"read_only": true``), and this turn's ``tools=`` schema
    (``_PLUGIN_TOOL_SCHEMAS``) — the generic, file-based-plugin counterpart
    to ``refresh_user_skills()``'s agent-authored one. A brand-new plugin
    folder (this session's ``coder_plugin``, or any future one) needs ZERO
    edits to this module to become dispatchable, capability-routed, and
    HITL-gated correctly — only its own ``manifest.json``.

    Rebuilds its own state fully on every call rather than only adding, so a
    plugin folder removed/renamed between calls correctly disappears too —
    same contract as ``refresh_user_skills()``. Called once at import time
    (bottom of this module); safe to call again later (e.g. a future
    hot-reload endpoint mirroring ``save_new_skill``'s).

    A plugin domain that collides with an existing hardcoded one (e.g.
    ``dana/plugins/freecad/manifest.json`` declaring domain ``"freecad"``,
    which ``_CAPABILITY_TOOL_IDS`` already owns for its 24 native handlers)
    UNIONS its NON-colliding tool ids into that domain rather than being
    skipped wholesale — ``_CAPABILITY_TOOL_IDS`` is never allowed to
    silently LOSE an existing built-in tool because of an unrelated
    plugin, but a manifest declaring the SAME domain as a native one is
    exactly how that native domain is meant to be extended (this is the
    same mechanism ``coder_plugin``'s brand-new ``"software_engineering"``
    domain uses — there just happens to also be a native ``"freecad"``
    domain already, which used to be treated as fully closed to
    manifest.json extension entirely; it no longer is).

    A plugin tool id that collides with ``_NATIVE_TOOL_IDS`` (a hard-wired
    handler defined directly in this module) is skipped with a logged
    warning, never registered — regardless of domain, a manifest can only
    ADD a new tool id, never silently replace an existing built-in
    handler's implementation under the same id. This is the real,
    pre-existing example that guard protects against: ``dana/plugins/
    freecad/manifest.json`` declares 5 tools under domain ``"freecad"``;
    3 of them (``create_freecad_box``/``create_freecad_cylinder``/
    ``create_freecad_extrusion``) share an id with an existing native
    handler and stay shadowed by it (the tested, hardened native
    implementation remains authoritative); the other 2
    (``modify_existing_freecad_document``/``execute_freecad_script``) are
    genuinely new ids with no native counterpart, so they union into
    ``_CAPABILITY_TOOL_IDS["freecad"]`` and become dispatchable.
    """
    global _LLM_TOOL_IDS
    from dana.plugins.plugin_manager import load_all_plugins_grouped

    for stale_id in _PLUGIN_TOOL_SCHEMAS:
        TOOL_HANDLERS.pop(stale_id, None)
        _PLUGIN_MUTATING_TOOL_IDS.discard(stale_id)
    _PLUGIN_TOOL_SCHEMAS.clear()

    domain_tool_ids: dict[str, set[str]] = {}
    grouped = load_all_plugins_grouped(force_refresh=True)
    for domain, tools in grouped.items():
        ids = domain_tool_ids.setdefault(domain, set())
        for spec, fn in tools:
            if spec.id in _NATIVE_TOOL_IDS:
                print(
                    f"[react_dispatch] WARNING: plugin domain {domain!r} declares tool_id "
                    f"{spec.id!r}, which collides with an existing built-in handler — skipping "
                    "the plugin's version; the built-in remains authoritative.",
                    flush=True,
                )
                continue
            TOOL_HANDLERS[spec.id] = _wrap_plugin_handler(fn)
            _PLUGIN_TOOL_SCHEMAS[spec.id] = to_openai_function_schema(spec)
            if not spec.read_only:
                _PLUGIN_MUTATING_TOOL_IDS.add(spec.id)
            ids.add(spec.id)

    for domain, ids in domain_tool_ids.items():
        _CAPABILITY_TOOL_IDS[domain] = _CAPABILITY_TOOL_IDS.get(domain, frozenset()) | frozenset(ids)

    _LLM_TOOL_IDS = _CORE_TOOL_IDS.union(*_CAPABILITY_TOOL_IDS.values())
    _tool_ids_for_plugins.cache_clear()
    _llm_tools_schema_cached.cache_clear()
    return {
        "domains": sorted(domain_tool_ids),
        "tool_count": len(_PLUGIN_TOOL_SCHEMAS),
        "mutating": sorted(_PLUGIN_MUTATING_TOOL_IDS),
    }


_llm_tool_registry_cache: dict[str, ToolSpec] | None = None


@lru_cache(maxsize=None)
def _tool_ids_for_plugins(active_plugins: frozenset[str] | None) -> frozenset[str]:
    """Core tools are always available; each active capability (whether it
    came from the frontend's active_plugins or the agent's own
    load_capability calls — dana.api.server._effective_capabilities merges
    both into the single frozenset this receives) unions in its own tool
    set. ``None`` means "caller isn't capability-aware" (e.g. the legacy
    ``parse_utterance`` wrapper) and is treated as "everything" — the
    original, pre-capability-routing behavior — NOT the same as passing an
    explicit empty set, which means "capability-aware caller, nothing active
    right now" and correctly yields only the core tools (including
    load_capability itself — the agent must always be able to ask for more).
    """
    if active_plugins is None:
        return _LLM_TOOL_IDS
    ids = _CORE_TOOL_IDS
    for plugin in active_plugins:
        ids = ids | _CAPABILITY_TOOL_IDS.get(plugin, frozenset())
    return ids


@lru_cache(maxsize=None)
def _llm_tools_schema_cached(tool_ids: frozenset[str]) -> tuple[dict[str, Any], ...]:
    """The actual openai_tools_schema() filter/build, cached per unique
    tool-id combination — every session sharing an active-plugin combination
    (the overwhelmingly common case: e.g. everyone with just "freecad"
    active) hits this cache after the first call, so routing costs one
    O(len(active_plugins)) set union (_tool_ids_for_plugins, also cached)
    plus an O(1) dict lookup, never a re-filter over the tool registry.
    """
    global _llm_tool_registry_cache
    if _llm_tool_registry_cache is None:
        _llm_tool_registry_cache = load_tool_registry()
    return tuple(openai_tools_schema(_llm_tool_registry_cache, tool_ids=tool_ids))


def _llm_tools_schema(
    active_plugins: frozenset[str] | None = None,
    *,
    query: str = "",
    sticky_ids: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Build this turn's ``tools=`` payload — capability-gated tool ids,
    optionally narrowed by semantic relevance to ``query`` (Pillar 1:
    ``dana.core.tool_retrieval.narrow_tool_ids_by_query``, a no-op when
    ``query`` is empty or the allowed set is already small), then minified
    (Pillar 2: ``dana.tools.schema_minify.minify_tool_schemas``, condenses
    verbose descriptions without touching the OpenAI wire contract).

    ``sticky_ids`` (tool ids already dispatched earlier in THIS turn's
    multi-step ReAct chain — see ``_sticky_tool_ids_from_messages``) always
    survive narrowing alongside ``_CORE_TOOL_IDS``, so a sequence like
    create-then-boolean-op never loses access to the tool it's already
    mid-sequence with just because the query text alone wouldn't have
    scored it highly.
    """
    tool_ids = _tool_ids_for_plugins(active_plugins)
    if active_plugins is not None:  # None = legacy "not capability-aware" caller — full set, no narrowing
        # The frontend's explicit "freecad"/"freecad_full" activation is
        # deliberately exempt from narrowing, same as it's exempt from the
        # older essential/full split (see _FREECAD_ESSENTIAL_TOOL_IDS's own
        # comment) — a real CAD-panel session gets the FULL tool set,
        # unchanged, exactly as before Pillar 1 existed. Narrowing still
        # applies to every other simultaneously-active domain (os_tools,
        # web_tools, vision_tools, user_skills, and any future plugin) —
        # that's where "scales to dozens of plugins" actually matters, since
        # CAD is the one domain a human already hand-tuned.
        #
        # Reads _CAPABILITY_TOOL_IDS live (not the bare _FREECAD_TOOL_IDS
        # literal) — "freecad" can grow past its original 24 at runtime via
        # refresh_plugin_tools()'s manifest.json union (e.g.
        # dana/plugins/freecad/manifest.json's modify_existing_freecad_
        # document/execute_freecad_script). Protecting only the ORIGINAL
        # literal left those manifest-added tools unprotected, so Pillar 1
        # could silently narrow them back out on the very query that most
        # needed a full, hand-tuned CAD domain — the exact "context drop"
        # bug class sticky_ids exists to prevent elsewhere in this module.
        protected = (
            (_CAPABILITY_TOOL_IDS.get("freecad", frozenset()) | _CAPABILITY_TOOL_IDS.get("freecad_full", frozenset()))
            if (active_plugins & {"freecad", "freecad_full"})
            else frozenset()
        )
        narrowed = narrow_tool_ids_by_query(
            tool_ids - protected, query, always_include=_CORE_TOOL_IDS | sticky_ids
        )
        tool_ids = narrowed | (protected & tool_ids)
    schemas = list(_llm_tools_schema_cached(tool_ids))
    # User-skill schemas live in _USER_SKILL_SCHEMAS, not dana/tools/tools.json
    # (what _llm_tools_schema_cached's registry is built from) — merged in
    # here, uncached, so a skill saved a moment ago is never missing from
    # this turn's tools= just because _llm_tools_schema_cached hasn't
    # re-run yet for this exact tool_ids combination.
    schemas.extend(schema for tool_id, schema in _USER_SKILL_SCHEMAS.items() if tool_id in tool_ids)
    # Manifest.json plugin schemas (dana.plugins.plugin_manager, refreshed
    # into _PLUGIN_TOOL_SCHEMAS by refresh_plugin_tools) — same "not in
    # tools.json, merge in uncached" reasoning as the user-skill step above.
    schemas.extend(schema for tool_id, schema in _PLUGIN_TOOL_SCHEMAS.items() if tool_id in tool_ids)
    return minify_tool_schemas(schemas)


_CORE_SYSTEM_PROMPT = """\
You are Dana, a general-purpose AI desktop assistant and autonomous \
software engineering agent. Additional capability domains (CAD modeling, \
software engineering, filesystem/OS access, web search, image analysis) may \
not be in your current tool set yet — that does NOT mean they are unavailable.

## Self-Resolving Missing Capabilities
- If the user asks for an action that needs a tool you don't currently see \
(e.g. "create a cylinder" with no create_freecad_* tool visible, or "refactor \
this function"/"fix this bug"/"write a test for this" with no analyze_codebase \
or execute_code_task tool visible), NEVER tell the user to go enable a plugin \
or domain themselves and stop there. Call load_capability with the matching \
domain FIRST, then call the actual tool you need in a later turn once it appears.
- check_plugin_registry only reports which tools EXIST across every domain — \
it does not make any of them callable. A tool it lists is still off-limits \
until you call load_capability with that exact tool's "domain" field; only \
then does the tool actually appear in your tool set and become callable.
- Any request to read, understand, edit, refactor, debug, or write code MUST \
be routed through the `software_engineering` domain, in this order: \
1) `load_capability(domain="software_engineering")`. \
2) `search_codebase` (Locate) — a regex `git grep` to find the relevant \
function/keyword by line number WITHOUT reading whole files. \
3) `analyze_codebase` (Read) — read only the specific files `search_codebase` \
turned up. \
4) `execute_code_task` (Edit) — make the actual edit once the change is \
scoped and confirmed. \
5) `run_verification_command` (Verify) — run pytest/flake8/mypy/`black \
--check` against what you just changed before ever presenting it as done.
- If `run_verification_command` returns an error or a traceback, you MUST \
NOT halt or summarize the task as finished. You MUST immediately call \
`execute_code_task` again with that exact traceback pasted into \
`task_description` so it can be fixed, then re-run `run_verification_command` \
to confirm the fix — repeat this Edit-then-Verify cycle until it passes \
before yielding the turn. Only stop early on the SAME error repeating twice \
in a row (the existing "don't retry an identical failure a third time" rule \
below still applies) and explain the terminal error instead.
- Only fall back to explaining what's missing if load_capability itself \
reports the domain doesn't exist or fails.

## Turn-Taking
- The user's request may describe several steps at once — you take exactly \
ONE tool call per turn, then stop and wait for its result before deciding \
the next step.
- Call a tool ONLY through the tool-calling mechanism. NEVER write a tool call \
(or a list of tool calls) as JSON text in your reply — that text is never executed.
- Only call a tool when the user is clearly asking for an action. If they're \
just chatting, asking a question, or every step they asked for is already \
done, reply in plain text without calling a tool.
- If a tool call returns an error, you may adjust its parameters and retry \
ONCE. If that retry returns the EXACT SAME error a second time, DO NOT try a \
third time — immediately stop, yield your turn, and explain the terminal \
error to the user instead of repeating the same failing call again.\
"""

# Always appended (see build_system_prompt below) — read_skill_source and
# save_new_skill are both CORE tools (always available, any domain/plugin
# active or not — see _CORE_TOOL_IDS), so this guidance stays unconditional
# too, rather than being duplicated into both _CORE_SYSTEM_PROMPT and
# _FREECAD_SYSTEM_PROMPT (the latter is verbatim/frozen — see its own
# docstring below).
_SKILL_DEBUGGING_SECTION = """\
## Debugging Your Own Skills
If a call to a skill you previously saved via `save_new_skill` fails and its \
tool result includes a "traceback" field, that is YOUR OWN Python code \
raising — use `read_skill_source` to see the skill's exact current code, \
find the line the traceback points at, then call `save_new_skill` again \
with the SAME skill_name and corrected python_code to overwrite and \
immediately hot-reload the fix. Never leave a broken skill registered, and \
never guess at a fix without reading the actual source first.\
"""

# Verbatim, byte-for-byte the module's original (pre-capability-routing)
# system prompt — used whenever "freecad" is in the active-plugin set, so a
# CAD session's LLM turn sees EXACTLY what it always has. Never edit this
# constant without also verifying tests/core/test_react_dispatch.py's
# build_system_prompt assertions and a live FreeCAD session still hold.
_FREECAD_SYSTEM_PROMPT = """\
You are Dana, a professional mechanical engineering CAD co-pilot for FreeCAD.

## Turn-Taking
- The user's request may describe several steps at once (e.g. "create a box, \
then check its size, then add a cylinder on top") — you take exactly ONE tool \
call per turn, then stop and wait for its result before deciding the next step.
- Call a tool ONLY through the tool-calling mechanism. NEVER write a tool call \
(or a list of tool calls) as JSON text in your reply — that text is never executed.
- Only call a tool when the user is clearly asking for an action. If they're \
just chatting, asking a question, or every step they asked for is already \
done, reply in plain text without calling a tool.

## Engineering Rules — apply in this order, every turn
1. NEVER HALLUCINATE HARDWARE DIMENSIONS. Before sketching or extruding any \
feature tied to a standard/off-the-shelf part (a named motor, a bolt/screw \
size, a bearing, etc.), call `query_engineering_standard` FIRST and use its \
returned numbers verbatim. When the part itself (not just its dimensions) is \
standard — a NEMA motor, an M3-M5 socket head screw, a ball bearing — prefer \
`insert_standard_part` over hand-sketching it. Only fall back to your own \
judgment once these tools report no match.
2. VERIFY BEFORE MUTATING. Before a risky edit — a fillet/chamfer, a boolean \
cut/union/intersect, or any change to an object you did not just create this \
turn — call `inspect_spatial_properties`, `get_freecad_bounding_box`, or \
`analyze_bounding_box_collisions` to confirm the object's ACTUAL current \
geometry rather than assuming it.
3. PREFER HIGH-LEVERAGE TOOLS over many small ones:
   - Use `create_freecad_sketch_extrude` (not `create_freecad_extrusion`) for \
any profile with a rounded/arc edge, a slot, or a shape a straight-edged \
polygon can't express.
   - Use `batch_pattern_array` (not one create_freecad_* call per copy) for \
3+ repeated features arranged in a line, grid, or circle — e.g. 4 bolt \
holes, an 8x8 tile grid.
   - Use `create_assembly_mate` (not `align_freecad_objects`) once you're \
building a real multi-part assembly — its concentric/coincident_planar/offset_axial \
mate types express true kinematic relationships, not just a bounding-box snap.
4. SELF-CORRECT ON ERROR — AND STOP AFTER A REPEATED FAILURE. A failed tool \
call never crashes the conversation — it returns `{"status": "error", \
"reason": ..., "suggestion": ...}`. Read both fields, adjust the specific \
parameter they point to, and retry. Never repeat an identical call that just \
failed. If you retry and get the EXACT SAME error a second time, DO NOT try a \
third time — immediately stop, yield your turn, and explain the terminal \
error to the user instead of guessing again.
5. VISUALLY CONFIRM COMPLEX RESULTS. After assembling several mated parts, or \
whenever you're not confident a sequence of edits looks right, call \
`take_canvas_screenshot` to see the actual rendered result before telling \
the user it's done or exporting it.\
"""


_PLAN_STATUS_MARKER: dict[str, str] = {"completed": "[x]", "active": "[>]", "pending": "[ ]"}


def _format_active_plan_for_prompt(plan: dict[str, Any]) -> str:
    """Renders the Task Planner's current plan
    (``dana.plugins.planning.task_board.get_active_plan``) as the
    "## Current Active Plan" system-prompt block — the LLM's own anchor
    for a long-horizon, multi-turn goal. Re-sent on EVERY ReAct iteration
    exactly like the core-memory block below it, so a model that just
    spent several turns on unrelated tool calls still opens its NEXT turn
    already reminded which task it's on, which are already done, and
    which are still ahead — rather than re-deriving (or hallucinating)
    its own place in a long plan from the raw conversation history alone.

    Returns ``""`` (never a heading with nothing under it) when no plan
    is active — ``format_core_memory_for_prompt``'s own empty-state
    convention.
    """
    tasks = plan.get("tasks") or []
    if not tasks:
        return ""
    objective = plan.get("objective") or "(no objective set)"
    lines = [f"## Current Active Plan\nObjective: {objective}"]
    for task in tasks:
        marker = _PLAN_STATUS_MARKER.get(task.get("status"), "[ ]")
        pointer = "  <-- YOU ARE HERE" if task.get("status") == "active" else ""
        lines.append(f"{marker} {task.get('id')}. {task.get('description')}{pointer}")
    lines.append(
        "This is YOUR OWN scratchpad — it does not update itself. Call "
        "mark_task_completed(task_id=..., next_task_id=...) as soon as a task is genuinely "
        "done, so the next turn (yours or another session's) sees accurate progress, not stale "
        "state. Call create_plan again to replace this plan entirely once its objective changes."
    )
    return "\n".join(lines)


def build_system_prompt(
    active_selection: dict[str, Any] | None,
    active_plugins: frozenset[str] | None = None,
    mounted_directories: list[str] | None = None,
    working_memory: str = "",
) -> str:
    """The dynamic context the LLM reasons over each turn — this is where
    the React 3D viewer's active-selection state enters the ReAct loop.

    This system prompt is re-sent on EVERY iteration of the multi-step
    ReAct loop (dana.api.server._run_react_loop), with the running
    conversation (including prior tool results) appended after it — so a
    request describing several steps at once gets to keep seeing this same
    "one step at a time" instruction, and the same engineering rulebook, as
    it progresses, not just on the first turn.

    ``active_plugins`` is the session's active-plugin set (see
    _tool_ids_for_plugins) — the CAD identity line and "## Engineering
    Rules" block are only included while "freecad" is among them. ``None``
    (the default, for callers not yet updated to pass session-scoped plugin
    state, e.g. parse_utterance) is treated the same as "freecad active" —
    this module's original, single-mode behavior — NOT the same as an
    explicit empty set, which correctly yields the lean general-assistant
    prompt for a real session with no plugins active.

    Always includes the "## Debugging Your Own Skills" section
    (``_SKILL_DEBUGGING_SECTION``) — read_skill_source/save_new_skill are
    both core tools, so this guidance is unconditional too, same as the
    "## Persistent Core Memory" block below it.

    Includes a "## Persistent Core Memory" block read fresh off disk
    (dana.plugins.memory.core_memory.format_core_memory_for_prompt) —
    whatever the agent itself has saved via update_core_memory across ANY
    past session, not just this one, so a restarted server still opens a
    turn already aware of it. Omitted entirely when core memory is empty,
    rather than showing an empty heading.

    ``mounted_directories`` (Dynamic Workspace Mounting, dana.api.workspace)
    is the current on-disk registry of external absolute directories the
    user has explicitly granted access to — listed here, by exact absolute
    path, so the model knows it may pass one of THESE paths (not one it
    invents) as an absolute ``path`` argument to list_directory/read_file/
    write_file. Omitted entirely when nothing is mounted, same as the
    core-memory section above.

    Always ENDS with a "## Current Active Plan" block
    (``_format_active_plan_for_prompt``, reading fresh from
    ``dana.plugins.planning.task_board.get_active_plan`` — global,
    in-memory, the exact same accessor ``dana.api.planner``'s REST API
    reads from) whenever ``create_plan`` has been called — deliberately
    the LAST thing in the prompt, so the agent's own executive-function
    anchor for a long-horizon goal is the freshest thing in its context on
    every single ReAct iteration, not buried above the (much larger)
    engineering rulebook. Omitted entirely when no plan is active, same
    empty-state convention as core memory/mounted directories above.
    """
    freecad_active = active_plugins is None or bool(
        active_plugins & {"freecad", "freecad_essential", "freecad_full"}
    )
    lines = [_FREECAD_SYSTEM_PROMPT if freecad_active else _CORE_SYSTEM_PROMPT, _SKILL_DEBUGGING_SECTION]
    centroid = active_selection.get("centroid") if active_selection else None
    normal = active_selection.get("normal") if active_selection else None
    if centroid:
        lines.append(
            f"Current active canvas selection: centroid {centroid}, normal {normal}. "
            "If the user refers to 'this', 'here', 'that spot', or 'the selected face', "
            "pass this centroid as target_position and this normal as target_normal "
            "(copy the numbers verbatim — do not invent your own coordinates)."
        )
    if mounted_directories:
        mounts_list = "\n".join(f"- {d}" for d in mounted_directories)
        lines.append(
            "## Mounted External Directories\n"
            "The user has explicitly granted access to these external directories, on top of "
            "your normal sandboxed workspace. list_directory/read_file/write_file may take an "
            "ABSOLUTE path under any of them (copy it verbatim — never invent or guess one):\n"
            f"{mounts_list}"
        )
    if working_memory.strip():
        # Pillar 3 (dana.core.context_distiller) — a per-SESSION rolling
        # summary distilled locally (RTX 2080, off the cloud hot path) from
        # turns this system prompt itself never carries forward otherwise:
        # a NEW user turn (dana.api.server._process_user_text) starts a
        # brand-new messages list, so without this the model has zero
        # memory of turn N-1 unless it explicitly wrote to Core Memory/the
        # Task Planner below.
        lines.append(
            "## Recent Session Context (auto-distilled, local model)\n"
            f"{working_memory.strip()}\n"
            "This is a compressed memory of earlier turns, not the verbatim conversation — "
            "if it seems to conflict with what the user just said, trust the user's latest message."
        )
    memory_section = format_core_memory_for_prompt()
    if memory_section:
        lines.append(memory_section)
    plan_section = _format_active_plan_for_prompt(_tb_get_active_plan())
    if plan_section:
        lines.append(plan_section)
    return "\n".join(lines)


def _resolve_camera_call(call: ToolCall, active_selection: dict[str, Any] | None) -> None:
    preset = str(call.arguments.get("preset") or "iso").strip().lower()
    preset = "iso" if preset.startswith("iso") else preset
    if preset not in _CAMERA_PRESETS:
        preset = "iso"
    target = active_selection.get("centroid") if active_selection else None
    target = target if isinstance(target, list) and len(target) == 3 else [0.0, 0.0, 0.0]
    call.arguments = {"position": list(_CAMERA_PRESETS[preset]), "target": target}


def _finalize_call_arguments(call: ToolCall, active_selection: dict[str, Any] | None) -> None:
    if call.tool_id == "manipulate_camera":
        _resolve_camera_call(call, active_selection)
        return
    if call.tool_id == "perform_freecad_edge_operation":
        # face_centroid isn't in this tool's LLM-facing schema at all (see
        # tools.json) — whether an edge op is face-targeted or whole-object
        # is decided here, by whether a canvas selection is currently
        # active, not by anything the LLM can express in its arguments.
        if "face_centroid" not in call.arguments and active_selection and active_selection.get("centroid"):
            call.arguments["face_centroid"] = active_selection.get("centroid")
        return
    if call.tool_id not in ("create_freecad_box", "create_freecad_cylinder", "create_freecad_extrusion"):
        return
    has_anchor = call.arguments.get("target_position") and call.arguments.get("target_normal")
    if not has_anchor and active_selection and _SELECTION_REFERENCE_PATTERN.search(call.raw_text or ""):
        call.arguments["target_position"] = active_selection.get("centroid")
        call.arguments["target_normal"] = active_selection.get("normal")


# Hard client-side ceiling on the PRIMARY model's turn — deliberately
# shorter than the Ollama HTTP layer's own 90s socket-stall timeout
# (dana.core.openai_tool_bridge.complete_openai_with_tools): a stalling/
# VRAM-fragmented local Ollama's real time-to-first-token is the failure
# signal that matters, not "did the whole request eventually finish" — a
# turn stuck at 60-90s is already a broken user experience even if it
# would have succeeded eventually.
#
# MUST stay above openai_tool_bridge's own TPM throttle-and-retry sleep
# ceiling (_MAX_REASONABLE_RETRY_AFTER_SEC + the 1s pad it adds — see
# complete_openai_with_tools) — this used to be a bare 22.0 and a real
# Groq 429 asking for a legitimate ~24-25s sleep got aborted mid-sleep by
# THIS timeout, logged as "primary model timed out", even though the
# bridge was behaving exactly as designed. Derived from that constant
# (rather than a second independent magic number) so the two can never
# drift out of sync like that again.
_LOCAL_TOOL_CALL_TIMEOUT_SEC = _MAX_REASONABLE_RETRY_AFTER_SEC + 15.0

# Separate, shorter ceiling for the fallback apology itself — this must
# never be allowed to hang the turn a second time, so it gets its own tight
# budget and a hardcoded final answer if even that expires.
_FALLBACK_TIMEOUT_SEC = 10.0

# A small, fast local model to ask for a plain apology once the primary
# coder model has already missed its deadline — never re-attempts the
# original multi-tool-schema request (a weaker model retrying the exact
# same request that just stalled is likely to fail the same way), just
# tells the user plainly so the turn ends instead of hanging.
_FALLBACK_LOCAL_MODEL = "llama3.2"

def _timeout_apology_text(primary_provider: str) -> str:
    """The last-resort, hardcoded apology shown once BOTH the primary
    model call AND the tiny local apology-generation fallback below have
    failed — phrased for whichever provider actually missed its deadline
    (``primary_provider``, from ``tool_calling_provider()`` at the ORIGINAL
    call site, not this fallback's own always-local model). Used to
    unconditionally say "The local model didn't respond in time" even when
    the primary provider was a cloud endpoint like Groq — misleading the
    user into restarting Ollama for a problem Ollama never had anything to
    do with.
    """
    if primary_provider == "ollama":
        return (
            "The local model didn't respond in time (it may be under memory "
            "pressure or stuck) — please try again. If this keeps happening, "
            "restarting Ollama usually clears it."
        )
    return (
        f"The cloud model ({primary_provider}) didn't respond in time — please "
        "try again shortly. This can happen during a provider-side rate-limit "
        "throttle or heavy load."
    )


async def _call_llm_timeout_fallback(
    api_keys: dict[str, str] | None, primary_provider: str
) -> dict[str, Any]:
    """Best-effort short apology once ``_call_llm_once``'s primary call has
    already missed ``_LOCAL_TOOL_CALL_TIMEOUT_SEC`` — the P2 "fail fast"
    half of the rescue plan. Tries a small, fast local model first (falling
    through to cloud automatically if ``DANA_ALLOW_CLOUD_FALLBACK`` is set —
    see ``ModelProvider.complete``'s own local-then-cloud logic), bounded by
    its own short timeout so a SECOND stalling call can never re-hang the
    turn; degrades to a hardcoded apology string (``_timeout_apology_text``,
    correctly attributed to ``primary_provider`` — the provider that ACTUALLY
    missed its deadline, not this fallback's own always-local model) if even
    that expires or raises. Always returns a "final" shape (empty
    ``tool_calls``) — this never attempts to chain another tool call.
    """
    fallback_messages = [
        {
            "role": "system",
            "content": (
                "The primary reasoning model just timed out. Reply with one short, "
                "plain-text apology telling the user to retry shortly. Do not call any tools."
            ),
        },
        {"role": "user", "content": "The previous request timed out — explain that briefly to the user."},
    ]
    try:
        provider = ModelProvider(local_model=_FALLBACK_LOCAL_MODEL, api_keys=api_keys)
        content = await asyncio.wait_for(
            asyncio.to_thread(provider.complete, fallback_messages, num_predict=128),
            timeout=_FALLBACK_TIMEOUT_SEC,
        )
        if content:
            # getattr, not provider.last_provider directly: this metadata
            # field is cosmetic (goes into a log/debug string, never
            # branched on) — a successful completion must never be thrown
            # away over a failure to read it. Caught live in stress-testing:
            # a test-double provider lacking the attribute made this whole
            # try block raise AttributeError AFTER a perfectly good
            # completion, silently discarding it for the generic hardcoded
            # apology below instead of the real answer just obtained.
            last_provider = getattr(provider, "last_provider", "unknown")
            return {"content": content, "tool_calls": [], "provider": f"fallback:{last_provider}"}
    except Exception:  # noqa: BLE001 — the fallback itself is best-effort; always degrade gracefully
        pass
    return {"content": _timeout_apology_text(primary_provider), "tool_calls": [], "provider": "fallback:static"}


def _sticky_tool_ids_from_messages(messages: list[dict[str, Any]]) -> frozenset[str]:
    """Tool ids that must survive Pillar 1's semantic-relevance narrowing
    regardless of how they score against this turn's query, because THIS
    turn's ReAct chain already has a concrete claim on them:

    1. Any tool id already invoked earlier in this chain (an assistant
       message's ``tool_calls``, as built by
       ``build_assistant_tool_call_message``) — so a multi-step sequence
       (e.g. create_freecad_box then perform_freecad_boolean referencing
       it) never loses a tool it's already mid-sequence with.
    2. Every tool id a ``load_capability`` call already unlocked earlier in
       this chain (read back from that call's own tool-result message,
       matched by ``tool_call_id`` so a same-shaped payload from an
       unrelated tool is never mistaken for one) — without this, the tool
       ``load_capability`` exists ONLY to unlock (e.g. ``execute_code_task``
       right after ``load_capability(domain="software_engineering")``) has
       itself never been "already invoked", so narrowing was free to drop
       it again on the very next turn: the one turn it's overwhelmingly
       likely to actually get called. ``load_capability``'s own handler
       (``_tool_load_capability``) is what puts ``unlocked_tools`` in the
       result payload in the first place.
    """
    ids: set[str] = set()
    call_id_to_name: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tc in message.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name")
            if name:
                ids.add(str(name))
                call_id_to_name[str(tc.get("id") or "")] = str(name)
    for message in messages:
        if message.get("role") != "tool":
            continue
        if call_id_to_name.get(str(message.get("tool_call_id") or "")) != "load_capability":
            continue
        try:
            payload = json.loads(message.get("content") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        for tool_id in payload.get("unlocked_tools") or ():
            if isinstance(tool_id, str):
                ids.add(tool_id)
    return frozenset(ids)


async def _call_llm_once(
    messages: list[dict[str, Any]],
    *,
    api_keys: dict[str, str] | None = None,
    active_plugins: frozenset[str] | None = None,
    raw_text: str = "",
) -> dict[str, Any]:
    """One raw LLM turn against the running ``messages`` history, via the
    existing OpenAI-tool-calling bridge (``dana.core.model_provider.
    ModelProvider`` + ``dana.tools.schema``) — no parsing/finalization of
    the result here, just the network call, so both the single-shot
    ``parse_utterance`` and the multi-step ``next_react_turn`` loop below
    share exactly one call site.

    ``api_keys`` (BYOK, from a session's SecretsMenu keys) is threaded into
    ``ModelProvider`` regardless of which provider this turn targets — the
    key is there, correctly prioritized over the environment variable, for
    whenever the resolved provider is one BYOK actually covers.

    Which provider this turn targets is decided by
    ``dana.core.model_provider.tool_calling_provider()``: local Ollama by
    default, or — when ``DANA_CLOUD_PRIMARY`` is set — a cloud OpenAI-
    compatible endpoint instead (Groq's free 70B by default), shifting the
    ReAct loop's actual per-turn tool-calling work off local VRAM entirely.
    Not hardcoded here so this one call site stays the single place that
    decision is made, regardless of which provider ends up selected.

    ``active_plugins`` (capability routing, from a session's active
    frontend plugins) narrows which tools this turn's ``tools=`` schema
    even offers the model — see ``_llm_tools_schema``/``_tool_ids_for_plugins``.

    A user turn built by ``build_user_message`` with attachments carries a
    multimodal ``content`` array (text + one or more ``image_url`` parts)
    instead of a plain string — ``ModelProvider.complete_with_tool_calls`` ->
    ``complete_openai_with_tools`` forwards ``messages`` into the request
    body verbatim (no per-message reshaping), so that array reaches the
    upstream OpenAI-wire-compatible endpoint exactly as built, with no
    special-casing needed here.

    ``messages`` is pruned (``dana.core.context_manager.
    prune_message_history``) into a SEPARATE list right before this HTTP
    call — older image attachments get replaced with a lightweight
    placeholder so a long conversation doesn't re-send every image it ever
    saw on every subsequent turn. The caller's own ``messages`` list (a
    session's real conversation history, resumed across HITL suspends and
    rendered turn-by-turn in the frontend) is never touched — only the
    payload actually posted to the model shrinks.

    Hard timeout + fallback (P2 of the local-agent rescue plan): the
    primary call is capped at ``_LOCAL_TOOL_CALL_TIMEOUT_SEC`` — well short
    of the HTTP layer's own 90s socket-stall timeout — since a stalling
    local model OR a cloud connection that drops mid-request (Groq/any
    OpenAI-compatible provider can stall or disconnect too) is a UX failure
    well before that. A timeout here never propagates as an "error" turn;
    it reroutes to ``_call_llm_timeout_fallback`` for a short apology
    instead, so the loop always ends in a clean "final" turn rather than
    the generic "I ran into a problem" message ``next_react_turn`` gives a
    genuine model/connection error.
    """
    provider = ModelProvider(api_keys=api_keys)
    pruned_messages = prune_message_history(messages)
    sticky_ids = _sticky_tool_ids_from_messages(messages)
    tools = _llm_tools_schema(active_plugins, query=raw_text, sticky_ids=sticky_ids)
    # Temporary diagnostic for the Semantic RAG "context drop" bug — proves
    # a tool load_capability just unlocked (sticky via _sticky_tool_ids_from_
    # messages) actually survives Pillar 1's narrowing into THIS turn's
    # payload, not just into the internal tool_ids set. Safe to remove once
    # that's re-confirmed live; deliberately a plain print (this module logs
    # everywhere else via print/stderr, not a logging.Logger) rather than a
    # new logging dependency for a debug line meant to be temporary.
    print(
        f"[Turn Context] Available tools for LLM: {sorted(t['function']['name'] for t in tools)}",
        file=sys.stderr,
        flush=True,
    )
    # Resolved ONCE and reused below — both the dispatch call and the
    # timeout handler must agree on which provider actually ran, so the
    # log line/apology text (see _timeout_apology_text) never blames the
    # wrong one (a real Groq TPM stall previously got logged/apologized
    # for as if it were "the local model").
    target_provider = tool_calling_provider()
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                provider.complete_with_tool_calls,
                pruned_messages,
                tools=tools,
                provider=target_provider,
            ),
            timeout=_LOCAL_TOOL_CALL_TIMEOUT_SEC,
        )
    except (TimeoutError, asyncio.TimeoutError):
        print(
            f"[ReAct] primary model ({target_provider}) timed out after "
            f"{_LOCAL_TOOL_CALL_TIMEOUT_SEC:.0f}s — falling back to a short apology",
            flush=True,
        )
        return await _call_llm_timeout_fallback(api_keys, target_provider)


async def parse_utterance(text: str, active_selection: dict[str, Any] | None = None) -> ToolCall | None:
    """One reason-then-act ReAct step: ask the local LLM whether this
    utterance needs a tool call. Kept as a single-turn convenience wrapper
    around ``next_react_turn`` for callers that only need one tool call and
    no multi-step loop (e.g. the existing test suite) — ``dana.api.server``'s
    ``/ws/chat`` handler drives the real multi-step loop directly via
    ``next_react_turn`` instead, so it can keep asking the LLM what to do
    next after each tool result.
    """
    text = (text or "").strip()
    if not text:
        return None

    messages = [
        {"role": "system", "content": build_system_prompt(active_selection)},
        {"role": "user", "content": text},
    ]
    turn = await next_react_turn(messages, active_selection, raw_text=text)
    return turn.call if turn.kind == "tool_call" else None


class ReactTurn:
    """Outcome of one ``next_react_turn`` iteration: either a final text
    answer (the loop should stop) or a resolved ``ToolCall`` ready for the
    caller to HITL-gate/dispatch and loop back with the result."""

    __slots__ = ("kind", "content", "call")

    def __init__(self, kind: str, content: str = "", call: ToolCall | None = None) -> None:
        self.kind = kind  # "final" | "tool_call" | "error"
        self.content = content
        self.call = call


# Keyword / intent auto-routing (P1 integration): a plain-chat session with
# no plugin tab active has no way to know a CAD request is coming until the
# model itself decides to call load_capability — meaning a request like
# "create a cylinder with radius 10" used to dead-end at "please enable the
# CAD plugin" (load_capability's tools.json enum didn't even offer
# "freecad" as a domain until this fix — see tools.json), or at best cost a
# whole extra turn just calling load_capability before the real action.
# This is a cheap, keyword-only pre-suggestion (never NLP/LLM-based) that
# folds "freecad" into THIS turn's tool schema up front whenever the user's
# own words plainly name a CAD/geometry action — so create_freecad_cylinder
# is simply already there for the model to call directly. Purely additive
# and ephemeral (recomputed fresh from raw_text every call, never written to
# session state): a false positive just means a few extra tool defs were
# offered for one turn; a false negative still leaves load_capability as a
# working fallback per the system prompt's new "Self-Resolving Missing
# Capabilities" section. If a suggested tool actually gets dispatched,
# dana.api.server's _touch_capability_domains/domains_for_tool_id already
# persists "freecad" into the session's real P1 decay tracking from that
# point on — no extra plumbing needed here for it to stick across turns.
_CAD_INTENT_KEYWORDS = frozenset(
    {
        "cad",
        "freecad",
        "cylinder",
        "extrude",
        "extrusion",
        "fillet",
        "chamfer",
        "sketch",
        "mesh",
        "solid",
        "assembly",
        "blueprint",
        "pyramid",
        "prism",
        "bounding box",
        "boolean",
        # "box" is deliberately absent — too common in ordinary English
        # ("check the box", "in a box") to use as a CAD signal on its own;
        # every other keyword here is specific enough not to false-positive
        # on everyday chat.
    }
)


def _keyword_suggested_domains(raw_text: str) -> frozenset[str]:
    lowered = (raw_text or "").lower()
    if any(keyword in lowered for keyword in _CAD_INTENT_KEYWORDS):
        return frozenset({"freecad"})
    return frozenset()


async def next_react_turn(
    messages: list[dict[str, Any]],
    active_selection: dict[str, Any] | None = None,
    *,
    raw_text: str = "",
    api_keys: dict[str, str] | None = None,
    active_plugins: frozenset[str] | None = None,
) -> ReactTurn:
    """One step of the multi-step ReAct loop: given the full running
    ``messages`` history (system + user + any prior assistant/tool turns
    already appended by the caller), ask the LLM what to do next.

    ``raw_text`` is the ORIGINAL user utterance for this turn (not the
    latest tool result) — ``_finalize_call_arguments``'s selection-reference
    check ("this"/"here"/...) matches against what the user actually said,
    which may be several tool calls back in a multi-step chain. Also feeds
    ``_keyword_suggested_domains`` below, folding "freecad" into THIS turn's
    effective capability set when the user's own words plainly name a CAD
    action — see that function's docstring for why.

    ``api_keys`` is the calling session's BYOK dict (``dana.api.server``'s
    ``session["api_keys"]``) — passed straight through to ``_call_llm_once``.

    ``active_plugins`` is the calling session's active-plugin set
    (capability routing) — passed to ``_call_llm_once`` (unioned with any
    keyword-suggested domain) so the model is never even offered a tool
    outside the effective set, AND re-checked below against the model's
    actual tool_calls response: a model can still name a tool from earlier
    conversation history that this turn's schema no longer offers (e.g. the
    user just deactivated a plugin mid-conversation) — treated exactly like
    an unknown tool_id, falling back to "final".
    """
    effective_plugins = active_plugins
    if active_plugins is not None:  # None means "not capability-aware" — leave as the full legacy set
        effective_plugins = active_plugins | _keyword_suggested_domains(raw_text)

    try:
        result = await _call_llm_once(
            messages, api_keys=api_keys, active_plugins=effective_plugins, raw_text=raw_text
        )
    except Exception as exc:  # noqa: BLE001 — Ollama unreachable/model missing surfaces as an error turn
        # str(exc) is all the UI-facing "error" turn carries onward (server.py
        # replaces it with a generic apology anyway) -- without this print,
        # a real provider rejection (bad request, rate limit, ...) leaves
        # ZERO trace anywhere once this function returns. openai_tool_bridge
        # already logs the raw HTTP body to stderr for cloud-provider
        # HTTPErrors; this print is the backstop for every OTHER exception
        # shape (local Ollama unreachable, a timeout that slipped through, etc.)
        # so this call site alone is never a silent sink.
        print(f"[ReAct] next_react_turn: LLM call failed -- {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return ReactTurn("error", content=str(exc))

    tool_calls = result.get("tool_calls") or []
    if not tool_calls:
        return ReactTurn("final", content=result.get("content") or "")
    call = tool_calls[0]  # one tool per LLM turn — the loop itself is what allows chaining several
    allowed_tool_ids = _tool_ids_for_plugins(effective_plugins)
    if call.tool_id not in TOOL_HANDLERS or call.tool_id not in allowed_tool_ids:
        # Real, observed quirk (live against qwen2.5-coder:7b): the model
        # can invent a plausible-but-nonexistent tool name (e.g.
        # "create_object" instead of "create_freecad_cylinder") even with
        # the correct tool sitting right in its own schema — not silence,
        # not a refusal, just the wrong name. One corrective retry, naming
        # the exact available tool ids, recovers this far more often than
        # silently falling back to a generic "I didn't think that needed a
        # tool call" — which is actively misleading here, since the model
        # clearly DID think a tool call was needed.
        retry_call = await _retry_after_unknown_tool_id(
            messages,
            call.tool_id,
            allowed_tool_ids,
            api_keys=api_keys,
            effective_plugins=effective_plugins,
            raw_text=raw_text,
        )
        if retry_call is None:
            return ReactTurn("final", content=result.get("content") or "")
        call = retry_call
    call.raw_text = raw_text
    _finalize_call_arguments(call, active_selection)
    return ReactTurn("tool_call", call=call)


async def _retry_after_unknown_tool_id(
    messages: list[dict[str, Any]],
    bad_tool_id: str,
    allowed_tool_ids: frozenset[str],
    *,
    api_keys: dict[str, str] | None,
    effective_plugins: frozenset[str] | None,
    raw_text: str = "",
) -> "ToolCall | None":
    """Exactly one corrective retry for ``next_react_turn`` when the model
    names a tool_id that isn't dispatchable (hallucinated, or a genuinely
    stale reference to a plugin the user just deactivated) — nudges with
    the precise available tool ids instead of silently giving up on the
    whole turn. Never mutates the caller's own ``messages`` list (appends
    the nudge to a fresh copy for this one extra call only); returns the
    corrected ``ToolCall``, or ``None`` if the retry doesn't recover a valid
    one either, in which case the caller falls back to "final" exactly as
    it would have without this retry.
    """
    nudge = {
        "role": "user",
        "content": (
            f"'{bad_tool_id}' is not a real tool — it doesn't exist. The exact tool names you can call "
            f"right now are: {', '.join(sorted(allowed_tool_ids))}. Call the correct one now with the same intent."
        ),
    }
    try:
        result = await _call_llm_once(
            messages + [nudge], api_keys=api_keys, active_plugins=effective_plugins, raw_text=raw_text
        )
    except Exception:  # noqa: BLE001 — the retry itself is best-effort, never worth failing the turn over
        return None
    retry_calls = result.get("tool_calls") or []
    if not retry_calls:
        return None
    retry_call = retry_calls[0]
    if retry_call.tool_id not in TOOL_HANDLERS or retry_call.tool_id not in allowed_tool_ids:
        return None
    return retry_call


def build_user_message(text: str, attachments: list[str] | None = None) -> dict[str, Any]:
    """OpenAI-wire user message for one chat turn — plain ``content: str`` when
    there are no attachments (unchanged shape every existing caller/test
    already expects), or a multimodal content-parts array (text part first,
    then one ``image_url`` part per attachment) when the frontend's
    attachment picker (see ``ChatPanel.tsx``) sent one or more images
    alongside the turn.

    ``attachments`` are already complete ``data:image/...;base64,...`` URIs
    (the frontend resizes+encodes client-side before sending — see
    dana.api.server's ``ws_chat``) so each one drops straight into
    ``image_url.url`` with no further encoding here. Anything that isn't a
    ``data:image/`` URI string is silently dropped rather than raising —
    a malformed/stale attachment must not take down an otherwise-valid text
    turn.
    """
    clean = [a for a in (attachments or []) if isinstance(a, str) and a.startswith("data:image/")]
    if not clean:
        return {"role": "user", "content": text}
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.extend({"type": "image_url", "image_url": {"url": a}} for a in clean)
    return {"role": "user", "content": content}


def extract_user_text(content: Any) -> str:
    """Recover the plain-text portion of a user message's OpenAI-wire
    ``content`` — either a plain string (the common case) or the multimodal
    content-parts array ``build_user_message`` produces when attachments are
    present. Used wherever a caller needs the ORIGINAL user utterance as a
    string (e.g. dana.api.server's ``_run_react_loop`` feeding ``raw_text``
    to ``next_react_turn``'s selection-reference fallback) regardless of
    which shape that turn's user message actually took.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text") or "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def build_assistant_tool_call_message(call: ToolCall) -> tuple[dict[str, Any], str]:
    """The OpenAI-wire assistant message announcing ``call``, to append to
    the running ``messages`` history so the model remembers deciding to
    call this tool — plus the synthetic ``tool_call_id`` the matching
    tool-result message must echo back.

    Dana's ``ToolCall`` IR has no id of its own (``openai_tool_calls_to_ir``
    doesn't preserve the wire ``id`` from the raw response), and round-
    tripping through this custom ``messages`` array only needs internal
    self-consistency between this message and its own tool-result reply —
    not the model's original id — so one is generated here.
    """
    call_id = f"call_{uuid.uuid4().hex[:24]}"
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": call.tool_id, "arguments": json.dumps(call.arguments)},
            }
        ],
    }
    return message, call_id


def build_tool_result_message(tool_call_id: str, result: "ToolResult") -> dict[str, Any]:
    """The OpenAI-wire ``tool`` role message reporting ``result`` back to
    the model for the next loop iteration, keyed to the assistant message
    that requested it via ``tool_call_id``.

    ``dispatch_tool_call`` already replaces a failed call's payload with
    ``digest_error``'s structured ``{status, reason, suggestion, raw_error}``
    shape, so that's used as-is; the bare ``{"ok": False, "error": ...}``
    reshape only kicks in as a fallback for a hand-built/empty-payload
    ``ToolResult`` that never went through dispatch (e.g. a synthetic
    failure a caller constructs directly).
    """
    payload = result.payload if (result.ok or result.payload) else {"ok": False, "error": result.message}
    return {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(payload)}


# Tools whose handler needs more than the uniform (arguments, engine,
# control_plane) signature — analyze_workspace_image and
# analyze_desktop_screen, both of which need the calling session's BYOK
# api_keys to reach ModelProvider the same way build_visual_inspection_result
# already gets one for the take_canvas_screenshot suspend path. Kept as an
# explicit, narrow allowlist rather than widening every handler's signature.
_TOOLS_NEEDING_API_KEYS = frozenset({"analyze_workspace_image", "analyze_desktop_screen"})

# Dynamic Workspace Mounting — the os_tools file/process septet, which need
# the session's currently-registered external mounts (dana.api.workspace's
# on-disk registry) threaded down to resolve_sandboxed_path's allowed_mounts
# param. Same narrow-allowlist pattern as _TOOLS_NEEDING_API_KEYS above, and
# disjoint from it — no handler needs both. stop_background_service/
# list_background_services are deliberately absent: neither takes a path at
# all (an alias-only lookup, and a no-argument status listing), so there is
# nothing for allowed_mounts to thread into.
_TOOLS_NEEDING_MOUNTS = frozenset(
    {
        "list_directory",
        "read_file",
        "write_file",
        "edit_file",
        "search_files",
        "execute_terminal_command",
        "start_background_service",
    }
)


def dispatch_tool_call(
    call: ToolCall,
    engine: Any,
    control_plane: Any,
    call_log: CadCallLog | None = None,
    api_keys: dict[str, str] | None = None,
    allowed_mounts: list[str] | None = None,
) -> ToolResult:
    """Dispatch ``call`` and, when ``call_log`` is given, append a
    ``CadCallRecord`` of the outcome — the single choke point every tool
    call passes through, so it's also the single place a session's ordered
    call history (``dana.plugins.freecad.py_export``'s "Show Your Work"
    export) needs to hook into, and the single place a failure — a raised
    exception, a FreeCAD/TopoShape kernel error, a timeout — gets converted
    via ``digest_error`` into a structured, LLM-actionable payload instead of
    a raw stderr/traceback dump the model can't act on.

    ``api_keys`` (BYOK, dana.api.server's session["api_keys"]) is threaded
    only into the tools in ``_TOOLS_NEEDING_API_KEYS``; ``allowed_mounts``
    (Dynamic Workspace Mounting's registered external directories, see
    dana.api.workspace) only into ``_TOOLS_NEEDING_MOUNTS`` — every other
    handler keeps the plain 3-argument call unchanged.

    Iterative Skill Debugging: when a tool in the "user_skills" domain
    (``_USER_SKILL_TOOL_IDS``) raises, the full ``traceback.format_exc()``
    — exact file/line, not just ``str(exc)`` — is attached to the failed
    payload as ``"traceback"``. This is the agent's OWN Python raising, so
    the trace points at code it can actually go read
    (``read_skill_source``) and fix (``save_new_skill``); a built-in tool's
    exception never gets this treatment — that would leak internal server
    implementation details the model has no business seeing and can't act
    on anyway.
    """
    handler = TOOL_HANDLERS.get(call.tool_id)
    if handler is None:
        return ToolResult(call.tool_id, False, {"ok": False, **digest_error(call.tool_id, "unknown tool_id")}, f"unknown tool_id '{call.tool_id}'", 0)
    start = time.perf_counter()
    skill_traceback: str | None = None
    try:
        if call.tool_id in _TOOLS_NEEDING_API_KEYS:
            payload = handler(call.arguments, engine, control_plane, api_keys=api_keys)
        elif call.tool_id in _TOOLS_NEEDING_MOUNTS:
            payload = handler(call.arguments, engine, control_plane, allowed_mounts=allowed_mounts)
        else:
            payload = handler(call.arguments, engine, control_plane)
        ok = bool(payload.get("ok", True))
        raw_error = None if ok else str(payload.get("error") or "tool reported failure")
    except Exception as exc:  # noqa: BLE001 — surface as a digested failure, never a crashed caller
        payload, ok, raw_error = {}, False, str(exc)
        if call.tool_id in _USER_SKILL_TOOL_IDS:
            skill_traceback = traceback.format_exc()
    duration_ms = int((time.perf_counter() - start) * 1000)
    if not ok:
        digested = digest_error(call.tool_id, raw_error)
        payload = {"ok": False, **digested}
        if skill_traceback:
            payload["traceback"] = skill_traceback
        message = digested["reason"]
    else:
        message = "ok"
        if isinstance(payload, dict) and payload.get("name") and payload.get("path"):
            _OBJECT_PATH_REGISTRY[str(payload["name"])] = str(payload["path"])
    if call_log is not None:
        call_log.record(call.tool_id, call.arguments, ok=ok, result=payload if ok else {}, error=None if ok else message)
    return ToolResult(call.tool_id, ok, payload, message, duration_ms)


def summarize_result(call: ToolCall, result: ToolResult) -> str:
    if not result.ok:
        return f"`{call.tool_id}` failed: {result.message}"
    payload = result.payload
    if call.tool_id in (
        "create_freecad_box",
        "create_freecad_cylinder",
        "create_freecad_extrusion",
        "create_freecad_pyramid",
        "create_freecad_star_prism",
        "perform_freecad_boolean",
        "perform_freecad_edge_operation",
        "create_freecad_pipe",
        "create_freecad_sketch_extrude",
        "batch_pattern_array",
        "insert_standard_part",
    ):
        driver = payload.get("driver", "win32/freecad")
        return (
            f"Created `{payload.get('type')}` named `{payload.get('name')}` via the "
            f"**{driver}** driver -> `{payload.get('path')}`."
        )
    if call.tool_id == "inspect_spatial_properties":
        return (
            f"Volume={payload.get('volume')}, area={payload.get('area')}, "
            f"valid={payload.get('is_valid')}, faces={payload.get('face_count')}, "
            f"edges={payload.get('edge_count')}."
        )
    if call.tool_id == "analyze_bounding_box_collisions":
        if payload.get("collision"):
            return f"`{payload.get('object_a')}` and `{payload.get('object_b')}` overlap — volume {payload.get('overlap_volume')}."
        return f"`{payload.get('object_a')}` and `{payload.get('object_b')}` do not overlap."
    if call.tool_id == "modify_freecad_parameter":
        return f"Set `{payload.get('name')}`.{payload.get('parameter_name')} = {payload.get('new_value')}mm."
    if call.tool_id == "align_freecad_objects":
        return (
            f"Aligned `{payload.get('name')}` ({payload.get('alignment_type')}) — "
            f"new placement {payload.get('placement')}."
        )
    if call.tool_id == "create_assembly_mate":
        return (
            f"Mated `{payload.get('name')}` to `{payload.get('fixed_object')}` "
            f"({payload.get('mate_type')}) — new placement {payload.get('placement')}."
        )
    if call.tool_id == "export_freecad_model":
        return f"Exported {payload.get('target_count')} object(s) as {str(payload.get('format')).upper()} -> `{payload.get('path')}`."
    if call.tool_id == "generate_2d_blueprint":
        views = ", ".join(payload.get("views", []))
        return f"Generated {str(payload.get('page_size')).upper()} blueprint ({views}) -> `{payload.get('path')}`."
    if call.tool_id == "get_freecad_bounding_box":
        return (
            f"Bounding box: x=[{payload.get('x_min')}, {payload.get('x_max')}], "
            f"y=[{payload.get('y_min')}, {payload.get('y_max')}], "
            f"z=[{payload.get('z_min')}, {payload.get('z_max')}]."
        )
    if call.tool_id == "resync_workspace":
        moved = payload.get("moved", [])
        return (
            f"Resynced workspace — {len(moved)} window(s) repositioned (zero-focus)."
            if moved
            else "Resynced workspace — nothing to move."
        )
    if call.tool_id == "check_plugin_registry":
        return "Active plugins: " + ", ".join(payload.get("plugins", [])) + "."
    if call.tool_id == "query_engineering_standard":
        if payload.get("ambiguous"):
            titles = ", ".join(m["title"] for m in payload.get("matches", []))
            return f"Ambiguous match for '{payload.get('query')}' — candidates: {titles}."
        return f"{payload.get('title')}: {payload.get('dimensions')}."
    if call.tool_id == "take_canvas_screenshot":
        return str(payload.get("summary") or payload.get("note") or "Captured the canvas viewport.")
    if call.tool_id == "execute_vision_analysis":
        return str(payload.get("summary") or "Analyzed the CAD viewport.")
    if call.tool_id == "manipulate_camera":
        return f"Moved the camera to {payload.get('position')}, looking at {payload.get('target')}."
    if call.tool_id == "system_state":
        return (
            f"control_plane={payload.get('control_plane')}, cad_engine={payload.get('cad_engine')}, "
            f"is_hf_space={payload.get('is_hf_space')}, dry_run={payload.get('dry_run')}."
        )
    return f"`{call.tool_id}` completed: {payload}"


def driver_state(engine: Any | None = None, control_plane: Any | None = None) -> dict[str, Any]:
    engine = engine or get_cad_engine()
    control_plane = control_plane or get_control_plane()
    return {
        "control_plane": type(control_plane).__name__,
        "cad_engine": type(engine).__name__,
        "is_hf_space": platform_factory.IS_HF_SPACE,
        "is_windows": platform_factory.IS_WINDOWS,
        "is_mac": platform_factory.IS_MAC,
        "dry_run": is_dry_run_enabled(),
    }


def plugin_registry_view() -> dict[str, Any]:
    """Introspection for the ``check_plugin_registry`` tool. Each tool entry
    includes its ``domain`` (via ``domains_for_tool_id``) — without this the
    model can see a tool EXISTS but has no way to know what string to pass
    to ``load_capability`` to actually unlock it, and gets stuck unable to
    call a tool it just saw listed here."""
    plugins = [p.name for p in discover_plugin_dirs()]
    tools = []
    for spec, _fn in load_all_plugins():
        domains = sorted(domains_for_tool_id(spec.id))
        tools.append({
            "id": spec.id,
            "description": spec.description_en,
            "domain": domains[0] if domains else None,
        })
    return {"plugins": plugins, "tools": tools}


# Picks up any skills already saved to disk from a previous run — so a
# freshly-started process is never missing a skill the agent taught itself
# in an earlier session, without needing a save_new_skill call to trigger
# the first load.
refresh_user_skills()

# Picks up every dana/plugins/*/manifest.json-declared plugin (e.g.
# coder_plugin) at process start — see refresh_plugin_tools's own docstring
# for exactly what this wires up with zero further edits to this module.
refresh_plugin_tools()


__all__ = (
    "TOOL_HANDLERS",
    "VISUAL_INSPECTION_TOOLS",
    "ReactTurn",
    "ToolResult",
    "build_assistant_tool_call_message",
    "build_system_prompt",
    "build_tool_result_message",
    "build_user_message",
    "build_visual_inspection_result",
    "describe_tool_call",
    "dispatch_tool_call",
    "driver_state",
    "extract_user_text",
    "is_mutating_tool",
    "is_visual_inspection_tool",
    "list_user_skills",
    "next_react_turn",
    "parse_utterance",
    "plugin_registry_view",
    "refresh_plugin_tools",
    "refresh_user_skills",
    "summarize_result",
)

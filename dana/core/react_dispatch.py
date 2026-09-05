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
import inspect
import json
import math
import re
import time
import traceback
import uuid
from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import Any, Literal

from dana.core import telemetry
from dana.core.context_manager import (
    compress_tool_output_history,
    prune_message_history,
    prune_tool_output_history,
)
from dana.core.model_provider import ModelProvider, tool_calling_provider
from dana.core.skill_loader import delete_skill, load_user_skills, read_skill_source, save_skill
from dana.core.tool_retrieval import narrow_tool_ids_by_query
from dana.session_context import get_session_id
from dana.tools.registry import get_tool_registry, is_bindable_tool
from dana.paths import CAPTURES_DIR
from dana.platform import factory as platform_factory
from dana.platform import get_cad_engine, get_control_plane
from dana.plugins.freecad.call_log import CadCallLog
from dana.plugins.freecad.error_digest import digest_error
from dana.plugins.freecad import skill_compiler
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
    ToolParameterSpec,
    ToolSpec,
    load_tool_registry,
    openai_tools_schema,
    to_openai_function_schema,
)
from dana.tools.schema_minify import minify_tool_schemas, should_strip_tool_schemas, strip_tool_schemas


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


def _tool_read_system_architecture(_args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    from dana.architecture import read_system_architecture

    return read_system_architecture()


def _tool_query_geometry_properties(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    from dana.tools.geometry_analyzer import query_geometry_properties

    return query_geometry_properties(str(args.get("mesh_path") or ""))


def _tool_generate_urdf_assembly(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    from dana.tools.urdf_builder import generate_urdf_assembly

    return json.loads(
        generate_urdf_assembly(
            str(args.get("robot_name") or "Robot"),
            list(args.get("links") or []),
            list(args.get("joints") or []),
        )
    )


def _tool_generate_3d_from_image(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    from dana.tools.image_to_3d import generate_3d_from_image

    image_path = str(args.get("image_path") or "").strip()
    if not image_path:
        return {"ok": False, "error": "generate_3d_from_image requires image_path"}
    return json.loads(generate_3d_from_image(image_path))


def _tool_import_and_solidify_mesh(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    """Bypasses the engine/control_plane driver abstraction entirely — same
    reasoning as ``_tool_insert_standard_part`` above (see
    ``dana.plugins.freecad.mesh_ops``'s own module docstring)."""
    from dana.plugins.freecad.mesh_ops import import_and_solidify_mesh

    mesh_path = str(args.get("mesh_path") or "").strip()
    if not mesh_path:
        return {"ok": False, "error": "import_and_solidify_mesh requires mesh_path"}
    object_name = str(args.get("object_name") or "").strip()
    if not object_name:
        return {"ok": False, "error": "import_and_solidify_mesh requires object_name"}
    return json.loads(import_and_solidify_mesh(mesh_path, object_name))


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

    Dynamic Domain Locking: also accepts ``tool_id`` INSTEAD OF ``domain`` —
    the inverse of ``unload_capability``'s own ``tool_id`` branch, un-hiding
    one specific tool_id this session previously hid (see that function's
    docstring for why domain granularity alone can't express "hide just
    this one tool"). Still stateless: server.py's ``_execute_and_continue``
    is what actually removes the id from ``session["hidden_tool_ids"]``
    once this reports success.
    """
    tool_id = str(args.get("tool_id") or "").strip()
    if tool_id:
        if tool_id not in TOOL_HANDLERS:
            return {"ok": False, "error": f"Unknown tool_id {tool_id!r} — nothing to unhide."}
        return {
            "ok": True,
            "unhidden_tool_id": tool_id,
            "message": f"Unhid '{tool_id}' — it's available now. Call it in your next tool call to continue.",
        }
    domain = str(args.get("domain") or "").strip()
    # "freecad" defaults to the trimmed essential set (see
    # _FREECAD_ESSENTIAL_TOOL_IDS) rather than all 24 FreeCAD tools — the
    # full set is heavy enough (~30KB of schemas) to blow a free-tier cloud
    # model's tokens-per-minute budget on the very next turn. This covers the
    # agent's own autonomous load_capability calls; the frontend's CAD-tab
    # activation is routed to "freecad_essential" too now (see
    # dana.api.server._PLUGIN_ID_TO_CAPABILITY), so both paths share the same
    # trimmed default. "freecad_full" is the explicit escalation domain —
    # callable again any time a task genuinely needs the heavier tools
    # (patterns, assembly mates, blueprints, standard parts,
    # engineering-standard lookups, camera control).
    resolved_domain = "freecad_essential" if domain == "freecad" else domain
    unlocked = _CAPABILITY_TOOL_IDS.get(resolved_domain)
    if unlocked is None:
        return {
            "ok": False,
            "error": f"Unknown capability domain {domain!r}. Available: {sorted(_CAPABILITY_TOOL_IDS)}",
        }
    tools = sorted(unlocked)
    # Message deliberately stays short — it's re-sent verbatim as this tool
    # call's own history on every later turn (dana.core.context_manager only
    # exempts the single MOST recent tool result from truncation), so
    # spelling out every tool name here was pure repeated token cost on top
    # of the already-compact `unlocked_tools` list below, which is what
    # dana.api.server/tests actually read programmatically. A large domain's
    # FULL tool set is still reported (nothing is hidden from the model or
    # from a caller inspecting `unlocked_tools`); only the wordy prose
    # enumeration is gone.
    message = f"Loaded '{resolved_domain}' — {len(tools)} tool(s) now available."
    if resolved_domain == "freecad_essential":
        message += (
            " Essential FreeCAD set. Call load_capability again with domain='freecad_full' "
            "if a heavier tool is needed."
        )
    return {
        "ok": True,
        "domain": resolved_domain,
        "unlocked_tools": tools,
        "message": message,
    }


def _tool_unload_capability(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    """Explicit release valve for autonomous semantic routing: lets the
    agent drop a capability domain it just finished using THIS session,
    instead of always waiting up to _CAPABILITY_DECAY_TURNS turns of disuse
    for it to decay out on its own (dana.api.server._effective_capabilities).
    Only ever affects a domain the agent unlocked via load_capability —
    never the frontend's own active_plugins (an open CAD/Coder tab stays
    under the human's explicit control and is untouched by this tool), the
    same boundary _effective_capabilities/_touch_capability_domains already
    draw between the two capability sources.

    Stateless, like _tool_load_capability: dana.api.server is what actually
    pops the domain from session["capability_unlocked_at_turn"] once
    dispatch_tool_call reports success (see server.py's
    _execute_and_continue) — this handler has no session object to mutate.

    Dynamic Domain Locking: also accepts ``tool_id`` INSTEAD OF ``domain`` —
    hides one specific tool_id from THIS session's tool schema even while
    its domain stays active (e.g. hide ``create_freecad_cylinder`` to force
    ``batch_pattern_array`` for a repeated-feature request) — something a
    domain boundary can't express on its own, since a domain's tools are
    always all-or-nothing (``create_freecad_cylinder`` and
    ``batch_pattern_array`` are both members of the same "freecad" domain).
    Stateless like the domain branch above: server.py's
    ``_execute_and_continue`` is what actually adds the id to
    ``session["hidden_tool_ids"]`` once this reports success.
    """
    tool_id = str(args.get("tool_id") or "").strip()
    if tool_id:
        if tool_id not in TOOL_HANDLERS:
            return {"ok": False, "error": f"Unknown tool_id {tool_id!r} — nothing to hide."}
        return {
            "ok": True,
            "hidden_tool_id": tool_id,
            "message": (
                f"Hid '{tool_id}' — it won't be offered starting next turn even though its domain "
                "stays active. Use the higher-leverage alternative instead. Call load_capability "
                f"with tool_id='{tool_id}' to unhide it if it's genuinely needed again."
            ),
        }
    domain = str(args.get("domain") or "").strip()
    if domain not in _CAPABILITY_TOOL_IDS:
        return {
            "ok": False,
            "error": f"Unknown capability domain {domain!r}. Available: {sorted(_CAPABILITY_TOOL_IDS)}",
        }
    return {
        "ok": True,
        "domain": domain,
        "message": (
            f"Unloaded '{domain}' — if it was only active via load_capability, its tools "
            "won't be sent starting next turn (call load_capability again if it's needed later). "
            "If it's also active as a frontend tab, it stays available until that tab closes."
        ),
    }


def _first_sentence(text: str, limit: int = 200) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return first[:limit].rstrip()


# Dropped from a search_tool_catalog query before lexical matching — common
# connectors that pad a natural-language query ("import a mesh FILE and
# convert it TO a solid") without ever appearing in a tool's own terse
# description. Deliberately small/conservative (not a full stop-word list)
# so a real content word never gets accidentally dropped.
_SEARCH_STOP_WORDS = frozenset({"and", "to", "a", "the", "file", "for", "with", "of", "in"})

# Fraction of a query's (stop-word-stripped) terms that must appear in a
# tool's name+description for _tool_search_tool_catalog's lexical stage to
# consider it a match — see that function's own docstring for why even
# stop-word filtering alone doesn't fully close its motivating recall gap
# (a genuine vocabulary mismatch, not just query padding, can still leave
# one term out of five unmatched).
_LEXICAL_MATCH_THRESHOLD = 0.75


def _tool_search_tool_catalog(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    """Lazy Loading, discovery half: searches the ENTIRE global tool
    registry (``dana.tools.registry.get_tool_registry`` — every tool from
    ``tools.json`` plus every plugin manifest, regardless of which
    capability domains are currently active) for ``query`` using a
    two-stage hybrid search, returning only a lightweight tool_id +
    one-sentence description per match — never a full schema, so a broad
    discovery search never itself costs the token budget the way actually
    loading a tool would. Pair with ``load_specific_tool`` to make a
    returned match callable.

    The old embedding-only search had a real recall gap: a live E2E run's
    query "import OBJ mesh file and convert to solid" failed to surface
    ``import_and_solidify_mesh`` at all, despite that tool's own
    description literally starting with "Imports a triangle mesh
    (.obj/.glb/.stl...)" — the agent then guessed a nonexistent tool_id and
    gave up (see ``_FREECAD_ESSENTIAL_TOOL_IDS``'s comment on the same
    incident, which is also why that tool moved into the essential tier).
    Two stages now run on every call, merged and deduplicated:

    1. **Lexical** — every registered tool whose name + description
       contains at least ``_LEXICAL_MATCH_THRESHOLD`` (75%) of ``query``'s
       significant terms (case-insensitive substring match, ``_STOP_WORDS``
       dropped first). Cheap (a linear scan over plain strings already held
       in memory, no embedding/network call) and catches exact-keyword
       queries the hash-embedding step alone can miss. Neither stop-word
       filtering nor a 100%-required threshold alone would have fully fixed
       the incident above: even with "and"/"to"/"file" dropped, the query's
       remaining "convert" never appears in ``import_and_solidify_mesh``'s
       own description (it says "combine"/"edge-operate", not "convert") —
       a plain ALL-terms match still misses it. A 75% threshold (4 of the 5
       remaining terms: import/obj/mesh/solid all hit, "convert" doesn't)
       is what actually closes the gap; ranked by match count so a tool
       matching more terms outranks one that barely clears the bar.
    2. **Semantic** — the existing vector-similarity search
       (``registry.retrieve``), for queries that don't share exact wording
       with a tool's own description.

    Lexical hits are listed first, best-match-count first (a literal
    keyword match is the stronger signal), then semantic hits fill any
    remaining slots up to 10 total.
    """
    query = str(args.get("query") or "").strip()
    if not query:
        return {"ok": False, "error": "query is required"}
    registry = get_tool_registry()

    raw_terms = [t for t in query.lower().split() if t]
    terms = [t for t in raw_terms if t not in _SEARCH_STOP_WORDS] or raw_terms
    lexical_hits = []
    if terms:
        required = max(1, math.ceil(len(terms) * _LEXICAL_MATCH_THRESHOLD))
        scored: list[tuple[int, Any]] = []
        for entry in registry.tools.values():
            if not is_bindable_tool(entry):
                continue
            haystack = f"{entry.name} {entry.description}".lower()
            matched = sum(1 for term in terms if term in haystack)
            if matched >= required:
                scored.append((matched, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        lexical_hits = [entry for _, entry in scored]

    semantic_hits = registry.retrieve(query, k=10)

    seen: set[str] = set()
    merged = []
    for entry in (*lexical_hits, *semantic_hits):
        if entry.name in seen:
            continue
        seen.add(entry.name)
        merged.append(entry)
        if len(merged) >= 10:
            break

    matches = [
        {"tool_id": entry.name, "description": _first_sentence(entry.description) or entry.name}
        for entry in merged
    ]
    return {
        "ok": True,
        "query": query,
        "matches": matches,
        "message": (
            f"Found {len(matches)} tool(s) matching {query!r}. Call load_specific_tool with a "
            "tool_id above to make it callable immediately in your next tool call."
            if matches
            else f"No tools matched {query!r}."
        ),
    }


def _tool_load_specific_tool(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    """Lazy Loading, fetch half: explicitly injects one specific tool_id's
    full schema into the active session for the NEXT turn, bypassing both
    the capability-domain gate (``_tool_ids_for_plugins``) and Pillar 1's
    semantic narrowing — see ``_sticky_tool_ids_from_messages`` (recognizes
    this tool's own ``unlocked_tools`` payload, same shape as
    ``load_capability``'s) and ``_call_llm_once``'s ``force_include`` wiring.
    Stateless, like ``_tool_load_capability``: this handler only validates
    the id exists somewhere in the global registry and reports it back; the
    actual schema-injection happens when ``_call_llm_once`` reads this call's
    own tool-result message back out of the running chain next turn.
    """
    tool_id = str(args.get("tool_id") or "").strip()
    if not tool_id:
        return {"ok": False, "error": "tool_id is required"}
    if get_tool_registry().get(tool_id) is None:
        return {
            "ok": False,
            "error": f"Unknown tool_id {tool_id!r} — call search_tool_catalog first to find a valid one.",
        }
    return {
        "ok": True,
        "tool_id": tool_id,
        "unlocked_tools": [tool_id],
        "message": f"Loaded '{tool_id}' — it's available now. Call it in your next tool call to continue.",
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
    # Deterministic LLM-Driven Mapping (Plan-and-Execute FSM, Phase 1
    # revision): each entry is normally a {"description": str,
    # "expected_tools": [str]} object (tools.json's create_plan schema) —
    # the PLANNER declares the exact tool(s) it intends for each task, since
    # it holds the real semantic context a k=3 embedding guess over a short
    # task description never reliably could. A bare string is still
    # accepted (a legacy caller, or a completion that ignores the object
    # shape) and treated as a task with no declared expected_tools — the
    # SAME "no fixed tool signature" case a genuinely tool-less task (a
    # pure visual inspection) already produces, handled identically below.
    descriptions: list[str] = []
    expected_tools_by_task: list[frozenset[str]] = []
    # Fix #1 — Strict Plan Validation (Kill Tool Hallucinations): collected
    # across EVERY task before anything is created, so one retry can name
    # every bad tool_id at once instead of a fix-one-fail-on-the-next loop.
    # A prior version of this function silently intersected each task's
    # declared expected_tools with `_RESTRICTED_GEOMETRY_TOOLS` below and
    # nothing else — a genuinely non-existent tool_id (e.g. the Planner
    # hallucinating "perform_freecad_pattern_array" for the real
    # "batch_pattern_array") landed in that intersection's "doesn't match
    # anything" bucket EXACTLY like a real-but-irrelevant tool_id would,
    # silently producing an EMPTY `expected_tool_ids` — which
    # `_tool_mark_task_completed`'s own False-Success-Blocker treats as "no
    # fixed tool signature", never gating that task's completion at all.
    # That silent drop is what let a hallucinated name bypass the blocker
    # entirely. Existence is now checked HERE, against TOOL_HANDLERS — the
    # actual live dispatch registry, not just this turn's capability-
    # narrowed schema — before the geometry-only intersection below ever
    # runs, so a hallucinated tool_id is a hard, plan-wide rejection instead
    # of a silently-accepted no-op task.
    hallucinated: list[tuple[int, str]] = []
    for i, entry in enumerate(raw_tasks if isinstance(raw_tasks, list) else [], start=1):
        if isinstance(entry, dict):
            descriptions.append(str(entry.get("description") or ""))
            raw_expected = entry.get("expected_tools")
            declared = frozenset(str(x) for x in raw_expected) if isinstance(raw_expected, list) else frozenset()
            hallucinated.extend((i, tool_id) for tool_id in sorted(declared) if tool_id not in TOOL_HANDLERS)
            expected_tools_by_task.append(declared)
        else:
            descriptions.append(str(entry))
            expected_tools_by_task.append(frozenset())

    if hallucinated:
        bad_ids = sorted({tool_id for _task_num, tool_id in hallucinated})
        detail = "; ".join(f"task {task_num}: '{tool_id}'" for task_num, tool_id in hallucinated)
        telemetry.log_error(stage="create_plan_rejected", hallucinated_tool_ids=bad_ids, detail=detail)
        raise RuntimeError(
            f"create_plan rejected — expected_tools names {bad_ids} which do not exist in the tool "
            f"catalog ({detail}). Nothing was created. Check the tools you have actually been "
            "offered (or call check_plugin_registry) for the EXACT tool_id and retry create_plan "
            "with corrected names."
        )

    result = _tb_create_plan(str(args.get("objective") or ""), descriptions)
    if result.get("ok"):
        # Plan-and-Execute Gatekeeper (Phase 6): a successful create_plan
        # unlocks this SESSION's geometry-mutating tools (see
        # _RESTRICTED_GEOMETRY_TOOLS/dispatch_tool_call) and anchors
        # `plan_text` into every later turn's system prompt (see
        # _get_active_plan/build_system_prompt) — formatted from the same
        # objective/tasks just sent to task_board, not re-derived from its
        # snapshot, so this stays independent of that module's own shape.
        objective = str(args.get("objective") or "").strip()
        plan_lines = [f"Objective: {objective}", *(f"{i}. {t}" for i, t in enumerate(descriptions, start=1))]
        _set_has_plan(True, "\n".join(plan_lines))
        # Session-scoped structured mirror of the same objective/tasks, kept
        # ONLY so `_get_active_plan` can render the Focused Plan Anchor (see
        # its own docstring) — `plan_text` above stays the flat verbatim
        # fallback for callers (tests included) that only ever called
        # `_set_has_plan` directly with a plain string and never populate
        # `tasks` at all.
        #
        # Plan-and-Execute FSM (Phase 1, revised): `expected_tool_ids` comes
        # straight from the Planner's OWN per-task `expected_tools`
        # declaration (parsed into `expected_tools_by_task` above), never a
        # semantic guess computed after the fact. The former approach —
        # `narrow_tool_ids_by_query(_RESTRICTED_GEOMETRY_TOOLS, t, k=3)`
        # scoring each task's short description against every geometry
        # tool's embedding — could and did silently misfire: a description
        # like "Export the final model as STEP" scoring create_freecad_box/
        # perform_freecad_boolean into ITS top-3 (short descriptions carry
        # too little signal for embeddings to reliably tell sequential CAD
        # steps apart). The Planner holds the real semantic context for its
        # own plan; declaring intent explicitly is strictly more reliable
        # than inferring it from a few words after the fact — and, since
        # Fix #1 above, every declared tool_id is ALREADY known to exist in
        # TOOL_HANDLERS by the time execution reaches here (a hallucinated
        # one aborted the whole create_plan call before this point).
        #
        # Universal FSM Enforcement (Geometry Filter REMOVED): a prior
        # revision intersected `declared` with `_RESTRICTED_GEOMETRY_TOOLS`
        # here — "the only tool_ids the FSM machinery consults" at the
        # time — so a Planner declaring a genuinely real, non-geometry tool
        # (e.g. `execute_freecad_script`, `batch_pattern_array` was still
        # geometry-scoped then too) came back with an EMPTY
        # `expected_tool_ids`, indistinguishable from "no fixed tool
        # signature at all". That was a second hallucination-shaped hole,
        # not just the first one Fix #1 closed: the model could declare
        # `execute_freecad_script`, have that silently dropped, build
        # something by an entirely different, UNDECLARED means, and still
        # pass `_tool_mark_task_completed`'s gate — which only ever fires
        # when `expected_tool_ids` is non-empty. `expected_tool_ids` now
        # perfectly mirrors what the Planner declared (already existence-
        # validated above) with NO further narrowing — `dispatch_tool_call`'s
        # out-of-order check and task-advancement tracking are correspondingly
        # broadened (see `_fsm_out_of_order_check`'s own module-level
        # comment) to enforce and track EVERY declared tool_id, not only
        # ones in `_RESTRICTED_GEOMETRY_TOOLS`.
        task_objs = [
            {
                "id": i,
                "description": t,
                "status": "active" if i == 1 else "pending",
                "expected_tool_ids": expected_tools_by_task[i - 1],
                # Fix #2 — The False Success Blocker: every tool_id
                # successfully dispatched while THIS task was active (see
                # _advance_fsm_on_dispatch), regardless of whether it
                # matched expected_tool_ids — checked against
                # expected_tool_ids by _tool_mark_task_completed before it
                # will let a manual completion through.
                "executed_tools": set(),
            }
            for i, t in enumerate(descriptions, start=1)
        ]
        _set_session_plan_tasks(objective, task_objs)
        entry = _PLAN_STATE_REGISTRY.setdefault(get_session_id(), {})
        entry["fsm_state"] = _fsm_transition(entry.get("fsm_state", "planning"), "create_plan_succeeded")
        telemetry.log_plan_created(objective=objective, task_count=len(task_objs))
    return result


def _tool_mark_task_completed(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
    """Manual half of task advancement — see the automatic half,
    ``_advance_fsm_on_dispatch``, and the Plan-and-Execute FSM's own
    module-level docstring for why BOTH exist (a task with no fixed
    ``expected_tool_ids`` can only ever be closed out manually).

    Fix #2 — The False Success Blocker: when the Planner declared
    ``expected_tools`` for this task (``_tool_create_plan``'s own per-task
    ``expected_tool_ids``), at least one of them must have been
    SUCCESSFULLY dispatched while the task was active (tracked in its
    ``executed_tools`` set by ``_advance_fsm_on_dispatch``) before this call
    is allowed to close it out. Without this, a task like "add the mounting
    hole pattern" (``expected_tools=["batch_pattern_array"]``) could be
    marked complete having never actually run the required operation — the
    exact hallucinated-completion failure mode this fix exists to block.
    Note this gate is normally moot for the auto-advance path: a
    successfully dispatched expected tool already closes the task out via
    ``_advance_fsm_on_dispatch`` itself, so reaching THIS function with the
    task still "active" and a non-empty ``expected_tool_ids`` means, by
    construction, that none of them has succeeded yet.
    """
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

    entry = _PLAN_STATE_REGISTRY.get(get_session_id())
    tasks = entry.get("tasks") if entry else None
    task = next((t for t in tasks if t.get("id") == task_id), None) if tasks else None
    if task is not None:
        expected = task.get("expected_tool_ids") or frozenset()
        executed = task.get("executed_tools") or set()
        if expected and not (expected & executed):
            missing = sorted(expected)
            telemetry.log_error(
                stage="mark_task_completed_rejected", task_id=task_id, missing_expected_tools=missing
            )
            return {
                "ok": False,
                "error": (
                    f"Task {task_id} declared expected_tools {missing} when the plan was created, "
                    "but none of them has been successfully executed yet. You must execute one of "
                    f"the required tool(s) (e.g. {missing[0]}) before calling mark_task_completed — "
                    "completing a task without running its required operation is not allowed."
                ),
            }

    result = _tb_mark_task_completed(task_id, next_task_id)
    if result.get("ok"):
        # Keep the session-scoped Focused Plan Anchor (_get_active_plan) in
        # sync with task_board's own global state — see _set_session_plan_tasks.
        _mark_session_task_completed(task_id, next_task_id)
        telemetry.log_task_state_change(task_id=task_id, status="completed", next_task_id=next_task_id)
    return result


# Composite Skill Compiler — turns a completed Plan-and-Execute run
# (create_plan -> N successful geometry calls) into a single, permanently
# reusable, parameterized tool. See dana.plugins.freecad.skill_compiler's
# own module docstring for the compiler itself; this handler is only the
# orchestration glue: slice this session's CadCallLog since its last
# create_plan, compile it, then hand the result to the SAME save_skill/
# refresh_user_skills pipeline save_new_skill already uses — a compiled
# skill IS a user skill, just machine-authored, so it gets hot-reload,
# HITL gating (see its "read_only" ABSENCE in tools.json), and now (see
# refresh_user_skills's own docstring) embedding-index registration for
# free, with zero new dispatch-side machinery.
#
# Always core (see _CORE_TOOL_IDS above): compiling what a plan just built
# must never be gated behind a specific plugin tab, same reasoning as
# create_plan/save_new_skill, which this composes.
def _tool_compile_plan_as_skill(
    args: dict[str, Any], _engine: Any, _cp: Any, *, call_log: CadCallLog | None = None
) -> dict[str, Any]:
    skill_name = str(args.get("skill_name") or "").strip()
    description = str(args.get("description") or "").strip()
    if not skill_name:
        return {"ok": False, "error": "compile_plan_as_skill requires skill_name"}
    if not description:
        return {"ok": False, "error": "compile_plan_as_skill requires description"}
    if skill_name in TOOL_HANDLERS and skill_name not in _USER_SKILL_TOOL_IDS:
        return {
            "ok": False,
            "error": f"'{skill_name}' collides with an existing built-in tool id — choose a different skill_name",
        }

    raw_tokens = args.get("expose_parameters")
    expose_tokens = [str(t) for t in raw_tokens] if isinstance(raw_tokens, list) else None

    records = skill_compiler.slice_records_since_plan(list(call_log.records) if call_log else [])
    compiled = skill_compiler.compile_call_log_to_skill(
        records, skill_name=skill_name, description=description, expose_tokens=expose_tokens
    )
    if not compiled.get("ok"):
        if compiled.get("invalid_expose_tokens"):
            # Fix #5 — Skill Compilation Integrity: surface the rejection as
            # a first-class ERROR event, not just a return value the caller
            # might not log at all.
            telemetry.log_error(
                stage="compile_plan_as_skill_rejected",
                skill_name=skill_name,
                invalid_expose_tokens=compiled["invalid_expose_tokens"],
            )
        return compiled

    save_result = save_skill(skill_name, compiled["python_code"], compiled["schema"])
    if not save_result.get("ok"):
        return save_result

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
        "parameter_count": compiled["parameter_count"],
        "step_count": compiled["step_count"],
        "skipped": compiled.get("skipped") or [],
        "message": (
            f"Compiled '{skill_name}' from {compiled['step_count']} step(s) with "
            f"{compiled['parameter_count']} exposed parameter(s) — available as a tool starting next turn."
        ),
    }


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


_CLOUD_SECURITY_RESTRICTION = {
    "ok": False,
    "error": "Security restriction: OS-level commands are disabled in the cloud demo environment.",
}


def _tool_write_file(
    args: dict[str, Any], _engine: Any, _cp: Any, *, allowed_mounts: list[str] | None = None
) -> dict[str, Any]:
    if platform_factory.IS_HF_SPACE:
        return _CLOUD_SECURITY_RESTRICTION
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
    if platform_factory.IS_HF_SPACE:
        return _CLOUD_SECURITY_RESTRICTION
    raw_args = args.get("args")
    script_args = [str(a) for a in raw_args] if isinstance(raw_args, list) else None
    return _fs_run_python_script(str(args.get("script_path") or ""), script_args)


def _tool_execute_terminal_command(
    args: dict[str, Any], _engine: Any, _cp: Any, *, allowed_mounts: list[str] | None = None
) -> dict[str, Any]:
    if platform_factory.IS_HF_SPACE:
        return _CLOUD_SECURITY_RESTRICTION
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


def _tool_create_freecad_polygon(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    sides = int(args.get("sides", 6))
    if sides < 3:
        return {"ok": False, "error": "create_freecad_polygon requires at least 3 sides"}
    return engine.create_polygon(
        sides,
        float(args.get("radius", 50)),
        float(args.get("height", 10)),
        name=str(args.get("name") or "Polygon"),
        placement=_extract_placement(args),
    )


# Maps a FreeCAD object's LLM-visible label (e.g. "Box") to the on-disk
# .FCStd/.stl path it was last saved to — every create_*/perform_freecad_boolean
# call spawns a brand-new document/subprocess (see dana.plugins.freecad.engine's
# module docstring), so there's no persistent ActiveDocument to fetch objects
# from by name across calls; this is that continuity, entirely dispatch-side so
# neither CAD engine driver needs to know about LLM-facing object names.
#
# Nested per session_id (outer key), not a flat dict — two different chat
# sessions each naming an object "Box" used to silently clobber each
# other's entry in one shared global dict (confirmed live: whichever
# session created its "Box" LAST won, and the other session's later
# perform_freecad_boolean/modify_freecad_parameter calls on "its own" Box
# would resolve to the wrong session's file). Always go through
# _object_registry() below, never this dict directly, so every read/write
# is scoped to the CURRENT turn's session_id (dana.session_context).
_OBJECT_PATH_REGISTRY: dict[str, dict[str, str]] = {}


def _object_registry() -> dict[str, str]:
    """THIS session's own slice of ``_OBJECT_PATH_REGISTRY`` — created
    empty on first use, same lazy semantics a flat dict's ``.get()``/``in``
    already had for a session's first-ever object."""
    return _OBJECT_PATH_REGISTRY.setdefault(get_session_id(), {})


# Topological Lineage Graph (TLG) — Phase 5: replaces the flat
# active_solid/consumed-ancestors scheme (Phase 4) with a real DAG the
# frontend's DAG Monitor can render directly, and which derives "is this
# name still usable" from actual graph structure instead of a
# hand-maintained consumed-set. {"nodes": {name: {"id", "label", "type"}},
# "edges": [{"source", "target"}, ...]} — session-scoped exactly like
# _OBJECT_PATH_REGISTRY, for the same cross-session-collision reason.
_TOPOLOGY_DAG_REGISTRY: dict[str, dict[str, Any]] = {}


def get_topology_dag(session_id: str | None = None) -> dict[str, Any]:
    """This session's Topological Lineage Graph. An explicit ``session_id``
    lets a caller outside the dispatch_tool_call call chain (e.g. the
    WebSocket "ready" handler seeding a reconnect) ask for a specific
    session's graph instead of relying on the ambient contextvar — the same
    override ``dana.session_context.session_scoped_dir`` already supports.
    Never returns ``None`` — a session with no geometry yet gets a fresh
    empty graph rather than a ``KeyError``.
    """
    sid = session_id if session_id is not None else get_session_id()
    return _TOPOLOGY_DAG_REGISTRY.setdefault(sid, {"nodes": {}, "edges": []})


def resolve_living_leaf(object_name: str) -> str:
    """Topological Amnesia mitigation (Phase 5): follows this session's
    topology_dag edges forward from ``object_name`` until reaching a node
    with no outgoing edge — a "living" leaf that hasn't itself been
    consumed by any later boolean/edge/feature operation. A name with no
    outgoing edge (never consumed, or not a known node at all) is returned
    unchanged — an unrecognized name is left for the caller's own existing
    "unknown object" validation to report, not something this should paper
    over.

    This is what makes a stale LLM reference silently self-heal with no
    tool-id-specific bookkeeping: a throwaway cutting-tool primitive (e.g. a
    Mounting_Hole cylinder) never GETS an outgoing edge until it's actually
    fed into a boolean as an input, so ``resolve_living_leaf("Mounting_Hole")``
    correctly returns it unchanged right up until that happens — unlike a
    single "most recently created" pointer, which hijacked onto it the
    instant it was created (the exact "cut the hole from itself, emit an
    empty STEP file" incident Phase 4's refinement had to special-case
    around with a tool-id allowlist).
    """
    dag = get_topology_dag()
    current = object_name
    seen = {current}
    while True:
        next_hop = None
        for edge in dag["edges"]:
            if edge["source"] == current:
                next_hop = edge["target"]  # last match wins: the most recent consumption
        if next_hop is None or next_hop in seen:
            return current
        seen.add(next_hop)
        current = next_hop


# tool_id -> the argument key names on THAT tool's call that reference an
# EXISTING object this session already created — eligible for silent
# resolve_living_leaf redirection, and recorded as a topology_dag input edge
# once the call succeeds. NOT every tool with a same-named key belongs here:
# import_and_solidify_mesh's own "object_name" names the BRAND-NEW object
# being created, not a reference to an existing one, so redirecting it would
# silently rename the caller's intended output — it's deliberately absent.
_TOPOLOGY_INPUT_ARG_KEYS: dict[str, tuple[str, ...]] = {
    "perform_freecad_boolean": ("base_object", "tool_object"),
    "perform_freecad_edge_operation": ("target_object",),
    "create_freecad_feature_on_face": ("object_name",),
    "modify_freecad_parameter": ("target_object",),
}
# export_freecad_model's target_objects is list-valued (not a single name)
# and never produces its own topology_dag node — an export is a terminal
# read, not a new object — so it's redirected separately in
# _apply_topology_redirects rather than folded into the dict above.
_EXPORT_TARGETS_ARG_KEY = "target_objects"

# Fix #3 — Kill Silent Auto-Redirection: perform_freecad_edge_operation
# (a fillet/chamfer radius) and create_freecad_feature_on_face (a feature
# anchored at named face coordinates) are TARGETED single-object
# operations — resolve_living_leaf's forward DAG walk exists to self-heal a
# STALE reference by retargeting onto whatever most recently consumed it
# (see its own docstring), but for a targeted op that "self-heal" is exactly
# the failure mode this fix removes: silently walking onto the WRONG
# consumer (e.g. an unrelated mounting hole that happens to share the same
# base object) operates on geometry the model never named, with only a
# warning buried in the result message to ever notice. perform_freecad_
# boolean/modify_freecad_parameter are deliberately NOT in this set — a
# boolean chain genuinely benefits from "the object I just referenced got
# consumed earlier this same turn" self-healing, and a wrong guess there is
# far more visible (the whole boolean fails or produces an obviously wrong
# shape) than a fillet silently landing on the wrong feature.
_STRICT_RESOLUTION_TOOLS = frozenset({"perform_freecad_edge_operation", "create_freecad_feature_on_face"})


def _resolve_strict_target(tool_id: str, key: str, requested: str) -> str:
    """Fix #3's actual resolution: the requested name must be EXACTLY a
    currently-live object in this session's topology_dag — never a fuzzy or
    probabilistic guess. Raises ``RuntimeError`` (caught by
    ``dispatch_tool_call``'s own try/except, surfaced as an ordinary
    digested tool failure) the instant either check fails, instead of
    returning some other name the caller never asked for:

    1. Unknown entirely (never created, or a typo) — nothing to resolve.
    2. Known but already CONSUMED by a later operation (has an outgoing
       topology_dag edge) — this is the exact "stale reference" case
       resolve_living_leaf used to silently walk past; now it's a hard
       failure naming the actual current object, so the model corrects its
       OWN next call instead of an invisible substitution correcting it.

    A UUID-prefixed alias (``f"{prefix}_{requested}"`` — skill_compiler's
    own ``_apply_name_prefix``/``execute_compiled_steps`` naming convention
    for a compiled skill's internal objects) is accepted when it resolves
    unambiguously (exactly one live match) — this is a deterministic,
    exact-suffix match against a known naming convention, not a fuzzy guess,
    and is the one case genuinely named in this fix's own requirements.
    """
    dag = get_topology_dag()
    nodes = dag["nodes"]
    if requested in nodes:
        live_name = requested
    else:
        alias_matches = [
            n for n in nodes if n.endswith(f"_{requested}") and len(n) > len(requested)
        ]
        if len(alias_matches) == 1:
            live_name = alias_matches[0]
        elif alias_matches:
            raise RuntimeError(
                f"{tool_id}: '{requested}' does not exist, and {len(alias_matches)} differently-"
                f"prefixed objects share that suffix ({sorted(alias_matches)}) — ambiguous, "
                "refusing to guess which one you mean. Name the exact object."
            )
        else:
            raise RuntimeError(
                f"{tool_id}: no object named '{requested}' exists in this session. Refusing to "
                "guess — verify the exact object name and retry with it exactly."
            )
    consumers = [edge["target"] for edge in dag["edges"] if edge["source"] == live_name]
    if consumers:
        raise RuntimeError(
            f"{tool_id}: '{live_name}' has already been consumed by a later operation "
            f"(-> '{consumers[-1]}') and no longer reflects the current geometry. If you mean "
            f"the current result, target '{consumers[-1]}' explicitly — it will never be "
            "substituted automatically."
        )
    return live_name


def _apply_topology_redirects(
    tool_id: str, arguments: dict[str, Any]
) -> tuple[dict[str, Any], list[str], str | None]:
    """The Silent Override, centralized: called once per dispatch, BEFORE
    the handler runs, so every consuming/terminal tool gets the same
    redirect logic instead of each duplicating its own call to
    resolve_living_leaf. Returns the (possibly rewritten) arguments the
    handler should actually run against, the resolved input object names
    this call consumes (for _record_topology_node's edge-recording once the
    call succeeds), and a combined warning string — surfaced back into the
    LLM-visible payload — for any redirect that actually fired.

    ``tool_id in _STRICT_RESOLUTION_TOOLS`` (Fix #3) never reaches
    resolve_living_leaf at all — see ``_resolve_strict_target``, which
    raises instead of substituting.
    """
    keys = _TOPOLOGY_INPUT_ARG_KEYS.get(tool_id, ())
    if not keys and tool_id != "export_freecad_model":
        return arguments, [], None

    resolved = dict(arguments)
    input_names: list[str] = []
    warnings: list[str] = []
    strict = tool_id in _STRICT_RESOLUTION_TOOLS

    for key in keys:
        raw = resolved.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        requested = raw.strip()
        if strict:
            live_name = _resolve_strict_target(tool_id, key, requested)
            resolved[key] = live_name
            input_names.append(live_name)
            continue
        living = resolve_living_leaf(requested)
        if living != requested:
            warnings.append(
                f"Note: Auto-redirected '{key}' from '{requested}' to '{living}' to reflect "
                "recent boolean/feature operations."
            )
        resolved[key] = living
        input_names.append(living)

    if tool_id == "export_freecad_model":
        raw_list = resolved.get(_EXPORT_TARGETS_ARG_KEY)
        if isinstance(raw_list, list):
            new_list = []
            for raw_name in raw_list:
                requested = str(raw_name).strip()
                living = resolve_living_leaf(requested) if requested else requested
                if living != requested:
                    warnings.append(
                        f"Note: Auto-redirected a target_objects entry from '{requested}' to "
                        f"'{living}' to reflect recent boolean/feature operations."
                    )
                new_list.append(living)
            resolved[_EXPORT_TARGETS_ARG_KEY] = new_list
            # Deliberately NOT added to input_names — export never produces
            # a topology_dag node of its own for these to be edges into.

    return resolved, input_names, (" ".join(warnings) or None)


def _record_topology_node(output_name: str, input_object_names: Iterable[str]) -> None:
    """Adds ``output_name`` as a topology_dag node and, for every input this
    call actually consumed (already resolved to ITS OWN living leaf by
    ``_apply_topology_redirects``), an edge recording that consumption. A
    self-edge (an in-place mutation like modify_freecad_parameter, whose
    output name equals its own input) is skipped — that's a same-object
    update, not a lineage step.

    Pure state mutation only — broadcasting the updated graph to the
    frontend is dana.api.server's job (mirroring how it already broadcasts
    plan_update/memory_update: a direct call site check right after
    dispatch_tool_call returns, not a callback registered here), since this
    module stays UI-agnostic and never imports the WebSocket layer.
    """
    dag = get_topology_dag()
    dag["nodes"][output_name] = {"id": output_name, "label": output_name, "type": "geometry"}
    for input_name in input_object_names:
        if input_name and input_name != output_name and input_name in dag["nodes"]:
            dag["edges"].append({"source": input_name, "target": output_name})


# Plan-and-Execute Gatekeeper — Phase 6: forces the LLM to call create_plan
# (outlining its step-by-step topological sequence, boolean strategy, and
# object naming convention) BEFORE it's allowed to dispatch any
# geometry-mutating tool, instead of improvising a boolean chain turn by
# turn and hallucinating object names along the way — the same failure mode
# the topology_dag/resolve_living_leaf machinery above heals AFTER the fact;
# this is the prevention half.
#
# Session-scoped exactly like topology_dag/_OBJECT_PATH_REGISTRY, and
# DELIBERATELY SEPARATE from dana.plugins.planning.task_board's own plan
# (the GLOBAL objective/tasks store PlanChecklist and GET /api/planner
# read): that store has no concept of "session" at all, so gating a
# session's own CAD workspace on it would let session B's geometry unlock
# the instant session A calls create_plan, and vice versa leave session B
# blocked forever behind a plan it never made. `plan_text` is stored
# verbatim (whatever _tool_create_plan formats it as) rather than derived
# from task_board's own snapshot, so this stays fully independent of that
# module's shape.
_PLAN_STATE_REGISTRY: dict[str, dict[str, Any]] = {}


def _get_has_plan(session_id: str | None = None) -> bool:
    sid = session_id if session_id is not None else get_session_id()
    return bool(_PLAN_STATE_REGISTRY.get(sid, {}).get("has_plan"))


def _set_has_plan(status: bool, plan_text: str, session_id: str | None = None) -> None:
    sid = session_id if session_id is not None else get_session_id()
    _PLAN_STATE_REGISTRY[sid] = {"has_plan": status, "plan_text": plan_text}


def _set_session_plan_tasks(
    objective: str, tasks: list[dict[str, Any]], session_id: str | None = None
) -> None:
    """Mirrors ``_tool_create_plan``'s objective/tasks into THIS session's
    entry of ``_PLAN_STATE_REGISTRY``, alongside (never replacing) the
    ``has_plan``/``plan_text`` keys ``_set_has_plan`` already wrote — this
    must run AFTER ``_set_has_plan`` in the same call, since that function
    overwrites the whole per-session dict rather than merging into it.

    Only ``_tool_create_plan`` ever populates ``tasks`` this way; a caller
    (mostly tests) that pokes ``_set_has_plan`` directly with a bare string
    leaves ``tasks`` unset, and ``_get_active_plan`` falls back to that
    plain ``plan_text`` string for them — see its own docstring.
    """
    sid = session_id if session_id is not None else get_session_id()
    entry = _PLAN_STATE_REGISTRY.setdefault(sid, {})
    entry["objective"] = objective
    entry["tasks"] = tasks


def _mark_session_task_completed(
    task_id: int, next_task_id: int | None, session_id: str | None = None
) -> None:
    """Applies the SAME status transition ``_tb_mark_task_completed`` just
    applied to task_board's global plan onto THIS session's own structured
    task list (see ``_set_session_plan_tasks``), so the next turn's Focused
    Plan Anchor (``_get_active_plan``) reflects it. A no-op if this session
    never called ``create_plan`` (no ``tasks`` key yet) — nothing to keep in
    sync.

    Also the MANUAL half of the Plan-and-Execute FSM's advancement (the
    AUTOMATIC half is ``_advance_fsm_on_dispatch``, below, called from
    ``dispatch_tool_call``'s own success path) — ``mark_task_completed``
    stays a callable tool specifically for a task with no fixed
    ``expected_tool_ids`` (a visual-only inspection, an under-specified
    task the planner's k=3 narrowing simply missed) — see this module's own
    Plan-and-Execute FSM docstring block below for the full design.
    """
    sid = session_id if session_id is not None else get_session_id()
    entry = _PLAN_STATE_REGISTRY.get(sid)
    tasks = entry.get("tasks") if entry else None
    if not tasks:
        return
    for task in tasks:
        if task.get("id") == task_id:
            task["status"] = "completed"
        elif next_task_id is not None and task.get("id") == next_task_id:
            task["status"] = "active"
    if entry.get("fsm_state") in ("executing", "validating"):
        entry["fsm_state"] = "executing" if any(t.get("status") in ("pending", "active") for t in tasks) else "done"


# ---------------------------------------------------------------------------
# Plan-and-Execute Finite State Machine (FSM) — the code-enforced replacement
# for relying on prompt text ("execute ONE task at a time", "call
# mark_task_completed") to keep the model from treating a multi-task plan as
# one big batch to blast through in a single unsupervised run (the
# "completionist reflex"). Four states, session-scoped in the SAME
# ``_PLAN_STATE_REGISTRY`` entry the plan itself already lives in (not a
# second parallel registry) since the two are never meaningfully separate —
# there is no FSM state without an active plan to be executing:
#
#   planning   -- no plan yet (or the previous one finished); the ONLY tool
#                 schema offered is create_plan + discovery tools (see
#                 next_react_turn's `hard_restrict_to`) — a geometry tool
#                 physically cannot be named because it isn't in the
#                 payload the model ever sees, not because a rule asked it
#                 not to.
#   executing  -- a plan exists; exactly one task is "active", and ONLY that
#                 task's own `expected_tool_ids` (plus core tools) are
#                 offered/protected from narrowing (next_react_turn's
#                 `narrowing_query`/`task_tool_ids`).
#   validating -- entered the instant a dispatched tool matches the active
#                 task's `expected_tool_ids` (Phase 1's "expected_tool_
#                 dispatched" event) — for that common, tool-mapped case
#                 this is DELIBERATELY momentary: `ok`/`failure` is already
#                 fully known at the exact point of dispatch, so
#                 `_advance_fsm_on_dispatch` resolves straight back to
#                 "executing" (next task, or "done") in the SAME function
#                 call, before any further LLM turn happens — there is
#                 nothing a dedicated extra turn could add for THIS case.
#                 It only genuinely PERSISTS across a turn boundary for a
#                 task with an EMPTY `expected_tool_ids` (nothing to
#                 auto-match against), where it waits for an explicit
#                 `mark_task_completed` — this is the one case
#                 `_build_validator_prompt` is actually shown for.
#   done       -- every task completed; geometry tools fall back to the
#                 SAME unrestricted, ad-hoc "idle" prompt/schema a session
#                 that never called create_plan gets (see _RESTRICTED_
#                 GEOMETRY_TOOLS's own docstring on why post-plan cleanup
#                 must stay reachable, not re-gated).
# ---------------------------------------------------------------------------

_FSM_TABLE: dict[tuple[str, str], str] = {
    ("planning", "create_plan_succeeded"): "executing",
    ("executing", "expected_tool_dispatched"): "validating",
    ("validating", "tool_ok"): "executing",
    ("validating", "tool_failed"): "executing",
    ("done", "create_plan_succeeded"): "executing",
}


def _fsm_transition(current: str, event: str) -> str:
    """Pure state-machine step — no session lookup, no side effect, so it's
    unit-testable with no FreeCAD/LLM/session machinery involved at all. An
    ``(current, event)`` pair with no table entry is a no-op (returns
    ``current`` unchanged) rather than a ``KeyError`` — an out-of-sequence
    or already-resolved event must never crash a turn, the same defensive
    posture ``task_board.mark_task_completed`` already takes for an unknown
    ``task_id``.
    """
    return _FSM_TABLE.get((current, event), current)


def _get_fsm_state(session_id: str | None = None) -> str:
    """This session's current FSM state — ``"planning"`` for any session
    with no entry yet (including one that only ever called ``_set_has_plan``
    directly with a bare string, never populating ``tasks``/``fsm_state`` at
    all — that legacy shape has no FSM concept, so it's treated exactly like
    "hasn't started planning"), never a ``KeyError``.
    """
    sid = session_id if session_id is not None else get_session_id()
    return _PLAN_STATE_REGISTRY.get(sid, {}).get("fsm_state", "planning")


def _active_task(session_id: str | None = None) -> dict[str, Any] | None:
    """The one task currently ``"active"`` in THIS session's structured plan,
    or ``None`` if there isn't one (no plan yet, or every task is done).
    """
    sid = session_id if session_id is not None else get_session_id()
    tasks = _PLAN_STATE_REGISTRY.get(sid, {}).get("tasks") or []
    return next((t for t in tasks if t.get("status") == "active"), None)


def _pending_task_count(session_id: str | None = None) -> int:
    """How many tasks remain AFTER the current active one — folded into the
    Executor prompt as "N task(s) remain after this one" instead of naming
    them, so the Executor knows there's more work without being tempted to
    look ahead at what it is (see ``_build_executor_prompt``).
    """
    sid = session_id if session_id is not None else get_session_id()
    tasks = _PLAN_STATE_REGISTRY.get(sid, {}).get("tasks") or []
    return sum(1 for t in tasks if t.get("status") == "pending")


# FSM Policy RESTORATION (Plan-and-Execute FSM, Phase 4, re-reverted): a
# geometry tool that ISN'T the active task's own `expected_tool_ids` USED to
# hard-block outright, then that hard block was REMOVED (see git history —
# the "FSM Policy Downgrade" this comment used to describe) because
# under-specifying a task's tools looked like a normal, recoverable mistake
# rather than evidence of running the plan out of order, and a wrong block
# risked refusing a tool the current task genuinely needed with nothing
# correct left to retry.
#
# That removal opened a WORSE hole in practice: with no out-of-order check
# at all, `dispatch_tool_call` never refused a dispatch on expected-tool
# grounds, so the model executed an entire multi-task CAD build while
# nominally still sitting on Task 1, then called `mark_task_completed` six
# times in a row at the very end — the FSM state never meaningfully tracked
# what the model was actually doing turn by turn, only rubber-stamped it
# retroactively. The risk the original removal was guarding against no
# longer applies: `expected_tool_ids` is now BOTH Planner-declared AND
# existence-validated at `create_plan` time (Fix #1 — Strict Plan
# Validation, above) rather than an embedding guess, so "this tool isn't in
# the active task's own declared set" is a reliable signal of running ahead
# of the plan, not of an under-specified task.
# `_fsm_out_of_order_check` below is that hard block, restored — called
# from `dispatch_tool_call` BEFORE the handler ever runs (see its own call
# site) so a mismatched tool never reaches FreeCAD (or any other plugin) at
# all, not just never auto-advances the plan.
#
# Universal FSM Enforcement (closes a SECOND hole the geometry-only scope
# left open): this check, its tracking counterpart `_advance_fsm_on_dispatch`,
# and `_tool_create_plan`'s own `expected_tool_ids` computation were ALL
# originally scoped to `_RESTRICTED_GEOMETRY_TOOLS` only. A Planner that
# declared a genuinely real, non-geometry tool for a task (e.g.
# `execute_freecad_script`, `batch_pattern_array` back when it was still
# geometry-scoped, `search_web`, ...) got that name silently narrowed away
# to an EMPTY `expected_tool_ids` — indistinguishable from "no fixed tool
# signature at all" — which let the model ignore its own declared tool
# entirely, do the work some other, undeclared way, and still pass
# `_tool_mark_task_completed`'s gate (which only fires when
# `expected_tool_ids` is non-empty). Both this check and
# `_advance_fsm_on_dispatch` now apply to EVERY tool_id except the
# always-available plan-management/discovery/self-teaching primitives in
# `_CORE_TOOL_IDS` (see `dispatch_tool_call`'s own call sites) — a
# Planner-declared tool is enforced no matter which plugin/domain it
# belongs to.


def _fsm_out_of_order_check(tool_id: str, session_id: str | None = None) -> str | None:
    """Hard-blocks a dispatch (any tool_id outside `_CORE_TOOL_IDS` — see
    the Universal FSM Enforcement comment above) that doesn't belong to the
    CURRENTLY ACTIVE task's own declared ``expected_tool_ids`` — returns a
    human-readable rejection reason for ``dispatch_tool_call`` to surface as
    an ordinary digested tool failure, or ``None`` if the dispatch may
    proceed to the handler.

    Deliberately still permissive for the two cases that are NOT "running
    ahead of the plan":

    - No active plan, or the FSM isn't in "executing" — nothing declared to
      check against yet (planning/done), or momentarily "validating" (see
      ``_advance_fsm_on_dispatch``'s own docstring on why that state is
      resolved within the SAME dispatch that entered it for a tool-mapped
      task — a later dispatch is never actually observed mid-"validating"
      for that case).
    - The active task's own ``expected_tool_ids`` came back EMPTY (a
      genuinely tool-less task, or one the Planner declared with no fixed
      tool signature) — there is nothing to enforce, so any non-core tool
      is let through exactly as before, parked in "validating" by
      ``_advance_fsm_on_dispatch`` once it succeeds.

    Everything else — a tool that belongs to neither "no fixed signature"
    nor the active task's own declared set (e.g. a later task's tool,
    dispatched early, or an undeclared tool used to route around a
    declared one) — is refused outright.
    """
    sid = session_id if session_id is not None else get_session_id()
    entry = _PLAN_STATE_REGISTRY.get(sid)
    if not entry or not entry.get("has_plan") or entry.get("fsm_state") != "executing":
        return None
    active = _active_task(sid)
    if active is None:
        return None
    expected = active.get("expected_tool_ids") or frozenset()
    if not expected or tool_id in expected:
        return None
    telemetry.log_error(
        stage="fsm_out_of_order_blocked",
        tool_id=tool_id,
        active_task_id=active["id"],
        expected_tool_ids=sorted(expected),
    )
    return (
        f"Execution blocked: '{tool_id}' is not one of task {active['id']}'s declared "
        f"expected_tools ({sorted(expected)}) — \"{active.get('description')}\". You are still on "
        f"task {active['id']}; call one of its declared tool(s) for the CURRENT task, or call "
        f"mark_task_completed(task_id={active['id']}, ...) to advance to the task that actually "
        "needs this tool first. Tools cannot be executed out of order."
    )


def _advance_fsm_on_dispatch(tool_id: str, ok: bool, session_id: str | None = None) -> str | None:
    """Deterministic Task Advancement (Plan-and-Execute FSM, Phase 4) — the
    code-enforced replacement for trusting the model to remember a SEPARATE
    ``mark_task_completed`` call after the work it was actually for. Called
    from ``dispatch_tool_call``'s own success/failure branch for every
    dispatch OUTSIDE ``_CORE_TOOL_IDS`` (Universal FSM Enforcement — see the
    module-level comment above ``_fsm_out_of_order_check``) that actually
    REACHED the handler; a no-op (returns ``None``) whenever this dispatch
    has nothing to do with FSM advancement at all — no plan, wrong phase, or
    a task with no fixed ``expected_tool_ids`` to match against. A tool that
    belongs to neither the active task's own declared set nor an empty
    ("no fixed signature") one never reaches here at all anymore —
    ``_fsm_out_of_order_check`` (see the module-level comment above it)
    refuses it BEFORE the handler runs, so this function only ever sees a
    tool_id that was either an exact match or unconstrained.

    Mirrors the update onto task_board's own GLOBAL plan too
    (``_tb_mark_task_completed``) — the SAME dual-write
    ``_tool_mark_task_completed`` already does for the manual path, so the
    frontend's PlanChecklist (reads task_board, not this session-scoped
    registry) never drifts out of sync with a session's own auto-advanced
    view.

    A task whose ``expected_tool_ids`` came back EMPTY at ``create_plan``
    time (the Planner omitted ``expected_tools`` for it — a pure visual
    inspection, say, or simply under-specified) can never satisfy the
    "tool_id in expected_tool_ids" match above, so it would otherwise never
    reach "validating" at all. For that case ONLY, any successfully
    dispatched non-core tool while it's active PARKS the FSM in
    "validating" (task status untouched — still "active") and records
    ``last_validation``, genuinely persisting across the next turn boundary
    so ``_build_validator_prompt`` has something concrete to show — the
    ONE case that prompt is actually reached for; see its own docstring.

    Returns a short human-readable note folded into the tool's own result
    payload — the model should always see WHY the plan did or didn't move,
    never a silent state change it has no way to notice.
    """
    sid = session_id if session_id is not None else get_session_id()
    entry = _PLAN_STATE_REGISTRY.get(sid)
    if not entry or not entry.get("has_plan") or entry.get("fsm_state") != "executing":
        return None
    tasks = entry.get("tasks") or []
    active = next((t for t in tasks if t.get("status") == "active"), None)
    if active is None:
        return None
    if ok:
        # Fix #2 — The False Success Blocker: record every tool successfully
        # dispatched while this task was active, regardless of whether it
        # matches expected_tool_ids — _tool_mark_task_completed's own gate
        # checks this set directly rather than inferring "was something run"
        # from FSM state alone.
        active.setdefault("executed_tools", set()).add(tool_id)
    expected = active.get("expected_tool_ids") or frozenset()
    if not expected:
        if not ok:
            return None  # a failed ad-hoc attempt has nothing to park a validation on
        entry["fsm_state"] = "validating"
        entry["last_validation"] = {"task_id": active["id"], "tool_id": tool_id, "ok": True}
        return (
            f"[Plan] Task {active['id']} has no fixed tool signature -- call "
            f"mark_task_completed(task_id={active['id']}) once you've confirmed it's done."
        )
    if tool_id not in expected:
        return None
    # "expected_tool_dispatched" -> validating, resolved immediately below —
    # see this block's own module-level docstring on why VALIDATING is
    # deliberately momentary for a tool-mapped task: ok/failure is already
    # fully known right here, before any further LLM turn.
    entry["fsm_state"] = _fsm_transition("executing", "expected_tool_dispatched")
    if not ok:
        entry["fsm_state"] = _fsm_transition(entry["fsm_state"], "tool_failed")
        return None  # nothing to advance -- the model retries the SAME active task
    active["status"] = "completed"
    next_task = next((t for t in tasks if t.get("status") == "pending"), None)
    _tb_mark_task_completed(active["id"], next_task["id"] if next_task else None)
    telemetry.log_task_state_change(
        task_id=active["id"], status="completed", auto_advanced_by=tool_id, next_task_id=next_task["id"] if next_task else None
    )
    if next_task is not None:
        next_task["status"] = "active"
        entry["fsm_state"] = _fsm_transition("validating", "tool_ok")
        return f"[Plan] Task {active['id']} complete -- now on task {next_task['id']}: {next_task['description']}."
    entry["fsm_state"] = "done"
    return f"[Plan] Task {active['id']} complete -- all tasks in the plan are now done."


def _format_focused_plan_anchor(objective: str, tasks: list[dict[str, Any]]) -> str:
    """Renders THIS session's structured plan as the Focused Plan Anchor —
    unlike task_board's own "## Current Active Plan" block (a flat dump of
    every task, done and pending alike — see ``_format_active_plan_for_prompt``),
    this deliberately DROPS completed tasks entirely and separates the
    single current task from the rest, so a model returning from several
    unrelated tool calls sees "what do I do RIGHT NOW", not the whole
    project's history, and isn't tempted to re-execute the entire plan in
    one turn.
    """
    pending = [t for t in tasks if t.get("status") == "pending"]
    active = next((t for t in tasks if t.get("status") == "active"), None)

    lines = [f"Objective: {objective}", ""]
    if pending:
        lines.append("Pending Tasks:")
        lines.extend(f"[ ] Task {t.get('id')}: {t.get('description')}" for t in pending)
        lines.append("")
    if active is not None:
        lines.append(f"--> CURRENT ACTIVE TASK: {active.get('id')}: {active.get('description')}")
        lines.append(
            "ACTION REQUIRED: Execute ONLY the geometry tools for this specific task, then "
            "call `mark_task_completed` to advance."
        )
    else:
        lines.append(
            "--> No task is currently marked active. If tasks remain pending, call "
            "mark_task_completed with a next_task_id to resume; otherwise the plan is done."
        )
    return "\n".join(lines)


def _get_active_plan(session_id: str | None = None) -> str:
    """This session's Focused Plan Anchor — ``build_system_prompt`` appends
    this (wrapped in ``=== ACTIVE PLAN ===`` markers) to every turn's system
    prompt while non-empty, so a model that just spent several turns on
    unrelated tool calls still opens its NEXT turn already reminded which
    ONE task is active right now, not just gated from deviating from a plan
    it can no longer see.

    Prefers the structured objective/tasks ``_set_session_plan_tasks``
    populated (rendered via ``_format_focused_plan_anchor``, which hides
    completed tasks and highlights only the current one) whenever this
    session's ``create_plan`` call went through the real tool handler.
    Falls back to the flat verbatim ``plan_text`` ``_set_has_plan`` stored
    for any caller (tests, mostly) that set that directly with a bare
    string and never populated structured tasks at all.
    """
    sid = session_id if session_id is not None else get_session_id()
    entry = _PLAN_STATE_REGISTRY.get(sid)
    if not entry or not entry.get("has_plan"):
        return ""
    tasks = entry.get("tasks")
    if tasks:
        return _format_focused_plan_anchor(entry.get("objective") or "", tasks)
    return entry.get("plan_text") or ""


# tool_ids that create new geometry or otherwise mutate the CAD workspace's
# topology — gated behind _get_has_plan() in dispatch_tool_call below.
# Deliberately excludes tools that only ADJUST already-existing geometry
# without creating new topology (modify_freecad_parameter,
# align_freecad_objects, create_assembly_mate) — those can't themselves
# cause a boolean-chain naming hallucination, so gating them too would only
# block legitimate post-plan cleanup with no corresponding benefit.
_RESTRICTED_GEOMETRY_TOOLS = frozenset(
    {
        "create_freecad_box",
        "create_freecad_cylinder",
        "create_freecad_extrusion",
        "create_freecad_pyramid",
        "create_freecad_star_prism",
        "create_freecad_polygon",
        "create_freecad_pipe",
        "create_freecad_sketch_extrude",
        "perform_freecad_boolean",
        "perform_freecad_edge_operation",
        "create_freecad_feature_on_face",
        "batch_pattern_array",
        "insert_standard_part",
        "import_and_solidify_mesh",
    }
)


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
    # base_object/tool_object have already been redirected to their
    # resolve_living_leaf by dispatch_tool_call's _apply_topology_redirects,
    # before this handler ever runs — see that function's docstring.
    #
    # Existence is still checked against _OBJECT_PATH_REGISTRY (populated by
    # dispatch_tool_call as a side effect of every successful create/modify
    # call), but the resolved PATH itself is no longer what's passed down —
    # every session-scoped creation tool now shares the SAME underlying
    # document, so a path alone can't tell two objects apart. apply_boolean
    # resolves base_object/tool_object by NAME against that shared session.
    if base_name not in _object_registry():
        return {
            "ok": False,
            "error": f"unknown base_object '{base_name}' — create it first with a create_freecad_* tool",
        }
    if tool_name not in _object_registry():
        return {
            "ok": False,
            "error": f"unknown tool_object '{tool_name}' — create it first with a create_freecad_* tool",
        }
    return engine.apply_boolean(operation, base_name, tool_name, name=str(args.get("name") or "").strip() or None)


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
    # target_object has already been redirected to its resolve_living_leaf
    # by dispatch_tool_call's _apply_topology_redirects, before this handler
    # ever runs — this is the exact "fillet the base object after a boolean
    # chain already consumed it" case that mechanism exists for.
    #
    # Existence check only — see perform_freecad_boolean's matching comment
    # on why the resolved PATH itself is no longer what gets passed down:
    # apply_edge_operation now resolves target_object by NAME against the
    # shared session, same as apply_boolean/modify_parameter.
    if target_name not in _object_registry():
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
        operation,
        target_name,
        value,
        face_centroid=face_centroid,
        name=str(args.get("name") or "").strip() or None,
    )


# "Placement"/"Placement.Base" are the one dimensional-ish property that
# isn't a bare number — a 3D translation — so modify_freecad_parameter
# accepts a [x, y, z] vector for either spelling instead of requiring a
# single float.
_VECTOR_PARAMETER_NAMES = frozenset({"placement", "placement.base"})


def _parse_vector_new_value(raw_value: Any) -> tuple[float, ...] | None:
    """Parse a 3-number ``[x, y, z]`` or 6-number ``[x, y, z, yaw, pitch,
    roll]`` vector from a list-like value or a JSON-array string such as
    ``"[30, 20, 10]"`` or ``"[30, 20, 10, 90, 0, 0]"``. Returns ``None`` on
    anything that isn't exactly 3 or 6 numbers — never raises. The 6-number
    form (mm position + DEGREES Euler rotation — see
    ``dana.plugins.freecad.engine.modify_parameter``'s own docstring for
    the live-confirmed units) lets a kinematic-assembly caller move AND
    orient a part in one call instead of two."""
    components = raw_value
    if isinstance(components, str):
        try:
            components = json.loads(components)
        except (TypeError, ValueError):
            return None
    if not isinstance(components, (list, tuple)) or len(components) not in (3, 6):
        return None
    try:
        return tuple(float(c) for c in components)
    except (TypeError, ValueError):
        return None


def _tool_modify_freecad_parameter(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    target_name = str(args.get("target_object") or "").strip()
    if not target_name:
        return {"ok": False, "error": "modify_freecad_parameter requires target_object"}
    # Existence check only — see perform_freecad_boolean's matching comment
    # on why the resolved PATH itself is no longer what gets passed down.
    if target_name not in _object_registry():
        return {
            "ok": False,
            "error": f"unknown target_object '{target_name}' — create it first with a create_freecad_* tool",
        }
    parameter_name = str(args.get("parameter_name") or "").strip()
    if not parameter_name:
        return {"ok": False, "error": "modify_freecad_parameter requires parameter_name"}
    raw_value = args.get("new_value")
    if parameter_name.lower() in _VECTOR_PARAMETER_NAMES:
        vector = _parse_vector_new_value(raw_value)
        if vector is None:
            return {
                "ok": False,
                "error": (
                    "modify_freecad_parameter: Placement new_value must be a 3-number "
                    "[x, y, z] vector (e.g. '[30, 20, 10]') or a 6-number "
                    "[x, y, z, yaw, pitch, roll] vector in mm + degrees "
                    f"(e.g. '[30, 20, 10, 90, 0, 0]') — got {raw_value!r}"
                ),
            }
        new_value: float | tuple[float, ...] = vector
    else:
        try:
            new_value = float(raw_value)
        except (TypeError, ValueError):
            return {"ok": False, "error": "modify_freecad_parameter requires a numeric new_value"}
    return engine.modify_parameter(target_name, parameter_name, new_value)


def _tool_get_freecad_bounding_box(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    target_name = str(args.get("target_object") or "").strip()
    if not target_name:
        return {"ok": False, "error": "get_freecad_bounding_box requires target_object"}
    target_path = _object_registry().get(target_name)
    if not target_path:
        return {
            "ok": False,
            "error": f"unknown target_object '{target_name}' — create it first with a create_freecad_* tool",
        }
    return engine.get_bounding_box(target_path, target_object=target_name)


def _tool_inspect_spatial_properties(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    target_name = str(args.get("target_object") or "").strip()
    if not target_name:
        return {"ok": False, "error": "inspect_spatial_properties requires target_object"}
    target_path = _object_registry().get(target_name)
    if not target_path:
        return {
            "ok": False,
            "error": f"unknown target_object '{target_name}' — create it first with a create_freecad_* tool",
        }
    return engine.inspect_spatial_properties(target_path, target_object=target_name)


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
    path_a = _object_registry().get(name_a)
    if not path_a:
        return {"ok": False, "error": f"unknown object_a '{name_a}' — create it first with a create_freecad_* tool"}
    path_b = _object_registry().get(name_b)
    if not path_b:
        return {"ok": False, "error": f"unknown object_b '{name_b}' — create it first with a create_freecad_* tool"}

    bbox_a = engine.get_bounding_box(path_a, target_object=name_a)
    if not bbox_a.get("ok"):
        return {"ok": False, "error": f"failed to read {name_a}'s bounding box: {bbox_a.get('error')}"}
    bbox_b = engine.get_bounding_box(path_b, target_object=name_b)
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

    source_path = _object_registry().get(source_name)
    if not source_path:
        return {
            "ok": False,
            "error": f"unknown source_object '{source_name}' — create it first with a create_freecad_* tool",
        }
    target_path = _object_registry().get(target_name)
    if not target_path:
        return {
            "ok": False,
            "error": f"unknown target_object '{target_name}' — create it first with a create_freecad_* tool",
        }
    return engine.align_objects(
        source_path, target_path, alignment_type, source_object=source_name, target_object=target_name
    )


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

    fixed_path = _object_registry().get(fixed_name)
    if not fixed_path:
        return {"ok": False, "error": f"unknown fixed_obj '{fixed_name}' — create it first with a create_freecad_* tool"}
    moving_path = _object_registry().get(moving_name)
    if not moving_path:
        return {
            "ok": False,
            "error": f"unknown moving_obj '{moving_name}' — create it first with a create_freecad_* tool",
        }

    mate_params = args.get("mate_params")
    if not isinstance(mate_params, dict):
        mate_params = {}
    return engine.create_assembly_mate(
        fixed_path, moving_path, mate_type, mate_params, fixed_object=fixed_name, moving_object=moving_name
    )


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

    # target_objects entries have already been redirected to their
    # resolve_living_leaf by dispatch_tool_call's _apply_topology_redirects,
    # before this handler ever runs — exporting "MotorBracket" after it's
    # been cut into "Cut_Mounting_Holes" is exactly the case that heals.
    target_paths = []
    resolved_names = []
    for raw_name in target_names:
        name = str(raw_name).strip()
        path = _object_registry().get(name)
        if not path:
            return {
                "ok": False,
                "error": f"unknown target object '{name}' — create it first with a create_freecad_* tool",
            }
        target_paths.append(path)
        resolved_names.append(name)

    return engine.export_model(target_paths, export_format, filename, target_objects=resolved_names)


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
    source_path = _object_registry().get(object_name)
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


def _tool_create_freecad_feature_on_face(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    object_name = str(args.get("object_name") or "").strip()
    if not object_name:
        return {"ok": False, "error": "create_freecad_feature_on_face requires object_name"}
    # object_name has already been redirected to its resolve_living_leaf by
    # dispatch_tool_call's _apply_topology_redirects, before this handler
    # ever runs — see that function's docstring.
    try:
        u = float(args.get("u"))
        v = float(args.get("v"))
        extent = float(args.get("extent"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "create_freecad_feature_on_face requires numeric u, v, extent"}
    radius = args.get("radius")
    width = args.get("width")
    length = args.get("length")
    try:
        radius = float(radius) if radius is not None else None
        width = float(width) if width is not None else None
        length = float(length) if length is not None else None
    except (TypeError, ValueError):
        return {"ok": False, "error": "create_freecad_feature_on_face: radius/width/length must be numeric if given"}
    return engine.create_feature_on_face(
        object_name,
        str(args.get("face") or ""),
        str(args.get("shape") or ""),
        u,
        v,
        extent,
        str(args.get("operation") or ""),
        radius=radius,
        width=width,
        length=length,
        name=str(args.get("name")) if args.get("name") else None,
    )


_PATTERN_TYPES = frozenset({"linear", "grid", "circular"})


def _tool_batch_pattern_array(args: dict[str, Any], engine: Any, _cp: Any) -> dict[str, Any]:
    source_name = str(args.get("source_object") or "").strip()
    if not source_name:
        return {"ok": False, "error": "batch_pattern_array requires source_object"}
    source_path = _object_registry().get(source_name)
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
        source_object=source_name,
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
    length = args.get("length")
    return json.loads(
        insert_standard_part(
            part_type,
            specification=str(args.get("specification") or ""),
            name=str(args.get("name") or "").strip() or None,
            placement=_extract_placement(args),
            fastener_type=str(args.get("fastener_type") or ""),
            size=str(args.get("size") or ""),
            length=float(length) if length is not None else None,
        )
    )


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any], Any, Any], dict[str, Any]]] = {
    "create_freecad_box": _tool_create_box,
    "generate_urdf_assembly": _tool_generate_urdf_assembly,
    "generate_3d_from_image": _tool_generate_3d_from_image,
    "import_and_solidify_mesh": _tool_import_and_solidify_mesh,
    "query_geometry_properties": _tool_query_geometry_properties,
    "read_system_architecture": _tool_read_system_architecture,
    "create_freecad_cylinder": _tool_create_cylinder,
    "create_freecad_pyramid": _tool_create_freecad_pyramid,
    "create_freecad_star_prism": _tool_create_freecad_star_prism,
    "create_freecad_polygon": _tool_create_freecad_polygon,
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
    "create_freecad_feature_on_face": _tool_create_freecad_feature_on_face,
    "batch_pattern_array": _tool_batch_pattern_array,
    "insert_standard_part": _tool_insert_standard_part,
    "resync_workspace": _tool_resync_workspace,
    "get_active_display": _tool_get_active_display,
    "prevent_focus_steal": _tool_prevent_focus_steal,
    "system_state": _tool_system_state,
    "check_plugin_registry": _tool_check_plugin_registry,
    "load_capability": _tool_load_capability,
    "unload_capability": _tool_unload_capability,
    "search_tool_catalog": _tool_search_tool_catalog,
    "load_specific_tool": _tool_load_specific_tool,
    "update_core_memory": _tool_update_core_memory,
    "create_plan": _tool_create_plan,
    "mark_task_completed": _tool_mark_task_completed,
    "compile_plan_as_skill": _tool_compile_plan_as_skill,
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
    if call.tool_id == "generate_3d_from_image":
        image_path = call.arguments.get("image_path", "an image")
        return (
            f"Send {image_path} to a free third-party Hugging Face Space (TripoSR, "
            "with CRM/zero123plus as fallbacks) to reconstruct it into a 3D mesh."
        )
    if call.tool_id == "import_and_solidify_mesh":
        mesh_path = call.arguments.get("mesh_path", "?")
        object_name = call.arguments.get("object_name", "?")
        return f"Import mesh '{mesh_path}' and solidify it as `{object_name}` in FreeCAD."
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
    if call.tool_id == "create_freecad_polygon":
        sides = call.arguments.get("sides", 6)
        radius = call.arguments.get("radius", 50)
        height = call.arguments.get("height", 10)
        return f"Create a regular {sides}-sided polygon (radius {radius}mm, thickness {height}mm) in FreeCAD."
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
        if str(param).lower() in _VECTOR_PARAMETER_NAMES:
            return f"Move `{target}` to {value} in FreeCAD."
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
    if call.tool_id == "create_freecad_feature_on_face":
        target = call.arguments.get("object_name", "?")
        face = str(call.arguments.get("face") or "?")
        shape = str(call.arguments.get("shape") or "?")
        op = "Cut" if call.arguments.get("operation") == "cut" else "Add"
        return f"{op} a {shape} on the {face} face of `{target}` in FreeCAD."
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
        if part_type == "fastener":
            spec = f"{call.arguments.get('fastener_type', '?')} {call.arguments.get('size', '?')}x{call.arguments.get('length', '?')}"
        else:
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
        "read_system_architecture",  # self-description (docs/architecture.md + tool schema
        # summary) — same "core capability with no plugin active at all" reasoning as
        # check_plugin_registry directly above; complements it rather than duplicating it.
        "load_capability",  # autonomous semantic routing — see below; always available so the
        # agent can retrieve a domain it wasn't handed, without needing a frontend plugin for it.
        "unload_capability",  # its explicit release valve — always available for the same reason,
        # so the agent can drop a domain it's done with without waiting out the decay clock.
        "search_tool_catalog",  # Lazy Loading, Layer 3 — discovery half; always available so the
        # agent can find a tool outside load_capability's fixed domain list without ever stalling.
        "load_specific_tool",  # Lazy Loading, Layer 3 — fetch half; always available for the same
        # reason, and it's the only way a tool found via search_tool_catalog becomes callable.
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
        "compile_plan_as_skill",  # Composite Skill Compiler — see
        # dana.plugins.freecad.skill_compiler; always available for the same "not gated behind
        # a specific plugin" reasoning as create_plan/save_new_skill, which this composes.
    }
)

_FREECAD_TOOL_IDS = frozenset(
    {
        "create_freecad_box",
        "create_freecad_cylinder",
        "create_freecad_extrusion",
        "create_freecad_pyramid",
        "create_freecad_star_prism",
        "create_freecad_polygon",
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
        "create_freecad_feature_on_face",
        "batch_pattern_array",
        "insert_standard_part",
        "manipulate_camera",
        "resync_workspace",
        "get_active_display",
        "prevent_focus_steal",
        "query_engineering_standard",
        "generate_urdf_assembly",  # Phase 2 Robotics: assembles create_freecad_*-generated
        # .stl parts into a kinematic URDF — same "freecad"/"freecad_full" domain as the
        # CAD primitives it consumes, not a separate "robotics" domain (which would need
        # its own load_capability/frontend wiring nothing currently activates).
        "query_geometry_properties",  # Spatial Awareness — the precise-joint-origin
        # companion to generate_urdf_assembly above; same domain for the same reason.
        "generate_3d_from_image",  # Image-to-3D via free Hugging Face Spaces
        # (dana.tools.image_to_3d) — its output feeds the same CAD pipeline
        # (inspect/export/assemble) create_freecad_*'s own output does, so
        # same domain rather than a separate one nothing currently activates.
        "import_and_solidify_mesh",  # Mesh-to-solid bridge (dana.plugins.freecad
        # .mesh_ops) — the mandatory next step for a generate_3d_from_image
        # result before any perform_freecad_*/create_assembly_mate call can
        # touch it; same domain as both.
    }
)

# Trimmed default shared by BOTH the agent's own autonomous
# `load_capability` call (via _tool_load_capability's "freecad" ->
# "freecad_essential" redirect above) and the frontend's CAD-tab activation
# (dana.api.server._PLUGIN_ID_TO_CAPABILITY maps "cad" -> "freecad_essential"
# directly). The raw "freecad" domain name still exists and still resolves
# to the full _FREECAD_TOOL_IDS set below — for whichever caller passes it
# explicitly, or for "freecad_full"'s own definition — it's just no longer
# what either normal activation path reaches by default. Root cause of the
# parse-0/parse-1 failure this addresses: sending all 24 FreeCAD tool
# schemas (~30KB / ~6.8-6.9k tokens serialized, confirmed by direct probe
# against Groq) as `tools=` on the VERY NEXT turn after load_capability
# reliably blows a free/on-demand-tier Groq model's tokens-per-minute
# ceiling (observed live: HTTP 429 "Rate limit reached ... tokens per
# minute (TPM): Limit 8000, Used 6895, Requested 6842" on
# openai/gpt-oss-120b) — a 400 "too many tools" was the original
# hypothesis, but the actual failure is TPM exhaustion; cutting the
# schema payload is still the correct fix either way. These 9 cover the
# overwhelming majority of "make me a basic part" requests (box/cylinder,
# one boolean op, one edge op, a parameter tweak, a bounding-box check,
# export, standard hardware, an AI-generated-mesh-to-solid bridge) without
# the model ever seeing the heavier/rarer schemas (patterns, assembly
# mates, blueprints, camera control). The agent can still reach the full
# set any time a task genuinely needs it by calling load_capability again
# with domain="freecad_full".
_FREECAD_ESSENTIAL_TOOL_IDS = frozenset(
    {
        "create_freecad_box",
        "create_freecad_cylinder",
        "perform_freecad_boolean",
        "perform_freecad_edge_operation",
        "modify_freecad_parameter",
        "get_freecad_bounding_box",
        "export_freecad_model",
        "insert_standard_part",  # frequently needed for basic hardware (bolts/nuts/
        # bearings/fasteners) and previously essential-only via a load_capability
        # ("freecad_full") hunt — the agent would spend several turns trying to reach it
        # (or give up) before ever calling it. Its own tools.json description already
        # says "Always prefer this over hand-modeling a named standard part," so it
        # needs to be reachable from turn one, same as the other essential primitives.
        "import_and_solidify_mesh",  # Mesh-to-solid bridge (dana.plugins.freecad
        # .mesh_ops) — the mandatory next step for a generate_3d_from_image result
        # before any perform_freecad_*/create_assembly_mate call can touch it.
        # Previously freecad_full-only: a live E2E run asked to solidify+union an
        # AI-generated mesh never found it (load_capability("freecad") redirects to
        # this essential set, and search_tool_catalog's old embedding-only search
        # failed to surface it), so the agent fabricated a "fused result" bounding
        # box without ever calling this tool or perform_freecad_boolean. Same
        # "reachable from turn one" reasoning as insert_standard_part above.
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

    Also (re)registers every loaded skill into ``dana.tools.registry
    .get_tool_registry()`` — the SEPARATE embedding-indexed catalog
    ``search_tool_catalog``/Pillar 1's ``narrow_tool_ids_by_query`` actually
    read from, distinct from the ``load_tool_registry()``-backed cache
    ``_llm_tools_schema_cached`` uses for the wire schema. Before this, a
    user skill was dispatchable and offered in ``tools=`` (universe #1
    above) but invisible to semantic search/narrowing (universe #2) — it
    could only survive a turn's narrowing pass via ``sticky_ids``/
    ``always_include``, never be FOUND by ``search_tool_catalog`` or a
    fresh top-K narrowing match on a cold turn. ``source="forge"`` marks
    these as runtime-synthesized (as opposed to ``"tools.json"``-defined)
    for anything downstream that cares about provenance; ``dynamic=False``
    (ToolSpec's default) keeps ``is_bindable_tool`` returning True even
    with no ``callable`` bound on the registry entry — actual dispatch
    still goes exclusively through ``TOOL_HANDLERS``/``dispatch_tool_call``,
    never ``ToolRegistry.execute()``.
    """
    global _LLM_TOOL_IDS
    result = load_user_skills()
    fresh_skills: dict[str, dict[str, Any]] = result.get("skills", {})

    tool_registry = get_tool_registry()
    for stale_id in _USER_SKILL_TOOL_IDS - fresh_skills.keys():
        TOOL_HANDLERS.pop(stale_id, None)
        _USER_SKILL_SCHEMAS.pop(stale_id, None)
        tool_registry.unregister(stale_id)

    _USER_SKILL_TOOL_IDS.clear()
    for tool_id, entry in fresh_skills.items():
        TOOL_HANDLERS[tool_id] = _wrap_skill_handler(entry["handler"])
        _USER_SKILL_SCHEMAS[tool_id] = entry["schema"]
        _USER_SKILL_TOOL_IDS.add(tool_id)
        tool_registry.register(
            _tool_spec_from_openai_schema(tool_id, entry["schema"]),
            source="forge",
            metadata={"origin": "user_skill"},
        )

    _CAPABILITY_TOOL_IDS["user_skills"] = frozenset(_USER_SKILL_TOOL_IDS)
    _LLM_TOOL_IDS = _CORE_TOOL_IDS.union(*_CAPABILITY_TOOL_IDS.values())
    _tool_ids_for_plugins.cache_clear()
    _llm_tools_schema_cached.cache_clear()
    return result


def _tool_spec_from_openai_schema(tool_id: str, schema: dict[str, Any]) -> ToolSpec:
    """Builds a ``dana.tools.schema.ToolSpec`` — the shape
    ``dana.tools.registry.ToolRegistry.register`` needs for its embedding
    index — from a full OpenAI-wire tool schema (``{"type": "function",
    "function": {...}}``, the shape both ``_USER_SKILL_SCHEMAS`` and a
    compiled skill's own schema already use). Populates ``parameters`` too
    (not just the description) so the embedded text
    (``dana.tools.registry._spec_schema_text``) includes each parameter's
    name/description/type — the same retrieval signal a natively
    ``tools.json``-defined tool already gets, not a degraded copy.
    """
    fn = schema.get("function") or {}
    props = ((fn.get("parameters") or {}).get("properties")) or {}
    required = set((fn.get("parameters") or {}).get("required") or ())
    params = tuple(
        ToolParameterSpec(
            name=str(name),
            type=str(prop.get("type") or "string"),
            required=name in required,
            enum=tuple(str(e) for e in (prop.get("enum") or [])),
            description_en=str(prop.get("description") or ""),
        )
        for name, prop in props.items()
    )
    return ToolSpec(
        id=tool_id,
        description_en=str(fn.get("description") or tool_id),
        description_fa="",
        parameters=params,
    )


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
    fn: Callable[..., dict[str, Any]],
) -> Callable[[dict[str, Any], Any, Any], dict[str, Any]]:
    """Adapts a manifest.json plugin's entrypoint to ``TOOL_HANDLERS``'s
    uniform ``(arguments, engine, control_plane)`` signature — identical
    reasoning to ``_wrap_skill_handler`` above: a generic plugin (a codebase
    search, a shell wrapper, a FreeCAD script runner, ...) has no business
    touching the CAD engine or control plane.

    Two different manifest-plugin calling conventions exist in this
    codebase, and this wrapper must serve both:
      - ``dana.plugins.coder_plugin.engine``'s functions each take exactly
        ONE parameter, literally named ``args`` (a dict), and do their own
        ``args.get(...)`` extraction internally.
      - ``dana/plugins/freecad/manifest.json``'s genuinely-new tool ids
        (``execute_freecad_script``, ``modify_existing_document``) instead
        take one named parameter PER manifest ``"parameters"`` entry — the
        same convention this module's own native FreeCAD handlers already
        use (e.g. ``_tool_create_box`` unpacking into
        ``engine.create_box(length=..., width=..., ...)``).

    Always calling ``fn(args)`` (passing the whole dict positionally) used
    to silently bind that dict to a freecad-style function's FIRST named
    parameter instead of raising — Python allows a dict as any type's
    positional argument — which is exactly how ``execute_freecad_script``
    ended up calling ``.strip()`` on a dict rather than the script string:
    ``execute_freecad_script({"python_script_str": "..."})`` bound the
    entire arguments dict to its one parameter, ``python_script_str``.

    Inspecting the wrapped function's OWN signature here (once, at
    registration time, not per call) resolves this for every current and
    future plugin instead of special-casing one tool id: a function
    declaring exactly one parameter literally named ``args`` gets the whole
    dict passed positionally (coder_plugin's convention); every other shape
    gets the dict unpacked as keyword arguments matching each parameter by
    name (freecad's, and any future plugin's, convention) — a missing
    required key or an unexpected extra one then raises a normal
    ``TypeError``, which ``dispatch_tool_call``'s existing try/except +
    ``digest_error`` already turns into a structured, LLM-actionable
    failure payload exactly like any other handler exception.
    """
    try:
        takes_single_args_dict = list(inspect.signature(fn).parameters) == ["args"]
    except (TypeError, ValueError):  # a builtin/C-extension callable with no inspectable signature
        takes_single_args_dict = False

    def _handler(args: dict[str, Any], _engine: Any, _cp: Any) -> dict[str, Any]:
        result = fn(args) if takes_single_args_dict else fn(**args)
        # dana.plugins.freecad.engine's functions (e.g. modify_existing_document,
        # execute_freecad_script — the two manifest.json tool ids that don't
        # collide with a native handler and so actually reach this wrapper at
        # runtime) return a JSON-encoded STRING, per that module's own "All
        # public functions return a JSON string" contract — built for the
        # older tool-broker stack, not this dict-expecting one.
        # dispatch_tool_call's payload.get("ok", True) crashes with
        # AttributeError on a raw string (confirmed live:
        # modify_existing_freecad_document), so parse it here, the one place
        # every manifest-plugin result already flows through, rather than in
        # every individual FreeCAD-engine function or at the dispatch
        # chokepoint itself. Left untouched for a plugin (e.g. coder_plugin)
        # whose functions already return a real dict.
        if isinstance(result, str):
            result = json.loads(result)
        return result

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
                telemetry.debug(
                    "plugin domain %r declares tool_id %r, which collides with an existing "
                    "built-in handler — skipping the plugin's version; the built-in remains "
                    "authoritative.",
                    domain,
                    spec.id,
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


# Explicitly-activated, CURATED-SMALL domains (dana.api.server.
# _PLUGIN_ID_TO_CAPABILITY — a human's own tab choice) whose tool set is
# small enough that per-turn semantic narrowing only ever misfires against
# it, never actually saves meaningful tokens. Confirmed live for both: an
# off-topic query (e.g. "hello") while only "software_engineering" (4 tools,
# coder_plugin) or "freecad_essential" (7 tools) was active narrowed EVERY
# domain tool away, silently breaking the Coder/CAD tab for that turn — the
# curated sets are already well under Groq's TPM ceiling on their own, so
# exempting them from narrowing costs nothing and fixes a real "context
# drop" bug, not a workaround for a token problem that still exists after
# the exemption.
#
# Deliberately does NOT include "freecad" (raw, ~26 tools) or "freecad_full"
# (the explicit heavy-tool escalation domain, same ~26): unlike the curated
# sets here, unioning either of these on top of an already-active
# "freecad_essential" (as load_capability(domain="freecad_full") does the
# moment a task needs one heavier tool) is exactly the "8000+ token" TPM
# failure this whole essential/full split exists to prevent — measured live:
# freecad_essential ∪ freecad_full = 6,131 tokens of tool schema alone, before
# the system prompt or any conversation history. Narrowing "freecad_full" by
# query relevance (same mechanism web_tools/vision_tools already get) still
# leaves the ONE relevant heavy tool reachable for the turn that actually
# asked for it — it just stops serializing all 26 at once for a domain that,
# unlike the curated sets here, was never meant to be small.
#
# "os_tools" (11 tools) was added alongside the two above once
# _TOOL_TOKEN_BUDGET was tightened from 4000 to 2000 (see that constant's own
# history) — at 4000 its full set plus core always fit comfortably so this
# never mattered in practice, but a tighter budget exposed the exact same
# "activated tab silently loses tools" bug the original two domains were
# exempted to prevent (list_directory — about as basic an os_tools call as
# exists — dropping out on a query-less/off-topic turn). Same size class as
# freecad_essential/software_engineering, same justification; not the
# same-order-of-magnitude case "freecad"/"freecad_full" above are excluded for.
_NARROWING_EXEMPT_DOMAINS = ("freecad_essential", "software_engineering", "os_tools")

# Per-TOOL narrowing exemption, one level more granular than the per-DOMAIN
# exemption above. All five of these are already members of
# _FREECAD_ESSENTIAL_TOOL_IDS, so in the common case (CAD tab open, or the
# agent's own load_capability("freecad"/"freecad_essential") redirect) they're
# ALREADY protected by "freecad_essential" being in _NARROWING_EXEMPT_DOMAINS.
# This is the backstop for the one case that isn't: a session where
# "freecad_essential" has decayed out of active_plugins (P1's per-session
# decay, dana.api.server._effective_capabilities) while "freecad_full" is
# still active on its own — that raw ~26-tool set is deliberately NOT
# narrowing-exempt (see the comment above _NARROWING_EXEMPT_DOMAINS on why),
# so without this, a query whose wording doesn't happen to score
# create_freecad_box/cylinder highly could silently narrow the model's own
# most basic geometry primitives out of tools= for that turn — indistinguishable,
# from the model's side, from those tools never having existed, and the exact
# shape of bug that produces a hallucinated tool name instead. Folded ONLY
# into narrow_tool_ids_by_query's `always_include` below (Pillar 1's
# query-relevance narrowing) — deliberately NOT into
# _cap_schemas_by_token_budget's `must_keep` (Pillar 1's hard per-turn TOKEN
# ceiling): that cap exists specifically to guarantee a Groq TPM 429 can
# never happen again (see the module comment above _TOOL_TOKEN_BUDGET), and
# `must_keep` is NEVER trimmed even when it alone busts the budget — adding
# 5 more (potentially large) FreeCAD schemas to that unconditional set
# measurably pushed the "freecad"/"freecad_full" regression tests past
# their ceiling. Only ever keeps ids already present in the candidate set,
# so listing one here that isn't currently eligible (no freecad domain
# active at all) is always a safe no-op, never a capability leak.
_ALWAYS_EXEMPT_TOOLS = frozenset(
    {
        "create_freecad_box",
        "create_freecad_cylinder",
        "perform_freecad_boolean",
        "perform_freecad_edge_operation",
        "export_freecad_model",
    }
)


def _llm_tools_schema(
    active_plugins: frozenset[str] | None = None,
    *,
    query: str = "",
    sticky_ids: frozenset[str] = frozenset(),
    force_include: frozenset[str] = frozenset(),
    hidden_tool_ids: frozenset[str] = frozenset(),
    task_tool_ids: frozenset[str] = frozenset(),
    hard_restrict_to: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Build this turn's ``tools=`` payload — capability-gated tool ids,
    optionally narrowed by semantic relevance to ``query`` (Pillar 1:
    ``dana.core.tool_retrieval.narrow_tool_ids_by_query``, a no-op when
    ``query`` is empty or the allowed set is already small), then minified
    (Pillar 2: ``dana.tools.schema_minify.minify_tool_schemas``, condenses
    verbose descriptions without touching the OpenAI wire contract), then —
    for this turn's resolved tool-calling provider only (local Ollama by
    default, or any provider when ``DANA_MINIFY_TOOL_SCHEMAS=true`` — see
    ``should_strip_tool_schemas``) — stripped further still
    (``dana.tools.schema_minify.strip_tool_schemas`` drops every description
    outright instead of condensing it, on top of the Pillar-2 pass above).

    ``sticky_ids`` (tool ids already dispatched earlier in THIS turn's
    multi-step ReAct chain — see ``_sticky_tool_ids_from_messages``) always
    survive narrowing alongside ``_CORE_TOOL_IDS``, so a sequence like
    create-then-boolean-op never loses access to the tool it's already
    mid-sequence with just because the query text alone wouldn't have
    scored it highly.

    ``force_include`` (Lazy Loading, Layer 3 — tool ids explicitly unlocked
    via ``load_specific_tool``) is unioned directly into the capability-gated
    ``tool_ids`` itself, BEFORE narrowing — unlike ``sticky_ids`` (which only
    protects an already-eligible id from being narrowed back out),
    ``force_include`` makes an id eligible in the first place even when no
    active domain covers it. Callers should also fold ``force_include`` into
    ``sticky_ids`` so the same id survives the token-budget cap below, not
    just domain gating.

    ``hidden_tool_ids`` (Dynamic Domain Locking — ``unload_capability``
    called with a ``tool_id`` instead of a ``domain``, see
    ``_tool_unload_capability``) is the inverse of ``force_include``:
    subtracted from the domain-gated set BEFORE ``force_include`` is
    unioned back in, so a hide can suppress an otherwise-reachable tool
    (e.g. ``create_freecad_cylinder`` while ``batch_pattern_array`` should
    be used instead) without needing its own domain boundary, while an
    explicit ``force_include``/``load_specific_tool`` call still wins as a
    deliberate override — the same escape hatch it already is for domain
    gating.

    ``task_tool_ids`` (Plan-and-Execute FSM, Phase 3 — the active plan
    task's own ``expected_tool_ids`` while ``fsm_state == "executing"``) is
    folded into narrowing's ``always_include`` alongside ``_ALWAYS_EXEMPT_
    TOOLS`` AND (unlike ``_ALWAYS_EXEMPT_TOOLS``) into the token-budget's
    ``must_keep`` too: it's normally at most k=3 ids per task (see
    ``_tool_create_plan``), negligible token cost, so — unlike the
    domain-wide protections `must_keep` deliberately excludes (see that
    comment below) — there's no TPM-budget reason not to unconditionally
    guarantee the one tool THIS task actually needs survives every cut.

    ``hard_restrict_to`` (Plan-and-Execute FSM, Phase 2 — the PLANNING
    phase) is a genuine hard allow-list, not one more input to the same
    best-effort pipeline every other caller goes through: when given, this
    bypasses domain gating, semantic narrowing, sticky/force_include, and
    the token budget entirely, returning schemas for EXACTLY that id set
    (intersected with what the registry actually has). The Planner must not
    be able to see — let alone hallucinate a call to — any geometry tool at
    all, and a soft narrowing bias (however strongly weighted) is not a
    structural guarantee of that; a fixed allow-list is.
    """
    if hard_restrict_to is not None:
        schemas = list(_llm_tools_schema_cached(frozenset(hard_restrict_to) - hidden_tool_ids))
        minified = minify_tool_schemas(schemas)
        return strip_tool_schemas(minified) if should_strip_tool_schemas(tool_calling_provider()) else minified
    tool_ids = (_tool_ids_for_plugins(active_plugins) - hidden_tool_ids) | force_include
    # Populated below only for a capability-aware caller with a narrowing-
    # exempt domain active; also folded into the FINAL token-budget's
    # must_keep (not just narrowing's always_include) below — a domain
    # exempted from narrowing because it's a hand-curated, human-activated
    # tab (see _NARROWING_EXEMPT_DOMAINS's own docstring) must stay exempt
    # from the budget cut too, or a tight budget silently reintroduces the
    # exact "context drop" bug the exemption exists to prevent.
    protected: frozenset[str] = frozenset()
    if active_plugins is not None:  # None = legacy "not capability-aware" caller — full set, no narrowing
        # Domains in _NARROWING_EXEMPT_DOMAINS (see its own docstring above)
        # are deliberately exempt from query-relevance narrowing — a real
        # CAD/Coder-panel session gets its FULL activated set, unchanged,
        # exactly as before Pillar 1 existed. Narrowing still applies to
        # every other simultaneously-active domain (os_tools, web_tools,
        # vision_tools, user_skills, and any future plugin) — that's where
        # "scales to dozens of plugins" actually matters, since these are
        # the domains a human already hand-tuned.
        #
        # Reads _CAPABILITY_TOOL_IDS live (not a frozen literal) — "freecad"
        # can grow past its original 24 at runtime via refresh_plugin_tools()'s
        # manifest.json union (e.g. dana/plugins/freecad/manifest.json's
        # modify_existing_freecad_document/execute_freecad_script). Protecting
        # only an original literal would leave those manifest-added tools
        # unprotected, so Pillar 1 could silently narrow them back out on the
        # very query that most needed a full, hand-tuned domain — the exact
        # "context drop" bug class sticky_ids exists to prevent elsewhere in
        # this module.
        protected = (
            frozenset().union(
                *(_CAPABILITY_TOOL_IDS.get(name, frozenset()) for name in _NARROWING_EXEMPT_DOMAINS)
            )
            if (active_plugins & set(_NARROWING_EXEMPT_DOMAINS))
            else frozenset()
        )
        narrowed = narrow_tool_ids_by_query(
            tool_ids - protected,
            query,
            always_include=_CORE_TOOL_IDS | sticky_ids | _ALWAYS_EXEMPT_TOOLS | task_tool_ids,
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
    minified = minify_tool_schemas(schemas)
    if should_strip_tool_schemas(tool_calling_provider()):
        minified = strip_tool_schemas(minified)
    if active_plugins is None:
        # Same "legacy, not capability-aware caller" bypass narrowing itself
        # already honors above — the token cap is conceptually part of the
        # same Pillar 1 system, so it follows the same contract: None means
        # the full, pre-capability-routing set, unconditionally.
        return minified
    return _cap_schemas_by_token_budget(
        minified, query, must_keep=_CORE_TOOL_IDS | sticky_ids | protected | task_tool_ids
    )


# --- Hard per-turn token ceiling, on top of domain-level narrowing above ---
#
# The narrowing above already trims WITHIN a single non-exempt domain
# (top-K by relevance) but has no ceiling ACROSS everything a turn could
# simultaneously assemble: _CORE_TOOL_IDS (always included), every domain in
# _NARROWING_EXEMPT_DOMAINS that happens to be active (each fully protected,
# by design), and narrowing's own top-K remainder from whatever else is
# eligible. Today that tops out well under Groq's TPM ceiling in practice —
# but "in practice, today" isn't a guarantee, and it stops being one the
# moment a session has the CAD tab open (protects freecad_essential) AND the
# agent has separately unlocked software_engineering (protects that too) AND
# narrowing pulls in a few more from a third domain. This is the actual
# backstop: measured on the REAL, already-minified schema bytes about to be
# sent (a tool-COUNT cap is a leaky proxy — a handful of parameter-heavy
# FreeCAD schemas can cost as much as a dozen simple ones), it caps the
# final payload at _TOOL_TOKEN_BUDGET tokens no matter how many domains fed
# into it.
_TOOL_TOKEN_BUDGET = 2000

# Secondary, purely defensive ceiling alongside the token budget — without
# it, a turn with many cheap/trivial schemas (rare, but not impossible) could
# stay under the token budget while still handing the model an unwieldy
# number of tools. In every measured scenario the token budget binds first;
# this just guarantees it always eventually does.
_MAX_ACTIVE_TOOLS = 30

_token_encoder: Any = None
_TOKEN_ENCODER_UNAVAILABLE = object()


def _count_tokens(text: str) -> int:
    """cl100k_base token count when tiktoken is importable; otherwise a
    deliberately conservative char-based estimate (//3, not //4 — JSON's
    quotes/braces/commas tokenize denser than prose) so the budget above
    stays a real ceiling even without tiktoken installed, just a slightly
    more conservative one."""
    global _token_encoder
    if _token_encoder is None:
        try:
            import tiktoken

            _token_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            _token_encoder = _TOKEN_ENCODER_UNAVAILABLE
    if _token_encoder is _TOKEN_ENCODER_UNAVAILABLE:
        return max(1, len(text) // 3)
    return len(_token_encoder.encode(text))


def _schema_tool_id(schema: dict[str, Any]) -> str:
    return str((schema.get("function") or {}).get("name") or "")


def _rank_by_relevance(tool_ids: Iterable[str], query: str) -> list[str]:
    """``tool_ids`` ordered by descending semantic relevance to ``query`` —
    reuses the SAME embedding registry narrowing already uses, over-fetching
    generously so the ranking covers the whole candidate pool, not just a
    small top-K slice of it. Falls back to a stable alphabetical order
    (never an arbitrary one) whenever there's no query text or retrieval
    itself fails — this is strictly a tie-breaking preference for WHICH
    tools fill the remaining budget, never a correctness requirement."""
    pool = list(tool_ids)
    query = (query or "").strip()
    if not query or not pool:
        return sorted(pool)
    try:
        registry = get_tool_registry()
        ranked_all = [entry.name for entry in registry.retrieve(query, k=len(pool) + 20)]
        wanted = set(pool)
        ranked = [t for t in ranked_all if t in wanted]
        ranked.extend(sorted(wanted - set(ranked)))  # unindexed tools, deterministic order
        return ranked
    except Exception:  # noqa: BLE001 — ranking is a pure optimization, never worth failing a turn over
        return sorted(pool)


def _cap_schemas_by_token_budget(
    schemas: list[dict[str, Any]], query: str, must_keep: frozenset[str]
) -> list[dict[str, Any]]:
    """Final ceiling on this turn's serialized ``tools=`` payload — see the
    module comment above ``_TOOL_TOKEN_BUDGET`` for why this exists on top
    of domain-level narrowing.

    ``must_keep`` (``_CORE_TOOL_IDS`` union this turn's ``sticky_ids``) is
    NEVER dropped, even in the pathological case where it alone exceeded the
    budget — a response the agent has no tools to act on at all is worse
    than one slightly over budget. Every other candidate is ranked by
    semantic relevance to ``query`` and added back, highest first, until the
    budget (or ``_MAX_ACTIVE_TOOLS``) is spent.
    """
    if len(schemas) <= 1:
        return schemas
    by_id = {_schema_tool_id(s): s for s in schemas if _schema_tool_id(s)}

    def _list_cost(ids: list[str]) -> int:
        # Measures the ACTUAL serialized list — summing each schema's own
        # individually-measured cost would miss the array's own structural
        # overhead (the `[`/`]`/`, ` separators, and any tokenizer merging
        # across adjacent schemas' boundaries), which is exactly the gap
        # between an approximate and a genuinely exact budget.
        return _count_tokens(json.dumps([by_id[t] for t in ids]))

    kept_ids = [t for t in by_id if t in must_keep]
    remaining = [t for t in by_id if t not in must_keep]
    if remaining and len(kept_ids) < _MAX_ACTIVE_TOOLS and _list_cost(kept_ids) < _TOOL_TOKEN_BUDGET:
        for tool_id in _rank_by_relevance(remaining, query):
            if len(kept_ids) >= _MAX_ACTIVE_TOOLS:
                break
            if _list_cost([*kept_ids, tool_id]) > _TOOL_TOKEN_BUDGET:
                continue  # doesn't fit — a smaller, lower-ranked tool still might
            kept_ids.append(tool_id)
    # Preserve `schemas`' original relative order rather than kept_ids'
    # append order, so output ordering stays stable/deterministic run to
    # run (matters for _llm_tools_schema_cached callers diffing turns).
    kept = set(kept_ids)
    return [s for s in schemas if _schema_tool_id(s) in kept]


_CORE_SYSTEM_PROMPT = """\
You are Dana, a general-purpose AI desktop assistant and autonomous \
software engineering agent. Capability domains (CAD modeling, software \
engineering, filesystem/OS access, web search, image analysis) not in your \
current tool set are gated, not unavailable.

## Self-Resolving Missing Capabilities
- If a needed tool isn't visible (e.g. no create_freecad_* for "create a \
cylinder", no analyze_codebase/execute_code_task for "fix this bug"), never \
tell the user to enable a domain themselves. Call `load_capability` with the \
matching domain FIRST, then call the needed tool once it appears.
- `check_plugin_registry` only lists which tools EXIST across domains — it \
does not unlock any of them. A listed tool stays off-limits until \
`load_capability` is called with its exact `"domain"` field.
- Any read/understand/edit/refactor/debug/write-code request MUST route \
through `software_engineering`, in order:
  1. `load_capability(domain="software_engineering")`.
  2. `search_codebase` (regex `git grep`, locate by line number without \
reading whole files) or `analyze_codebase` (read the specific files found).
  3. `execute_code_task` to make the scoped edit. If a verification test \
exists, pass it as `test_command` (e.g. "pytest tests/test_foo.py") so the \
coding engine runs and self-repairs it in this one call.
  4. Only fall back to a standalone `run_verification_command` if no \
`test_command` existed, or the change still needs confirming.
- If step 4's `run_verification_command` errors, do NOT stop — call \
`execute_code_task` again with the exact traceback in `task_description` \
(same `test_command`), then re-verify. Repeat until it passes. Stop only on \
the SAME error twice in a row and explain it instead.
- Fall back to explaining what's missing only if `load_capability` itself \
reports the domain doesn't exist or fails.

## Turn-Taking
- ONE tool call per turn, even for a multi-step request — stop and wait for \
its result before deciding the next step.
- Call tools ONLY via the tool-calling mechanism. NEVER write a tool call as \
JSON text in your reply — it will not execute.
- Only call a tool for a clear action request; otherwise reply in plain text.
- CRITICAL: a result with `"ok": false` or `"status": "error"` means that \
step did NOT happen — never claim it succeeded or proceed as if it did. \
Adjust the parameters and retry ONCE. On the EXACT SAME error a second time, \
stop immediately and explain the terminal error instead of trying a third \
time (never silently continue as if it had worked).\
"""

# Always appended (see build_system_prompt below) — read_skill_source and
# save_new_skill are both CORE tools (always available, any domain/plugin
# active or not — see _CORE_TOOL_IDS), so this guidance stays unconditional
# too, rather than being duplicated into both _CORE_SYSTEM_PROMPT and
# _FREECAD_SYSTEM_PROMPT (the latter is verbatim/frozen — see its own
# docstring below).
_SKILL_DEBUGGING_SECTION = """\
## Debugging Your Own Skills
- A saved skill failing with a "traceback" field is YOUR OWN code raising.
- Call `read_skill_source` to see its current code and find the failing line.
- Call `save_new_skill` again with the SAME skill_name and corrected \
python_code to overwrite and hot-reload the fix.
- Never leave a broken skill registered. Never guess a fix without reading \
the actual source first.\
"""

# Used whenever "freecad" is in the active-plugin set. No longer the
# module's original byte-for-byte text (compressed once already — see git
# history — then extended with rules #7/#8 for the local-model tool-
# fixation fix, and rules #11-13 for boolean-succession tracking, object-
# naming, and the ambiguity-override behavior), but still the ONE prompt
# every CAD session's LLM turn sees.
# Never edit this constant without also verifying
# tests/core/test_react_dispatch.py's build_system_prompt assertions and a
# live FreeCAD session still hold.
_FREECAD_SYSTEM_PROMPT = """\
You are Dana, a professional mechanical engineering CAD co-pilot for FreeCAD.

## Turn-Taking
- ONE tool call per turn, even for a multi-step request (e.g. "create a box, \
check its size, add a cylinder on top") — stop and wait for its result \
before deciding the next step.
- Call tools ONLY via the tool-calling mechanism. NEVER write a tool call as \
JSON text in your reply — it will not execute.
- Only call a tool for a clear action request; otherwise reply in plain text.

## Engineering Rules — apply in this order, every turn
1. PLAN MULTI-STEP TASKS FIRST. If the user asks for more than one shape, \
feature, or action, your VERY FIRST tool call must be `create_plan` to break \
the request into distinct, sequential steps. Do not generate any geometry \
until the plan is created.
2. NEVER HALLUCINATE HARDWARE DIMENSIONS. Before sketching/extruding a \
standard/off-the-shelf part (named motor, bolt/screw size, bearing), call \
`query_engineering_standard` FIRST and use its numbers verbatim. If the part \
itself is standard (NEMA motor, M3-M5 socket head screw, ball bearing), \
prefer `insert_standard_part` over hand-sketching it. Fall back to your own \
judgment only once these tools report no match.
3. VERIFY BEFORE MUTATING. Before a fillet/chamfer, boolean cut/union/\
intersect, or any edit to an object you did not just create this turn, call \
`inspect_spatial_properties`, `get_freecad_bounding_box`, or \
`analyze_bounding_box_collisions` to confirm the ACTUAL current geometry — \
never assume it.
4. PREFER HIGH-LEVERAGE TOOLS over many small ones:
   - `create_freecad_sketch_extrude` (not `create_freecad_extrusion`) for \
any profile with a rounded/arc edge, a slot, or a shape straight edges can't express.
   - `batch_pattern_array` (not one create_freecad_* call per copy) for 3+ \
repeated features in a line, grid, or circle — e.g. 4 bolt holes, an 8x8 tile grid. \
If you catch YOURSELF about to dispatch the same create_freecad_* call a 2nd time \
for what is really a repeated pattern, STOP and call `unload_capability` with \
`tool_id` set to that exact create_freecad_* tool name FIRST — this hides it from \
your own next turn so you cannot lazily repeat it, then call `batch_pattern_array` \
instead. Call `load_capability` with the same `tool_id` afterward if you \
genuinely need that primitive again later.
   - COMBINE BASIC PRIMITIVES (boxes, cylinders, polygons) INTO ONE SOLID \
via `perform_freecad_boolean` with `operation="union"` — NEVER \
`create_assembly_mate` for this. `create_assembly_mate` is STRICTLY for a \
real multi-part KINEMATIC assembly (parts that stay SEPARATE objects, \
positioned relative to each other by a concentric/coincident_planar/\
offset_axial relationship) — prefer it over `align_freecad_objects` for \
THAT case only; it is never a substitute for actually fusing primitives \
into one merged solid, which only `perform_freecad_boolean` does.
   - `base_object`/`tool_object`/`fixed_object`/`moving_object` must \
always be an OBJECT name a create_freecad_*/insert_standard_part/prior \
perform_freecad_boolean call actually returned (e.g. `"base_plate"`) — \
NEVER `"Session_Active"`, which is the shared SESSION DOCUMENT's own \
name, not an object inside it, and will always fail to resolve.
   - NEVER hand-compute a 3D placement/rotation to put a hole, boss, \
pocket, or tab on a vertical/angled face (e.g. the front or side of a \
box). Call `create_freecad_feature_on_face` instead — give it the \
object name, a face label (`"top"`/`"bottom"`/`"front"`/`"back"`/\
`"left"`/`"right"`), a shape (`"circle"`/`"rectangle"`), and local \
(u, v) coordinates ON that face; it resolves the face's real 3D \
position/orientation itself. Like `perform_freecad_boolean`, this \
CONSUMES `object_name` — only its returned result name is valid \
afterward (see the CRITICAL TOPOLOGY RULE below).
5. SELF-CORRECT ON ERROR — STOP AFTER A REPEATED FAILURE. CRITICAL: a result \
with `"ok": false` or `"status": "error"` means that step did NOT happen — \
never treat it as done, move on, or tell the user it succeeded. A failure \
returns `{"status": "error", "reason": ..., "suggestion": ...}` — read both \
fields, adjust the parameter they point to, and retry. Never repeat an \
identical call that just failed. On the EXACT SAME error a second time, DO \
NOT try a third time — stop, yield your turn, and explain the terminal \
error instead of guessing again (never silently continue as if it worked).
6. LOOK BEFORE YOU PROCEED. Every successful create_freecad_*/perform_freecad_*/\
modify_freecad_parameter/insert_standard_part/batch_pattern_array/\
align_freecad_objects/create_assembly_mate call already returns a \
`screenshot_path` (auto-captured, headless, right after your edit) and a \
`visual_verification` summary read off it by a vision model — no separate \
tool call needed. Check `visual_verification` looks physically correct \
before proceeding or telling the user it's done — a numerically successful \
result can still look wrong on screen. If it contradicts your intent (wrong \
shape, missing feature, bad proportion), investigate with \
`get_freecad_bounding_box`/`inspect_spatial_properties` or correct it — \
never ignore a visual mismatch just because `"ok": true`. If verification \
itself was unavailable (no FreeCAD window, no vision model reachable), \
that's a convenience miss, not a failure — proceed on the numeric result. \
`take_canvas_screenshot` remains available separately for an on-request \
interactive view; it needs the live 3D viewer and is not what this \
automatic check uses.
7. After creating or modifying solid geometry, ensure it recomputes so the \
mesh pipeline exports the updated geometry.
8. MATCH THE TOOL TO THE GEOMETRY — NEVER REPEAT ONE TOOL OUT OF HABIT. Do \
not repeatedly call the same geometric tool (e.g. creating 5 boxes in a row) \
unless the user explicitly asked for identical shapes. Select the specific \
tool that matches the actual physical geometry requested, not whichever tool \
is already loaded or easiest to repeat.
9. DISCOVER MISSING GEOMETRY TOOLS BEFORE IMPROVISING. Do not attempt to \
mathematically derive complex polygons (e.g. hexagons, gears, star prisms) \
using the 2D sketch tool. If a specific 3D shape is requested and no \
dedicated primitive tool is loaded, call `search_tool_catalog` to find it, \
then `load_specific_tool` to load it. This applies to OPERATIONS, not just \
shapes: if the user requests a specific operation (e.g. "pattern array", \
"sweep") and you do not know the EXACT tool_id for it, you MUST call \
`search_tool_catalog` BEFORE calling `create_plan` — you are FORBIDDEN from \
guessing a tool name. Do not substitute an incorrect tool you already know \
(e.g. declaring `perform_freecad_boolean` when a pattern array is \
requested) just to force a plan to validate — your plan's `expected_tools` \
must match the user's TRUE geometric intent, never whatever tool is \
already loaded or convenient. You are FORBIDDEN from using \
`execute_freecad_script` as a workaround to avoid finding the correct \
native CAD tool. You must use the native primitive and modifier tools.
10. COMPLETE MULTI-STEP REQUESTS ONE TASK AT A TIME. Never stop after \
completing only one part of a multi-step request, but never blast through \
several tasks in one turn either — see === CAD AGENT PROTOCOL === below. \
Execute the CURRENT ACTIVE TASK's tool(s), call `mark_task_completed`, then \
let the NEXT turn's refreshed plan anchor tell you what's active now. The \
request isn't done until every task the plan lists has been completed.
11. CRITICAL TOPOLOGY RULE: FreeCAD consumes base objects during boolean \
operations. After any boolean cut, intersection, or union, you MUST \
exclusively reference the newly generated output object name for all \
subsequent modifications or exports. Never reference the consumed base \
object again.
12. NAMING CONVENTION: You must never use spaces in object names. Always use \
PascalCase or snake_case (e.g. BaseCylinder or base_cylinder).
13. AUTONOMY OVERRIDE: If a user requests a modification without specifying \
exact object names or spatial coordinates, DO NOT halt the workflow to ask \
for clarification. Immediately use the get_freecad_bounding_box or \
inspect_spatial_properties tools to assess the workspace, make the most \
logical engineering assumption, execute the tool, and document your \
assumption in your final response to the user.\
"""

# Appended right after _FREECAD_SYSTEM_PROMPT (see build_system_prompt) —
# the tool-hallucination/task-tracking-amnesia fix: rule 10 above already
# points here for the step-by-step contract, this is the literal, load-
# bearing directive text the model re-reads every single turn. Kept as its
# own constant/section (rather than folded into the numbered rules) so it
# reads as an unconditional PROTOCOL, not one more rule among thirteen.
#
# DETERMINISTIC TASK VERIFICATION / ZERO-TRUST ORCHESTRATION (below) close
# the gap the code-enforced FSM (dispatch_tool_call/_fsm_out_of_order_check/
# _tool_mark_task_completed) cannot close by itself: the backend can verify
# WHICH tool ran and that it returned `"ok": true`, but it cannot judge
# whether the geometry that tool produced actually matches the user's
# intent (e.g. a `perform_freecad_boolean` union of an object to itself
# succeeds structurally while accomplishing nothing geometrically). That
# judgment has to happen here, in the model's own reasoning, using the
# deterministic post-condition data (Fix #4 — bounding box/volume) every
# geometry TOOL_RESULT already carries.
_CAD_AGENT_PROTOCOL = """\
=== CAD AGENT PROTOCOL ===
- TOOL VERIFICATION: If you need to create basic geometry but primitive \
tools (e.g. `create_freecad_box`, `create_freecad_cylinder`) are missing \
from your Available Tools list, your VERY FIRST action must be to call \
`load_capability` with domain='freecad_essential'.
- STRICT STEP-BY-STEP EXECUTION: You are a sequential executor. You must \
ONLY execute the CAD tools required for your CURRENT ACTIVE TASK. Do NOT \
execute the entire plan at once.
- DETERMINISTIC TASK VERIFICATION: Before calling `mark_task_completed`, \
you MUST evaluate the deterministic geometric post-conditions (length, \
width, height, volume) returned in the TOOL_RESULT. If asked to pattern a \
feature 20 times, the volume and bounding box MUST mathematically reflect \
20 features, not 1. If it does not, your operation failed and you must \
correct it before advancing. Do not self-report false success.
- ZERO-TRUST ORCHESTRATION: Never assume an operation worked just because \
it did not throw a Python error. A union of an object to itself will \
return "success" but accomplishes nothing geometrically. Verify the \
physical dimensions with absolute certainty.
- TASK LIFECYCLE: Immediately after successfully generating the geometry \
for your current task, you MUST call `mark_task_completed` to advance the \
plan before taking any other action.
===========================\
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


# ---------------------------------------------------------------------------
# Plan-and-Execute FSM, Phase 2: State-Isolated Prompt Builders.
#
# _FREECAD_SYSTEM_PROMPT (above) stays BYTE-FOR-BYTE UNTOUCHED — its own
# comment already warns never to edit it without re-verifying a live FreeCAD
# session, and it remains the correct, complete prompt for the "idle" path
# (no plan active yet for a legacy/non-capability-aware caller, or a plan
# that's already fully "done" — ad-hoc post-plan cleanup, same reasoning
# _RESTRICTED_GEOMETRY_TOOLS's own docstring gives for never re-gating that
# case). The engineering domain-knowledge rules that still apply ONE TASK AT
# A TIME during EXECUTING (standard-parts lookup, verify-before-mutating,
# topology, naming convention, ...) are duplicated here in trimmed form
# rather than sliced out of that frozen constant by string surgery — the two
# are allowed to drift in wording (this one can get leaner over time; the
# idle one must not change at all).
_EXECUTOR_ENGINEERING_RULES = """\
## Engineering Rules — apply in this order, every turn
1. NEVER HALLUCINATE HARDWARE DIMENSIONS. Before sketching/extruding a \
standard/off-the-shelf part (named motor, bolt/screw size, bearing), call \
`query_engineering_standard` FIRST and use its numbers verbatim. Prefer \
`insert_standard_part` over hand-sketching a standard part. Fall back to \
your own judgment only once these report no match.
2. VERIFY BEFORE MUTATING. Before a fillet/chamfer, boolean cut/union/\
intersect, or any edit to an object you did not just create this turn, call \
`inspect_spatial_properties`/`get_freecad_bounding_box`/`analyze_bounding_box_\
collisions` to confirm the ACTUAL current geometry — never assume it.
3. PREFER HIGH-LEVERAGE TOOLS: `create_freecad_sketch_extrude` (not \
`create_freecad_extrusion`) for any rounded/arc/slotted profile; \
`batch_pattern_array` (not one create_freecad_* call per copy) for 3+ \
repeated features; `perform_freecad_boolean(operation="union")` (never \
`create_assembly_mate`) to fuse primitives into one solid — \
`create_assembly_mate` is strictly for a real multi-part KINEMATIC assembly \
that must stay separate objects. `base_object`/`tool_object` must be an \
OBJECT name a prior tool call actually returned, never `"Session_Active"` \
(the shared session document's own name, not an object in it). Never \
hand-compute a placement for a hole/boss/pocket on a face — call \
`create_freecad_feature_on_face` instead.
4. SELF-CORRECT ON ERROR, STOP AFTER A REPEATED FAILURE. `"ok": false` means \
that step did NOT happen — never treat it as done. Read the `reason`/\
`suggestion` fields, adjust, and retry ONCE. On the EXACT SAME error twice, \
stop and explain instead of trying a third time.
5. LOOK BEFORE YOU PROCEED. A successful geometry call's own \
`visual_verification` (auto-captured, vision-model-read) must look \
physically correct before you proceed or report done — a numerically \
successful result can still look wrong. Investigate/correct on a mismatch; \
a missing/unavailable verification is a convenience miss, not a failure.
6. Ensure solid geometry recomputes after every create/modify so the mesh \
pipeline exports current state.
7. MATCH THE TOOL TO THE GEOMETRY. Don't repeat one tool out of habit — \
select the specific tool the physical shape actually needs.
8. DISCOVER MISSING TOOLS BEFORE IMPROVISING. Don't hand-derive a complex \
polygon (hexagon, gear, star) from the 2D sketch tool — call \
`search_tool_catalog` then `load_specific_tool` for a dedicated primitive. \
Same for OPERATIONS: if you don't know the EXACT tool_id for what the user \
asked (e.g. "pattern array", "sweep"), call `search_tool_catalog` BEFORE \
`create_plan` — guessing a tool name is forbidden. Never declare a tool you \
already know (e.g. `perform_freecad_boolean`) in place of the one the \
request actually needs (e.g. a pattern array) just to force a plan to \
validate — `expected_tools` must match the user's true geometric intent. \
You are FORBIDDEN from using `execute_freecad_script` as a workaround to \
avoid finding the correct native CAD tool. You must use the native \
primitive and modifier tools.
9. CRITICAL TOPOLOGY RULE: a boolean cut/union/intersect CONSUMES its base \
objects — reference only the newly generated output name afterward, never \
the consumed base again.
10. NAMING CONVENTION: no spaces in object names — PascalCase or \
snake_case (e.g. BaseCylinder or base_cylinder).
11. AUTONOMY OVERRIDE: if a modification doesn't specify exact names/\
coordinates, don't halt to ask — use `get_freecad_bounding_box`/\
`inspect_spatial_properties` to assess the workspace, make the most logical \
engineering assumption, execute, and document the assumption in your final \
reply.\
"""


def _build_planner_prompt() -> str:
    """PLANNING — used whenever a freecad-active, capability-aware session
    has no active plan yet. Deliberately names no geometry tool and carries
    no CAD rulebook: ``next_react_turn``'s ``hard_restrict_to`` (Phase 2/3)
    ensures the schema THIS prompt accompanies physically contains nothing
    but ``create_plan``/``search_tool_catalog``/``check_plugin_registry``/
    ``_CORE_TOOL_IDS`` — the model cannot call a geometry tool here not
    because a rule tells it not to, but because it was never in the payload
    it saw this turn.
    """
    return (
        "=== PLANNING PHASE ===\n"
        "You are Dana. This session is in the PLANNING phase of a "
        "Plan-and-Execute pipeline — no CAD/geometry tool is offered to you "
        "this turn, by design.\n"
        "Your ONLY job right now is to call `create_plan` with an `objective` "
        "(one sentence) and a `tasks` list — one entry per distinct shape, "
        "cut, feature, or action the request needs, in the exact order they "
        "must happen. Do not describe HOW to build anything yet; each task's "
        "own tools become visible only once that task is active, one at a "
        "time.\n"
        "Even a single-action request still needs a one-item `tasks` list — "
        "every geometry tool stays gated behind an active plan.\n"
        "`search_tool_catalog`/`check_plugin_registry` are available if you "
        "need to confirm what's possible before committing to a plan.\n"
        "==========================="
    )


def _build_executor_prompt(objective: str, active_task: dict[str, Any], remaining_count: int) -> str:
    """EXECUTING — the ONLY plan content this receives is the objective (one
    line, context only) and the SINGLE current active task; every other
    pending task is reduced to a bare count ("N task(s) remain after this
    one") rather than named, so the model knows there's more work without
    being tempted to look ahead and execute it early — the literal
    completionist-reflex bug this whole FSM exists to prevent.

    Explicitly warns about the restored out-of-order hard block
    (``_fsm_out_of_order_check``, called from ``dispatch_tool_call`` before
    any handler runs, Universal FSM Enforcement): ANY tool outside
    ``_CORE_TOOL_IDS`` that ISN'T one of this task's own
    ``expected_tool_ids`` (declared by the Planner itself at ``create_plan``
    time, see ``_tool_create_plan`` — never just a geometry tool) is now
    REFUSED outright, not silently skipped past — the model must not assume
    it can "get ahead" and run a later task's tool early, OR route around a
    declared tool by reaching for an undeclared one instead. A task with NO
    fixed tool signature at all (e.g. a visual-only inspection) is
    unaffected by the block, but still requires an explicit
    ``mark_task_completed`` since there's nothing to auto-advance on.
    """
    remaining_note = (
        f"{remaining_count} task(s) remain after this one."
        if remaining_count
        else "This is the LAST task in the plan."
    )
    return (
        f"{_EXECUTOR_ENGINEERING_RULES}\n\n"
        "=== EXECUTING PHASE ===\n"
        f"Objective (context only — do not act on later steps yet): {objective}\n\n"
        f"--> CURRENT ACTIVE TASK {active_task.get('id')}: {active_task.get('description')}\n"
        f"{remaining_note}\n"
        "Execute ONLY the tool(s) needed for THIS task. The backend automatically "
        "advances the plan the instant you successfully call one of this task's own "
        "expected tools — you do NOT need to (and normally should not) call "
        "`mark_task_completed` yourself in that case. If this task has a fixed tool "
        "signature, calling any OTHER tool (one that belongs to a LATER task, or any "
        "undeclared tool you reach for instead of the one you committed to) is BLOCKED "
        "outright — you cannot get ahead of the plan, and you cannot substitute an "
        "undeclared tool for a declared one. You must either call this task's own "
        "declared tool(s) exactly, or call `mark_task_completed` first if you believe "
        "this task is already done. Only a task with NO fixed tool signature at all "
        "(e.g. a visual-only inspection/adjustment) lets any tool through without "
        "auto-advancing — for that case, you MUST call `mark_task_completed` yourself "
        "once you've confirmed it's genuinely done, or the plan will stall here even "
        "though the work is finished.\n"
        "==========================="
    )


def _build_validator_prompt(
    objective: str, active_task: dict[str, Any], last_validation: dict[str, Any] | None
) -> str:
    """VALIDATING — reached only for a task with NO ``expected_tool_ids``
    (the k=3 semantic mapping at ``create_plan`` time found nothing
    tool-shaped for its description, e.g. a pure visual inspection), parked
    here awaiting an explicit ``mark_task_completed`` since there is no
    automatic tool-success signal to advance on. The common, tool-mapped
    case never actually reaches this prompt — see
    ``_advance_fsm_on_dispatch``'s own docstring for why that resolves
    synchronously, in the same dispatch that entered "validating", before
    any further LLM turn.
    """
    last_result_note = (
        f"Last tool result recorded for this task: {json.dumps(last_validation)[:1000]}"
        if last_validation
        else "No tool result has been recorded for this task yet."
    )
    return (
        "=== VALIDATING PHASE ===\n"
        f"Objective (context only): {objective}\n"
        f"Task awaiting confirmation {active_task.get('id')}: {active_task.get('description')}\n"
        f"{last_result_note}\n"
        "This task has no single fixed tool signature, so the plan cannot auto-advance "
        "it. Inspect the current state (get_freecad_bounding_box/inspect_spatial_"
        "properties), make a further corrective tool call for THIS task if needed, then "
        "call `mark_task_completed(task_id=...)` once you're satisfied it's genuinely "
        "done.\n"
        "==========================="
    )


def build_system_prompt(
    active_selection: dict[str, Any] | None,
    active_plugins: frozenset[str] | None = None,
    mounted_directories: list[str] | None = None,
    working_memory: str = "",
    session_id: str | None = None,
) -> str:
    """The dynamic context the LLM reasons over each turn — this is where
    the React 3D viewer's active-selection state enters the ReAct loop.

    ``session_id``, like ``mounted_directories``/``working_memory``, is
    passed explicitly rather than read off the ambient
    ``dana.session_context`` contextvar: this is called from
    ``dana.api.server._process_user_text`` at the very START of a turn,
    BEFORE ``_execute_and_continue`` has ever called ``set_session_id`` for
    it — on a freshly (re)connected socket's first message, the contextvar
    would still read its process-wide default here, silently showing the
    Plan-and-Execute Anchor (below) for the wrong session, or none at all
    for a resumed session that already has one. Omitted (``None``) falls
    back to the ambient value anyway, for any caller that genuinely doesn't
    care (e.g. ``parse_utterance``, already documented above as tolerating
    the single-session-shaped defaults).

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

    Ends with one further block AFTER that — the Plan-and-Execute
    Gatekeeper's own Focused Plan Anchor (``_get_active_plan``, Phase 6) —
    whenever THIS session has one. Distinct from the "## Current Active
    Plan" block above: that one is task_board's live, GLOBAL, per-task
    pending/active/completed snapshot (shared with PlanChecklist/GET
    /api/planner); this one is THIS session's own objective/tasks (as given
    to its own ``create_plan`` call), rendered by ``_format_focused_plan_anchor``
    to show ONLY the pending-task list and the single current task —
    deliberately NOT the full plan dump task_board's block above already is,
    so the freshest, LAST thing in the prompt is "what do I do right now",
    not one more copy of the whole plan to get lost re-deriving progress
    from. Falls back to a flat verbatim string for any caller (tests) that
    set the session's plan text directly without structured tasks.

    Plan-and-Execute FSM (Phase 2, State-Isolated Prompt Builders): the base
    prompt (this function's very first line) is chosen by PHASE, not just by
    ``freecad_active`` — ``_build_planner_prompt``/``_build_executor_prompt``/
    ``_build_validator_prompt`` (see their own docstrings) replace
    ``_FREECAD_SYSTEM_PROMPT`` + ``_CAD_AGENT_PROTOCOL`` once a capability-
    aware (``active_plugins is not None``) freecad-active session has
    actually engaged the Plan-and-Execute Gatekeeper — i.e. exactly the same
    population ``_RESTRICTED_GEOMETRY_TOOLS`` already requires a plan from.
    A legacy caller (``active_plugins is None``, e.g. ``parse_utterance``) or
    a session with no plan yet AND not capability-aware never enters this
    branch at all — same "None means the original, single-mode behavior,
    unconditionally" contract every other capability-routing decision in
    this module already honors. A ``"done"`` plan (every task completed)
    ALSO falls back to this same idle path — ad-hoc post-plan cleanup must
    stay reachable, not re-gated (see ``_RESTRICTED_GEOMETRY_TOOLS``'s own
    docstring).
    """
    freecad_active = active_plugins is None or bool(
        active_plugins & {"freecad", "freecad_essential", "freecad_full"}
    )
    fsm_phase = "idle"
    if freecad_active and active_plugins is not None:
        if not _get_has_plan(session_id):
            fsm_phase = "planning"
        else:
            state = _get_fsm_state(session_id)
            fsm_phase = state if state in ("executing", "validating") else "idle"

    if fsm_phase == "planning":
        lines = [_build_planner_prompt()]
    elif fsm_phase in ("executing", "validating"):
        active_task = _active_task(session_id)
        objective = _PLAN_STATE_REGISTRY.get(
            session_id if session_id is not None else get_session_id(), {}
        ).get("objective") or ""
        if active_task is None:
            # Defensive fallback only — _get_has_plan()/_get_fsm_state() both
            # already returning "there's an executing/validating plan" with
            # no actual active task would mean this session's structured
            # tasks list and its fsm_state have desynced; safer to fall back
            # to the full idle prompt (still fully functional, if verbose)
            # than to build a prompt around a task that doesn't exist.
            lines = [_FREECAD_SYSTEM_PROMPT, _CAD_AGENT_PROTOCOL]
        elif fsm_phase == "validating":
            sid_for_lookup = session_id if session_id is not None else get_session_id()
            last_validation = _PLAN_STATE_REGISTRY.get(sid_for_lookup, {}).get("last_validation")
            lines = [_build_validator_prompt(objective, active_task, last_validation)]
        else:
            lines = [_build_executor_prompt(objective, active_task, _pending_task_count(session_id))]
    else:
        lines = [_FREECAD_SYSTEM_PROMPT if freecad_active else _CORE_SYSTEM_PROMPT]
        if freecad_active:
            lines.append(_CAD_AGENT_PROTOCOL)
    lines.append(_SKILL_DEBUGGING_SECTION)
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
    # Plan-and-Execute Gatekeeper's own anchor (Phase 6) — see this
    # function's docstring for how this differs from plan_section above.
    # Deliberately the LAST thing appended, so it's the freshest text in
    # the model's context on every turn.
    active_plan = _get_active_plan(session_id)
    if active_plan:
        lines.append(
            f"\n=== ACTIVE PLAN ===\n{active_plan}\n===================\n"
            "Stick to this naming convention and topological strategy."
        )
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


# Hard client-side ceiling on the PRIMARY model's turn — matches the Ollama
# HTTP layer's own socket-stall timeout (dana.core.openai_tool_bridge.
# complete_openai_with_tools), so this asyncio.wait_for is purely a
# "did the connection ever answer at all" backstop rather than a second,
# tighter latency budget racing the HTTP layer's own.
#
# No longer derived from openai_tool_bridge's own TPM throttle-and-retry
# sleep ceiling — that sleep/retry loop was removed once cloud tool-calling
# moved to a direct single-provider call (dana.core.model_provider.
# tool_calling_provider — OpenRouter by default), whose own server-side
# ``models`` fallback array (see complete_openai_with_tools's
# fallback_models) retries the next model upstream in milliseconds. A
# 429/5xx now fails this call fast instead of sleeping, so this timeout
# only needs to bound a genuinely stalling connection (local Ollama or the
# cloud provider itself), not a legitimate multi-second rate-limit wait.
#
# Raised from 45s to 600s: a heavier local model (e.g. a 14B-parameter
# qwen2.5-coder, run for exact-artifact accuracy over speed — see
# DANA_LOCAL_MODEL) can legitimately take 2-3 minutes per turn, which the
# old 45s ceiling killed outright as a false "stall" before the model ever
# had a chance to finish. This is now a genuine "the connection is dead"
# backstop, not a UX-latency budget — lower it only for a fast
# cloud-primary setup where a multi-minute hang really would indicate a
# stalled connection rather than a slow-but-working local generation.
_LOCAL_TOOL_CALL_TIMEOUT_SEC = 600.0

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


def _sticky_tool_ids_from_messages(messages: list[dict[str, Any]], raw_text: str = "") -> frozenset[str]:
    """Tool ids that must survive Pillar 1's semantic-relevance narrowing
    regardless of how they score against this turn's query, because THIS
    turn's ReAct chain already has a concrete claim on them:

    1. Any tool id already invoked earlier in this chain (an assistant
       message's ``tool_calls``, as built by
       ``build_assistant_tool_call_message``) — so a multi-step sequence
       (e.g. create_freecad_box then perform_freecad_boolean referencing
       it) never loses a tool it's already mid-sequence with.
    2. Every tool id a ``load_capability`` OR ``load_specific_tool`` call
       already unlocked earlier in this chain (read back from that call's
       own tool-result message, matched by ``tool_call_id`` so a
       same-shaped payload from an unrelated tool is never mistaken for
       one) — without this, the tools ``load_capability``/
       ``load_specific_tool`` exist ONLY to unlock (e.g. ``execute_code_task``
       right after ``load_capability(domain="software_engineering")``, or
       any single tool right after ``load_specific_tool``) have themselves
       never been "already invoked", so narrowing was free to drop them
       again on the very next turn: the one turn they're overwhelmingly
       likely to actually get called. Both handlers
       (``_tool_load_capability``/``_tool_load_specific_tool``) put
       ``unlocked_tools`` in their result payload for exactly this reason.

    A large ``unlocked_tools`` list (in practice only ``load_capability``'s
    "freecad"/"freecad_full" domain-unlock, ~24-26 tools — ``load_specific_tool``
    always unlocks exactly one) is ranked by relevance to ``raw_text`` and
    capped to ``_KEYWORD_DOMAIN_TOP_K``, same treatment
    ``_keyword_suggested_tool_ids`` gives a large keyword-matched domain and
    for the identical reason: ``_cap_schemas_by_token_budget``'s ``must_keep``
    is NEVER trimmed even when it alone exceeds the budget, so blindly
    stickying an entire large domain the moment it's unlocked silently
    defeated the whole token budget on the very next turn — measured live,
    unlocking "freecad_full" this way alone produced an 8,966-token request.
    The domain stays fully ELIGIBLE regardless (``dana.api.server``'s
    ``_effective_capabilities`` keeps it in ``active_plugins`` from the
    ``"domain"`` field alone, independent of ``unlocked_tools``) — only the
    unconditional STICKINESS of every one of its tools is capped; ordinary
    narrowing/the budget's own relevance-ranked greedy fill still reaches the
    rest as normal. A small list (at/under the threshold) is kept whole,
    exactly as before.
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
        if call_id_to_name.get(str(message.get("tool_call_id") or "")) not in (
            "load_capability",
            "load_specific_tool",
        ):
            continue
        try:
            payload = json.loads(message.get("content") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        unlocked = [t for t in (payload.get("unlocked_tools") or ()) if isinstance(t, str)]
        if len(unlocked) > _KEYWORD_DOMAIN_NARROW_MIN:
            ids.update(_rank_by_relevance(unlocked, raw_text)[:_KEYWORD_DOMAIN_TOP_K])
        else:
            ids.update(unlocked)
    return frozenset(ids)


async def _call_llm_once(
    messages: list[dict[str, Any]],
    *,
    api_keys: dict[str, str] | None = None,
    active_plugins: frozenset[str] | None = None,
    raw_text: str = "",
    hidden_tool_ids: frozenset[str] = frozenset(),
    narrowing_query: str | None = None,
    task_tool_ids: frozenset[str] = frozenset(),
    hard_restrict_to: frozenset[str] | None = None,
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

    ``narrowing_query``/``task_tool_ids``/``hard_restrict_to`` (Plan-and-
    Execute FSM, Phases 2-3 — all three computed by ``next_react_turn`` from
    this session's own FSM state, never by a caller directly) override how
    ``_llm_tools_schema`` narrows/restricts this turn's tool schema: see that
    function's own docstring for what each one does.

    A user turn built by ``build_user_message`` with attachments carries a
    multimodal ``content`` array (text + one or more ``image_url`` parts)
    instead of a plain string — ``ModelProvider.complete_with_tool_calls`` ->
    ``complete_openai_with_tools`` forwards ``messages`` into the request
    body verbatim (no per-message reshaping), so that array reaches the
    upstream OpenAI-wire-compatible endpoint exactly as built, with no
    special-casing needed here.

    ``messages`` is pruned (``dana.core.context_manager.
    prune_message_history`` + ``prune_tool_output_history``) into a SEPARATE
    list right before this HTTP call — older image attachments get replaced
    with a lightweight placeholder, and older tool-result messages (e.g. a
    verbose ``search_codebase``/``execute_code_task`` result from several
    turns back in this same ReAct chain) have their ``content`` truncated
    once they're no longer among the most recent couple of tool executions —
    so a long conversation, or a long multi-step ReAct chain, doesn't
    re-send every image or every stale tool output it ever saw on every
    subsequent turn. For local Ollama specifically (``tool_calling_provider()
    == "ollama"``) the tool-output step uses ``compress_tool_output_history``
    instead — JSON-structure-aware compression (drops large/low-signal
    fields, keeps every file/object-name, numeric-geometric-parameter, and
    outcome field verbatim, never touches an unresolved error) rather than a
    blind character slice, since a local model's context/VRAM budget is
    tighter than a cloud provider's. Message COUNT and ORDER are never
    changed by any of these steps (only a ``content`` payload is ever
    rewritten), which is what keeps every tool_calls/tool-result pair
    OpenAI/Groq's API requires intact even after pruning. The caller's own
    ``messages`` list (a session's real conversation history, resumed across
    HITL suspends and rendered turn-by-turn in the frontend) is never
    touched — only the payload actually posted to the model shrinks.

    Hard timeout + fallback (P2 of the local-agent rescue plan): the
    primary call is capped at ``_LOCAL_TOOL_CALL_TIMEOUT_SEC`` (matching the
    HTTP layer's own socket-stall timeout — see openai_tool_bridge's
    ``timeout`` default) — a genuinely dead connection (Groq/any
    OpenAI-compatible provider can stall or disconnect too, not just local
    Ollama) still needs a hard ceiling, it's just no longer tighter than a
    heavy local model's real generation time. A timeout here never
    propagates as an "error" turn;
    it reroutes to ``_call_llm_timeout_fallback`` for a short apology
    instead, so the loop always ends in a clean "final" turn rather than
    the generic "I ran into a problem" message ``next_react_turn`` gives a
    genuine model/connection error.
    """
    provider = ModelProvider(api_keys=api_keys)
    deduped_messages = prune_message_history(messages)
    pruned_messages = (
        compress_tool_output_history(deduped_messages)
        if tool_calling_provider() == "ollama"
        else prune_tool_output_history(deduped_messages)
    )
    # _keyword_suggested_tool_ids folded in alongside the existing "already
    # invoked"/"load_capability-unlocked" sticky sources — a domain the
    # user's own words just named (see _keyword_suggested_domains, called
    # again here on the exact same raw_text next_react_turn already used to
    # widen active_plugins) needs its tools protected from narrowing too, or
    # pre-warming the domain wouldn't guarantee its own tools actually reach
    # this turn's schema.
    # Layer 3 (Lazy Loading): tool ids explicitly unlocked via
    # load_specific_tool (or load_capability) earlier in this chain must
    # bypass the capability-domain gate itself, not just narrowing — see
    # _llm_tools_schema's force_include param. Folded into sticky_ids too so
    # the same ids also survive the token-budget cap once they're in.
    #
    # Skipped entirely when `hard_restrict_to` is set (Plan-and-Execute FSM's
    # PLANNING phase, see next_react_turn) — these are exactly the leak
    # vector a hard allow-list exists to close: a stale/keyword-matched
    # geometry tool_id from earlier conversation history (e.g. the user's
    # own raw text mentions "box") would otherwise get unioned straight into
    # `tool_ids` via `force_include`, defeating the whole point of
    # "the Planner cannot see a geometry tool at all."
    if hard_restrict_to is not None:
        tools = _llm_tools_schema(active_plugins, hidden_tool_ids=hidden_tool_ids, hard_restrict_to=hard_restrict_to)
    else:
        force_include_ids = _sticky_tool_ids_from_messages(messages, raw_text) | _anchor_suggested_tool_ids(raw_text)
        sticky_ids = force_include_ids | _keyword_suggested_tool_ids(raw_text)
        tools = _llm_tools_schema(
            active_plugins,
            query=narrowing_query if narrowing_query is not None else raw_text,
            sticky_ids=sticky_ids,
            force_include=force_include_ids,
            hidden_tool_ids=hidden_tool_ids,
            task_tool_ids=task_tool_ids,
        )
    # Temporary diagnostic for the Semantic RAG "context drop" bug — proves
    # a tool load_capability just unlocked (sticky via _sticky_tool_ids_from_
    # messages) actually survives Pillar 1's narrowing into THIS turn's
    # payload, not just into the internal tool_ids set. Safe to remove once
    # that's re-confirmed live. DEBUG tier only (Three-Tiered Telemetry,
    # Fix #1) — an available-tools list is exactly the per-turn noise INFO
    # must never carry; a developer chasing this bug opts in via
    # DANA_LOG_LEVEL=DEBUG instead of it flooding every normal run.
    telemetry.debug(
        "[Turn Context] Available tools for LLM: %s", sorted(t["function"]["name"] for t in tools)
    )
    # Resolved ONCE and reused below — both the dispatch call and the
    # timeout handler must agree on which provider actually ran, so the
    # log line/apology text (see _timeout_apology_text) never blames the
    # wrong one (a real Groq TPM stall previously got logged/apologized
    # for as if it were "the local model").
    target_provider = tool_calling_provider()
    # Cloud-primary turns call whichever single provider
    # tool_calling_provider() resolved (OpenRouter by default) directly —
    # no local gateway process in between. OpenRouter's own server-side
    # ``models`` fallback array (DANA_OPENROUTER_MODEL, comma-separated —
    # see model_provider._resolve_openai_endpoint's "openrouter" branch)
    # is what retries a 429/5xx against the next model upstream now.
    def _run_completion() -> dict[str, Any]:
        return provider.complete_with_tool_calls(pruned_messages, tools=tools, provider=target_provider)

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_completion),
            timeout=_LOCAL_TOOL_CALL_TIMEOUT_SEC,
        )
    except (TimeoutError, asyncio.TimeoutError):
        telemetry.log_error(
            stage="model_call_timeout",
            provider=target_provider,
            timeout_sec=_LOCAL_TOOL_CALL_TIMEOUT_SEC,
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
    caller to HITL-gate/dispatch and loop back with the result.

    ``usage_info`` (Cost Tracking) is ``None`` for an "error" turn (no
    result came back to read it from) and otherwise
    ``{"model": str, "provider": str, "prompt_tokens": int,
    "completion_tokens": int, "cost_usd": float | None}`` straight off this
    iteration's ``ModelProvider.complete_with_tool_calls`` result — see
    ``dana.api.server``'s ``_run_react_loop``, the one place this gets
    broadcast to the frontend as a ``usage_update`` event.
    """

    __slots__ = ("kind", "content", "call", "usage_info")

    def __init__(
        self,
        kind: str,
        content: str = "",
        call: ToolCall | None = None,
        usage_info: dict[str, Any] | None = None,
    ) -> None:
        self.kind = kind  # "final" | "tool_call" | "error"
        self.content = content
        self.call = call
        self.usage_info = usage_info


# Keyword / intent auto-routing (P1 integration): a plain-chat session with
# no plugin tab active has no way to know a CAD request is coming until the
# model itself decides to call load_capability — meaning a request like
# "create a cylinder with radius 10" used to dead-end at "please enable the
# CAD plugin" (load_capability's tools.json enum didn't even offer
# "freecad" as a domain until this fix — see tools.json), or at best cost a
# whole extra turn just calling load_capability before the real action. See
# _keyword_suggested_domains below (this is one entry in its
# _DOMAIN_INTENT_KEYWORDS table) for the general mechanism this now feeds.
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
        # Polygon-shape names for create_freecad_polygon — a live stress
        # test (2026-08-30) showed a request naming a specific polygon
        # (e.g. "hexagonal collar") never actually surfaced this tool: it
        # exists in the ~26-tool "freecad" domain, but Pillar 1's semantic
        # narrowing didn't rank it into the turn's top-K on its own, and the
        # model fell back to relabeling a plain cylinder instead of
        # discovering the real tool via search_tool_catalog. These keywords
        # make it STICKY (see _keyword_suggested_tool_ids) the moment the
        # user's own words name a specific polygon, the same fix already
        # applied to "pyramid"/"prism" above.
        "polygon",
        "hexagon",
        "hexagonal",
        "pentagon",
        "pentagonal",
        "octagon",
        "octagonal",
        "heptagon",
        "nonagon",
        "decagon",
        # "box" is deliberately absent — too common in ordinary English
        # ("check the box", "in a box") to use as a CAD signal on its own;
        # every other keyword here is specific enough not to false-positive
        # on everyday chat.
    }
)

# Same reasoning as _CAD_INTENT_KEYWORDS, for coder_plugin's manifest-driven
# "software_engineering" domain (search_codebase/analyze_codebase/
# run_verification_command/execute_code_task) — a request like "refactor this
# function to fix the bug in utils.py" used to dead-end into a wasted Turn 0
# that only called load_capability(domain="software_engineering") before any
# real action; this folds the domain in up front so search_codebase/
# analyze_codebase are simply already there to call. Deliberately skips this
# domain if coder_plugin never actually loaded (see the "software_engineering"
# in _CAPABILITY_TOOL_IDS check in _keyword_suggested_domains below) — no
# point suggesting a domain manifest loading failed to register.
_SOFTWARE_ENGINEERING_INTENT_KEYWORDS = frozenset(
    {
        "refactor",
        "codebase",
        "traceback",
        "stack trace",
        "pytest",
        "flake8",
        "mypy",
        "git diff",
        "git commit",
        "git status",
        "syntax error",
        "compile error",
        "unit test",
        "source code",
        "function signature",
        "fix the bug",
        "fix this bug",
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        "python file",
        "typescript file",
        "javascript file",
    }
)

# _WEB_TOOLS_TOOL_IDS ("search_web", "read_webpage") — narrow on purpose:
# generic verbs like "search" or "look up" are common outside of an actual
# web-search request, so these lean on phrases that name the web itself.
_WEB_INTENT_KEYWORDS = frozenset(
    {
        "search the web",
        "google",
        "look it up online",
        "browse to",
        "webpage",
        "website",
        "http://",
        "https://",
        "www.",
    }
)

# _VISION_TOOLS_TOOL_IDS ("analyze_workspace_image") — screen/image-specific
# phrases only; a bare "look" or "see" is too generic to use alone.
_VISION_INTENT_KEYWORDS = frozenset(
    {
        "screenshot",
        "screen capture",
        "what's on my screen",
        "whats on my screen",
        "look at my screen",
        "analyze this image",
        "analyze the image",
        "what is on the screen",
    }
)

# _OS_TOOLS_TOOL_IDS (filesystem/shell access) — the last of the four
# curated domains to get pre-router coverage; without this, "read this file"/
# "run this script" used to dead-end at a wasted Turn 0 the same way CAD/
# coder requests once did, before load_capability got called autonomously.
# "list" / "read" / "run" alone are too generic for everyday chat, so every
# phrase here names the filesystem/process action explicitly.
_OS_TOOLS_INTENT_KEYWORDS = frozenset(
    {
        "list directory",
        "read file",
        "write file",
        "edit file",
        "search files",
        "run this script",
        "run python script",
        "terminal command",
        "shell command",
        "background service",
        "file system",
        "filesystem",
    }
)

# Domain name (a _CAPABILITY_TOOL_IDS key) -> its keyword set. Checked live
# against _CAPABILITY_TOOL_IDS in _keyword_suggested_domains rather than
# assumed present, since "software_engineering" only exists once
# refresh_plugin_tools() has actually loaded coder_plugin's manifest.json —
# see _CAPABILITY_TOOL_IDS's own module-import-time refresh_plugin_tools()
# call at the bottom of this file for why that's reliably true by the time
# any real turn runs, without hardcoding the assumption here too.
_DOMAIN_INTENT_KEYWORDS: dict[str, frozenset[str]] = {
    "freecad": _CAD_INTENT_KEYWORDS,
    "software_engineering": _SOFTWARE_ENGINEERING_INTENT_KEYWORDS,
    "web_tools": _WEB_INTENT_KEYWORDS,
    "vision_tools": _VISION_INTENT_KEYWORDS,
    "os_tools": _OS_TOOLS_INTENT_KEYWORDS,
}


def _keyword_suggested_domains(raw_text: str) -> frozenset[str]:
    """Cheap, keyword-only pre-suggestion (never NLP/LLM-based) unioning in
    every domain in ``_DOMAIN_INTENT_KEYWORDS`` whose keywords plainly appear
    in ``raw_text`` — purely additive and ephemeral (recomputed fresh from
    ``raw_text`` on every call, never written to session state): a false
    positive just means a few extra tool defs were offered for one turn (its
    own tool ids are protected from Pillar 1's narrowing too — see
    ``_keyword_suggested_tool_ids`` below — so a false positive costs a
    handful of token defs, never a dropped tool), and a false negative still
    leaves ``load_capability`` as a working fallback per the system prompt's
    "Self-Resolving Missing Capabilities" section. If a suggested tool
    actually gets dispatched, ``dana.api.server``'s
    ``_touch_capability_domains``/``domains_for_tool_id`` already persists
    the domain into the session's real P1 decay tracking from that point on
    — no extra plumbing needed here for it to stick across turns.
    """
    lowered = (raw_text or "").lower()
    if not lowered:
        return frozenset()
    return frozenset(
        domain
        for domain, keywords in _DOMAIN_INTENT_KEYWORDS.items()
        if domain in _CAPABILITY_TOOL_IDS and any(keyword in lowered for keyword in keywords)
    )


# Matches dana.core.tool_retrieval._MIN_TOOLS_TO_NARROW's own "not worth
# narrowing a small pool" threshold, and _DEFAULT_TOP_K's own top-K size —
# kept as separate constants (not imported) since this is a distinct
# narrowing pass over one domain's own tool set, not Pillar 1's general
# narrow_tool_ids_by_query call.
_KEYWORD_DOMAIN_NARROW_MIN = 8
_KEYWORD_DOMAIN_TOP_K = 6


def _keyword_suggested_tool_ids(raw_text: str) -> frozenset[str]:
    """Tool ids for every domain ``_keyword_suggested_domains`` flags for
    ``raw_text`` — folded into ``_call_llm_once``'s ``sticky_ids`` so these
    tools are GUARANTEED to survive Pillar 1's semantic-relevance narrowing
    this turn, the same way an already-dispatched or ``load_capability``-
    unlocked tool id is (see ``_sticky_tool_ids_from_messages``). Without
    this, unioning a keyword-suggested domain into ``effective_plugins``
    alone only makes its tools ELIGIBLE — narrowing could still score e.g.
    ``search_codebase``/``analyze_codebase`` below the RAG top-K threshold
    for a phrasing like "refactor this function to fix the bug" and drop
    them anyway, defeating the entire point of pre-warming them for Turn 0.

    A domain at or under ``_KEYWORD_DOMAIN_NARROW_MIN`` tool ids is included
    WHOLE, same as before (software_engineering, freecad_essential, os_tools,
    web_tools, vision_tools all qualify). A LARGER domain — in practice just
    the raw "freecad" domain (~26 tools) — is trimmed to its ``_rank_by_
    relevance``-ranked top ``_KEYWORD_DOMAIN_TOP_K`` first: measured live, a
    CAD prompt's keyword match used to make the ENTIRE 26-tool domain sticky,
    and ``_cap_schemas_by_token_budget``'s ``must_keep`` is NEVER trimmed even
    when it alone exceeds the budget (by design — see that function's own
    docstring) — so a large keyword-matched domain silently defeated the
    whole token budget instead of merely being protected from narrowing
    within it.

    Deliberately reuses ``_rank_by_relevance`` (over-fetches
    ``registry.retrieve(query, k=len(pool)+20)`` then filters/orders by
    membership in ``pool``) rather than ``dana.core.tool_retrieval.
    narrow_tool_ids_by_query`` here: that function's OWN top-K retrieval
    (``registry.retrieve_specs(query, k=6)``) searches the GLOBAL tool index,
    not scoped to this domain — for a domain that's a small fraction of the
    full registry, a real top-6 measured live against "freecad" returned
    ZERO tools actually IN the freecad domain, so its own empty-result
    fallback (return the input unchanged, to avoid ever returning zero
    tools) silently defeated narrowing entirely for exactly the large-domain
    case this exists to fix. ``_rank_by_relevance``'s much wider over-fetch
    (`len(pool)+20`, i.e. the domain's own full size plus slack) reliably
    covers the whole domain regardless of embedding quality, and falls back
    to a deterministic alphabetical order (never empty) if retrieval itself
    fails.
    """
    ids: set[str] = set()
    for domain in _keyword_suggested_domains(raw_text):
        domain_ids = _CAPABILITY_TOOL_IDS.get(domain, frozenset())
        if len(domain_ids) <= _KEYWORD_DOMAIN_NARROW_MIN:
            ids |= domain_ids
        else:
            ids |= set(_rank_by_relevance(domain_ids, raw_text)[:_KEYWORD_DOMAIN_TOP_K])
    return frozenset(ids)


# Direct phrase -> tool_id anchors, for a primitive whose plain-English name
# is too generic to trust as a DOMAIN-level trigger (_CAD_INTENT_KEYWORDS
# above deliberately keeps "box" out for exactly that reason) but still
# needs a way to GUARANTEE its own tool reaches this turn's model even once
# the "freecad" domain is otherwise already active.
#
# _keyword_suggested_tool_ids's own domain-wide _rank_by_relevance top-K cut
# is not reliable enough on its own for this: a live stress test (2026-08-30)
# showed a multi-clause prompt whose OTHER clauses used strong/distinctive
# words ("cylinder", "hexagonal") out-ranked create_freecad_box's much more
# generic description for a bare "base plate" clause — create_freecad_box
# was ELIGIBLE (the freecad domain was already active) but never made the
# domain's top-6 embedding-similarity cut, so the model never saw it at all.
#
# Each entry is a BOUNDED, multi-word phrase (never a bare generic word like
# "box" or "plate" alone) mapped directly to the one tool_id it should force
# — bypassing _rank_by_relevance's ranking entirely for these specific,
# high-confidence anchors, rather than hoping a wider net of loose keywords
# happens to rank well enough against whatever else is in the same prompt.
_PRIMITIVE_SHAPE_TOOL_ANCHORS: dict[str, str] = {
    "base plate": "create_freecad_box",
    "baseplate": "create_freecad_box",
    "mounting plate": "create_freecad_box",
    "flat plate": "create_freecad_box",
    "build a box": "create_freecad_box",
    "create a box": "create_freecad_box",
    "make a box": "create_freecad_box",
    "rectangular box": "create_freecad_box",
    "cuboid": "create_freecad_box",
    # create_freecad_polygon hasn't been proven in a live multi-step chain
    # yet either (the last run never got past step 1) — anchoring it here
    # too means THIS same ranking bottleneck can't silently reproduce for
    # the hexagon step the moment box's own fix lets the turn get further.
    "hexagon": "create_freecad_polygon",
    "hexagonal": "create_freecad_polygon",
    "pentagon": "create_freecad_polygon",
    "pentagonal": "create_freecad_polygon",
    "octagon": "create_freecad_polygon",
    "octagonal": "create_freecad_polygon",
}


def _anchor_suggested_tool_ids(raw_text: str) -> frozenset[str]:
    """Direct phrase -> tool_id anchors (see _PRIMITIVE_SHAPE_TOOL_ANCHORS)
    — folded into _call_llm_once's force_include_ids/sticky_ids alongside
    _keyword_suggested_tool_ids, so a matched phrase guarantees its exact
    tool id specifically, rather than only widening the DOMAIN it belongs
    to and hoping relevance-ranking keeps it."""
    lowered = (raw_text or "").lower()
    if not lowered:
        return frozenset()
    return frozenset(
        tool_id for phrase, tool_id in _PRIMITIVE_SHAPE_TOOL_ANCHORS.items() if phrase in lowered
    )


async def next_react_turn(
    messages: list[dict[str, Any]],
    active_selection: dict[str, Any] | None = None,
    *,
    raw_text: str = "",
    api_keys: dict[str, str] | None = None,
    active_plugins: frozenset[str] | None = None,
    hidden_tool_ids: frozenset[str] = frozenset(),
    session_id: str | None = None,
) -> ReactTurn:
    """One step of the multi-step ReAct loop: given the full running
    ``messages`` history (system + user + any prior assistant/tool turns
    already appended by the caller), ask the LLM what to do next.

    ``raw_text`` is the ORIGINAL user utterance for this turn (not the
    latest tool result) — ``_finalize_call_arguments``'s selection-reference
    check ("this"/"here"/...) matches against what the user actually said,
    which may be several tool calls back in a multi-step chain. Also feeds
    ``_keyword_suggested_domains`` below, folding any keyword-matched domain
    (CAD, software engineering, web, vision, ...) into THIS turn's effective
    capability set when the user's own words plainly name that kind of
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

    ``hidden_tool_ids`` (Dynamic Domain Locking, ``session["hidden_tool_ids"]``
    — see ``_tool_unload_capability``) is subtracted from BOTH this turn's
    offered schema (via ``_call_llm_once``) AND the ``allowed_tool_ids``
    check below — a hidden tool must never be dispatchable even if the
    model names it anyway from earlier conversation history, the same
    "re-checked, not just un-offered" treatment a deactivated plugin's
    tools already get.

    ``session_id`` (Plan-and-Execute FSM) is what THIS function reads its
    own FSM state from (``_get_fsm_state``/``_active_task``) to decide how
    to narrow/restrict this turn's tool schema — passed explicitly rather
    than relying on the ambient ``dana.session_context`` contextvar for the
    EXACT SAME reason ``build_system_prompt`` already takes it explicitly:
    this runs from ``dana.api.server._run_react_loop``, which calls it
    BEFORE that turn's own ``_execute_and_continue`` has set the contextvar
    for a freshly (re)connected session. ``None`` falls back to the ambient
    value for callers that don't care (``parse_utterance``, tests) —
    identical fallback contract to every other ``session_id`` parameter in
    this module.
    """
    effective_plugins = active_plugins
    if active_plugins is not None:  # None means "not capability-aware" — leave as the full legacy set
        effective_plugins = active_plugins | _keyword_suggested_domains(raw_text)

    # Plan-and-Execute FSM (Phases 2-3): decide THIS turn's tool-schema
    # restriction from the session's own FSM state, once, so both the
    # primary call below and its one retry (_retry_after_unknown_tool_id)
    # honor the identical restriction — a retry that quietly offered the
    # FULL domain again would undo the whole point of a hard planning-phase
    # allow-list. Only engages for a genuinely capability-aware, freecad-
    # active caller with a plan already in play; every other caller
    # (active_plugins=None, a non-CAD domain, or freecad active but no plan
    # yet — that last case is exactly next_react_turn's own hard_restrict_to
    # for the PLANNING phase, handled the same way) computes these as the
    # neutral "no FSM override" defaults, so this is purely additive.
    hard_restrict_to: frozenset[str] | None = None
    narrowing_query: str | None = None
    task_tool_ids: frozenset[str] = frozenset()
    freecad_domain_active = effective_plugins is not None and bool(
        effective_plugins & {"freecad", "freecad_essential", "freecad_full"}
    )
    if freecad_domain_active:
        if not _get_has_plan(session_id):
            # PLANNING: no geometry tool is even offered -- see _llm_tools_
            # schema's hard_restrict_to docstring for why this is a hard
            # allow-list, not a narrowing bias.
            hard_restrict_to = _CORE_TOOL_IDS | {"search_tool_catalog", "check_plugin_registry"}
        elif _get_fsm_state(session_id) == "executing":
            active_task = _active_task(session_id)
            if active_task is not None:
                narrowing_query = active_task.get("description") or raw_text
                task_tool_ids = active_task.get("expected_tool_ids") or frozenset()
        # "validating" (a task with no expected_tool_ids, parked awaiting a
        # manual mark_task_completed) and "done" (plan finished, ad-hoc
        # cleanup) both fall through to the neutral defaults above --
        # exactly today's full-domain, whole-objective-narrowed behavior.

    try:
        result = await _call_llm_once(
            messages,
            api_keys=api_keys,
            active_plugins=effective_plugins,
            raw_text=raw_text,
            hidden_tool_ids=hidden_tool_ids,
            narrowing_query=narrowing_query,
            task_tool_ids=task_tool_ids,
            hard_restrict_to=hard_restrict_to,
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
        telemetry.log_error(stage="next_react_turn", exc_type=type(exc).__name__, detail=str(exc))
        return ReactTurn("error", content=str(exc))

    # Cost Tracking: read off whichever provider actually answered this
    # iteration (result["model"]/["usage"]/["cost_usd"] — see
    # ModelProvider.complete_with_tool_calls) regardless of which of the
    # three ReactTurn outcomes below this call ends up producing.
    usage_info = {
        "model": result.get("model"),
        "provider": result.get("provider"),
        "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens", 0),
        "completion_tokens": (result.get("usage") or {}).get("completion_tokens", 0),
        "cost_usd": result.get("cost_usd"),
    }

    tool_calls = result.get("tool_calls") or []
    if not tool_calls:
        telemetry.debug("[ReAct] LLM finished with final response (no tool call)")
        return ReactTurn("final", content=result.get("content") or "", usage_info=usage_info)
    call = tool_calls[0]  # one tool per LLM turn — the loop itself is what allows chaining several
    # PLANNING's hard_restrict_to is the actual offered/allowed set in that
    # phase -- _tool_ids_for_plugins(effective_plugins) would report the
    # FULL domain (create_plan is gated on has_plan, not on schema
    # visibility, elsewhere), which would wrongly let a hallucinated
    # geometry tool_id through this check even though it was never in the
    # schema the model actually saw this turn.
    allowed_tool_ids = (
        (hard_restrict_to - hidden_tool_ids)
        if hard_restrict_to is not None
        else (_tool_ids_for_plugins(effective_plugins) - hidden_tool_ids)
    )
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
            hidden_tool_ids=hidden_tool_ids,
            narrowing_query=narrowing_query,
            task_tool_ids=task_tool_ids,
            hard_restrict_to=hard_restrict_to,
        )
        if retry_call is None:
            return ReactTurn("final", content=result.get("content") or "", usage_info=usage_info)
        call = retry_call
    call.raw_text = raw_text
    _finalize_call_arguments(call, active_selection)
    telemetry.log_tool_call(call.tool_id, arguments=call.arguments)
    return ReactTurn("tool_call", call=call, usage_info=usage_info)


async def _retry_after_unknown_tool_id(
    messages: list[dict[str, Any]],
    bad_tool_id: str,
    allowed_tool_ids: frozenset[str],
    *,
    api_keys: dict[str, str] | None,
    effective_plugins: frozenset[str] | None,
    raw_text: str = "",
    hidden_tool_ids: frozenset[str] = frozenset(),
    narrowing_query: str | None = None,
    task_tool_ids: frozenset[str] = frozenset(),
    hard_restrict_to: frozenset[str] | None = None,
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
            messages + [nudge],
            api_keys=api_keys,
            active_plugins=effective_plugins,
            raw_text=raw_text,
            hidden_tool_ids=hidden_tool_ids,
            narrowing_query=narrowing_query,
            task_tool_ids=task_tool_ids,
            hard_restrict_to=hard_restrict_to,
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


# Tools whose result payload always comes from engine.py's own `_ok(name=,
# type=, bounding_box=, ...)` helper — the same uniform geometry-result shape
# dana.api.server._CAD_CREATE_TOOLS scopes its own mesh/artifact handling to
# (mirrored here, not imported, since dana.api.server already imports FROM
# this module — importing back would cycle). Every OTHER tool (search_codebase,
# execute_code_task, read_file, a user skill, resync_workspace, ...) has a
# completely different, unrelated result shape that _GEOMETRY_RESULT_KEEP_KEYS
# would gut if applied to it (e.g. execute_code_task's stdout/stderr, a user
# skill's own traceback, resync_workspace's "moved" list) — this allowlist is
# deliberately scoped to ONLY these tools.
_GEOMETRY_RESULT_TOOL_IDS = frozenset(
    {
        "create_freecad_box",
        "create_freecad_cylinder",
        "create_freecad_extrusion",
        "create_freecad_pyramid",
        "create_freecad_star_prism",
        "create_freecad_polygon",
        "perform_freecad_boolean",
        "perform_freecad_edge_operation",
        "modify_freecad_parameter",
        "create_freecad_pipe",
        "align_freecad_objects",
        "create_assembly_mate",
        "create_freecad_sketch_extrude",
        "create_freecad_feature_on_face",
        "batch_pattern_array",
        "insert_standard_part",
        "import_and_solidify_mesh",
    }
)

_GEOMETRY_RESULT_KEEP_KEYS = frozenset(
    {
        "ok",
        "name",
        "type",
        "dimensions",
        "placement",
        "bounding_box",
        "error",
        # A failed geometry-tool payload has no top-level "error" key at all —
        # dispatch_tool_call replaces it with digest_error's own structured
        # shape ({status, tool_id, reason, suggestion, raw_error}) specifically
        # so the model can tell WHY it failed and how to fix it. Dropping
        # these to match the plain success-shape allowlist above would blind
        # the model to every failure's cause, defeating error_digest.py's
        # entire purpose.
        "status",
        "tool_id",
        "reason",
        "suggestion",
        "raw_error",
    }
)


def _slim_geometry_tool_result(tool_id: str, payload: Any) -> Any:
    """Drop every key outside ``_GEOMETRY_RESULT_KEEP_KEYS`` from a geometry
    tool's result payload, e.g. the absolute ``.FCStd``/``.stl`` path (re-sent
    in full on every subsequent turn of a multi-step create/modify/boolean
    chain), ``gui_shown``, or a driver-specific ``note`` string — none of
    which the model needs to keep reasoning about the geometry it just built.

    Scoped to ``_GEOMETRY_RESULT_TOOL_IDS`` and a dict payload only; anything
    else is returned unchanged. Never mutates ``payload`` — this only affects
    the copy that enters the LLM-facing message history, not
    ``ToolResult.payload`` itself (still read unpruned by ``dispatch_tool_call``'s
    ``_OBJECT_PATH_REGISTRY`` bookkeeping and the frontend's own
    ``"tool_result"`` websocket event, both of which run before/independently
    of this).
    """
    if tool_id not in _GEOMETRY_RESULT_TOOL_IDS or not isinstance(payload, dict):
        return payload
    return {k: v for k, v in payload.items() if k in _GEOMETRY_RESULT_KEEP_KEYS}


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

    A geometry tool's payload (``_GEOMETRY_RESULT_TOOL_IDS``) is further
    slimmed to ``_GEOMETRY_RESULT_KEEP_KEYS`` before it enters this message —
    see ``_slim_geometry_tool_result``.
    """
    payload = result.payload if (result.ok or result.payload) else {"ok": False, "error": result.message}
    payload = _slim_geometry_tool_result(result.tool_id, payload)
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

# The Composite Skill Compiler's own tool needs the session's ordered
# CadCallLog to compile FROM — the one piece of session state no other
# handler needs directly (every other tool reads only its own `args` plus
# the shared `engine`/`control_plane`). Same narrow-allowlist pattern as
# _TOOLS_NEEDING_API_KEYS/_TOOLS_NEEDING_MOUNTS above, and disjoint from
# both — no handler needs more than one of these three extras.
_TOOLS_NEEDING_CALL_LOG = frozenset({"compile_plan_as_skill"})


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

    Plan-and-Execute Gatekeeper (Phase 6): the VERY FIRST check, before even
    the handler lookup or any topology redirect — a geometry-mutating
    tool_id (``_RESTRICTED_GEOMETRY_TOOLS``) is refused outright unless this
    session has already called ``create_plan`` (``_get_has_plan``), forcing
    the LLM to outline its topological sequence, boolean strategy, and
    object naming convention up front instead of improvising (and
    hallucinating names for) a boolean chain turn by turn.

    Plan-and-Execute FSM, Phase 4 (RESTORED out-of-order hard block,
    Universal FSM Enforcement): a tool outside ``_CORE_TOOL_IDS`` that
    doesn't belong to the CURRENT active task's own ``expected_tool_ids`` is
    refused BEFORE the handler lookup even happens — see
    ``_fsm_out_of_order_check`` and the module-level comment right above it
    for why this hard block was removed, then restored once
    ``expected_tool_ids`` became existence-validated at ``create_plan`` time
    (Fix #1) rather than an embedding guess, and why its scope then had to
    widen past ``_RESTRICTED_GEOMETRY_TOOLS`` alone (a Planner-declared
    non-geometry tool, e.g. ``execute_freecad_script``, was a second
    hallucination-shaped hole otherwise). A task with no fixed tool
    signature at all (``expected_tool_ids`` empty) is unaffected — any
    non-core tool is still let through for it, exactly as before.

    Deterministic Task Advancement, Phase 4: on the SUCCESS path below,
    ``_advance_fsm_on_dispatch`` auto-advances the plan the instant the
    dispatched tool matches the active task's own ``expected_tool_ids`` —
    the code-enforced replacement for trusting a separate
    ``mark_task_completed`` call; see that function's own docstring.
    """
    if call.tool_id in _RESTRICTED_GEOMETRY_TOOLS and not _get_has_plan():
        reason = (
            "Execution blocked. You MUST call 'create_plan' to explicitly outline your "
            "step-by-step topological sequence, boolean strategy, and object naming "
            "conventions before generating any geometry."
        )
        return ToolResult(call.tool_id, False, {"ok": False, **digest_error(call.tool_id, reason)}, reason, 0)

    # Universal FSM Enforcement: governs every tool EXCEPT the always-
    # available plan-management/discovery/self-teaching primitives in
    # _CORE_TOOL_IDS (create_plan, mark_task_completed, load_capability,
    # ...) — not just _RESTRICTED_GEOMETRY_TOOLS. A Planner-declared
    # non-geometry tool (execute_freecad_script, search_web, ...) must be
    # enforced exactly like a geometry one; see _fsm_out_of_order_check's
    # own module-level comment for the exploit this closes.
    if call.tool_id not in _CORE_TOOL_IDS:
        out_of_order_reason = _fsm_out_of_order_check(call.tool_id)
        if out_of_order_reason is not None:
            return ToolResult(
                call.tool_id,
                False,
                {"ok": False, **digest_error(call.tool_id, out_of_order_reason)},
                out_of_order_reason,
                0,
            )

    handler = TOOL_HANDLERS.get(call.tool_id)
    if handler is None:
        return ToolResult(call.tool_id, False, {"ok": False, **digest_error(call.tool_id, "unknown tool_id")}, f"unknown tool_id '{call.tool_id}'", 0)
    start = time.perf_counter()
    skill_traceback: str | None = None
    # Silent Override, centralized (Phase 5 — Topological Lineage Graph):
    # rewrite any stale-but-known input-object argument to its
    # resolve_living_leaf BEFORE the handler ever sees it — see
    # _apply_topology_redirects's docstring. `resolved_arguments` (not
    # call.arguments) is what actually gets dispatched from here on;
    # `input_object_names` is threaded through to _record_topology_node once
    # the call succeeds, and `topology_warning` gets merged into the
    # LLM-visible payload below.
    # Defaults for the (rare) case _apply_topology_redirects itself raises
    # below (Fix #3 — Kill Silent Auto-Redirection) — call_log.record at the
    # bottom of this function always needs SOME value for these three, even
    # when the redirect step is what failed, never the handler.
    resolved_arguments, input_object_names, topology_warning = call.arguments, [], None
    try:
        # Fix #3 — Kill Silent Auto-Redirection: this call now RAISES
        # (RuntimeError) instead of silently substituting a guessed object
        # for perform_freecad_edge_operation/create_freecad_feature_on_face
        # (see _apply_topology_redirects's own docstring) — folded into this
        # SAME try/except so that failure is surfaced as an ordinary
        # digested tool failure below, exactly like a handler exception,
        # rather than crashing this function's caller.
        resolved_arguments, input_object_names, topology_warning = _apply_topology_redirects(
            call.tool_id, call.arguments
        )
        if call.tool_id in _TOOLS_NEEDING_API_KEYS:
            payload = handler(resolved_arguments, engine, control_plane, api_keys=api_keys)
        elif call.tool_id in _TOOLS_NEEDING_MOUNTS:
            payload = handler(resolved_arguments, engine, control_plane, allowed_mounts=allowed_mounts)
        elif call.tool_id in _TOOLS_NEEDING_CALL_LOG:
            payload = handler(resolved_arguments, engine, control_plane, call_log=call_log)
        else:
            payload = handler(resolved_arguments, engine, control_plane)
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
        if call.tool_id not in _CORE_TOOL_IDS:  # Universal FSM Enforcement — see the gate above
            _advance_fsm_on_dispatch(call.tool_id, ok=False)
    else:
        message = "ok"
        if isinstance(payload, dict) and payload.get("name") and payload.get("path"):
            _object_registry()[str(payload["name"])] = str(payload["path"])
            _record_topology_node(str(payload["name"]), input_object_names)
        if topology_warning and isinstance(payload, dict) and payload.get("ok", True):
            existing = payload.get("message")
            payload["message"] = f"{existing} {topology_warning}".strip() if existing else topology_warning
        if call.tool_id not in _CORE_TOOL_IDS and isinstance(payload, dict):  # Universal FSM Enforcement
            fsm_note = _advance_fsm_on_dispatch(call.tool_id, ok=True)
            if fsm_note:
                existing = payload.get("message")
                payload["message"] = f"{existing} {fsm_note}".strip() if existing else fsm_note
    telemetry.log_tool_result(call.tool_id, ok, payload=payload, message=message, duration_ms=duration_ms)
    if call_log is not None:
        # resolved_arguments, not call.arguments: the "Show Your Work" export
        # should record what actually ran (post Silent-Override redirect),
        # not a stale pre-redirect name that never touched the CAD engine.
        call_log.record(
            call.tool_id, resolved_arguments, ok=ok, result=payload if ok else {}, error=None if ok else message
        )
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
        "create_freecad_polygon",
        "perform_freecad_boolean",
        "perform_freecad_edge_operation",
        "create_freecad_pipe",
        "create_freecad_sketch_extrude",
        "create_freecad_feature_on_face",
        "batch_pattern_array",
        "insert_standard_part",
        "import_and_solidify_mesh",
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
        param = str(payload.get("parameter_name") or "")
        if param.lower() in _VECTOR_PARAMETER_NAMES:
            return f"Moved `{payload.get('name')}` to {payload.get('new_value')}."
        return f"Set `{payload.get('name')}`.{param} = {payload.get('new_value')}mm."
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
    "get_topology_dag",
    "is_mutating_tool",
    "is_visual_inspection_tool",
    "list_user_skills",
    "next_react_turn",
    "parse_utterance",
    "plugin_registry_view",
    "refresh_plugin_tools",
    "refresh_user_skills",
    "resolve_living_leaf",
    "summarize_result",
)

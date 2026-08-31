"""Headless FastAPI server — the single backend for the Tauri/React frontend.

Replaces the Gradio UI's in-process callbacks with a ``/ws/chat`` WebSocket
that streams ReAct-style tool-dispatch events to any connected client.
Each user turn runs a genuine multi-step ReAct while-loop (see
``_run_react_loop`` below): the LLM can chain several tool calls within one
turn, each one's result appended back into the running conversation so the
NEXT LLM call actually sees it (e.g. read a bounding box, then use those
numbers to place a new object) — not just one tool_call -> tool_result ->
assistant_message per turn. All dispatch logic still lives in
``dana.core.react_dispatch`` and goes through
``dana.platform.get_control_plane()`` / ``get_cad_engine()`` exactly as the
legacy UI did — only the transport changed, not the drivers.

Boot locally with ``scripts/launchers/launch_api_server.py``. In production
(e.g. a Hugging Face Space Docker deploy) the built ``frontend/dist`` bundle
is mounted at ``/`` automatically if present, so this single process serves
both the API and the static React app.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Loaded here, before any dana.* import below — the one module every entry
# point (start_dana.py, app.py, scripts/launchers/launch_api_server.py,
# run_e2e_cad.py, pytest) already imports — so DANA_HEADLESS/TRIPO_API_KEY/
# etc. are guaranteed present in os.environ for the live web app too, not
# just the standalone E2E runner (which loads its own copy of .env for the
# same reason: several checks in this module and dana.plugins.freecad.engine
# read os.environ directly, not through
# dana.core.model_provider.ensure_dotenv_loaded(), so nothing upstream of
# them was guaranteed to have loaded .env first). Same explicit-path-then-
# default-search double call ensure_dotenv_loaded() itself uses, for the same
# reason: reliable regardless of this process's current working directory at
# launch. Must precede the dana.* imports just below, not merely follow the
# stdlib ones — several of those modules read os.environ at import time.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from dana.api import artifacts_registry  # noqa: E402
from dana.api.cad import router as _cad_router  # noqa: E402
from dana.api.sessions import derive_title, is_valid_session_id, load_session, new_session_id, save_session  # noqa: E402
from dana.api.sessions import router as _sessions_router  # noqa: E402
from dana.api.system import router as _system_router  # noqa: E402
from dana.api.workspace import load_mounted_directories  # noqa: E402
from dana.api.workspace import router as _workspace_router  # noqa: E402
from dana.core.model_provider import tool_calling_provider  # noqa: E402
from dana.plugins.planning.task_board import create_plan as _tb_create_plan  # noqa: E402
from dana.plugins.planning.task_board import get_active_plan as _tb_get_active_plan  # noqa: E402
from dana.core.react_dispatch import (  # noqa: E402
    ToolResult,
    build_assistant_tool_call_message,
    build_system_prompt,
    build_tool_result_message,
    build_user_message,
    build_visual_inspection_result,
    describe_tool_call,
    dispatch_tool_call,
    domains_for_tool_id,
    driver_state,
    extract_user_text,
    is_mutating_tool,
    is_visual_inspection_tool,
    next_react_turn,
    plugin_registry_view,
)
from dana.core.context_distiller import schedule_distillation  # noqa: E402
from dana.paths import CAPTURES_DIR  # noqa: E402
from dana.session_context import set_session_id  # noqa: E402
from dana.platform import get_cad_engine, get_control_plane  # noqa: E402
from dana.platform.factory import IS_HF_SPACE  # noqa: E402
from dana.plugins.freecad.call_log import CadCallLog  # noqa: E402
from dana.plugins.freecad.py_export import write_macro_script  # noqa: E402
from dana.plugins.memory.core_memory import read_core_memory  # noqa: E402
from dana.security.dry_run import is_dry_run_enabled  # noqa: E402
from dana.plugins.os.desktop_vision import _capture_primary_monitor_jpeg_b64  # noqa: E402
from dana.services.voice_service import VoiceService, VoiceState  # noqa: E402

_FRONTEND_DIST = _REPO_ROOT / "frontend" / "dist"

# Dev origins for `npm run tauri dev` / `npm run dev` (Vite default 5173,
# Tauri's dev webview origin, and the packaged app's custom scheme).
_DEV_ORIGINS = (
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "tauri://localhost",
    "http://tauri.localhost",
)

# Geometry meshes are generated to arbitrary temp paths by the CAD engine;
# this registry maps opaque tokens to those paths so /api/mesh/{token} never
# has to accept (and validate) a raw filesystem path from a client.
_MESH_REGISTRY: dict[str, Path] = {}

# Same pattern, for synthesized TTS WAV files (see _speak_reply) — opaque
# token in, /api/audio/{token}.wav out, never a raw filesystem path.
_AUDIO_REGISTRY: dict[str, Path] = {}

# Capability routing (dana.core.react_dispatch's _tool_ids_for_plugins/
# build_system_prompt) keys off the underlying capability/tool DOMAIN name
# ("freecad", matching dana/plugins/freecad/*), not the frontend's UI-facing
# plugin id ("cad", see frontend/src/plugins/registry.ts) — this map bridges
# the two so react_dispatch never has to know what the frontend calls its
# own tabs, and the frontend is free to rename/add plugin ids later without
# this module's tool-routing logic changing at all. Unrecognized ids pass
# through unchanged (a future plugin's domain name might just BE its id).
_PLUGIN_ID_TO_CAPABILITY: dict[str, str] = {
    # "freecad_essential" (dana.core.react_dispatch._FREECAD_ESSENTIAL_TOOL_IDS,
    # 7 tools) instead of the full ~24-tool "freecad" domain: sending all of
    # "freecad" as `tools=` on every turn is what previously blew a free-tier
    # Groq model's 8000 TPM ceiling (observed live: HTTP 429, "Used 6895,
    # Requested 6842"). The agent's own autonomous load_capability("freecad")
    # call already redirects to this same essential set for the identical
    # reason — this just brings the frontend's CAD-tab activation path in
    # line with it instead of leaving it as the one unmitigated route to the
    # full schema. Nothing is walled off: the agent can still self-escalate
    # to domain="freecad_full" via load_capability whenever a task genuinely
    # needs a heavier tool (patterns, assembly mates, blueprints, standard
    # parts, engineering-standard lookup, camera control).
    "cad": "freecad_essential",
    "freecad": "freecad",
    "coder": "software_engineering",
    "software_engineering": "software_engineering",
}


def _normalize_active_plugins(raw: Any) -> frozenset[str]:
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(_PLUGIN_ID_TO_CAPABILITY.get(p, p) for p in raw if isinstance(p, str) and p)


# P1 of the local-agent rescue plan: agent_loaded_capabilities used to be a
# frozenset that only ever grew for the life of a session — every
# load_capability call permanently added another domain's tools to every
# subsequent turn's schema (dana/tools/tools.json is 113KB+ across ~44
# tools), so a long session's prompt-eval time (and therefore TTFT) crept
# up turn over turn even though the actual chat history doesn't accumulate
# in `messages` (each new user turn starts a fresh system+user pair — see
# _process_user_text). This is how many session TURNS (not ReAct-loop
# iterations within one turn — see _MAX_REACT_ITERATIONS) a domain stays in
# the effective set without being touched again before it's dropped back
# out of the schema. Not a permanent loss: the agent can always call
# load_capability again if it turns out to still need it.
_CAPABILITY_DECAY_TURNS = 4


def _effective_capabilities(session: dict[str, Any]) -> frozenset[str]:
    """The set actually passed to react_dispatch's tool/prompt routing —
    the union of the frontend's UI-driven ``active_plugins`` and whatever
    the agent has autonomously unlocked via the ``load_capability`` tool
    this session AND touched recently enough to still count.

    ``session["capability_unlocked_at_turn"]`` (a domain -> the
    ``session["turn_counter"]`` value it was last unlocked/used at) replaces
    the old ``agent_loaded_capabilities`` frozenset-that-only-grows: a
    domain whose last touch is more than ``_CAPABILITY_DECAY_TURNS`` turns
    ago is excluded from the return value AND pruned from the dict here —
    so it disappears from the tool schema, not just the returned set,
    instead of silently accumulating dead entries for the rest of the
    session. Refreshed in two places: ``_tool_load_capability`` succeeding
    (re-unlocking it), and ``_execute_and_continue`` dispatching any tool
    that belongs to it (``domains_for_tool_id``) — so a domain the agent is
    actively using never expires mid-task.

    Computed in exactly one place so a plugin the frontend just deactivated
    never retroactively strips a domain the agent loaded on its own
    initiative — "update_context" (see ws_chat) only ever writes
    ``active_plugins``, never touches ``capability_unlocked_at_turn``.
    """
    active_plugins = session.get("active_plugins") or frozenset()
    unlocked_at: dict[str, int] = session.setdefault("capability_unlocked_at_turn", {})
    turn = session.get("turn_counter", 0)
    fresh = {
        domain: last_turn
        for domain, last_turn in unlocked_at.items()
        if turn - last_turn <= _CAPABILITY_DECAY_TURNS
    }
    if len(fresh) != len(unlocked_at):
        session["capability_unlocked_at_turn"] = fresh
    return active_plugins | frozenset(fresh)


def _touch_capability_domains(session: dict[str, Any], domains: frozenset[str]) -> None:
    """Stamps ``domains`` as freshly used at the session's current turn —
    shared by a successful ``load_capability`` call and any tool dispatch
    belonging to an already-unlocked domain (see ``_execute_and_continue``),
    so either one resets that domain's decay clock.
    """
    if not domains:
        return
    turn = session.get("turn_counter", 0)
    unlocked_at: dict[str, int] = session.setdefault("capability_unlocked_at_turn", {})
    for domain in domains:
        unlocked_at[domain] = turn

# Every open /ws/chat connection's per-session context (currently just
# `active_selection`), keyed by the websocket itself — lets the headless
# VoiceService (running on its own thread, with no browser tab of its own)
# feed a finalized transcript through the same dispatch path each connected
# client's typed chat messages go through.
_active_sessions: dict[WebSocket, dict[str, Any]] = {}
_voice_service: VoiceService | None = None
_event_loop: asyncio.AbstractEventLoop | None = None


async def _broadcast(message: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    for ws in list(_active_sessions):
        try:
            await ws.send_json(message)
        except Exception:  # noqa: BLE001 — connection already gone; reap it below
            dead.append(ws)
    for ws in dead:
        _active_sessions.pop(ws, None)


class _BroadcastStream:
    """Terminal History plumbing: tees every line written to ``stream``
    (stdout or stderr — every ``print()``/traceback in this process, from
    any thread) out to every connected ``/ws/chat`` client as a
    ``server_log`` event, so the frontend's floating Terminal History panel
    shows live backend output without a separate log-tailing mechanism.
    The original stream is always written first — this never replaces real
    console output, only mirrors it.

    Buffers partial lines (``write`` can be called with partial/multi-line
    chunks) and only emits complete lines. Broadcasting itself hops onto the
    asyncio event loop via ``run_coroutine_threadsafe`` since this can be
    called from a worker thread (e.g. VoiceService, a tool's background
    thread) with no running loop of its own.
    """

    def __init__(self, original: Any, stream_name: str) -> None:
        self._original = original
        self._stream_name = stream_name
        self._buffer = ""

    def write(self, s: str) -> int:
        self._original.write(s)
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._emit(line)
        return len(s)

    def flush(self) -> None:
        self._original.flush()

    def isatty(self) -> bool:
        return False

    def _emit(self, line: str) -> None:
        loop = _event_loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                _broadcast({"type": "server_log", "stream": self._stream_name, "line": line}), loop
            )
        except RuntimeError:  # noqa: BLE001 — event loop already closed at shutdown
            pass


def _on_voice_state(state: VoiceState, transcript: str) -> None:
    """Runs on VoiceService's worker thread — hop back onto the event loop.

    VoiceService hands off a finalized transcript on the "processing"
    transition (state stays "processing" but now carries the transcript —
    see VoiceService's class docstring), not "speaking": "speaking" is
    reserved for dana.api.server's own TTS-playback state (_speak_reply),
    driven by the assistant's reply, not the user's STT input. Conflating
    the two used to mean the orb flashed "speaking" for the user's own
    transcript instead of the assistant's actual voice reply.
    """
    loop = _event_loop
    if loop is None:
        return

    async def _emit() -> None:
        await _broadcast({"type": "voice_state", "state": state, "transcript": transcript})
        if state == "processing" and transcript:
            for ws, session in list(_active_sessions.items()):
                await _process_user_text(ws, session, transcript)

    asyncio.run_coroutine_threadsafe(_emit(), loop)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _voice_service, _event_loop
    _event_loop = asyncio.get_running_loop()
    _voice_service = VoiceService(on_state=_on_voice_state)
    _voice_service.start()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _BroadcastStream(original_stdout, "stdout")  # type: ignore[assignment]
    sys.stderr = _BroadcastStream(original_stderr, "stderr")  # type: ignore[assignment]
    sweep_task = asyncio.create_task(_sweep_stale_suspensions())
    try:
        yield
    finally:
        sweep_task.cancel()
        sys.stdout, sys.stderr = original_stdout, original_stderr
        if _voice_service is not None:
            _voice_service.stop()


app = FastAPI(title="Dana API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_DEV_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(_workspace_router)
app.include_router(_sessions_router)
app.include_router(_system_router)
app.include_router(_cad_router)


def _register_mesh(path: str) -> str:
    token = uuid.uuid4().hex
    _MESH_REGISTRY[token] = Path(path)
    return token


def _register_audio(path: str) -> str:
    token = uuid.uuid4().hex
    _AUDIO_REGISTRY[token] = Path(path)
    return token


@app.get("/api/health")
def health() -> dict[str, Any]:
    # "provider" reuses the exact same source of truth _call_llm_once
    # resolves its own ReAct-loop provider from (dana.core.model_provider.
    # tool_calling_provider) — this is a status read, never a second
    # decision, so the frontend's model-indicator badge can never disagree
    # with what a real turn is actually about to do.
    return {"ok": True, "provider": tool_calling_provider(), **driver_state()}


@app.get("/api/plugins")
def plugins() -> dict[str, Any]:
    return {"ok": True, **plugin_registry_view()}


CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/api/vision", StaticFiles(directory=str(CAPTURES_DIR)), name="vision")


@app.get("/api/mesh/{token}.stl")
def get_mesh(token: str) -> FileResponse:
    path = _MESH_REGISTRY.get(token)
    if path is None or not path.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="mesh not found")
    return FileResponse(path, media_type="model/stl", filename=f"{token}.stl")


@app.get("/api/mesh/{token}.glb")
def get_mesh_glb(token: str) -> FileResponse:
    """Same opaque-token registry/route shape as ``get_mesh`` above, for a
    ``generate_3d_from_image`` result that came back as a ``.glb`` rather
    than a ``.stl`` — see that route's own reasoning for why the extension
    is baked into the path rather than a single ``{ext}`` route param."""
    path = _MESH_REGISTRY.get(token)
    if path is None or not path.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="mesh not found")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{token}.glb")


@app.get("/api/mesh/{token}.obj")
def get_mesh_obj(token: str) -> FileResponse:
    """Same as ``get_mesh_glb`` above, for a ``generate_3d_from_image``
    result that came back as a ``.obj`` instead."""
    path = _MESH_REGISTRY.get(token)
    if path is None or not path.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="mesh not found")
    return FileResponse(path, media_type="model/obj", filename=f"{token}.obj")


@app.get("/api/mesh/{token}.urdf")
def get_urdf(token: str) -> FileResponse:
    """Same opaque-token registry/route shape as ``get_mesh`` above, one
    extension over — reused so ``tool_result.mesh_url`` stays the single
    field Viewer3D watches regardless of whether a turn produced a plain
    mesh or a full URDF assembly; the frontend picks the loader by the
    URL's own file extension (see Viewer3D.tsx).
    """
    path = _MESH_REGISTRY.get(token)
    if path is None or not path.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="urdf not found")
    return FileResponse(path, media_type="application/xml", filename=f"{token}.urdf")


@app.get("/api/audio/{token}.wav")
def get_audio(token: str) -> FileResponse:
    path = _AUDIO_REGISTRY.get(token)
    if path is None or not path.is_file():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(path, media_type="audio/wav", filename=f"{token}.wav")


# Tools whose successful result is a solid FreeCAD/mock CAD document that
# should be tessellated and pushed to the 3D viewer as an STL mesh — kept
# as one constant since dag-node coloring and mesh export both key off it.
_CAD_CREATE_TOOLS = frozenset(
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
        "batch_pattern_array",
        "insert_standard_part",
        "import_and_solidify_mesh",
        # get_freecad_bounding_box, inspect_spatial_properties,
        # analyze_bounding_box_collisions, export_freecad_model, and
        # take_canvas_screenshot are intentionally absent: those reads have
        # no new/changed geometry, and an export's own output (.stl/.step,
        # possibly multi-object) isn't a single FreeCAD document
        # export_mesh_stl can re-tessellate for the viewer the way every
        # other tool here produces.
    }
)


def _node_type_for(tool_id: str) -> str:
    if tool_id in ("execute_vision_analysis", "take_canvas_screenshot"):
        return "vision"
    if tool_id in _CAD_CREATE_TOOLS or tool_id == "generate_urdf_assembly":
        return "tool"
    return "agent"


async def _dag_start(websocket: WebSocket, node_id: str, label: str, node_type: str, inputs: dict[str, Any]) -> None:
    await websocket.send_json(
        {"type": "dag_node_start", "node_id": node_id, "label": label, "node_type": node_type, "inputs": inputs}
    )


async def _dag_complete(
    websocket: WebSocket, node_id: str, status: str, output: dict[str, Any], duration_ms: int
) -> None:
    await websocket.send_json(
        {"type": "dag_node_complete", "node_id": node_id, "status": status, "output": output, "duration_ms": duration_ms}
    )


async def _send_tool_dispatch_start(websocket: WebSocket, node_id: str, call: Any) -> None:
    """Consolidated replacement for the old dag_node_start(dispatch) +
    "tool_call" + "tool_start" trio — ONE event per tool dispatch instead of
    three (WebSocket Consolidation). Carries everything either consumer used
    to read off two separate messages: DAG Monitor's node_id/label/node_type
    (same shape _dag_start already sent for a "dispatch-N" node) and
    ChatPanel's Agent Activity feed's tool_name/args_summary (previously
    "tool_start"'s own fields) — plus the dispatch arguments themselves
    (previously the standalone "tool_call" event).

    Deliberately NOT used for a "parse-N" node (the LLM's own reasoning
    step, still `_dag_start`/`_dag_complete` — see _run_react_loop) — this
    is only for an actual tool dispatch, exactly the three call sites
    `_dag_start`+"tool_call" used to cover: _run_react_loop's immediate
    non-mutating dispatch, _resolve_react_hitl's post-approval dispatch, and
    _resolve_visual_capture's take_canvas_screenshot resolution.
    """
    await websocket.send_json(
        {
            "type": "tool_dispatch_start",
            "node_id": node_id,
            "label": call.tool_id,
            "node_type": _node_type_for(call.tool_id),
            "tool_name": call.tool_id,
            "arguments": call.arguments,
            "args_summary": _summarize_tool_args(call.tool_id, call.arguments),
        }
    )


async def _send_tool_dispatch_end(
    websocket: WebSocket, node_id: str, call: Any, result: ToolResult, mesh_url: str | None
) -> None:
    """Consolidated replacement for the old "tool_complete" +
    dag_node_complete(dispatch) + "tool_result" trio — the result payload is
    now serialized on the wire exactly ONCE (`output`), where it used to go
    out twice (dag_node_complete.output and tool_result.payload were always
    byte-identical). `status`/`output`/`duration_ms`/`mesh_url` together
    cover every field either consumer (DAG Monitor, Agent Activity, the
    mesh-viewer's meshUrl state) used to read off the three separate events.

    Also carries `message` — NOT in this consolidation's original field
    list, added back deliberately: CoderPlugin.tsx's error banner falls back
    to it (`payload.error ?? message`) when a tool's own failure payload
    doesn't use the "error" key (many use ``digest_error``'s "reason" key
    instead, or something else entirely) — `result.message` is the one
    field `dispatch_tool_call` unconditionally populates with a meaningful
    failure summary regardless of which shape the tool's own payload used,
    so dropping it would have silently degraded that fallback.

    Timing note: the old "tool_complete" fired immediately after
    ``dispatch_tool_call`` returned, before mesh/STEP export and visual
    verification ran — Agent Activity's spinner-to-checkmark flip was
    therefore slightly faster than DAG Monitor's own status update, which
    already waited for all of that (same call site as this function). This
    consolidation makes both consumers wait for the same single event, so
    Agent Activity's status now resolves at DAG Monitor's (later) pace
    instead of its own faster one — a deliberate, minor trade-off of this
    consolidation, not an oversight.
    """
    await websocket.send_json(
        {
            "type": "tool_dispatch_end",
            "node_id": node_id,
            "tool_id": call.tool_id,
            "status": "success" if result.ok else "error",
            "output": result.payload,
            "message": result.message,
            "duration_ms": result.duration_ms,
            "mesh_url": mesh_url,
        }
    )


async def _broadcast_usage_update(
    websocket: WebSocket, session: dict[str, Any], usage_info: dict[str, Any]
) -> None:
    """Cost Tracking: accumulates one ``next_react_turn`` iteration's LLM
    cost (``dana.core.react_dispatch.ReactTurn.usage_info``, ultimately from
    ``ModelProvider.complete_with_tool_calls`` -> ``dana.core.pricing``) onto
    this SESSION's running total, then pushes a ``usage_update`` event to
    the frontend's CostBar. Called once per ``_run_react_loop`` iteration
    (right after ``next_react_turn`` returns), so a long multi-tool-call
    turn reports cost incrementally rather than only once the whole turn
    finishes.

    ``cost_usd`` is ``None`` whenever the model isn't in ``dana.core.
    pricing``'s table (every local Ollama model, by construction) — the
    running total only ever accumulates a KNOWN cost, never silently
    treats an unpriced model as free.
    """
    model = usage_info.get("model") or "unknown"
    cost_usd = usage_info.get("cost_usd")
    tracking = session.setdefault("cost_tracking", {"total_usd": 0.0, "by_model": {}})
    if cost_usd is not None:
        tracking["total_usd"] += cost_usd
        tracking["by_model"][model] = tracking["by_model"].get(model, 0.0) + cost_usd
    await websocket.send_json(
        {
            "type": "usage_update",
            "model": model,
            "tokens": {
                "prompt": usage_info.get("prompt_tokens", 0),
                "completion": usage_info.get("completion_tokens", 0),
            },
            "cost_usd": cost_usd,
            "session_total_usd": tracking["total_usd"],
            "by_model": dict(tracking["by_model"]),
        }
    )


async def _broadcast_plan_update(websocket: WebSocket, plan: dict[str, Any] | None) -> None:
    """Pushes the Task Planner's current state (dana.plugins.planning.
    task_board's global plan — the exact ``{"objective", "tasks",
    "current_task_id"}`` shape ``get_active_plan``/``create_plan``/
    ``mark_task_completed`` all already share, same as the "## Current
    Active Plan" system-prompt block reads) to the frontend's
    PlanChecklist, as a ``plan_update`` event.

    Two call sites, both guarded on ``result.ok``/``plan_result.get("ok")``
    before ever reaching here: ``_execute_and_continue`` (the MODEL calling
    create_plan/mark_task_completed as an ordinary tool dispatch) and
    ``_process_user_text`` (the structural create_plan override that
    bypasses the model entirely — see its own docstring). ``plan=None``
    (a caller passing one through unconditionally) is a no-op here too, so
    a stray call after a FAILED mutation never broadcasts a missing plan.
    """
    if not plan:
        return
    await websocket.send_json({"type": "plan_update", "plan": plan})


async def _broadcast_memory_update(websocket: WebSocket, memory: dict[str, str] | None) -> None:
    """Pushes the Core Memory's current state (dana.plugins.memory.
    core_memory's read_core_memory() dict — the exact ``{section: content}``
    shape write_core_memory/replace_core_memory both produce and
    format_core_memory_for_prompt reads) to the frontend's MemoryViewer,
    as a ``memory_update`` event.

    Called from ``_execute_and_continue`` whenever the MODEL successfully
    calls update_core_memory as an ordinary tool dispatch. ``memory=None``
    (a caller passing one through unconditionally) is a no-op here, so
    a stray call after a FAILED mutation never broadcasts missing memory.
    """
    if memory is None:
        return
    await websocket.send_json({"type": "memory_update", "core_memory": memory})


async def _speak_reply(websocket: WebSocket, text: str) -> None:
    """Synthesizes ``text`` through the existing Piper/pyttsx3 pipeline
    (dana.audio.multi_voice_tts, unchanged — same module the legacy
    desktop-assistant loop already used) and hands the resulting WAV to
    this one connected client as a fetchable URL, exactly like a CAD tool's
    mesh_url. Drives the AssistiveOrb into "speaking" for every connected
    window (all windows should show the same assistant is-talking state);
    the actual audio itself only goes to the client whose turn this is —
    every open window hearing the same reply simultaneously would be a
    cacophony, not "in sync".

    Synthesis is a blocking call (loads/uses the ONNX model) so it's pushed
    to a worker thread — this coroutine must not block the event loop the
    way every other websocket session on this server shares.

    The import is deliberately deferred (not a top-level import) and
    wrapped in the same broad except as the rest of this function: the
    dana.audio package pulls in dana.core.shared_state, which has its own
    pre-existing hard dependency on a diagnostics-only module not on
    sys.path outside pytest's conftest. Importing eagerly at server-module
    load time would take the whole server down at boot; VoiceService's own
    lazy/guarded imports (see start_whisper_background_load) already work
    around the same issue the same way.
    """
    try:
        from dana.audio.multi_voice_tts import synthesize_speech

        path = await asyncio.to_thread(synthesize_speech, text, voice_id="dana")
    except Exception:  # noqa: BLE001 — TTS is best-effort; never fail the turn over a synth error
        return
    token = _register_audio(str(path))
    await _broadcast({"type": "voice_state", "state": "speaking", "transcript": text})
    await websocket.send_json({"type": "assistant_audio", "audio_url": f"/api/audio/{token}.wav"})


# Safety counter for _run_react_loop — forcefully stops the loop after this
# many tool-executing iterations within one user turn, so a model stuck
# re-deciding to call tools (a hallucination loop) can't run forever. Raised
# from 13 to 30: a complex multi-part CAD assembly (several primitives, a
# mesh import, multiple booleans, edge ops, verification) can legitimately
# need more turns than a simple "build one box" scenario ever did.
_MAX_REACT_ITERATIONS = 30

# Permanently HITL-exempt, every session, no prior approval needed —
# narrow parametric FreeCAD geometry CRUD (create/modify/boolean/pattern/
# align a primitive, insert a standard part) that only ever touches its own
# generated .FCStd output. Explicit tool_id list, NOT a tool_id substring
# match: a "contains freecad" check would also silently exempt
# execute_freecad_script and modify_existing_freecad_document — both run an
# ARBITRARY caller-supplied Python/FreeCAD script, not a parametric
# geometry op, so both stay behind is_mutating_tool's normal HITL gate
# (session-allowlist-eligible, but never permanently pre-approved) no
# matter how this set changes.
_HITL_ALWAYS_APPROVED_TOOLS: frozenset[str] = frozenset(
    {
        "create_freecad_box",
        "create_freecad_cylinder",
        "create_freecad_pyramid",
        "create_freecad_star_prism",
        "create_freecad_extrusion",
        "create_freecad_pipe",
        "create_freecad_sketch_extrude",
        "perform_freecad_boolean",
        "perform_freecad_edge_operation",
        "modify_freecad_parameter",
        "align_freecad_objects",
        "insert_standard_part",
        # Local-only geometry CRUD, same as every other entry here — imports
        # an already-downloaded mesh FILE (never a network call of its own)
        # into the shared .FCStd session doc. Unlike generate_3d_from_image
        # (which DOES reach a third-party service and is deliberately NOT
        # here), there's nothing about this specific step for a human to
        # gate on beyond what create_freecad_box's own presence here already
        # implies is fine.
        "import_and_solidify_mesh",
    }
)

# P3 of the local-agent rescue plan — a suspended react_state/visual_state
# (a mutating tool awaiting HITL approval, or take_canvas_screenshot
# awaiting the frontend's R3F capture) normally resumes the instant the
# matching hitl_response/visual_capture_response arrives. If the frontend
# never replies — a dropped message, a closed plugin tab, a crashed
# renderer — the turn would otherwise stay parked forever with no path back
# for the user short of reconnecting. _sweep_stale_suspensions is the
# wall-clock backstop: any suspension older than this is auto-cancelled
# with a synthetic failure reply instead of hanging indefinitely.
#
# Raised 60s -> 300s: 60s was killing real CAD-review HITL turns outright
# (auto-"Cancelled" mid-review) whenever a user actually paused to read a
# non-trivial approval card instead of clicking immediately — this is a
# dead-frontend backstop, not meant to rush a genuinely-present user.
_SUSPENDED_TURN_TIMEOUT_SEC = 300.0

# How often the sweep checks every connected session — cheap (a dict scan
# over at most a handful of local sessions), so this can run often without
# it costing anything meaningful.
_SUSPENSION_SWEEP_INTERVAL_SEC = 15.0

# Hard cap on "tool_dispatch_start"'s args_summary — a human-glance label
# for the ChatPanel's inline Agent Activity feed (see
# _send_tool_dispatch_start), NOT a payload dump. The full arguments already
# go out in that same event's own "arguments" field for the DAG-Monitor/
# HITL-facing consumers, so a single oversized value here (write_file's full
# "content", run_python_script's script text, ...) is hard-truncated rather
# than blowing up this lightweight status line.
_ARGS_SUMMARY_MAX_CHARS = 80

# Which argument best identifies a given tool_id's call at a glance, for
# args_summary — falls back to the first argument present (in insertion
# order) for any tool_id not listed here, so this never has to be updated
# in lockstep with every new tool added to react_dispatch.TOOL_HANDLERS.
_ARGS_SUMMARY_KEY: dict[str, str] = {
    "search_web": "query",
    "read_webpage": "url",
    "read_file": "path",
    "write_file": "path",
    "list_directory": "path",
    "run_python_script": "script_path",
    "update_core_memory": "section",
    "analyze_workspace_image": "query",
    "query_engineering_standard": "query",
    "load_capability": "domain",
}


def _summarize_tool_args(tool_id: str, arguments: dict[str, Any]) -> str:
    """One short, human-glance string for the ChatPanel's inline Agent
    Activity feed's ``tool_start`` event — never the full arguments dict.
    """
    key = _ARGS_SUMMARY_KEY.get(tool_id)
    value = arguments.get(key) if key else None
    if value is None and arguments:
        value = next(iter(arguments.values()), None)
    text = "" if value is None else str(value)
    if len(text) > _ARGS_SUMMARY_MAX_CHARS:
        text = text[: _ARGS_SUMMARY_MAX_CHARS - 1] + "…"
    return text


async def _execute_and_continue(
    websocket: WebSocket,
    session: dict[str, Any],
    messages: list[dict[str, Any]],
    loop_count: int,
    call: Any,
    tool_call_id: str,
    node_id: str,
    last_failure: tuple[str, str] | None = None,
    last_call: tuple[str, str, int] | None = None,
) -> None:
    """Runs an already-approved (or never-gated) tool call, streams its
    dispatch/DAG/mesh/camera events, appends its result back into
    ``messages`` as a ``tool`` role reply, then loops back into
    ``_run_react_loop`` for the NEXT iteration — this loop-back, not just
    the dispatch itself, is what lets the LLM actually see the result and
    decide whether to chain another tool call.

    This is also the ONE choke point every actual tool dispatch passes
    through regardless of path (immediate for a non-mutating tool, or
    post-approval for a HITL-gated one — see _run_react_loop/
    _resolve_react_hitl), so it's the single place to emit
    "tool_dispatch_end" (WebSocket Consolidation — see _send_tool_dispatch_end),
    read by BOTH the frontend's ChatPanel Agent Activity feed (status/
    args_summary-adjacent fields) and the CadPlugin's DAG Monitor
    (node_id/output/duration_ms) off the same one event, rather than two
    separate ones each consumer used to get.

    ``last_failure`` is ``(tool_id, error_message)`` from the PREVIOUS
    iteration's dispatch, if it failed — the deterministic backstop for the
    "insane retry loop" bug (a model blindly re-issuing the exact same
    failing non-mutating call, e.g. generate_2d_blueprint hitting a file-
    lock error, over and over until _MAX_REACT_ITERATIONS). A same-tool,
    same-message failure TWICE in a row stops the turn immediately with a
    clear explanation instead of looping a third time — this is a runtime
    guarantee, not just the system prompt's new "don't retry a repeated
    identical error" instruction, since a quirky local model can't be
    trusted to reliably follow that on its own (see this same session's
    earlier live findings on tool-name hallucination). Only threaded
    through the auto-dispatched (non-mutating) path — a HITL-gated mutating
    tool already has a human in the loop re-approving each retry, which is
    its own safety valve.

    ``last_call`` is ``(tool_id, args_signature, repeat_count)`` from the
    PREVIOUS iteration — a broader backstop than ``last_failure``: a live
    FreeCAD run had the local-Ollama fallback call
    ``load_capability({"domain": "freecad_full"})`` over 20 times in a row
    without ever acting on an unlocked tool. That call SUCCEEDS every time
    (``_tool_load_capability`` is stateless and always returns ``ok: True``
    for a valid domain), so ``last_failure`` — which only tracks identical
    FAILURES — never catches it; the loop would otherwise burn through every
    remaining iteration up to ``_MAX_REACT_ITERATIONS`` making zero progress.
    The same tool_id with the same arguments three times in a row (success
    or failure) stops the turn immediately instead.
    """
    # Ambient session_id for the whole synchronous dispatch_tool_call ->
    # _tool_* handler -> dana.plugins.freecad.engine.* call chain below —
    # see dana.session_context's own module docstring for why a
    # contextvar, not a new parameter threaded through ~40 handler/~20
    # engine function signatures, is what actually makes CAD workspace
    # isolation (one Session_Active.FCStd, one object-name registry per
    # chat session) work. This is the ONE production call site of
    # dispatch_tool_call (see this function's own docstring), so setting
    # it here covers every dispatch path — immediate and post-HITL-
    # approval alike.
    set_session_id(session["session_id"])
    engine = get_cad_engine()
    control_plane = get_control_plane()

    result = dispatch_tool_call(
        call,
        engine,
        control_plane,
        call_log=session.get("call_log"),
        api_keys=session.get("api_keys"),
        allowed_mounts=load_mounted_directories(),
    )

    if call.tool_id == "load_capability" and result.ok:
        # Autonomous semantic routing: the domain _tool_load_capability just
        # reported as unlocked (result.payload["domain"]) becomes visible to
        # every later turn THIS session via _effective_capabilities — the
        # very next next_react_turn()/build_system_prompt() call already
        # sees it, since both are computed fresh per turn, never cached
        # across turns (only the tool-schema-per-combination step is
        # @lru_cache'd, keyed on the resulting frozenset's value). Stamped
        # via _touch_capability_domains (not a permanent union) so it also
        # starts its P1 decay clock at THIS turn, not turn 0.
        domain = result.payload.get("domain")
        if isinstance(domain, str) and domain:
            _touch_capability_domains(session, frozenset({domain}))
    elif call.tool_id == "unload_capability" and result.ok:
        # Mirror of the load_capability branch above: immediate eviction
        # instead of stamping a fresh decay-clock turn. Only ever pops from
        # capability_unlocked_at_turn (the agent's own autonomously-loaded
        # domains) — session["active_plugins"] (the frontend's own tab
        # state) is a completely separate dict this never touches, same
        # boundary _effective_capabilities already draws.
        domain = result.payload.get("domain")
        if isinstance(domain, str) and domain:
            session.get("capability_unlocked_at_turn", {}).pop(domain, None)
    elif result.ok:
        # Any OTHER successful dispatch belonging to an already-unlocked
        # capability domain (e.g. list_directory while "os_tools" is
        # active) resets that domain's decay clock too — a domain the
        # agent is actively working with must never expire mid-task just
        # because the turns are ticking by; only genuine disuse decays.
        _touch_capability_domains(session, domains_for_tool_id(call.tool_id))

    if call.tool_id in ("create_plan", "mark_task_completed") and result.ok:
        # Closes the loop on Cost Tracking's sibling event: the MODEL just
        # mutated the Task Planner's global plan itself (as opposed to
        # _process_user_text's own structural override, a separate call
        # site) — push it to PlanChecklist right away rather than waiting
        # for this tool's own tool_result (which ChatPanel never renders).
        await _broadcast_plan_update(websocket, result.payload.get("plan"))

    if call.tool_id == "update_core_memory" and result.ok:
        # Core Memory UI: the MODEL just mutated the agent's persistent
        # memory via update_core_memory — push the updated state to
        # MemoryViewer right away rather than waiting for this tool's own
        # tool_result (which ChatPanel never renders). Uses the same
        # result.payload["memory"] shape write_core_memory already produces.
        await _broadcast_memory_update(websocket, result.payload.get("memory"))

    mesh_url = None
    if result.ok and call.tool_id in _CAD_CREATE_TOOLS:
        # target_object scopes both exports to the ONE object this call
        # actually produced — result.payload["name"] — rather than every
        # object in result.payload["path"]'s document. Necessary now that
        # path can be the shared Session_Active.FCStd (create_box/
        # create_cylinder/apply_boolean/modify_parameter/align_objects/
        # create_assembly_mate all write there): without a name, both
        # export_mesh_stl and export_model previously grabbed every
        # sibling object (or, pre-fix, the WRONG single object) in that
        # document, not just the one this tool call was about.
        result_name = result.payload.get("name")
        mesh = engine.export_mesh_stl(result.payload["path"], name=result_name, target_object=result_name)
        if mesh.get("ok"):
            token = _register_mesh(mesh["path"])
            mesh_url = f"/api/mesh/{token}.stl"
            artifacts_registry.register_artifact(
                mesh["path"], format="stl", source="generated", session_id=session["session_id"]
            )
        # Best-effort STEP sibling — rule 6 of _FREECAD_SYSTEM_PROMPT asks
        # the LLM to keep geometry recomputed for "the mesh pipeline"; this
        # is that pipeline's other half, run automatically instead of
        # waiting on the LLM to call export_freecad_model itself. Silently
        # skipped (never surfaced as a tool error) when unsupported — the
        # mock (trimesh) engine always reports ok=False here by design (no
        # B-rep/STEP writer; see MockFreeCADEngine.export_model), which is
        # an expected, honest limitation of that driver, not a failure of
        # this turn's actual tool call.
        try:
            step = engine.export_model(
                [result.payload["path"]], "step", result_name or "model", target_objects=[result_name]
            )
        except Exception:  # noqa: BLE001 — best-effort; a driver-level failure here must never fail the turn
            step = {"ok": False}
        if step.get("ok"):
            artifacts_registry.register_artifact(
                step["path"], format="step", source="generated", session_id=session["session_id"]
            )

        # Automatic Visual Verification — headless, no live R3F/Tauri canvas
        # needed (unlike take_canvas_screenshot, which requires that
        # frontend round-trip and so cannot be silently auto-triggered
        # mid-dispatch of an unrelated tool call). Screenshots the actual
        # FreeCAD GUI window (already on-screen via _auto_show) and reads it
        # back with a VLM, merged directly into THIS tool's own result
        # payload — the next turn's next_react_turn call sees it as part of
        # the same observation, no separate message/multimodal plumbing
        # needed. Uses verify_visual_operation (direct cloud VLM call via
        # OpenRouter), NOT analyze_cad_blueprint — this hook fires after
        # EVERY geometry tool call, and analyze_cad_blueprint's own
        # local-Ollama-first policy was the actual VRAM/latency cost this
        # was rewritten to eliminate; analyze_cad_blueprint itself is
        # untouched and still used by the separate, on-demand
        # verify_cad_rendering tool. Best-effort in both stages (capture,
        # then VLM read): a missing/failed screenshot or an unreachable
        # cloud VLM must never fail the geometry operation itself — same
        # "convenience miss, not a tool failure" philosophy
        # dana.plugins.freecad.engine._auto_show and
        # build_visual_inspection_result already use. Skipped entirely in
        # dry-run mode (tests, CI) — same flag every other OS/FreeCAD-touching
        # operation in this codebase already respects, so a test suite never
        # triggers a real OS screen capture or a live vision-model HTTP call.
        #
        # call.tool_id/result_name (the SAME object name already resolved
        # above for the export step) are passed through as the prompt's
        # tool/object context — a live E2E run had this hook's VLM
        # confidently describing a "CutResult" object that no longer even
        # existed in the document, because the old prompt asked a fully
        # generic "does this look successful?" question with nothing for
        # the model to actually check the screenshot against.
        try:
            if os.getenv("DANA_HEADLESS", "false").lower() == "true":
                # Headless mode: show_in_freecad_gui's own fix (always
                # terminate + relaunch the GUI fresh, so a stale document
                # can never taint a screenshot — see engine.py's
                # _terminate_freecad_gui) means every geometry-mutating
                # call now visibly closes/reopens the FreeCAD window. That
                # flashing is fine for a supervised interactive session but
                # unacceptable for an unattended/CI run — this skips the
                # GUI relaunch, screenshot, and VLM call entirely rather
                # than just suppressing the visible symptom, since
                # _auto_show (which triggers the relaunch) runs earlier,
                # inside each create/modify tool itself, not here.
                result.payload["visual_verification"] = "Visual verification skipped (Headless Mode active)."
            elif not is_dry_run_enabled():
                from dana.tools.cad_vision import capture_cad_viewport, verify_visual_operation

                capture = await asyncio.to_thread(capture_cad_viewport)
                if capture.get("ok") and capture.get("path"):
                    result.payload["screenshot_path"] = capture["path"]
                    result.payload["visual_verification"] = await asyncio.to_thread(
                        verify_visual_operation, capture["path"], call.tool_id, result_name or ""
                    )
        except Exception:  # noqa: BLE001 — best-effort; a vision-pipeline failure here must never fail the turn
            pass
    elif result.ok and call.tool_id == "export_freecad_model":
        # An explicit LLM-invoked export — same registry, so the Export
        # dropdown/Gradio "artifacts" endpoint see it alongside the
        # automatic entries above regardless of which path produced it.
        path = result.payload.get("path")
        if isinstance(path, str) and path:
            artifacts_registry.register_artifact(
                path,
                format=str(result.payload.get("format") or "").lower(),
                source="exported",
                session_id=session["session_id"],
            )
    elif result.ok and call.tool_id == "generate_urdf_assembly":
        # The .urdf IS the artifact here (unlike _CAD_CREATE_TOOLS, there's
        # no separate FreeCAD document to tessellate) — register it and
        # push it straight to the live viewer the same way, reusing the
        # mesh_url field/get_urdf route above; Viewer3D picks URDFLoader vs
        # STLLoader off the URL's own file extension.
        path = result.payload.get("path")
        if isinstance(path, str) and path:
            artifacts_registry.register_artifact(
                path, format="urdf", source="generated", session_id=session["session_id"]
            )
            token = _register_mesh(path)
            mesh_url = f"/api/mesh/{token}.urdf"
    elif result.ok and call.tool_id == "generate_3d_from_image":
        # Deliberately NOT in _CAD_CREATE_TOOLS: its payload key is
        # "mesh_path" (not "path"), and the file it names is already a raw
        # .obj/.glb mesh — not a FreeCAD .FCStd document, which is what
        # export_mesh_stl/export_model both require (they open their source
        # via FreeCAD's own App.openDocument, which only reads FreeCAD's
        # native document format). Same "it's already the artifact, just
        # register+serve it" pattern as generate_urdf_assembly right above.
        path = result.payload.get("mesh_path")
        if isinstance(path, str) and path:
            mesh_format = Path(path).suffix.lstrip(".").lower()
            artifacts_registry.register_artifact(
                path, format=mesh_format, source="generated", session_id=session["session_id"]
            )
            token = _register_mesh(path)
            mesh_url = f"/api/mesh/{token}.{mesh_format}"

    await _send_tool_dispatch_end(websocket, node_id, call, result, mesh_url)

    if call.tool_id == "manipulate_camera" and result.ok:
        await websocket.send_json(
            {"type": "camera_animate", "position": result.payload["position"], "target": result.payload["target"]}
        )

    messages.append(build_tool_result_message(tool_call_id, result))

    current_failure = None if result.ok else (call.tool_id, result.message)
    if current_failure is not None and current_failure == last_failure:
        # Same tool, same exact error message, two turns in a row — stop
        # here rather than trusting the model to notice and self-correct.
        await _finish_turn(
            websocket,
            session,
            messages,
            f"'{call.tool_id}' failed with the same error twice in a row, so I stopped instead of "
            f"retrying again: {result.message}",
        )
        return

    # Broader backstop than the failure check above: catches a call that
    # keeps SUCCEEDING with zero progress (see this function's own
    # docstring -- load_capability(domain="freecad_full") is exactly this
    # shape, always ok:True). Same tool_id + same arguments three times in
    # a row, success or failure, stops the turn instead of burning through
    # every remaining iteration up to _MAX_REACT_ITERATIONS.
    args_signature = json.dumps(call.arguments, sort_keys=True, default=str)
    current_call = (call.tool_id, args_signature)
    repeat_count = (
        last_call[2] + 1
        if last_call is not None and last_call[0] == current_call[0] and last_call[1] == current_call[1]
        else 1
    )
    if repeat_count >= 3:
        await _finish_turn(
            websocket,
            session,
            messages,
            f"Called '{call.tool_id}' with the same arguments {repeat_count} times in a row without "
            "making progress, so I stopped instead of continuing.",
        )
        return

    if not result.ok:
        # A blunt, turn-specific reinforcement on top of the standing
        # "SELF-CORRECT ON ERROR" system-prompt rule (dana.core.react_dispatch's
        # _CORE_SYSTEM_PROMPT/_FREECAD_SYSTEM_PROMPT) — this fires only on an
        # ACTUAL failure the loop is about to hand back to the model (never on
        # the repeated-failure branch above, which already ends the turn
        # instead of asking the model to act on anything), so the very next
        # next_react_turn call opens with an unmissable directive instead of
        # relying solely on a general rule stated once, several messages back.
        messages.append(
            {
                "role": "system",
                "content": (
                    "SYSTEM OVERRIDE: The last tool call failed. You must fix the error "
                    "and retry. Do not move on to the next step."
                ),
            }
        )
    else:
        # Success path reinforcement, now tied to structural planning
        messages.append(
            {
                "role": "system",
                "content": (
                    "SYSTEM OVERRIDE: The last tool call succeeded. Check your active plan (if created) "
                    "or the original request. If there are pending steps remaining, output the EXACT next "
                    "JSON tool call immediately. Do NOT stop or output conversational text until all steps are finished."
                ),
            }
        )

    print(
        f"[ReAct] Continuing loop after '{call.tool_id}' -> iteration {loop_count + 1}",
        file=sys.stderr,
        flush=True,
    )
    await _run_react_loop(
        websocket,
        session,
        messages,
        loop_count + 1,
        last_failure=current_failure,
        last_call=(current_call[0], current_call[1], repeat_count),
    )


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    """The ORIGINAL user utterance for this turn, recovered from an
    OpenAI-wire ``messages`` list (this turn's own, or a suspended
    react_state/visual_state's stashed copy) — shared by ``_finish_turn``'s
    persistence hook and ``_run_react_loop``'s own ``raw_text`` (used by
    ``next_react_turn``'s selection-reference fallback).
    """
    return next((extract_user_text(m["content"]) for m in messages if m.get("role") == "user"), "")


def _persist_turn(session: dict[str, Any], user_text: str, assistant_text: str) -> None:
    """Appends one user/assistant exchange onto this session's on-disk chat
    history (dana.api.sessions) — Local Chat Session Persistence's
    auto-save hook. Deriving the sidebar title from the FIRST user message
    only happens once per session (``session["session_title"]`` starts
    ``None`` for a genuinely new session, hydrated from disk otherwise —
    see ``ws_chat``); every later turn reuses whatever title was already
    set, on this connection or a prior one for the same session_id.
    """
    history: list[dict[str, str]] = session.setdefault("chat_history", [])
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    if session.get("session_title") is None:
        session["session_title"] = derive_title(user_text)
    record = save_session(
        session["session_id"],
        title=session["session_title"],
        created_at=session.get("session_created_at"),
        messages=history,
    )
    # Pins the ORIGINAL created_at from here on — save_session only stamps
    # a fresh "now" when it receives None, so this must be captured back
    # onto the session after a session's very first save.
    session["session_created_at"] = record["created_at"]


async def _finish_turn(websocket: WebSocket, session: dict[str, Any], messages: list[dict[str, Any]], content: str) -> None:
    """Sends the turn-ending "assistant_message" AND persists the exchange
    to disk — the single choke point every terminal branch of the ReAct
    loop (final text, model error, max-iterations, HITL cancel, abort)
    funnels through, so on-disk history can never desync from what the
    user actually saw for a completed turn. Skips persistence (but still
    sends the reply) if ``messages`` carries no user turn at all — should
    not happen for a real turn, but a defensive no-op is cheaper than a
    KeyError reaching a client-facing coroutine.
    """
    await websocket.send_json({"type": "assistant_message", "content": content})
    user_text = _first_user_text(messages)
    if user_text:
        _persist_turn(session, user_text, content)
        # Pillar 3 (dana.core.context_distiller) — fire-and-forget local-GPU
        # summary update; never awaited, so a slow/unreachable local model
        # can't delay a reply the user has already been sent above.
        schedule_distillation(session, user_text, content)


async def _sweep_stale_suspensions() -> None:
    """Background loop (started in ``_lifespan``, for the life of the
    process) — P3's wall-clock enforcement for a suspended
    ``react_state``/``visual_state`` that the frontend never replied to.
    Auto-cancels anything older than ``_SUSPENDED_TURN_TIMEOUT_SEC`` with
    the same "Cancelled"-shaped ``_finish_turn`` reply an explicit HITL
    rejection already produces, so the model sees a clean, actionable turn
    ending rather than the session staying silently parked forever.

    Runs for every currently-connected session (``_active_sessions``) each
    tick — cheap even at a short interval, since a local single-user app
    only ever has a handful of open connections. A session whose websocket
    has already dropped (send raises) is skipped rather than crashing the
    sweep loop; ``ws_chat``'s own ``finally`` block is what actually reaps
    ``_active_sessions`` for a closed connection.
    """
    while True:
        await asyncio.sleep(_SUSPENSION_SWEEP_INTERVAL_SEC)
        now = time.monotonic()
        for websocket, session in list(_active_sessions.items()):
            for state_key in ("react_state", "visual_state"):
                state = session.get(state_key)
                if state is None:
                    continue
                if now - state["created_at"] < _SUSPENDED_TURN_TIMEOUT_SEC:
                    continue
                session[state_key] = None
                try:
                    await _finish_turn(
                        websocket,
                        session,
                        state["messages"],
                        f"This action timed out waiting for a response (over "
                        f"{_SUSPENDED_TURN_TIMEOUT_SEC:.0f}s) and was cancelled automatically.",
                    )
                except Exception:  # noqa: BLE001 — a dead/closing socket must never crash the sweep loop
                    pass


# Substrings _acknowledges_failure checks a proposed "final" answer for,
# case-insensitively, when the last tool actually dispatched failed —
# deliberately plain/generic (not tied to any one tool's error wording),
# since the point is only to tell "the model is aware something went
# wrong" from "the model is confidently reporting success that never
# happened", not to grade the quality of its explanation.
_FAILURE_ACKNOWLEDGEMENT_MARKERS: tuple[str, ...] = ("failed", "error", "cannot", "unable")


def _acknowledges_failure(content: str) -> bool:
    """True if ``content`` (a model's proposed final answer) contains any of
    ``_FAILURE_ACKNOWLEDGEMENT_MARKERS`` — used by _run_react_loop's
    verification gate to tell whether a final answer, offered right after
    the last dispatched tool call failed, actually admits that rather than
    silently reporting success."""
    lowered = (content or "").lower()
    return any(marker in lowered for marker in _FAILURE_ACKNOWLEDGEMENT_MARKERS)


async def _run_react_loop(
    websocket: WebSocket,
    session: dict[str, Any],
    messages: list[dict[str, Any]],
    loop_count: int,
    last_failure: tuple[str, str] | None = None,
    last_call: tuple[str, str, int] | None = None,
) -> None:
    """The multi-step ReAct loop for one user turn: ask the LLM what to do
    next given the running ``messages`` history, either finish with plain
    text or dispatch (HITL-gating mutating tools) one tool call and loop
    back with its result — up to ``_MAX_REACT_ITERATIONS`` times.

    A mutating tool call SUSPENDS the loop here: it stashes everything
    needed to resume (``messages``, ``loop_count``, the pending call) on
    ``session["react_state"]`` and returns, letting ``ws_chat``'s own
    receive loop pick the next inbound message back up — normal control
    flow for a coroutine, not a real generator suspend, but the effect is
    the same: this turn's reasoning picks up exactly where it left off once
    ``_resolve_react_hitl`` sees the matching ``hitl_response``.

    Global Abort: ``session["abort_requested"]`` is checked here (top of
    the loop, before asking the LLM anything) AND again right after
    ``next_react_turn`` returns (before acting on whatever it decided) —
    the two points in this coroutine's own control flow that best mirror
    "top of the while loop" and "immediately after an await point" for a
    recursive-async-call loop rather than a literal ``while``. Either check
    firing consumes the flag (resets it to ``False``) and ends the turn
    with a "Generation aborted by user." reply instead of continuing.
    A tool already dispatched by ``_execute_and_continue`` is never
    interrupted mid-call (there is no hook to do that safely without OS-
    level thread killing) — its result is simply never acted on, since the
    very next thing that coroutine does is recurse back into THIS
    function, where the abort is caught before any further reasoning.
    """
    if session.get("abort_requested"):
        session["abort_requested"] = False
        await _finish_turn(websocket, session, messages, "Generation aborted by user.")
        return

    if loop_count >= _MAX_REACT_ITERATIONS:
        await _finish_turn(
            websocket,
            session,
            messages,
            f"Reached the maximum number of reasoning steps ({_MAX_REACT_ITERATIONS}) — stopping here.",
        )
        return

    node_id = f"parse-{loop_count}"
    await _dag_start(websocket, node_id, "Parse intent", "agent", {"step": loop_count})
    parse_start = time.perf_counter()
    raw_text = _first_user_text(messages)
    turn = await next_react_turn(
        messages,
        session.get("active_selection"),
        raw_text=raw_text,
        api_keys=session.get("api_keys"),
        active_plugins=_effective_capabilities(session),
    )
    parse_ms = int((time.perf_counter() - parse_start) * 1000)

    if turn.usage_info:
        await _broadcast_usage_update(websocket, session, turn.usage_info)

    if session.get("abort_requested"):
        session["abort_requested"] = False
        await _dag_complete(websocket, node_id, "error", {"aborted": True}, parse_ms)
        await _finish_turn(websocket, session, messages, "Generation aborted by user.")
        return

    if turn.kind == "error":
        await _dag_complete(websocket, node_id, "error", {"matched": False}, parse_ms)
        # dana.core.openai_tool_bridge already catches urllib.error.HTTPError
        # (502s, 400s, any status) and urllib.error.URLError (stalled/refused
        # connections), so a proxy/network failure never crashes this process
        # — it surfaces here as next_react_turn's ReactTurn("error", content=
        # the specific failure, e.g. "cloud HTTP 402: Payment Required -- ...",
        # or, once ModelProvider's Ollama fallback also fails, a combined
        # "cloud AND local fallback both failed" message). That detail is
        # exactly str(exc) from whatever HTTP/provider client raised it —
        # provider-internal wording (status codes, endpoint URLs, sometimes
        # response bodies) that has no business reaching the chat bubble, so
        # it's logged server-side for whoever's debugging instead of handed
        # to _finish_turn, which both replies to the user AND feeds
        # dana.core.context_distiller's working-memory summary that every
        # later turn's system prompt is built from (`messages` itself is NOT
        # the right place for it: a fresh `messages` list is built from
        # scratch for every new user turn — see _handle_user_message above —
        # so anything appended to THIS turn's list is discarded the moment
        # this function returns).
        error_detail = turn.content or "no further detail available"
        print(f"[react-loop] model call failed: {error_detail}", file=sys.stderr, flush=True)
        reply = "I ran into a problem talking to the model — please try again."
        await _finish_turn(websocket, session, messages, reply)
        return

    if turn.kind == "final":
        content = turn.content

        # Guard against raw tool regurgitation (JSON mimicry or leaked tokens)
        if content and ("<tool_response>" in content or content.strip().startswith('{"status": "error"')):
            content = (
                "I successfully completed the earlier steps, but I encountered an error "
                "trying to generate the final geometry and got stuck."
            )
        elif not content:
            content = (
                "I didn't think that needed a tool call — try asking for a specific "
                "action (e.g. \"refactor foo.py to use snake_case\" or \"build a box 60x40x20\")."
            )

        if last_failure is not None and not _acknowledges_failure(content):
            # Verification gate against hallucinated success: the LAST tool
            # actually dispatched failed (last_failure is threaded through
            # every _run_react_loop/_execute_and_continue recursion — see
            # _execute_and_continue's own docstring), and the model's
            # proposed final answer doesn't even mention that in plain
            # language. A live E2E run hit exactly this: import_and_
            # solidify_mesh/perform_freecad_boolean were never successfully
            # called, yet the model's "final" answer confidently reported a
            # fabricated "fused result" bounding box. Rejecting the
            # termination and bouncing back into the loop (bounded by the
            # existing _MAX_REACT_ITERATIONS check at the top of this
            # function — this never loops forever) forces the model to
            # either retry, try something else, or actually say it failed.
            await _dag_complete(websocket, node_id, "error", {"final": True, "rejected": "unacknowledged_failure"}, parse_ms)
            failed_tool_id, failure_message = last_failure
            print(
                f"[ReAct] Rejecting unacknowledged-failure termination after '{failed_tool_id}' "
                f"failed ({failure_message!r}) -> iteration {loop_count + 1}",
                file=sys.stderr,
                flush=True,
            )
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "SYSTEM OVERRIDE: Your last tool action failed. You cannot declare the "
                        "task successful. You must either retry the action, try an alternative "
                        "tool, or explicitly explain the failure to the user."
                    ),
                }
            )
            await _run_react_loop(
                websocket, session, messages, loop_count + 1, last_failure=last_failure, last_call=last_call
            )
            return

        await _dag_complete(websocket, node_id, "success", {"final": True}, parse_ms)
        messages.append({"role": "assistant", "content": content})
        await _finish_turn(websocket, session, messages, content)
        await _speak_reply(websocket, content)
        return

    # turn.kind == "tool_call"
    await _dag_complete(websocket, node_id, "success", {"tool_id": turn.call.tool_id}, parse_ms)
    call = turn.call
    assistant_message, tool_call_id = build_assistant_tool_call_message(call)
    messages.append(assistant_message)

    if is_visual_inspection_tool(call.tool_id):
        # Needs an external actor (the R3F canvas, live in the Tauri
        # frontend) to produce anything at all — suspends the loop exactly
        # like a mutating tool suspends for HITL approval, but on its own
        # `visual_state` slot (never dispatch_tool_call'd directly; see
        # dana.core.react_dispatch._tool_take_canvas_screenshot), resolved
        # by _resolve_visual_capture once the frontend's screenshot arrives.
        request_id = uuid.uuid4().hex
        session["visual_state"] = {
            "messages": messages,
            "loop_count": loop_count,
            "call": call,
            "tool_call_id": tool_call_id,
            "request_id": request_id,
            # P3 — the wall-clock deadline _sweep_stale_suspensions checks
            # against; time.monotonic() rather than time.time() since this
            # is a duration measurement, immune to a system clock change
            # mid-suspend.
            "created_at": time.monotonic(),
        }
        await websocket.send_json({"type": "visual_capture_request", "payload": {"request_id": request_id}})
        return

    # Session HITL allowlist: a tool_id the user already approved once this
    # session (see _resolve_react_hitl, which populates it on approval)
    # skips the prompt for every subsequent call to that SAME tool_id — the
    # gate is still per tool_id, not "approve everything from now on".
    # _HITL_ALWAYS_APPROVED_TOOLS is the same idea, permanently, for a small
    # explicit set of narrow geometry-CRUD tools (see its own comment for
    # why execute_freecad_script/modify_existing_freecad_document are
    # deliberately never in it).
    if (
        is_mutating_tool(call.tool_id)
        and call.tool_id not in _HITL_ALWAYS_APPROVED_TOOLS
        and call.tool_id not in session.get("hitl_approved_tools", set())
    ):
        print(
            f"[ReAct] '{call.tool_id}' is mutating -> suspending loop for HITL approval "
            "(app.py's _GradioSocket auto-approves this immediately on the HF Space path)",
            file=sys.stderr,
            flush=True,
        )
        request_id = uuid.uuid4().hex
        session["react_state"] = {
            "messages": messages,
            "loop_count": loop_count,
            "call": call,
            "tool_call_id": tool_call_id,
            "request_id": request_id,
            "created_at": time.monotonic(),  # P3 — see the matching visual_state comment above
        }
        await websocket.send_json(
            {
                "type": "hitl_approval_required",
                "payload": {
                    "request_id": request_id,
                    "action_name": call.tool_id,
                    "description": describe_tool_call(call),
                    "parameters": call.arguments,
                },
            }
        )
        return

    # Non-mutating: "tool_dispatch_start" fires only now, right as execution
    # actually begins — same contract a mutating tool gets below, post-
    # approval, in _resolve_react_hitl (never pre-approval — a client must
    # not see a dispatch-start event for something that hasn't run yet).
    dispatch_node_id = f"dispatch-{loop_count}"
    await _send_tool_dispatch_start(websocket, dispatch_node_id, call)
    await _execute_and_continue(
        websocket,
        session,
        messages,
        loop_count,
        call,
        tool_call_id,
        dispatch_node_id,
        last_failure=last_failure,
        last_call=last_call,
    )


async def _resolve_react_hitl(websocket: WebSocket, session: dict[str, Any], response: dict[str, Any]) -> None:
    state = session.get("react_state")
    if state is None or response.get("request_id") != state["request_id"]:
        return  # stale or unknown request_id — ignore
    session["react_state"] = None

    call = state["call"]
    if not response.get("approved"):
        # Never dispatched — no "tool_call"/dag node was ever opened for a
        # call awaiting approval, so there's nothing to dag-complete either.
        print(f"[ReAct] HITL request for '{call.tool_id}' was rejected — turn ends here", file=sys.stderr, flush=True)
        await _finish_turn(websocket, session, state["messages"], "Cancelled — no changes were made.")
        return

    print(f"[ReAct] HITL request for '{call.tool_id}' approved -> dispatching", file=sys.stderr, flush=True)
    # Remembered for the rest of this session only (session dict, never
    # persisted) — the next call to THIS SAME tool_id skips the approval
    # prompt entirely (see the is_mutating_tool check in _run_react_loop).
    # A "Modify" approval (an edited-parameters override) still counts as
    # approving the tool itself, not just this one call's specific args.
    session.setdefault("hitl_approved_tools", set()).add(call.tool_id)

    override = response.get("parameters")
    if isinstance(override, dict):
        call.arguments.update(override)

    loop_count = state["loop_count"]
    dispatch_node_id = f"dispatch-{loop_count}"
    await _send_tool_dispatch_start(websocket, dispatch_node_id, call)

    await _execute_and_continue(
        websocket, session, state["messages"], loop_count, call, state["tool_call_id"], dispatch_node_id
    )


async def _resolve_visual_capture(websocket: WebSocket, session: dict[str, Any], response: dict[str, Any]) -> None:
    """Resolves a suspended take_canvas_screenshot call once the frontend's
    ``visual_capture_response`` arrives — the visual-inspection counterpart
    to ``_resolve_react_hitl``, on its own ``visual_state`` slot rather than
    ``react_state`` since this isn't a human approval, it's an external data
    dependency (the R3F canvas render) the ReAct loop had to wait on.
    """
    state = session.get("visual_state")
    if state is None or response.get("request_id") != state["request_id"]:
        return  # stale or unknown request_id — ignore
    session["visual_state"] = None

    call = state["call"]
    api_key = (session.get("api_keys") or {}).get("openai")
    payload = build_visual_inspection_result(response.get("image_b64"), error=response.get("error"), api_key=api_key)
    message = str(payload.get("summary") or payload.get("note") or payload.get("error") or "ok")
    result = ToolResult(call.tool_id, bool(payload.get("ok")), payload, message, 0)

    dispatch_node_id = f"dispatch-{state['loop_count']}"
    await _send_tool_dispatch_start(websocket, dispatch_node_id, call)
    # The client already has the image bytes it just sent — no need to blast
    # the full base64 back over the wire a second time (same trim the old
    # standalone "tool_result" send already applied here).
    trimmed_result = ToolResult(
        result.tool_id,
        result.ok,
        {k: v for k, v in payload.items() if k != "image_b64"},
        result.message,
        result.duration_ms,
    )
    await _send_tool_dispatch_end(websocket, dispatch_node_id, call, trimmed_result, None)

    messages = state["messages"]
    messages.append(build_tool_result_message(state["tool_call_id"], result))
    await _run_react_loop(websocket, session, messages, state["loop_count"] + 1)


def _capture_desktop_context_data_uri() -> str | None:
    """Implicit Screen Awareness: reuses the exact same capture routine the
    HITL-gated ``analyze_desktop_screen`` tool call uses
    (``dana.plugins.os.desktop_vision``), just invoked directly and BEFORE
    the ReAct loop starts, instead of as a dispatched tool call mid-loop.

    This is a deliberately different consent model from that tool's own
    HITL gate (``dana.core.react_dispatch.is_mutating_tool``), not a
    bypass of it: ``analyze_desktop_screen``
    stays HITL-gated for when the AGENT decides on its own initiative to
    look at the screen. Here it's the FRONTEND toggle (ChatPanel's Screen
    Awareness button — see ``ws_chat``'s ``include_desktop_context``
    handling) that is the user's own explicit, continuously-visible choice
    to attach a screenshot to every message while it's on; that standing
    opt-in is the consent, so no per-message approval click is raised for
    it. Returns ``None`` (never raises) on any capture failure, so a
    screen-capture error degrades to "send the text with no screenshot"
    rather than failing the whole turn.
    """
    try:
        image_b64 = _capture_primary_monitor_jpeg_b64()
    except Exception:  # noqa: BLE001 — capture failure must never crash the turn
        return None
    return f"data:image/jpeg;base64,{image_b64}"


# Structural Task-Planner forcing: a live stress test (2026-08-30, 7B local
# model) confirmed the system prompt's "call create_plan first for a
# multi-step request" rule (dana.core.react_dispatch._FREECAD_SYSTEM_PROMPT
# Rule #1) is simply ignored in practice — the model went straight to
# geometry tool calls every time. Rather than lean further on prompt
# wording, a genuinely multi-step request gets its plan created HERE, in
# plain Python, before the model is ever asked anything — so the "## Current
# Active Plan" block (dana.core.react_dispatch._format_active_plan_for_prompt)
# is already populated on this very first turn regardless of whether the
# model would have called create_plan itself.
_MULTI_STEP_SPLIT_RE = re.compile(
    r"\s*(?:,?\s+and\s+then\s+|,?\s+then\s+|;\s*then\s+|,\s+and\s+|;\s+)\s*",
    re.IGNORECASE,
)


def _looks_multi_step(user_text: str) -> bool:
    """Cheap keyword heuristic, not NLP. A false positive just pre-creates a
    harmless single-task plan (get_active_plan-gated, see the call site
    below, so it never clobbers a plan already in progress); a false
    negative just leaves the model's own Rule #1 as the only defense —
    exactly the status quo before this existed."""
    lowered = f" {(user_text or '').lower()} "
    return any(
        marker in lowered
        for marker in (" then ", " and then ", ", then", "; then", " after that ")
    )


def _split_into_steps(user_text: str) -> list[str]:
    """Naive clause split on the same connectors _looks_multi_step scans
    for — good enough to give create_plan distinct, ordered task strings
    without needing a real NLP pass; falls back to the whole utterance as
    one task if the split doesn't actually yield 2+ pieces."""
    parts = [p.strip(" .") for p in _MULTI_STEP_SPLIT_RE.split(user_text.strip()) if p.strip(" .")]
    return parts if len(parts) >= 2 else [user_text.strip()]


async def _process_user_text(
    websocket: WebSocket, session: dict[str, Any], user_text: str, attachments: list[str] | None = None
) -> None:
    """Starts a fresh multi-step ReAct loop for one chat utterance.

    Used both for a client's own typed messages and for a finalized
    ``VoiceService`` transcript replayed across every connected session —
    same loop, same DAG/HITL events, regardless of the source. Bounces
    with a nudge instead of starting a second loop if one is already
    mid-flight (suspended on a pending HITL approval) for this session —
    starting a new ``messages`` history here would silently orphan that
    pending approval and desync session["react_state"].

    ``attachments`` (BYOK-free, no session state needed) are the data URIs
    ``ws_chat`` pulled off this turn's ``chat_message`` payload — see
    ``build_user_message`` for how they turn into the OpenAI-wire multimodal
    content array the LLM actually sees. ``None``/absent for a voice-relayed
    transcript, which never carries an attachment of its own.
    """
    if session.get("react_state") is not None or session.get("visual_state") is not None:
        await websocket.send_json(
            {
                "type": "assistant_message",
                "content": "Please wait for the pending action above to resolve before sending another message.",
            }
        )
        return
    # A genuinely NEW turn always starts with a clean abort flag — guards
    # against an "abort_turn" that arrived just after the PREVIOUS turn had
    # already finished (nothing left in flight for it to cancel) from
    # bleeding forward and wrongly cancelling this unrelated one.
    session["abort_requested"] = False
    # P1's decay clock — one tick per NEW user turn, not per ReAct-loop
    # iteration within it (see the "turn_counter" session field's own
    # comment above).
    session["turn_counter"] = session.get("turn_counter", 0) + 1

    # Structural planner forcing (see _looks_multi_step above) — only ever
    # auto-creates a plan into a genuinely IDLE planner slot; a plan already
    # active from an earlier turn in this same session is never clobbered
    # by a later turn's heuristic guess.
    if _looks_multi_step(user_text) and not _tb_get_active_plan().get("tasks"):
        plan_result = _tb_create_plan(objective=user_text, tasks=_split_into_steps(user_text))
        if plan_result.get("ok"):
            await _broadcast_plan_update(websocket, plan_result.get("plan"))

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": build_system_prompt(
                session.get("active_selection"),
                active_plugins=_effective_capabilities(session),
                mounted_directories=load_mounted_directories(),
                working_memory=(session.get("working_memory") or {}).get("summary", ""),
            ),
        },
        build_user_message(user_text, attachments),
    ]
    await _run_react_loop(websocket, session, messages, 0)


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket, session_id: str | None = None) -> None:
    """``?session_id=`` hydrates this connection from a previously-saved
    chat (dana.api.sessions) if one exists — resuming a conversation the
    user picked from the frontend's ChatSidebar. Omitted, or an invalid/
    unrecognized id, just generates a fresh one instead (a brand-new chat,
    or a "New Chat" click, which the frontend deliberately connects with no
    session_id at all) — never an error, since a stale/deleted id must not
    fail the whole connection.
    """
    await websocket.accept()
    resolved_session_id = session_id if (session_id and is_valid_session_id(session_id)) else new_session_id()
    stored = load_session(resolved_session_id)
    await websocket.send_json(
        {
            "type": "ready",
            "session_id": resolved_session_id,
            "driver_state": driver_state(),
            "plugins": plugin_registry_view(),
            # PlanChecklist's initial seed: the Task Planner is a single
            # GLOBAL plan (dana.plugins.planning.task_board), not per-session
            # — a reconnect/page-refresh mid-plan must see it immediately
            # here, not wait for the next create_plan/mark_task_completed
            # mutation's own "plan_update" broadcast (_broadcast_plan_update).
            "active_plan": _tb_get_active_plan(),
            # MemoryViewer's initial seed: Core Memory is a single GLOBAL
            # store (dana.plugins.memory.core_memory), not per-session —
            # a reconnect/page-refresh must see the current state immediately
            # here, not wait for the next update_core_memory mutation's own
            # "memory_update" broadcast (_broadcast_memory_update).
            "core_memory": read_core_memory(),
        }
    )
    session: dict[str, Any] = {
        "active_selection": None,
        "react_state": None,
        "visual_state": None,
        "call_log": CadCallLog(),
        # Local Chat Session Persistence (dana.api.sessions) — this
        # connection's on-disk identity, its already-saved transcript (if
        # resuming), and the bookkeeping _persist_turn needs to keep saving
        # to the SAME file/title/created_at as this session accumulates
        # more turns. "session_title"/"session_created_at" start None for a
        # genuinely new session — _persist_turn derives+stamps them once,
        # on this session's very first completed turn.
        "session_id": resolved_session_id,
        "chat_history": list(stored["messages"]) if stored else [],
        "session_title": stored["title"] if stored else None,
        "session_created_at": stored["created_at"] if stored else None,
        # BYOK — {"openai": "...", "anthropic": "..."}, populated by
        # "update_secrets" below (sent by the frontend's SecretsContext on
        # connect and on every change). Never logged; see next_react_turn/
        # build_visual_inspection_result for where this actually gets used.
        "api_keys": {},
        # Capability routing — frozenset of capability-domain names (e.g.
        # {"freecad"}), populated by "update_context" below (sent by the
        # frontend on connect and whenever the active plugin tab changes).
        # Starts EXPLICITLY empty, not None — see build_system_prompt/
        # _tool_ids_for_plugins for why that distinction matters (empty set
        # = "plugin-aware session, nothing active yet" -> lean core prompt +
        # tools; None = "caller doesn't participate in this feature at all"
        # -> full legacy CAD-everything behavior, for un-migrated callers).
        "active_plugins": frozenset(),
        # Autonomous semantic routing WITH decay (P1) — capability domains
        # the AGENT itself unlocked this session via the load_capability
        # tool, each stamped with the turn_counter value it was last
        # unlocked/used at (distinct from active_plugins, which only
        # "update_context" ever writes). See _effective_capabilities for
        # how this decays back out of the tool schema after
        # _CAPABILITY_DECAY_TURNS turns of disuse, and
        # _execute_and_continue/_touch_capability_domains for where this
        # actually gets mutated. "update_context" unloading a plugin must
        # never clear this.
        "capability_unlocked_at_turn": {},
        # Pillar 3 (dana.core.context_distiller) — rolling, local-model-
        # distilled summary of prior turns in THIS session. Populated by
        # schedule_distillation (called from _finish_turn) after each turn
        # completes; read by _process_user_text into the NEXT turn's system
        # prompt. Starts empty — build_system_prompt's empty-state
        # convention (same as core memory/the active plan) omits the
        # section entirely until there's something to show.
        "working_memory": {"summary": "", "turn": 0},
        # Incremented once per NEW user turn (_process_user_text) — the
        # clock _effective_capabilities' decay logic measures "turns of
        # disuse" against. Deliberately NOT incremented per ReAct-loop
        # iteration (_MAX_REACT_ITERATIONS/loop_count) — a capability used
        # several times within ONE turn's tool-chaining should only cost
        # one tick of staleness, not several.
        "turn_counter": 0,
        # Global Abort — set True by the "abort_turn" handler below while a
        # turn is actively iterating (no HITL/visual suspension currently
        # pending); consumed (and reset to False) by _run_react_loop's own
        # checks. Reset to False at the START of every fresh turn in
        # _process_user_text too, so a stale True left over from an
        # abort_turn that arrived just after its turn already finished
        # can never bleed into wrongly cancelling the NEXT, unrelated turn.
        "abort_requested": False,
        # Session-scoped HITL allowlist — tool_ids the user has already
        # approved once this session (see _run_react_loop's is_mutating_tool
        # check and _resolve_react_hitl, which adds to this on approval).
        # In-memory only, never persisted to disk/chat_history — a fresh
        # connection (even resuming the same on-disk chat) starts empty, so
        # this never silently skips approval on a brand-new session.
        "hitl_approved_tools": set(),
    }
    _active_sessions[websocket] = session
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "canvas_selection":
                session["active_selection"] = data.get("payload") or {}
                continue

            if msg_type == "update_context":
                # Frontend plugin activation/deactivation — see
                # dana.core.react_dispatch's capability routing
                # (_tool_ids_for_plugins/build_system_prompt).
                session["active_plugins"] = _normalize_active_plugins(data.get("active_plugins"))
                continue

            if msg_type == "update_secrets":
                # Deliberately not logged — do not add a print()/log() here.
                keys = data.get("keys")
                if isinstance(keys, dict):
                    session["api_keys"] = {
                        k: v for k, v in keys.items() if isinstance(k, str) and isinstance(v, str) and v
                    }
                continue

            if msg_type == "voice_control":
                # AssistiveOrb click/hotkey — see VoiceService's push-to-talk
                # docstring. "listen" no-ops if a cycle is already in
                # flight; "cancel" no-ops unless currently listening.
                if _voice_service is not None:
                    action = data.get("action")
                    if action == "listen":
                        _voice_service.request_listen()
                    elif action == "cancel":
                        _voice_service.cancel()
                continue

            if msg_type == "audio_playback_complete":
                # The client's <audio> element finished playing the last
                # assistant_audio clip (see _speak_reply) — close out the
                # "speaking" state and re-arm VoiceService for the next
                # request_listen(). Harmless to call even for a turn that
                # was never voice-triggered in the first place.
                if _voice_service is not None:
                    _voice_service.finish_turn()
                await _broadcast({"type": "voice_state", "state": "idle", "transcript": ""})
                continue

            if msg_type == "abort_turn":
                # Global Abort (the "Stop Generating" button). A turn
                # SUSPENDED on a pending HITL/visual-capture approval has no
                # active loop iteration left to catch a flag on its own —
                # cancel it directly, right here, so Stop works instantly
                # regardless of whether the loop is actively iterating or
                # paused waiting on the user/frontend for something else.
                # Exactly one of these two branches ever fires per click, so
                # exactly one "Generation aborted by user." message is ever
                # sent for it (either now, or later from _run_react_loop's
                # own check once an actively-iterating turn reaches it).
                pending = session.get("react_state") or session.get("visual_state")
                if pending is not None:
                    session["react_state"] = None
                    session["visual_state"] = None
                    session["abort_requested"] = False
                    await _finish_turn(websocket, session, pending["messages"], "Generation aborted by user.")
                else:
                    session["abort_requested"] = True
                continue

            if msg_type == "hitl_response":
                payload = data.get("payload") or {}
                await _resolve_react_hitl(websocket, session, payload)
                continue

            if msg_type == "visual_capture_response":
                payload = data.get("payload") or {}
                await _resolve_visual_capture(websocket, session, payload)
                continue

            if msg_type == "export_python_script":
                call_log = session.get("call_log")
                if not call_log:
                    await websocket.send_json(
                        {
                            "type": "assistant_message",
                            "content": "Nothing to export yet — no FreeCAD tool calls have run this session.",
                        }
                    )
                    continue
                path = write_macro_script(call_log, filename=str(data.get("filename") or "dana_session_macro"))
                await websocket.send_json({"type": "python_script_exported", "path": path})
                continue

            user_text = str(data.get("text") or "").strip()
            if not user_text:
                continue

            raw_attachments = data.get("attachments")
            attachments = [a for a in raw_attachments if isinstance(a, str)] if isinstance(raw_attachments, list) else None

            if data.get("include_desktop_context") is True:
                desktop_capture = _capture_desktop_context_data_uri()
                if desktop_capture is not None:
                    attachments = [*(attachments or []), desktop_capture]

            await _process_user_text(websocket, session, user_text, attachments=attachments)
    except WebSocketDisconnect:
        pass
    finally:
        _active_sessions.pop(websocket, None)


# Serve the built React/Tauri web bundle, if present. Mounted last so it
# never shadows the /api or /ws routes above. Absent in a bare
# `python -m dana.api.server` dev loop where only the API is being exercised.
#
# On a HF Space, app.py mounts Gradio at "/" itself (HF's readiness probe for
# `sdk: gradio` Spaces polls GET /config at the Space root, which only exists
# if Gradio owns "/") — so the frontend moves to /ui there instead, to avoid
# both mounts competing for "/". Everywhere else (Tauri's bundled build,
# a standalone Docker image) nothing polls /config, so it keeps serving at
# the root as before.
_FRONTEND_MOUNT_PATH = "/ui" if IS_HF_SPACE else "/"
if _FRONTEND_DIST.is_dir():
    app.mount(_FRONTEND_MOUNT_PATH, StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")


__all__ = ("app",)

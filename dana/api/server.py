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
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dana.api.memory import router as _memory_router
from dana.api.planner import router as _planner_router
from dana.api.services import router as _services_router
from dana.api.skills import router as _skills_router
from dana.api.sessions import derive_title, is_valid_session_id, load_session, new_session_id, save_session
from dana.api.sessions import router as _sessions_router
from dana.api.workspace import load_mounted_directories
from dana.api.workspace import router as _workspace_router
from dana.core.react_dispatch import (
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
from dana.core.context_distiller import schedule_distillation
from dana.paths import CAPTURES_DIR
from dana.platform import get_cad_engine, get_control_plane
from dana.plugins.freecad.call_log import CadCallLog
from dana.plugins.freecad.py_export import write_macro_script
from dana.plugins.os.desktop_vision import _capture_primary_monitor_jpeg_b64
from dana.services.voice_service import VoiceService, VoiceState

_REPO_ROOT = Path(__file__).resolve().parents[2]
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
    "cad": "freecad",
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
    """Debug Terminal plumbing: tees every line written to ``stream`` (stdout
    or stderr — every ``print()``/traceback in this process, from any
    thread) out to every connected ``/ws/chat`` client as a ``server_log``
    event, so the frontend's floating Debug Terminal shows live backend
    output without a separate log-tailing mechanism. The original stream is
    always written first — this never replaces real console output, only
    mirrors it.

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
app.include_router(_memory_router)
app.include_router(_sessions_router)
app.include_router(_skills_router)
app.include_router(_services_router)
app.include_router(_planner_router)


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
    return {"ok": True, **driver_state()}


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
        "perform_freecad_boolean",
        "perform_freecad_edge_operation",
        "modify_freecad_parameter",
        "create_freecad_pipe",
        "align_freecad_objects",
        "create_assembly_mate",
        "create_freecad_sketch_extrude",
        "batch_pattern_array",
        "insert_standard_part",
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
    if tool_id in _CAD_CREATE_TOOLS:
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
# re-deciding to call tools (a hallucination loop) can't run forever.
_MAX_REACT_ITERATIONS = 13

# P3 of the local-agent rescue plan — a suspended react_state/visual_state
# (a mutating tool awaiting HITL approval, or take_canvas_screenshot
# awaiting the frontend's R3F capture) normally resumes the instant the
# matching hitl_response/visual_capture_response arrives. If the frontend
# never replies — a dropped message, a closed plugin tab, a crashed
# renderer — the turn would otherwise stay parked forever with no path back
# for the user short of reconnecting. _sweep_stale_suspensions is the
# wall-clock backstop: any suspension older than this is auto-cancelled
# with a synthetic failure reply instead of hanging indefinitely.
_SUSPENDED_TURN_TIMEOUT_SEC = 60.0

# How often the sweep checks every connected session — cheap (a dict scan
# over at most a handful of local sessions), so this can run often without
# it costing anything meaningful.
_SUSPENSION_SWEEP_INTERVAL_SEC = 15.0

# Hard cap on "tool_start"'s args_summary — a human-glance label for the
# ChatPanel's inline Agent Activity feed (see _execute_and_continue), NOT a
# payload dump. The full arguments already go out on the existing
# "tool_call"/"dag_node_start" events for the DAG-Monitor/HITL-facing
# consumers, so a single oversized value here (write_file's full "content",
# run_python_script's script text, ...) is hard-truncated rather than
# blowing up this lightweight status line.
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
    _resolve_react_hitl), so it's the single place to emit the lightweight
    "tool_start"/"tool_complete" pair the frontend's ChatPanel renders as
    an inline Agent Activity feed — plugin-agnostic, unlike the existing
    "dag_node_start"/"tool_call"/"dag_node_complete" events, which are only
    ever rendered by the CadPlugin's DAG Monitor.

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
    """
    engine = get_cad_engine()
    control_plane = get_control_plane()

    await websocket.send_json(
        {
            "type": "tool_start",
            "tool_name": call.tool_id,
            "args_summary": _summarize_tool_args(call.tool_id, call.arguments),
        }
    )

    result = dispatch_tool_call(
        call,
        engine,
        control_plane,
        call_log=session.get("call_log"),
        api_keys=session.get("api_keys"),
        allowed_mounts=load_mounted_directories(),
    )

    await websocket.send_json(
        {
            "type": "tool_complete",
            "tool_name": call.tool_id,
            "status": "success" if result.ok else "error",
        }
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
    elif result.ok:
        # Any OTHER successful dispatch belonging to an already-unlocked
        # capability domain (e.g. list_directory while "os_tools" is
        # active) resets that domain's decay clock too — a domain the
        # agent is actively working with must never expire mid-task just
        # because the turns are ticking by; only genuine disuse decays.
        _touch_capability_domains(session, domains_for_tool_id(call.tool_id))

    mesh_url = None
    if result.ok and call.tool_id in _CAD_CREATE_TOOLS:
        mesh = engine.export_mesh_stl(result.payload["path"], name=result.payload.get("name"))
        if mesh.get("ok"):
            token = _register_mesh(mesh["path"])
            mesh_url = f"/api/mesh/{token}.stl"

    await _dag_complete(websocket, node_id, "success" if result.ok else "error", result.payload, result.duration_ms)
    await websocket.send_json(
        {
            "type": "tool_result",
            "tool_id": call.tool_id,
            "ok": result.ok,
            "payload": result.payload,
            "message": result.message,
            "duration_ms": result.duration_ms,
            "mesh_url": mesh_url,
        }
    )

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

    await _run_react_loop(websocket, session, messages, loop_count + 1, last_failure=current_failure)


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


async def _run_react_loop(
    websocket: WebSocket,
    session: dict[str, Any],
    messages: list[dict[str, Any]],
    loop_count: int,
    last_failure: tuple[str, str] | None = None,
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

    if session.get("abort_requested"):
        session["abort_requested"] = False
        await _dag_complete(websocket, node_id, "error", {"aborted": True}, parse_ms)
        await _finish_turn(websocket, session, messages, "Generation aborted by user.")
        return

    if turn.kind == "error":
        await _dag_complete(websocket, node_id, "error", {"matched": False}, parse_ms)
        await _finish_turn(
            websocket, session, messages, "I ran into a problem talking to the model — please try again."
        )
        return

    if turn.kind == "final":
        await _dag_complete(websocket, node_id, "success", {"final": True}, parse_ms)
        content = turn.content or (
            "I didn't think that needed a tool call — try asking for a specific "
            "action (e.g. \"refactor foo.py to use snake_case\" or \"build a box 60x40x20\")."
        )
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

    if is_mutating_tool(call.tool_id):
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

    # Non-mutating: "dag_node_start"/"tool_call" fire only now, right as
    # execution actually begins — same contract a mutating tool gets below,
    # post-approval, in _resolve_react_hitl (never pre-approval — a client
    # must not see "tool_call" for something that hasn't run yet).
    dispatch_node_id = f"dispatch-{loop_count}"
    await _dag_start(websocket, dispatch_node_id, call.tool_id, _node_type_for(call.tool_id), call.arguments)
    await websocket.send_json({"type": "tool_call", "tool_id": call.tool_id, "arguments": call.arguments})
    await _execute_and_continue(
        websocket, session, messages, loop_count, call, tool_call_id, dispatch_node_id, last_failure=last_failure
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
        await _finish_turn(websocket, session, state["messages"], "Cancelled — no changes were made.")
        return

    override = response.get("parameters")
    if isinstance(override, dict):
        call.arguments.update(override)

    loop_count = state["loop_count"]
    dispatch_node_id = f"dispatch-{loop_count}"
    await _dag_start(websocket, dispatch_node_id, call.tool_id, _node_type_for(call.tool_id), call.arguments)
    await websocket.send_json({"type": "tool_call", "tool_id": call.tool_id, "arguments": call.arguments})

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
    await _dag_start(websocket, dispatch_node_id, call.tool_id, _node_type_for(call.tool_id), call.arguments)
    await _dag_complete(websocket, dispatch_node_id, "success" if result.ok else "error", payload, 0)
    await websocket.send_json(
        {
            "type": "tool_result",
            "tool_id": call.tool_id,
            "ok": result.ok,
            # The client already has the image bytes it just sent — no need
            # to blast the full base64 back over the wire a second time.
            "payload": {k: v for k, v in payload.items() if k != "image_b64"},
            "message": result.message,
            "duration_ms": 0,
            "mesh_url": None,
        }
    )

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


# Serve the built React/Tauri web bundle, if present, at the app root. Mounted
# last so it never shadows the /api or /ws routes above. Absent in a bare
# `python -m dana.api.server` dev loop where only the API is being exercised.
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")


__all__ = ("app",)

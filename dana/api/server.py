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

from dana.core.react_dispatch import (
    build_assistant_tool_call_message,
    build_system_prompt,
    build_tool_result_message,
    describe_tool_call,
    dispatch_tool_call,
    driver_state,
    is_mutating_tool,
    next_react_turn,
    plugin_registry_view,
)
from dana.paths import CAPTURES_DIR
from dana.platform import get_cad_engine, get_control_plane
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


def _on_voice_state(state: VoiceState, transcript: str) -> None:
    """Runs on VoiceService's worker thread — hop back onto the event loop."""
    loop = _event_loop
    if loop is None:
        return

    async def _emit() -> None:
        await _broadcast({"type": "voice_state", "state": state, "transcript": transcript})
        if state == "speaking" and transcript:
            for ws, session in list(_active_sessions.items()):
                await _process_user_text(ws, session, transcript)

    asyncio.run_coroutine_threadsafe(_emit(), loop)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _voice_service, _event_loop
    _event_loop = asyncio.get_running_loop()
    _voice_service = VoiceService(on_state=_on_voice_state)
    _voice_service.start()
    try:
        yield
    finally:
        if _voice_service is not None:
            _voice_service.stop()


app = FastAPI(title="Dana API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_DEV_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _register_mesh(path: str) -> str:
    token = uuid.uuid4().hex
    _MESH_REGISTRY[token] = Path(path)
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
        # get_freecad_bounding_box is intentionally absent — a read-only
        # query with no new/changed geometry, so it shouldn't trigger
        # another mesh export on every call.
    }
)


def _node_type_for(tool_id: str) -> str:
    if tool_id == "execute_vision_analysis":
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


# Safety counter for _run_react_loop — forcefully stops the loop after this
# many tool-executing iterations within one user turn, so a model stuck
# re-deciding to call tools (a hallucination loop) can't run forever.
_MAX_REACT_ITERATIONS = 5


async def _execute_and_continue(
    websocket: WebSocket,
    session: dict[str, Any],
    messages: list[dict[str, Any]],
    loop_count: int,
    call: Any,
    tool_call_id: str,
    node_id: str,
) -> None:
    """Runs an already-approved (or never-gated) tool call, streams its
    dispatch/DAG/mesh/camera events, appends its result back into
    ``messages`` as a ``tool`` role reply, then loops back into
    ``_run_react_loop`` for the NEXT iteration — this loop-back, not just
    the dispatch itself, is what lets the LLM actually see the result and
    decide whether to chain another tool call.
    """
    engine = get_cad_engine()
    control_plane = get_control_plane()
    result = dispatch_tool_call(call, engine, control_plane)

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
    await _run_react_loop(websocket, session, messages, loop_count + 1)


async def _run_react_loop(
    websocket: WebSocket, session: dict[str, Any], messages: list[dict[str, Any]], loop_count: int
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
    """
    if loop_count >= _MAX_REACT_ITERATIONS:
        await websocket.send_json(
            {
                "type": "assistant_message",
                "content": f"Reached the maximum number of reasoning steps ({_MAX_REACT_ITERATIONS}) — stopping here.",
            }
        )
        return

    node_id = f"parse-{loop_count}"
    await _dag_start(websocket, node_id, "Parse intent", "agent", {"step": loop_count})
    parse_start = time.perf_counter()
    raw_text = next((m["content"] for m in messages if m.get("role") == "user"), "")
    turn = await next_react_turn(messages, session.get("active_selection"), raw_text=raw_text)
    parse_ms = int((time.perf_counter() - parse_start) * 1000)

    if turn.kind == "error":
        await _dag_complete(websocket, node_id, "error", {"matched": False}, parse_ms)
        await websocket.send_json(
            {"type": "assistant_message", "content": "I ran into a problem talking to the model — please try again."}
        )
        return

    if turn.kind == "final":
        await _dag_complete(websocket, node_id, "success", {"final": True}, parse_ms)
        content = turn.content or (
            "I didn't think that needed a tool call — try asking for a specific "
            "action (e.g. \"build a box 60x40x20\")."
        )
        messages.append({"role": "assistant", "content": content})
        await websocket.send_json({"type": "assistant_message", "content": content})
        return

    # turn.kind == "tool_call"
    await _dag_complete(websocket, node_id, "success", {"tool_id": turn.call.tool_id}, parse_ms)
    call = turn.call
    assistant_message, tool_call_id = build_assistant_tool_call_message(call)
    messages.append(assistant_message)

    if is_mutating_tool(call.tool_id):
        request_id = uuid.uuid4().hex
        session["react_state"] = {
            "messages": messages,
            "loop_count": loop_count,
            "call": call,
            "tool_call_id": tool_call_id,
            "request_id": request_id,
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
    await _execute_and_continue(websocket, session, messages, loop_count, call, tool_call_id, dispatch_node_id)


async def _resolve_react_hitl(websocket: WebSocket, session: dict[str, Any], response: dict[str, Any]) -> None:
    state = session.get("react_state")
    if state is None or response.get("request_id") != state["request_id"]:
        return  # stale or unknown request_id — ignore
    session["react_state"] = None

    call = state["call"]
    if not response.get("approved"):
        # Never dispatched — no "tool_call"/dag node was ever opened for a
        # call awaiting approval, so there's nothing to dag-complete either.
        await websocket.send_json({"type": "assistant_message", "content": "Cancelled — no changes were made."})
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


async def _process_user_text(websocket: WebSocket, session: dict[str, Any], user_text: str) -> None:
    """Starts a fresh multi-step ReAct loop for one chat utterance.

    Used both for a client's own typed messages and for a finalized
    ``VoiceService`` transcript replayed across every connected session —
    same loop, same DAG/HITL events, regardless of the source. Bounces
    with a nudge instead of starting a second loop if one is already
    mid-flight (suspended on a pending HITL approval) for this session —
    starting a new ``messages`` history here would silently orphan that
    pending approval and desync session["react_state"].
    """
    if session.get("react_state") is not None:
        await websocket.send_json(
            {
                "type": "assistant_message",
                "content": "Please resolve the pending approval above (Proceed/Cancel) before sending another message.",
            }
        )
        return
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(session.get("active_selection"))},
        {"role": "user", "content": user_text},
    ]
    await _run_react_loop(websocket, session, messages, 0)


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "ready",
            "driver_state": driver_state(),
            "plugins": plugin_registry_view(),
        }
    )
    session: dict[str, Any] = {"active_selection": None, "react_state": None}
    _active_sessions[websocket] = session
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "canvas_selection":
                session["active_selection"] = data.get("payload") or {}
                continue

            if msg_type == "hitl_response":
                payload = data.get("payload") or {}
                await _resolve_react_hitl(websocket, session, payload)
                continue

            user_text = str(data.get("text") or "").strip()
            if not user_text:
                continue

            await _process_user_text(websocket, session, user_text)
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

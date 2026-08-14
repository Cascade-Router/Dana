"""Headless FastAPI server — the single backend for the Tauri/React frontend.

Replaces the Gradio UI's in-process callbacks with a ``/ws/chat`` WebSocket
that streams ReAct-style tool-dispatch events (tool_call -> tool_result ->
assistant_message) to any connected client. All dispatch logic lives in
``dana.core.react_dispatch`` and goes through ``dana.platform.get_control_plane()``
/ ``get_cad_engine()`` exactly as the legacy UI did — only the transport
changed, not the drivers.

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
    describe_tool_call,
    dispatch_tool_call,
    driver_state,
    is_mutating_tool,
    parse_utterance,
    plugin_registry_view,
    summarize_result,
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


def _node_type_for(tool_id: str) -> str:
    if tool_id == "execute_vision_analysis":
        return "vision"
    if tool_id in ("create_freecad_box", "create_freecad_cylinder", "create_freecad_extrusion"):
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


async def _dispatch_and_emit(websocket: WebSocket, call: Any) -> None:
    """Runs an already-approved (or never-gated) tool call and streams the
    dispatch/DAG/mesh/camera events for it. Shared by the plain-chat path
    and the post-approval resume path so both emit an identical sequence."""
    await _dag_start(websocket, "dispatch", call.tool_id, _node_type_for(call.tool_id), call.arguments)
    await websocket.send_json({"type": "tool_call", "tool_id": call.tool_id, "arguments": call.arguments})

    engine = get_cad_engine()
    control_plane = get_control_plane()
    result = dispatch_tool_call(call, engine, control_plane)

    mesh_url = None
    if result.ok and call.tool_id in ("create_freecad_box", "create_freecad_cylinder", "create_freecad_extrusion"):
        mesh = engine.export_mesh_stl(result.payload["path"], name=result.payload.get("name"))
        if mesh.get("ok"):
            token = _register_mesh(mesh["path"])
            mesh_url = f"/api/mesh/{token}.stl"

    await _dag_complete(
        websocket, "dispatch", "success" if result.ok else "error", result.payload, result.duration_ms
    )
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

    await websocket.send_json({"type": "assistant_message", "content": summarize_result(call, result)})


async def _resolve_hitl(websocket: WebSocket, call: Any, response: dict[str, Any]) -> None:
    if not response.get("approved"):
        await _dag_complete(websocket, "dispatch", "error", {"cancelled": True}, 0)
        await websocket.send_json({"type": "assistant_message", "content": "Cancelled — no changes were made."})
        return
    override = response.get("parameters")
    if isinstance(override, dict):
        call.arguments.update(override)
    await _dispatch_and_emit(websocket, call)


async def _process_user_text(websocket: WebSocket, session: dict[str, Any], user_text: str) -> None:
    """Parse + dispatch one chat utterance for ``session``'s connection.

    Used both for a client's own typed messages and for a finalized
    ``VoiceService`` transcript replayed across every connected session —
    same dispatch path, same DAG/HITL events, regardless of the source.
    """
    await _dag_start(websocket, "parse", "Parse intent", "agent", {"text": user_text})
    parse_start = time.perf_counter()
    call = await parse_utterance(user_text, session.get("active_selection"))
    parse_ms = int((time.perf_counter() - parse_start) * 1000)

    if call is None:
        await _dag_complete(websocket, "parse", "error", {"matched": False}, parse_ms)
        await websocket.send_json(
            {
                "type": "assistant_message",
                "content": (
                    "I didn't think that needed a tool call — try asking for a specific "
                    "action (e.g. \"build a box 60x40x20\")."
                ),
            }
        )
        return
    await _dag_complete(websocket, "parse", "success", {"tool_id": call.tool_id}, parse_ms)

    if is_mutating_tool(call.tool_id):
        request_id = uuid.uuid4().hex
        session["pending_hitl"] = {"request_id": request_id, "call": call}
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

    await _dispatch_and_emit(websocket, call)


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
    session: dict[str, Any] = {"active_selection": None, "pending_hitl": None}
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
                pending = session.get("pending_hitl")
                if pending is None or payload.get("request_id") != pending["request_id"]:
                    continue  # stale or unknown request_id — ignore
                session["pending_hitl"] = None
                await _resolve_hitl(websocket, pending["call"], payload)
                continue

            user_text = str(data.get("text") or "").strip()
            if not user_text:
                continue
            if session.get("pending_hitl") is not None:
                await websocket.send_json(
                    {
                        "type": "assistant_message",
                        "content": "Please resolve the pending approval above (Proceed/Cancel) before sending another message.",
                    }
                )
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

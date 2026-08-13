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

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dana.core.react_dispatch import (
    dispatch_tool_call,
    driver_state,
    parse_utterance,
    plugin_registry_view,
    summarize_result,
)
from dana.paths import CAPTURES_DIR
from dana.platform import get_cad_engine, get_control_plane

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

app = FastAPI(title="Dana API")

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
    try:
        while True:
            data = await websocket.receive_json()
            user_text = str(data.get("text") or "").strip()
            if not user_text:
                continue

            call = parse_utterance(user_text)
            if call is None:
                await websocket.send_json(
                    {
                        "type": "assistant_message",
                        "content": (
                            "I didn't match that to a registered tool. Try phrasing it "
                            "like a command (e.g. \"build a box 60x40x20\")."
                        ),
                    }
                )
                continue

            await websocket.send_json(
                {"type": "tool_call", "tool_id": call.tool_id, "arguments": call.arguments}
            )

            engine = get_cad_engine()
            control_plane = get_control_plane()
            result = dispatch_tool_call(call, engine, control_plane)

            mesh_url = None
            if result.ok and call.tool_id in ("create_freecad_box", "create_freecad_cylinder"):
                mesh = engine.export_mesh_stl(result.payload["path"], name=result.payload.get("name"))
                if mesh.get("ok"):
                    token = _register_mesh(mesh["path"])
                    mesh_url = f"/api/mesh/{token}.stl"

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
            await websocket.send_json(
                {"type": "assistant_message", "content": summarize_result(call, result)}
            )
    except WebSocketDisconnect:
        pass


# Serve the built React/Tauri web bundle, if present, at the app root. Mounted
# last so it never shadows the /api or /ws routes above. Absent in a bare
# `python -m dana.api.server` dev loop where only the API is being exercised.
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")


__all__ = ("app",)

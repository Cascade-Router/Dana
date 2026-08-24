"""Hugging Face Space entry point (sdk: gradio) — pure Gradio chat.

Reverted from the FastAPI+React-behind-Gradio setup: mounting a custom
ASGI app via gr.mount_gradio_app was found to conflict with ZeroGPU's own
ASGI middleware injection on this Space's hardware tier. This is a pure
`gr.ChatInterface` instead — no `gr.mount_gradio_app`, no manually-run
uvicorn, no exported FastAPI `app` — so HF's native Gradio launcher owns
the whole process and ZeroGPU's middleware has nothing to conflict with.

Trade-off: the React UI, CAD 3D viewport, and DAG monitor are gone for this
HF-hosted build — text chat only. The Tauri desktop app is unaffected (it
never went through this file).

Rather than reimplementing the ReAct loop's HITL approval, tool dispatch,
capability-decay, and retry-backstop logic from scratch — which would
immediately start drifting from the real implementation — this drives
dana.api.server's actual `_process_user_text`/`_run_react_loop` machinery
through `_GradioSocket`, a duck-typed stand-in for the `WebSocket` those
functions normally stream events to. Every mocked-CAD/mocked-control-plane
behavior those functions already have for `IS_HF_SPACE` (dana.platform.
factory) comes along for free, unchanged.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
import spaces

from dana.api.server import _process_user_text, _resolve_react_hitl, _resolve_visual_capture
from dana.api.sessions import new_session_id
from dana.plugins.freecad.call_log import CadCallLog


def _new_session() -> dict[str, Any]:
    """Same shape as dana.api.server.ws_chat's own per-connection `session`
    dict, minus fields this text-only front end has no source for (a live
    3D canvas selection, a resumed on-disk chat)."""
    return {
        "active_selection": None,
        "react_state": None,
        "visual_state": None,
        "call_log": CadCallLog(),
        "session_id": new_session_id(),
        "chat_history": [],
        "session_title": None,
        "session_created_at": None,
        "api_keys": {},
        "active_plugins": frozenset(),
        "capability_unlocked_at_turn": {},
        "working_memory": {"summary": "", "turn": 0},
        "turn_counter": 0,
        "abort_requested": False,
    }


class _GradioSocket:
    """Duck-typed stand-in for the `WebSocket` dana.api.server's ReAct loop
    streams events to via `await websocket.send_json(...)`.

    Two event types normally SUSPEND the loop waiting for something this
    text-only chat has no source for, so they're resolved immediately
    instead of actually suspending:
    - hitl_approval_required: auto-approved. Safe specifically because
      IS_HF_SPACE already swaps in Mock* CAD/control-plane drivers
      everywhere in this environment — every "mutating" action a human
      would normally approve here is already a simulated no-op, never a
      real desktop/file action.
    - visual_capture_request: resolved with an error (no live screen to
      capture) — same as any other tool failure the model can recover from.

    Every other event type (dag_node_start/tool_call/tool_start/
    tool_complete/tool_result/camera_animate/...) is streaming UI telemetry
    with no consumer here — this chat only surfaces the final reply.
    """

    def __init__(self, session: dict[str, Any]) -> None:
        self._session = session
        self.final_reply: str | None = None

    async def send_json(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type")
        if kind == "assistant_message":
            self.final_reply = payload["content"]
        elif kind == "hitl_approval_required":
            request_id = payload["payload"]["request_id"]
            await _resolve_react_hitl(self, self._session, {"request_id": request_id, "approved": True})
        elif kind == "visual_capture_request":
            request_id = payload["payload"]["request_id"]
            await _resolve_visual_capture(
                self,
                self._session,
                {"request_id": request_id, "error": "Screen capture isn't available in this hosted environment."},
            )


async def _dana_chat(message: str, history: list, session: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if session is None:
        session = _new_session()
    socket = _GradioSocket(session)
    await _process_user_text(socket, session, message)
    return socket.final_reply or "(no reply — something went wrong)", session


# ZeroGPU is only compatible with the Gradio SDK (per HF's own docs) and
# ties into Gradio's *registered event handlers* — a @spaces.GPU function
# that's never bound to a Gradio component/event never shows up in the
# Blocks' own dependency list, so its startup check can't find it even
# though it's a plain top-level decorated function. Binding it to the
# page's `load` event below (fires once per visitor, does nothing) is what
# actually registers it. This app never needs a GPU otherwise.
@spaces.GPU
def _dummy_gpu_function():
    pass


_CUSTOM_CSS = """
footer {display: none !important;}
.gradio-container {padding: 0 !important; margin: 0 !important; max-width: 900px !important;}
"""

with gr.Blocks(css=_CUSTOM_CSS, title="Dānā") as demo:
    _session_state = gr.State(None)
    gr.ChatInterface(
        fn=_dana_chat,
        additional_inputs=[_session_state],
        additional_outputs=[_session_state],
        # Stable REST/JS-client contract (frontend/src/lib/gradioChatClient.ts
        # calls this by name) — the default would otherwise be the private,
        # refactor-fragile "_dana_chat" (this function's own Python name).
        api_name="chat",
        title="Dānā",
        description=(
            "Hugging Face Space build — text chat only. Desktop/vision actuators and real CAD "
            "are mocked in this environment; the Tauri desktop app has the full FreeCAD + "
            "screen-vision experience."
        ),
    )
    demo.load(_dummy_gpu_function)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

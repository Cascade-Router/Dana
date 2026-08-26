"""Hugging Face Space entry point (sdk: gradio).

Plain `gr.Blocks` instead of `gr.ChatInterface` — ChatInterface's `fn`
contract is exactly one string in, one string out, with no room for a
second output. The Vercel frontend needs the STL mesh a CAD tool call
produces alongside the reply text, so the "chat" API endpoint here binds
two dedicated, hidden outputs (`text_out`, `file_out`) instead.

No `gr.mount_gradio_app`, no manually-run uvicorn, no exported FastAPI
`app` — mounting a custom ASGI app that way was found to conflict with
ZeroGPU's own ASGI middleware injection on this Space's hardware tier
(see git history). HF's native Gradio launcher owns the whole process.

Trade-off: the React UI, CAD 3D viewport, and DAG monitor are gone for
this HF-hosted build — this page is a plain chat + mesh-preview smoke
test, with the real UI living on Vercel instead, talking to the "chat"
endpoint. The Tauri desktop app is unaffected (it never went through
this file).

Rather than reimplementing the ReAct loop's HITL approval, tool dispatch,
capability-decay, and retry-backstop logic from scratch — which would
immediately start drifting from the real implementation — this drives
dana.api.server's actual `_process_user_text`/`_run_react_loop` machinery
through `_GradioSocket`, a duck-typed stand-in for the `WebSocket` those
functions normally stream events to. Every mocked-CAD/mocked-control-plane
behavior those functions already have for `IS_HF_SPACE` (dana.platform.
factory) comes along for free, unchanged — including which mesh format a
CAD tool call actually produces (dana.plugins.freecad's export_mesh_stl:
always `.stl`, there is no `.glb` export anywhere in this codebase).
`gr.Model3D` accepts `.stl` natively, so this serves that real file
directly rather than converting it to a format nothing here produces.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gradio as gr
import spaces

from dana.api import artifacts_registry
from dana.api.server import _MESH_REGISTRY, _process_user_text, _resolve_react_hitl, _resolve_visual_capture
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

    tool_result is inspected (not just discarded) for `mesh_url` — the
    REST-style path (e.g. "/api/mesh/<token>.stl") dana.api.server would
    normally serve, meaningless here since this app never mounts that
    FastAPI app at all. `_MESH_REGISTRY[token]` (same process, same
    dict — _register_mesh populates it synchronously before that event is
    even sent) resolves it back to the real local file path instead.

    dag_node_start/dag_node_complete ARE also captured now (into
    `dag_events`, in arrival order) — this is exactly what frontend/src/
    components/DAGMonitor.tsx's buildGraph() consumes to render the
    Execution Graph over the WS path; without this the Gradio-mode
    frontend hook (useGradioChat.ts) had no source for it at all and the
    graph stayed permanently empty ("(0)"), even though dana.api.server's
    ReAct loop was emitting these events into this same _GradioSocket the
    whole time. tool_call/tool_start/tool_complete/camera_animate/... are
    still discarded — no consumer for those on this text-only chat.
    """

    def __init__(self, session: dict[str, Any]) -> None:
        self._session = session
        self.final_reply: str | None = None
        self.mesh_path: str | None = None
        self.dag_events: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type")
        if kind == "assistant_message":
            self.final_reply = payload["content"]
        elif kind == "tool_result":
            mesh_url = payload.get("mesh_url")
            if mesh_url:
                token = mesh_url.rsplit("/", 1)[-1].removesuffix(".stl")
                path = _MESH_REGISTRY.get(token)
                if path is not None:
                    self.mesh_path = str(path)
        elif kind in ("dag_node_start", "dag_node_complete"):
            self.dag_events.append(payload)
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


async def _respond(message: str, chatbot_history: list, session: dict[str, Any] | None):
    """Bound to `api_name="chat"` below. Outputs are ordered
    `(text_out, file_out, chatbot, mesh_preview, session_state, graph_out)`
    — text_out/file_out FIRST specifically so a REST/JS caller's
    `result.data[0]`/`[1]` are the plain reply string and mesh path,
    never this function's internal chatbot-message-list bookkeeping
    (frontend/src/lib/gradioChatClient.ts already reads `data[0]` as a
    plain string; reordering these would silently break it). `graph_out`
    is appended LAST rather than inserted earlier for the same reason —
    every existing positional read stays valid.

    This yields twice, not once — but that's an honest "show a pending
    state, then the real one" UI improvement, not a token-level LLM
    stream: dana.api.server's `_run_react_loop` resolves a whole turn in
    one shot (`next_react_turn` is a plain `async def` returning a
    complete result, never a generator), so there is no token stream to
    relay here even in principle.
    """
    trimmed = (message or "").strip()
    if not trimmed:
        yield "", None, chatbot_history, None, session, []
        return
    # Session-persistence trace (item 3): `session` is whatever gr.State
    # handed back for THIS browser tab's session_hash — a fresh dict only on
    # the very first turn, the SAME dict object (by id()) every turn after,
    # since Gradio round-trips gr.State's value for a given session_hash
    # unchanged. If a mesh/tool created in turn 1 seems to "vanish" by turn
    # 2, this line is what tells you whether it's actually a fresh session
    # (a new session_id/id() each turn — a real state-loss bug) or a bug
    # elsewhere despite the same session persisting correctly.
    is_new_session = session is None
    if is_new_session:
        session = _new_session()
    print(
        f"[app.py] Chat turn start. session_id={session['session_id']} "
        f"new_session={is_new_session} session_obj_id={id(session)} turn={session['turn_counter']}",
        file=sys.stderr,
        flush=True,
    )

    pending_history = chatbot_history + [
        {"role": "user", "content": trimmed},
        {"role": "assistant", "content": "…"},
    ]
    yield "…", None, pending_history, None, session, []

    socket = _GradioSocket(session)
    await _process_user_text(socket, session, trimmed)
    reply = socket.final_reply or "(no reply — something went wrong)"
    final_history = chatbot_history + [
        {"role": "user", "content": trimmed},
        {"role": "assistant", "content": reply},
    ]
    print(
        f"[app.py] Chat turn complete. Exported mesh path: {socket.mesh_path}, "
        f"Registry items: {len(artifacts_registry.list_artifacts())}, "
        f"DAG events: {len(socket.dag_events)}",
        file=sys.stderr,
        flush=True,
    )
    yield reply, socket.mesh_path, final_history, socket.mesh_path, session, socket.dag_events


def _list_artifact_files() -> list[str]:
    """Bound to the hidden "artifacts" api endpoint below — the Gradio-mode
    equivalent of ``dana.api.cad``'s ``GET /api/cad/artifacts``, which the
    Vercel frontend's CadToolbar can't reach here at all (see this file's own
    docstring: no FastAPI app is mounted on this Space). Returns a plain list
    of existing file paths so the bound ``gr.File`` output FileData-ifies
    each one into a real fetchable URL (same mechanism ``_respond`` already
    relies on for ``file_out``/``mesh_preview`` above) — the JS client reads
    each entry's ``.url``/``.orig_name`` directly, no second round trip
    needed to resolve a path into something fetchable.
    """
    return [a["path"] for a in artifacts_registry.list_artifacts() if Path(a["path"]).is_file()]


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
    gr.Markdown(
        "# ⚙️ Dana AI Copilot (Headless Backend)\n\n"
        "⚠️ **Demo Purposes Only:** This interface is the headless backend for Dana. "
        "The actual application UI is hosted externally on Vercel. You can interact "
        "with the agent here for testing, but the full 3D viewport, workspace, and "
        "execution graph are only available on the main client."
    )
    # type="messages" is load-bearing, not the modern default it looks like:
    # the Space's pinned sdk_version (deploy/space_README.md) is Gradio
    # 5.49.1, where an unset `type` silently falls back to the legacy
    # "tuples" format (confirmed directly in that version's source) — then
    # chokes on the {"role": ..., "content": ...} dicts _respond below
    # yields, with exactly this crash: "Data incompatible with tuples
    # format. Each message should be a list of length 2." Newer Gradio
    # (installed locally during dev) dropped `type` entirely because
    # messages-format became the only option, which is exactly how this
    # went unnoticed in local testing — dev and the deployed Space were on
    # different major versions the whole time.
    chatbot = gr.Chatbot(label="Dānā", height=420, type="messages")
    mesh_preview = gr.Model3D(label="Generated mesh (.stl)")
    # label="message" is load-bearing, not cosmetic (show_label=False hides
    # it visually only): frontend/src/lib/gradioChatClient.ts already calls
    # client.predict("/chat", { message }) — the JS/Python clients resolve a
    # keyed call's keys against each parameter's label, and gr.ChatInterface
    # (the previous version of this file) gave its equivalent input that
    # exact label implicitly. A relabel/no-label here would silently break
    # that existing call without ever raising an error.
    msg = gr.Textbox(label="message", placeholder="Ask Dana to do something...", show_label=False)
    send_btn = gr.Button("Send")

    session_state = gr.State(None)
    # Hidden, API-only outputs — see _respond's own docstring for why these
    # come first in the bound outputs list below.
    text_out = gr.Textbox(visible=False)
    file_out = gr.Model3D(visible=False)
    # This turn's dag_node_start/dag_node_complete events (see
    # _GradioSocket), as plain JSON — gradioChatClient.ts reads this as
    # `data[5]` and feeds it straight into DAGMonitor.tsx's buildGraph(),
    # the same shape the WS path already produces. Appended last in
    # _outputs so it doesn't renumber any existing positional read.
    graph_out = gr.JSON(visible=False)

    _outputs = [text_out, file_out, chatbot, mesh_preview, session_state, graph_out]

    msg.submit(_respond, inputs=[msg, chatbot, session_state], outputs=_outputs, api_name="chat").then(
        lambda: "", None, msg
    )
    # Same handler, same effect — not a second public endpoint (api_name
    # left unset here defaults to hidden/private since this event's inputs
    # match one already registered above under "chat").
    send_btn.click(_respond, inputs=[msg, chatbot, session_state], outputs=_outputs, api_name=False).then(
        lambda: "", None, msg
    )

    # Hidden trigger for the "artifacts" api endpoint — CadToolbar.tsx (via
    # gradioChatClient.ts's fetchGradioArtifacts) calls this by api_name
    # exactly like `msg.submit`'s "chat" above, never by actually clicking
    # it. gr.File FileData-ifies each returned path into a real fetchable
    # URL, so the frontend needs no separate download endpoint either.
    artifacts_trigger = gr.Button(visible=False)
    artifacts_out = gr.File(visible=False, file_count="multiple")
    artifacts_trigger.click(_list_artifact_files, None, artifacts_out, api_name="artifacts")

    demo.load(_dummy_gpu_function)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)

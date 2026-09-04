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
CAD tool call actually produces by default (dana.plugins.freecad's
export_mesh_stl: always `.stl`). `gr.Model3D` accepts `.stl` natively, so
this serves that real file directly by default; `_convert_step_to_mesh`
below additionally converts the turn's "best-effort STEP sibling" artifact
(dana/api/server.py's automatic `export_model(..., "step", ...)` call) into
a `.glb` for preview instead, whenever headless FreeCAD is actually
available — a no-op today (the mock engine never produces a usable
`.step`), functional once packages.txt's `freecad` apt package is wired in
as this Space's real driver.

Sandbox Hardening: `_harden_tool_registry()` below permanently strips
`execute_terminal_command`/`execute_code_task`/`search_codebase` out of
the shared tool registry before `gr.Blocks` boots — even with every
IS_HF_SPACE mock already in place, these three should never be queuable at
all on this entry point, not just fail closed once dispatched.

Backend tracing (FSM/dispatch/DAG prints) is also tee'd into a bounded
buffer and surfaced live via the visible `logs_out` textbox — see
`_TeeStream`/`_tail_logs` below.
"""

from __future__ import annotations

import collections
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import spaces

from dana.api import artifacts_registry
from dana.api.server import _MESH_REGISTRY, _process_user_text, _resolve_react_hitl, _resolve_visual_capture
from dana.api.sessions import new_session_id
from dana.core import react_dispatch
from dana.plugins.freecad.call_log import CadCallLog


def _harden_tool_registry() -> None:
    """Sandbox Hardening — called once below, before `with gr.Blocks(...)`
    boots. Even with mock CAD/control-plane drivers, the highest-risk tools
    must not be QUEUABLE at all here, not just fail closed once dispatched.
    Reuses the exact registries react_dispatch.refresh_plugin_tools() itself
    already mutates at import time (TOOL_HANDLERS, _PLUGIN_TOOL_SCHEMAS)
    rather than building a second, parallel allowlist mechanism.

    Two layers, deliberately not just one:
    - `TOOL_HANDLERS.pop(...)` is the actual DISPATCH gate (see
      react_dispatch's turn loop: `if call.tool_id not in TOOL_HANDLERS or
      call.tool_id not in allowed_tool_ids: ... retry`) — this alone makes
      all three permanently undispatchable here, full stop, regardless of
      what the model's tool schema still says.
    - `_PLUGIN_TOOL_SCHEMAS.pop(...)` additionally stops execute_code_task/
      search_codebase from being OFFERED to the model at all, so a turn
      never wastes a round-trip on a call that would just come back
      "unknown tool_id".
      execute_terminal_command doesn't need this second step: it's a
      native, tools.json-sourced schema gated behind the "os_tools"
      capability domain, which a fresh Gradio session never activates by
      default (see _new_session's empty active_plugins/capability_unlocked_
      at_turn below) — and its handler ALREADY hard-refuses via
      IS_HF_SPACE regardless (react_dispatch._tool_execute_terminal_
      command). Popping it from TOOL_HANDLERS here is redundant
      defense-in-depth for that one specifically, not a gap-fix — kept
      anyway since "the tool doesn't exist" is a strictly stronger
      guarantee than "the tool exists but refuses when called".

    Only effective against tools registered by the time this runs (import
    time, before any user turn) — there is no hot-reload endpoint on this
    Gradio entry point that could call refresh_plugin_tools() again mid
    -process and silently undo this, but if one is ever added here, it
    would need to re-apply this hardening afterward.
    """
    for tool_id in ("execute_terminal_command", "execute_code_task", "search_codebase"):
        react_dispatch.TOOL_HANDLERS.pop(tool_id, None)
    for tool_id in ("execute_code_task", "search_codebase"):
        react_dispatch._PLUGIN_TOOL_SCHEMAS.pop(tool_id, None)
    react_dispatch._tool_ids_for_plugins.cache_clear()
    react_dispatch._llm_tools_schema_cached.cache_clear()


_harden_tool_registry()


class _TeeStream:
    """Tees every line written to `stream` out to `_LOG_BUFFER` (a bounded
    deque) in addition to writing it through unchanged — the Gradio-mode
    counterpart to dana.api.server's `_BroadcastStream` (which broadcasts
    over `/ws/chat` instead), reimplemented locally rather than reused
    because installing it requires starting FastAPI's lifespan, which this
    file's own module docstring already rules out (ZeroGPU ASGI conflict).

    Captures ALL stdout/stderr lines, not a real logging-level filter: this
    codebase's own backend tracing is almost entirely plain `print(...)`
    calls (this file's own `_respond` included) rather than a leveled
    `logging` hierarchy, so "the INFO logs" and "everything written to
    stdout/stderr" are, in practice, the same thing on this path today.
    """

    def __init__(self, original: Any) -> None:
        self._original = original
        self._buffer = ""

    def write(self, s: str) -> int:
        self._original.write(s)
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                _LOG_BUFFER.append(line)
        return len(s)

    def flush(self) -> None:
        self._original.flush()

    def isatty(self) -> bool:
        # Uvicorn's colored logging setup (via click) checks this on
        # sys.stdout/sys.stderr before configuring itself — a plain
        # AttributeError here (this class had no isatty at all) crashed
        # demo.launch() before the Gradio interface ever came up.
        # getattr(..., lambda: False) rather than a bare self._original.
        # isatty() call: self._original is always a real sys.stdout/stderr
        # here in practice (see _install_log_tee), which always has this
        # method, but delegating defensively costs nothing and means a
        # genuine local TTY still reports its real color-capable state
        # while any stream lacking the method safely reports False instead
        # of raising.
        return getattr(self._original, "isatty", lambda: False)()


_LOG_BUFFER: collections.deque[str] = collections.deque(maxlen=400)


def _install_log_tee() -> None:
    if not isinstance(sys.stdout, _TeeStream):
        sys.stdout = _TeeStream(sys.stdout)
    if not isinstance(sys.stderr, _TeeStream):
        sys.stderr = _TeeStream(sys.stderr)


_install_log_tee()


_FREECAD_STEP_TO_MESH_MACRO = """
import sys
import FreeCAD
import Part
import Mesh
import MeshPart

step_path, out_path = sys.argv[1], sys.argv[2]
doc = FreeCAD.newDocument("dana_convert")
Part.insert(step_path, doc.Name)
doc.recompute()

combined = Mesh.Mesh()
for obj in doc.Objects:
    if hasattr(obj, "Shape") and obj.Shape.Volume > 0:
        combined.addMesh(MeshPart.meshFromShape(Shape=obj.Shape, LinearDeflection=0.1, AngularDeflection=0.5))

if combined.CountFacets == 0:
    sys.exit(1)
combined.write(out_path)
"""


def _convert_step_to_mesh(step_path: str, out_format: str = "glb") -> dict[str, Any]:
    """Converts a `.step`/`.stp` CAD file into a browser-renderable mesh
    (`.glb` by default) for `gr.Model3D` — a two-stage pipeline since
    neither half can do this alone: trimesh (already a hard dependency,
    see requirements.txt) reads mesh formats but not STEP/BREP; FreeCAD
    reads STEP but this codebase's headless HF Space path is mock-only
    today (dana/platform/factory.py's IS_HF_SPACE branch still hardcodes
    MockFreeCADEngine, whose export_model always returns ok=False for
    "step" — no B-rep writer, by design).

    Contingent on packages.txt's `freecad` apt package actually being
    installed on this container AND a real FreeCAD-backed engine
    eventually being wired in as the IS_HF_SPACE driver — returns a clear,
    honest error rather than crashing when `freecadcmd` isn't on PATH,
    which is the current default reality (the mock engine never produces a
    usable `.step` file for this function to even be called on in the
    first place; see the STEP-artifact check in `_respond` below).
    """
    freecadcmd = shutil.which("freecadcmd") or shutil.which("FreeCADCmd")
    if freecadcmd is None:
        return {
            "ok": False,
            "error": "freecadcmd not found on PATH — headless FreeCAD isn't installed/wired in this environment yet.",
        }

    source = Path(step_path)
    if not source.is_file():
        return {"ok": False, "error": f"_convert_step_to_mesh: source not found: {step_path}"}

    with tempfile.TemporaryDirectory() as tmp:
        macro_path = Path(tmp) / "step_to_mesh.py"
        macro_path.write_text(_FREECAD_STEP_TO_MESH_MACRO, encoding="utf-8")
        stl_path = Path(tmp) / "converted.stl"
        try:
            proc = subprocess.run(
                [freecadcmd, str(macro_path), str(source), str(stl_path)],
                capture_output=True,
                text=True,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "_convert_step_to_mesh: freecadcmd timed out after 90s"}
        if proc.returncode != 0 or not stl_path.is_file():
            return {"ok": False, "error": f"_convert_step_to_mesh: freecadcmd failed: {proc.stderr.strip()[-500:]}"}

        try:
            import trimesh

            mesh = trimesh.load(stl_path, force="mesh")
            out_fd, out_name = tempfile.mkstemp(suffix=f".{out_format}")
            os.close(out_fd)
            out_path = Path(out_name)
            mesh.export(out_path)
        except Exception as exc:  # noqa: BLE001 — best-effort; a re-export failure must never crash the turn
            return {"ok": False, "error": f"_convert_step_to_mesh: mesh re-export failed: {exc}"}

    return {"ok": True, "path": str(out_path)}


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
        # Unused on this path in practice — _GradioSocket.send_json below
        # auto-approves every hitl_approval_required unconditionally — but
        # _resolve_react_hitl is shared code that reads/writes this key
        # regardless of caller, so it's initialized here too for the same
        # "same shape as ws_chat's session dict" contract this docstring
        # already promises.
        "hitl_approved_tools": set(),
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

    tool_dispatch_end is inspected (not just discarded) for `mesh_url` —
    the REST-style path (e.g. "/api/mesh/<token>.stl") dana.api.server
    would normally serve, meaningless here since this app never mounts
    that FastAPI app at all. `_MESH_REGISTRY[token]` (same process, same
    dict — _register_mesh populates it synchronously before that event is
    even sent) resolves it back to the real local file path instead.
    (This used to key off a "tool_result" event — server.py's
    _send_tool_dispatch_end consolidated "tool_complete"/dag_node_complete/
    "tool_result" into one "tool_dispatch_end" event, but this socket was
    never updated to match, so every mesh_url silently stopped arriving
    here — mesh_preview/file_out stayed None even though the tool
    succeeded and its .stl was registered in artifacts_registry just fine.)

    dag_node_start/dag_node_complete ARE also captured now (into
    `dag_events`, in arrival order) — this is exactly what frontend/src/
    components/DAGMonitor.tsx's buildGraph() consumes to render the
    Execution Graph over the WS path; without this the Gradio-mode
    frontend hook (useGradioChat.ts) had no source for it at all and the
    graph stayed permanently empty ("(0)"), even though dana.api.server's
    ReAct loop was emitting these events into this same _GradioSocket the
    whole time. tool_dispatch_start/camera_animate/... are still
    discarded — no consumer for those on this text-only chat.
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
        elif kind == "tool_dispatch_end":
            mesh_url = payload.get("mesh_url")
            if mesh_url:
                # `_MESH_REGISTRY` is keyed by a bare uuid4 hex token with NO
                # extension (see dana/api/server.py's _register_mesh) — the
                # extension is only appended when building `mesh_url`
                # itself, and it isn't always ".stl" (a URDF assembly is
                # "{token}.urdf", generate_3d_from_image can be "{token}.obj"
                # /".glb"). A bare `.removesuffix(".stl")` here silently left
                # the extension attached (and therefore never matched the
                # registry) for every one of those non-STL cases — Path.stem
                # strips whichever suffix is actually present.
                token = Path(mesh_url).stem
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


def _tail_logs(n: int = 40) -> str:
    """Last `n` captured stdout/stderr lines (see `_TeeStream`/`_LOG_BUFFER`
    above) — read fresh on every yield so `logs_out` shows this turn's own
    FSM/dispatch/DAG tracing, not a stale snapshot from page load."""
    return "\n".join(list(_LOG_BUFFER)[-n:])


async def _respond(message: str, chatbot_history: list, session: dict[str, Any] | None):
    """Bound to `api_name="chat"` below. Outputs are ordered
    `(text_out, file_out, chatbot, mesh_preview, session_state, logs_out,
    graph_out)` — text_out/file_out FIRST specifically so a REST/JS caller's
    `result.data[0]`/`[1]` are the plain reply string and mesh path,
    never this function's internal chatbot-message-list bookkeeping
    (frontend/src/lib/gradioChatClient.ts already reads `data[0]` as a
    plain string; reordering these would silently break it). `graph_out`
    is LAST, specifically — not just "appended after everything so far" —
    because gradioChatClient.ts reads it as `dataArr[dataArr.length - 1]`,
    never a fixed index, since `session_state` (a gr.State) never reaches
    the client at all (gr.State.skip_api is hardcoded True in Gradio) and
    the client-visible array shrinks accordingly. `logs_out` was added
    later than graph_out but goes BEFORE it here for that exact reason:
    appending it after graph_out (as originally done) silently broke the
    DAG view, since the client then read the raw log text where it
    expected graph JSON and dropped every dag_events payload.

    This yields twice, not once — but that's an honest "show a pending
    state, then the real one" UI improvement, not a token-level LLM
    stream: dana.api.server's `_run_react_loop` resolves a whole turn in
    one shot (`next_react_turn` is a plain `async def` returning a
    complete result, never a generator), so there is no token stream to
    relay here even in principle.
    """
    trimmed = (message or "").strip()
    if not trimmed:
        yield "", None, chatbot_history, None, session, _tail_logs(), []
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
    yield "…", None, pending_history, None, session, _tail_logs(), []

    socket = _GradioSocket(session)
    await _process_user_text(socket, session, trimmed)
    reply = socket.final_reply or "(no reply — something went wrong)"
    final_history = chatbot_history + [
        {"role": "user", "content": trimmed},
        {"role": "assistant", "content": reply},
    ]

    # Prefer a freshly-converted STEP artifact over the plain tessellated
    # STL, if one exists AND headless FreeCAD is actually available (see
    # _convert_step_to_mesh's own docstring) — on the current mock-only
    # deployment this is a cheap no-op every turn (shutil.which() finds
    # nothing, no subprocess ever spawned); it only starts doing real work
    # once packages.txt's `freecad` apt package is actually wired in as
    # this Space's driver.
    mesh_path = socket.mesh_path
    step_artifacts = [
        a
        for a in artifacts_registry.list_artifacts(session["session_id"])
        if a.get("format") == "step"
    ]
    if step_artifacts:
        newest_step = max(step_artifacts, key=lambda a: a["modified_at"])
        converted = _convert_step_to_mesh(newest_step["path"])
        if converted.get("ok"):
            mesh_path = converted["path"]
        else:
            print(f"[app.py] STEP mesh preview skipped: {converted.get('error')}", file=sys.stderr, flush=True)

    print(
        f"[app.py] Chat turn complete. Exported mesh path: {mesh_path}, "
        f"Registry items: {len(artifacts_registry.list_artifacts())}, "
        f"DAG events: {len(socket.dag_events)}",
        file=sys.stderr,
        flush=True,
    )
    yield reply, mesh_path, final_history, mesh_path, session, _tail_logs(), socket.dag_events


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
    # Label says both formats now: this still serves the mock engine's
    # plain .stl by default, but prefers a freshly-converted .glb instead
    # whenever a real STEP artifact exists AND headless FreeCAD is actually
    # wired in (see _convert_step_to_mesh) — gr.Model3D accepts either
    # natively, no component-level change needed for that upgrade path.
    mesh_preview = gr.Model3D(label="Generated mesh (.stl / .glb)")
    logs_out = gr.Textbox(
        label="Backend Log (FSM / Topological DAG)",
        interactive=False,
        lines=8,
        max_lines=8,
        autoscroll=True,
    )
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
    # `dataArr[dataArr.length - 1]` (the actual LAST element, not a fixed
    # index — see that file's own comment) and feeds it straight into
    # DAGMonitor.tsx's buildGraph(), the same shape the WS path already
    # produces. MUST stay the last entry in `_outputs` below no matter what
    # else gets added later — see logs_out's comment just below.
    graph_out = gr.JSON(visible=False)

    # Declared here, after graph_out, but placed BEFORE it in `_outputs`
    # below: graph_out has to stay the actual last element (see its own
    # comment above), so anything added after graph_out already existed
    # goes earlier in the list, never appended past it.
    _outputs = [text_out, file_out, chatbot, mesh_preview, session_state, logs_out, graph_out]

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

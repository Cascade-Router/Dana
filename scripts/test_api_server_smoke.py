"""One-off smoke test for dana/api/server.py — run manually, not part of pytest.

Exercises the WebSocket ReAct loop in-process (no real socket/uvicorn needed)
via Starlette's TestClient, so it works even without the frontend toolchain.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient

from dana.api.server import app


def _drain_until(ws: Any, msg_type: str, limit: int = 40) -> dict[str, Any]:
    """Read messages until one of ``msg_type`` arrives, discarding anything
    else along the way (e.g. ``dag_node_start`` events for the "parse-N"
    reasoning step, which precede ``tool_dispatch_start`` in the real
    ``_run_react_loop`` sequence — see the same pattern in
    ``tests/api/test_ws_chat.py``).

    A weaker local model can chain an extra, unwanted mutating tool call
    (e.g. proposing to mate a freshly-created box to itself) after the one
    this script actually asked for. The server suspends the turn on a
    ``hitl_approval_required`` event and waits for a real client's
    approve/reject click — which never comes here — so this declines it
    on sight, the same way a human clicking "Cancel" would, letting the
    turn wrap up instead of hanging forever."""
    for _ in range(limit):
        msg = ws.receive_json()
        print(f"({msg.get('type')}) ->", msg)
        if msg.get("type") == "hitl_approval_required":
            request_id = msg["payload"]["request_id"]
            print(f"  -> declining HITL request {request_id} for {msg['payload']['action_name']!r}")
            ws.send_json({"type": "hitl_response", "payload": {"request_id": request_id, "approved": False}})
            continue
        if msg.get("type") == "visual_capture_request":
            # A local model may call take_canvas_screenshot even with no
            # live R3F canvas behind this in-process TestClient — answer
            # with an error so the suspended turn resolves instead of
            # hanging forever waiting for a frontend that doesn't exist here.
            request_id = msg["payload"]["request_id"]
            print(f"  -> answering visual_capture_request {request_id} with no-canvas error")
            ws.send_json(
                {
                    "type": "visual_capture_response",
                    "payload": {"request_id": request_id, "error": "no live canvas in this smoke test"},
                }
            )
            continue
        if msg.get("type") == msg_type:
            return msg
    raise AssertionError(f"never received a {msg_type!r} message")


def _drain_until_tool_dispatch_start(ws: Any, tool_id: str, limit: int = 40) -> dict[str, Any]:
    """Drain ``tool_dispatch_start`` events until ``tool_id`` shows up
    (WebSocket Consolidation — this used to be a separate ``tool_call``
    event; see dana/api/server.py's ``_send_tool_dispatch_start``).

    A local model may first call ``load_capability`` to unlock the
    ``freecad`` domain (session-scoped capability decay — the CAD toolset
    starts unloaded and only appears in the LLM's tool list after this
    warm-up call) before actually dispatching the real tool. Any dispatch
    for a different tool_id along the way is a legitimate warm-up step, not
    a failure."""
    for _ in range(limit):
        call = _drain_until(ws, "tool_dispatch_start", limit=limit)
        if call["tool_name"] == tool_id:
            return call
    raise AssertionError(f"never received a tool_dispatch_start for {tool_id!r}")


def main() -> int:
    client = TestClient(app)

    health = client.get("/api/health").json()
    print("GET /api/health ->", health)
    assert health["ok"] is True

    with client.websocket_connect("/ws/chat") as ws:
        ready = ws.receive_json()
        print("ready ->", ready)
        assert ready["type"] == "ready"

        ws.send_json({"text": "build a box 60x40x20"})

        dispatch_start = _drain_until_tool_dispatch_start(ws, "create_freecad_box")
        assert dispatch_start["type"] == "tool_dispatch_start"
        assert dispatch_start["tool_name"] == "create_freecad_box"
        # Only check the dimensions requested in the prompt — different
        # models fill in the box tool's optional args (name, placement,
        # target_normal, ...) differently and may emit numbers instead of
        # numeric strings, neither of which is a defect.
        args = dispatch_start["arguments"]
        assert str(args.get("length")) == "60"
        assert str(args.get("width")) == "40"
        assert str(args.get("height")) == "20"

        dispatch_end = _drain_until(ws, "tool_dispatch_end")
        assert dispatch_end["type"] == "tool_dispatch_end"
        assert dispatch_end["status"] == "success"
        assert dispatch_end["mesh_url"], "expected a mesh_url for a created box"

        mesh_resp = client.get(dispatch_end["mesh_url"])
        print("GET", dispatch_end["mesh_url"], "->", mesh_resp.status_code, len(mesh_resp.content), "bytes")
        assert mesh_resp.status_code == 200
        assert len(mesh_resp.content) > 0

        assistant_message = _drain_until(ws, "assistant_message")
        assert assistant_message["type"] == "assistant_message"

        # An utterance that matches no tool should just get a plain reply,
        # not a crashed connection.
        ws.send_json({"text": "asdkjfh nonsense"})
        no_match = _drain_until(ws, "assistant_message")
        assert no_match["type"] == "assistant_message"

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

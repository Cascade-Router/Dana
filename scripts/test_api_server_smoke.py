"""One-off smoke test for dana/api/server.py — run manually, not part of pytest.

Exercises the WebSocket ReAct loop in-process (no real socket/uvicorn needed)
via Starlette's TestClient, so it works even without the frontend toolchain.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient

from dana.api.server import app


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

        tool_call = ws.receive_json()
        print("tool_call ->", tool_call)
        assert tool_call["type"] == "tool_call"
        assert tool_call["tool_id"] == "create_freecad_box"
        assert tool_call["arguments"] == {"length": "60", "width": "40", "height": "20"}

        tool_result = ws.receive_json()
        print("tool_result ->", tool_result)
        assert tool_result["type"] == "tool_result"
        assert tool_result["ok"] is True
        assert tool_result["mesh_url"], "expected a mesh_url for a created box"

        mesh_resp = client.get(tool_result["mesh_url"])
        print("GET", tool_result["mesh_url"], "->", mesh_resp.status_code, len(mesh_resp.content), "bytes")
        assert mesh_resp.status_code == 200
        assert len(mesh_resp.content) > 0

        assistant_message = ws.receive_json()
        print("assistant_message ->", assistant_message)
        assert assistant_message["type"] == "assistant_message"

        # An utterance that matches no tool should just get a plain reply,
        # not a crashed connection.
        ws.send_json({"text": "asdkjfh nonsense"})
        no_match = ws.receive_json()
        print("no_match ->", no_match)
        assert no_match["type"] == "assistant_message"

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

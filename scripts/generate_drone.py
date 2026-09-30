"""Drive one Dana turn over /ws/chat for a quadcopter frame, then export the
session's FreeCAD macro. Based on the client in DEMO_SCRIPT.md, extended for
an unattended run: answers visual-capture requests (no React canvas here),
prints each tool result, and enforces an overall timeout.

Usage (backend already running on localhost:8000):
    python scripts/generate_drone.py [--url ws://localhost:8000/ws/chat] [--timeout 1500]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import websockets

PROMPT = (
    "Design a quadcopter drone frame body in an X-configuration. The frame should have a 210mm "
    "wheelbase. The center body must accommodate a standard 30.5 x 30.5 mm flight controller, and "
    "the four arms should radiate outward, ending in motor mounting plates that each have a 5mm "
    "center shaft hole."
)


def _short(value: object, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def run(url: str, filename: str, prompt: str) -> int:
    started = time.monotonic()
    async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
        ready = json.loads(await ws.recv())
        print(f"[drone] connected, session {ready['session_id']}")
        print(f"[drone] > {prompt}\n")
        await ws.send(json.dumps({"text": prompt}))

        tool_calls = 0
        while True:
            msg = json.loads(await ws.recv())
            kind = msg.get("type")
            elapsed = f"{time.monotonic() - started:6.1f}s"
            if kind == "tool_dispatch_start":
                tool_calls += 1
                print(f"{elapsed}  -> {msg.get('tool_name')}  {_short(msg.get('args_summary') or '')}")
            elif kind == "tool_dispatch_end":
                detail = msg.get("message") or (msg.get("output") or {}).get("error") or ""
                print(f"{elapsed}     {msg.get('status')}: {_short(detail)}")
            elif kind == "hitl_approval_required":
                payload = msg["payload"]
                print(f"{elapsed}  [auto-approve] {payload['action_name']}")
                await ws.send(json.dumps(
                    {"type": "hitl_response", "payload": {"request_id": payload["request_id"], "approved": True}}
                ))
            elif kind == "visual_capture_request":
                # take_canvas_screenshot needs the React/R3F canvas, which this client doesn't have.
                print(f"{elapsed}  [visual capture] no canvas in this client; declining")
                await ws.send(json.dumps({
                    "type": "visual_capture_response",
                    "payload": {"request_id": msg["payload"]["request_id"], "error": "no canvas available in headless client"},
                }))
            elif kind == "assistant_message":
                print(f"\n[dana] {msg['content']}\n")
                break

        print(f"[drone] turn finished after {tool_calls} tool call(s), {time.monotonic() - started:.0f}s")
        await ws.send(json.dumps({"type": "export_python_script", "filename": filename}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "python_script_exported":
                print(f"[drone] macro written to {msg['path']}")
                return 0
            if msg.get("type") == "assistant_message":  # nothing to export
                print(f"[drone] {msg['content']}")
                return 1


def main() -> int:
    # Replies contain non-cp1252 characters (e.g. "→"); a redirected Windows
    # console would otherwise crash here before the macro export is requested.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="ws://localhost:8000/ws/chat")
    parser.add_argument("--filename", default="drone_frame")
    parser.add_argument("--timeout", type=float, default=1500.0, help="overall seconds before giving up")
    parser.add_argument("--prompt", default=PROMPT, help="override the design prompt")
    args = parser.parse_args()
    try:
        return asyncio.run(asyncio.wait_for(run(args.url, args.filename, args.prompt), timeout=args.timeout))
    except asyncio.TimeoutError:
        print(f"[drone] gave up after {args.timeout:.0f}s")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

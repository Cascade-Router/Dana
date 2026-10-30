"""Backend for the Playwright suite (frontend/playwright.config.ts).

Runs the real ``dana.api.server`` (or, with ``--gradio``, the Hugging Face
Space's ``app.py``) with:

* the trimesh mock CAD engine — what CI and the Space use;
* a throwaway workspace: every ``freecad_output/``, ``exports/`` and
  ``agent_workspace/`` path is redirected to a temp directory
  (tests/workspace_isolation.py), so a UI test never touches the real ones;
* a scripted LLM instead of a provider — no API key, deterministic turns. It
  streams its replies slowly enough for the UI's streaming to be observable
  and reports token usage and cost, so the CostBar gets real usage_update
  events.

Usage::

    .venv/Scripts/python tests/e2e/mock_backend.py --port 8765
    .venv/Scripts/python tests/e2e/mock_backend.py --gradio --port 7861
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

for _key, _value in {
    "DANA_HEADLESS": "true",
    "DANA_OS_DRY_RUN": "1",
    "DANA_CONTEXT_DISTILL": "0",
}.items():
    os.environ[_key] = _value
for _key in ("DANA_CLOUD_PRIMARY", "DANA_PRINTER_IP", "SPACE_ID"):
    os.environ.pop(_key, None)

MODEL = "e2e/scripted-model"
USAGE = {"prompt_tokens": 1200, "completion_tokens": 60}
CHUNK_DELAY_S = 0.06

# Keyword in the user's message -> (plan tasks, the tool calls to make in
# order, the final streamed reply). The scripted model drives the same
# gates a real one must pass: it loads the CAD capability, creates a plan
# and loads any tool it isn't offered yet (tasks complete automatically
# once their expected tools have run).
SCRIPTS: dict[str, tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]], str]] = {
    "plate": (
        [{"description": "Build the chamfered plate", "expected_tools": ["create_freecad_box", "perform_freecad_edge_operation"]}],
        [
            ("create_freecad_box", {"name": "Plate", "length": 60, "width": 40, "height": 5}),
            (
                "perform_freecad_edge_operation",
                {"operation": "chamfer", "target_object": "Plate", "value": 1, "face_centroid": [30, 20, 5]},
            ),
        ],
        "Built a 60 x 40 x 5 mm plate and chamfered its top edges by 1 mm. "
        "Every dimension is editable in the Parameters tab.",
    ),
    "bom": (
        [{"description": "Model and cost the stand", "expected_tools": [
            "create_freecad_box", "create_freecad_cylinder", "create_freecad_assembly", "add_parts_to_assembly",
            "generate_assembly_bom"]}],
        [
            ("create_freecad_box", {"name": "Base", "length": 40, "width": 40, "height": 10}),
            ("create_freecad_cylinder", {"name": "Post", "radius": 5, "height": 30,
                                         "placement_x": 20, "placement_y": 20, "placement_z": 10}),
            ("create_freecad_assembly", {"name": "Stand"}),
            ("add_parts_to_assembly", {"assembly_name": "Stand", "part_names": ["Base", "Post"]}),
            ("generate_assembly_bom", {"assembly_name": "Stand"}),
        ],
        "The stand's bill of materials is ready: two parts in PLA.",
    ),
}


def _offered(tools: Any) -> set[str]:
    return {(t.get("function") or {}).get("name") or t.get("name") for t in tools or [] if isinstance(t, dict)}


def _issued(messages: list[dict[str, Any]]) -> list[str]:
    return [
        call["function"]["name"]
        for m in messages
        if m.get("role") == "assistant"
        for call in m.get("tool_calls") or []
    ]


class ScriptedProvider:
    """Stands in for dana.core.model_provider.ModelProvider."""

    def complete_with_tool_calls(self, messages: list[dict[str, Any]], *, tools: Any, provider: Any = None,
                                 **kwargs: Any) -> dict[str, Any]:
        from dana.core import openai_tool_bridge as bridge
        from dana.tools.schema import ToolCall

        text = next((str(m.get("content")) for m in messages if m.get("role") == "user"), "").lower()
        tasks, calls, reply = next(
            (s for key, s in SCRIPTS.items() if key in text), ([], [], "Hello from the scripted model.")
        )
        sequence: list[tuple[str, dict[str, Any]]] = []
        if calls:
            sequence = [
                ("load_capability", {"domain": "freecad_full"}),
                ("create_plan", {"objective": text, "tasks": tasks}),
                *calls,
            ]
        issued = [t for t in _issued(messages) if t != "load_specific_tool"]
        result = {"model": MODEL, "provider": "openrouter", "usage": USAGE, "cost_usd": 0.0015}
        on_delta = kwargs.get("on_delta")
        if len(issued) < len(sequence):
            tool_id, arguments = sequence[len(issued)]
            if tool_id not in _offered(tools) and tool_id not in ("load_capability", "create_plan"):
                tool_id, arguments = "load_specific_tool", {"tool_id": tool_id}
            if on_delta:
                on_delta(f"Calling {tool_id}")
                bridge.reset_stream(on_delta)
            return {**result, "content": "", "tool_calls": [ToolCall(tool_id, dict(arguments))]}
        for i in range(0, len(reply), 6):
            if on_delta:
                on_delta(reply[i : i + 6])
            time.sleep(CHUNK_DELAY_S)
        return {**result, "content": reply, "tool_calls": []}


def _install(workspace: Path) -> None:
    import dana.api.server as server
    import dana.core.model_provider as model_provider
    import dana.core.react_dispatch as rd
    import dana.plugins.os.file_system as file_system
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine
    from workspace_isolation import import_path_holders, workspace_redirects

    import_path_holders()
    for module, attr, value in list(workspace_redirects(workspace)):
        setattr(module, attr, value)
    Path(file_system._SANDBOX_ROOT).mkdir(parents=True, exist_ok=True)

    # A developer's .env must not switch providers or the HEADLESS lock back mid-run.
    model_provider.ensure_dotenv_loaded = lambda: None
    server.get_cad_engine = lambda: MockFreeCADEngine()
    server.get_control_plane = lambda: MockControlPlane()
    server._default_auto_approve = lambda: True
    rd.ModelProvider = lambda **_k: ScriptedProvider()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gradio", action="store_true", help="serve app.py (the Space) instead of the API server")
    args = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="dana-e2e-"))
    print(f"[mock_backend] workspace: {workspace}", flush=True)
    try:
        if args.gradio:
            import app  # the Space entry point; imports dana.api.server itself

            _install(workspace)
            app.demo.queue().launch(server_name="127.0.0.1", server_port=args.port, ssr_mode=False)
        else:
            import uvicorn
            from fastapi.middleware.cors import CORSMiddleware

            import dana.api.server as server

            _install(workspace)
            # The suite's own Vite instance (playwright.config.ts) is on 1421,
            # not the 1420 the server's dev CORS list knows.
            server.app.add_middleware(
                CORSMiddleware,
                allow_origins=["http://localhost:1421", "http://127.0.0.1:1421"],
                allow_methods=["*"],
                allow_headers=["*"],
            )
            uvicorn.run(server.app, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()

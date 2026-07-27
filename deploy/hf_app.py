"""Stage 9.2 — Gradio wrapper for Dānā LangGraph on Hugging Face Spaces.

Exposes ``gr.Interface`` (text → text) so Astro's client can POST::

    POST /api/predict
    {"data": ["user prompt"]}

Matches ``website/src/utils/hf_api.ts``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure CAMGRASPER repo root is importable when Space cwd is deploy/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.cloud_bridge import apply_cloud_mode, run_text_command  # noqa: E402

apply_cloud_mode()


def predict(message: str) -> str:
    """Single-turn prediction — Astro / Gradio ``/api/predict`` entrypoint.

    Signature must stay ``(str) -> str`` so the JSON body is ``{"data": [prompt]}``.
    """
    return run_text_command(message)


def build_demo():
    import gradio as gr

    return gr.Interface(
        fn=predict,
        inputs=gr.Textbox(
            lines=3,
            label="Command to Dānā",
            placeholder="Type a command to Dānā…",
        ),
        outputs=gr.Textbox(lines=8, label="Dānā"),
        title="Dānā — LangGraph Cloud",
        description=(
            "Hugging Face Space wrapper for the Dānā ReAct / LangGraph backend. "
            "Desktop vision and OS actuators are mocked. "
            "REST: POST /api/predict with {\"data\": [prompt]}."
        ),
        flagging_mode="never",
        analytics_enabled=False,
    )


demo = build_demo()


if __name__ == "__main__":
    # Spaces inject PORT; local smoke uses 7860. share=False — HF hosts ingress.
    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or "7860")
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )

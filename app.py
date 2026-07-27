"""Hugging Face Spaces entrypoint (Stage 9.2).

Spaces that expect ``app.py`` at the repo root import the Gradio ``demo``
from ``deploy/hf_app.py``. Astro REST client: POST ``/api/predict`` with
``{"data": [prompt]}``.
"""

from __future__ import annotations

from deploy.hf_app import demo, predict

__all__ = ["demo", "predict"]

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or "7860")
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )

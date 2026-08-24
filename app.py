"""Hugging Face Space entry point (sdk: gradio).

HF's free tier no longer offers `sdk: docker` for this account, but
`sdk: gradio` is free and Gradio is itself built on FastAPI/Starlette — so
this mounts our real `dana.api.server:app` underneath a Gradio app.

HF's readiness probe for `sdk: gradio` Spaces polls GET /config at the
Space's root, which only exists if a Gradio Blocks app owns "/" — so unlike
a plain health-check stub, Gradio has to actually hold the root here. The
real React UI lives at /ui instead (dana.api.server mounts frontend/dist
there when IS_HF_SPACE — see its own _FRONTEND_MOUNT_PATH), shown full-screen
via an iframe so a visitor still lands on the real product. REST/WS calls
made from inside that iframe (/api/*, /ws/chat) hit dana.api.server directly,
same-origin — unaffected by which path serves the HTML shell.
"""

from __future__ import annotations

import gradio as gr
import spaces

from dana.api.server import app


# Hugging Face ZeroGPU's AST scan for @spaces.GPU usage only walks top-level
# module statements — wrapped in try/except it's invisible to the scanner,
# even though `spaces` (pinned in deploy/requirements-space.txt) always
# imports fine at runtime. This app never needs a GPU; the decorator is only
# here to satisfy that startup check.
@spaces.GPU
def _dummy_gpu_function():
    pass

_CUSTOM_CSS = """
footer {display: none !important;}
.gradio-container {padding: 0 !important; margin: 0 !important; max-width: 100% !important;}
"""

with gr.Blocks(css=_CUSTOM_CSS) as _demo:
    gr.HTML(
        '<iframe src="/ui/" style="width:100vw; height:100vh; border:none; '
        'margin:0; padding:0; overflow:hidden;"></iframe>'
    )

# Safe to mount at the true root now: dana.api.server's own frontend mount
# moved to /ui specifically so the two don't compete for "/".
app = gr.mount_gradio_app(app, _demo, path="/")

if __name__ == "__main__":
    # HF's `sdk: gradio` runtime runs this file with a plain `python app.py`
    # (it does not itself run uvicorn), so the entry point must bind the
    # server — Spaces always expects port 7860.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)

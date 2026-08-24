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


# ZeroGPU is only compatible with the Gradio SDK (per HF's own docs) and
# ties into Gradio's *registered event handlers* — a @spaces.GPU function
# that's never bound to a Gradio component/event never shows up in the
# Blocks' own dependency list, so ZeroGPU's startup check can't find it even
# though it's a plain top-level decorated function. Binding it to the page's
# `load` event below (fires once per visitor, does nothing) is what actually
# registers it. This app never needs a GPU otherwise.
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
    _demo.load(_dummy_gpu_function)

# Safe to mount at the true root now: dana.api.server's own frontend mount
# moved to /ui specifically so the two don't compete for "/".
app = gr.mount_gradio_app(app, _demo, path="/")

if __name__ == "__main__":
    # HF's `sdk: gradio` runtime runs this file with a plain `python app.py`
    # (it does not itself run uvicorn), so the entry point must bind the
    # server — Spaces always expects port 7860.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)

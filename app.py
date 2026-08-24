"""Hugging Face Space entry point (sdk: gradio).

HF's free tier no longer offers `sdk: docker` for this account, but
`sdk: gradio` is free and Gradio is itself built on FastAPI/Starlette — so
this mounts a trivial Gradio app onto our real `dana.api.server:app` purely
to satisfy the Space's Gradio health check, while the actual traffic (the
React UI + REST/WS API) is served by the existing FastAPI app underneath.

``dana.api.server`` already mounts the built ``frontend/dist`` at ``/`` when
present (see its own ``_FRONTEND_DIST`` block) and already branches on
``dana.platform.factory.IS_HF_SPACE`` to swap in mock CAD/control-plane
drivers, so importing it here is enough to get the full app.
"""

from __future__ import annotations

import gradio as gr

from dana.api.server import app

_demo = gr.Blocks()
with _demo:
    gr.Markdown("# Dānā backend is running.\n\nThe app itself is served at the Space root.")

app = gr.mount_gradio_app(app, _demo, path="/gradio")

# `mount_gradio_app` appends the "/gradio" mount to the END of app.routes.
# dana.api.server already mounts the built frontend at "/" (a catch-all that
# matches every path, added when frontend/dist exists) — routes are matched
# in list order, so as appended "/gradio" would sit AFTER that catch-all and
# never be reached. Move it to just before the "/" mount so both are live.
_routes = app.router.routes
_gradio_route = _routes.pop()
_root_index = next((i for i, r in enumerate(_routes) if getattr(r, "path", None) == ""), len(_routes))
_routes.insert(_root_index, _gradio_route)

if __name__ == "__main__":
    # HF's `sdk: gradio` runtime runs this file with a plain `python app.py`
    # (it does not itself run uvicorn), so the entry point must bind the
    # server — Spaces always expects port 7860.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)

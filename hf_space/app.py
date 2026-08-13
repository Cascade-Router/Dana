"""Dana AI Copilot — Hugging Face Space entry point.

Thin launcher only: the actual UI lives in ``dana.ui.unified_app`` so this
Space and ``scripts/launchers/launch_gradio_local.py`` run the exact same
Gradio codebase. Tool dispatch there always goes through
``dana.platform.get_control_plane()`` / ``get_cad_engine()``, which resolve
to the mock drivers here (``SPACE_ID`` is set on every HF Space) and to the
real Win32/FreeCAD drivers on a local desktop.

``dana/`` is staged alongside this file by ``.github/workflows/deploy_hf.yml``
— it is not a subpath import trick, it is a sibling package at the Space's
repo root.
"""

from __future__ import annotations

from dana.ui.unified_app import demo

if __name__ == "__main__":
    demo.launch(ssr_mode=False)

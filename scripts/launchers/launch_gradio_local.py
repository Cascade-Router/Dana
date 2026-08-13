"""Local launcher for the unified Gradio UI (``dana/ui/unified_app.py``).

Boots the exact same Gradio app that runs on the Hugging Face Space
(``hf_space/app.py``) — same code, real Win32/FreeCAD drivers instead of the
mock ones, since ``dana.platform.factory`` picks the driver from the
environment (no ``SPACE_ID`` + ``sys.platform == "win32"`` ->
``Win32ControlPlane`` / ``RealFreeCADEngine``).

Usage:
    python scripts/launchers/launch_gradio_local.py
    python scripts/launchers/launch_gradio_local.py --app
    python scripts/launchers/launch_gradio_local.py --host 0.0.0.0 --port 7861
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host (default {DEFAULT_HOST})")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="Open in a native desktop window (pywebview) as a container wrapper around the "
        "local Gradio server, instead of a browser tab. Falls back to opening a browser tab "
        "with a warning if pywebview isn't installed.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open a browser tab (ignored with --app).",
    )
    return parser.parse_args(argv)


def _launch_in_native_window(url: str) -> None:
    try:
        import webview
    except ImportError:
        print(
            "[launch_gradio_local] --app requires the 'pywebview' package "
            "(pip install pywebview) — falling back to opening a browser tab instead.",
            file=sys.stderr,
        )
        import webbrowser

        webbrowser.open(url)
        return
    webview.create_window("Dana AI Co-Pilot", url)
    webview.start()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from dana.ui.unified_app import demo

    if args.app:
        # The Gradio server must be up before pywebview can point at it —
        # launch without blocking, hand the URL to the native window, then
        # block on the server thread once the window is showing.
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            inbrowser=False,
            prevent_thread_lock=True,
            ssr_mode=False,
        )
        _launch_in_native_window(f"http://{args.host}:{args.port}")
        demo.block_thread()
        return 0

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        ssr_mode=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

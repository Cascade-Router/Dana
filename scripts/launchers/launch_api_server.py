"""Local launcher for the headless API (``dana/api/server.py``).

Boots the Uvicorn/FastAPI server that the Tauri/React frontend talks to over
``ws://.../ws/chat``. Real Win32/FreeCAD drivers run here exactly as they did
under the now-deleted Gradio UI (``dana.platform.factory`` picks the driver
from the environment, not from which launcher started the process).

Usage:
    python scripts/launchers/launch_api_server.py
    python scripts/launchers/launch_api_server.py --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host (default {DEFAULT_HOST})")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Autoreload on source change (dev only)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import uvicorn

    uvicorn.run(
        "dana.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(_ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

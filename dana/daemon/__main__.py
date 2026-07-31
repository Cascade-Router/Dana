"""``python -m dana.daemon`` entrypoint."""

from __future__ import annotations

from dana.daemon.engine import main

if __name__ == "__main__":
    raise SystemExit(main())

"""python -m dana.cli → monitor by default."""

from __future__ import annotations

from dana.cli.monitor import main

if __name__ == "__main__":
    raise SystemExit(main())

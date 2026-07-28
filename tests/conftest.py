"""Pytest bootstrap: keep CAMGRASPER repo root on ``sys.path``.

Tests live under ``tests/``; packages ``dana`` / ``donna_security`` stay at repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_root_s = str(_ROOT)
if _root_s not in sys.path:
    sys.path.insert(0, _root_s)

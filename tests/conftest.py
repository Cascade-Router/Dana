"""Pytest bootstrap: keep CAMGRASPER repo root on ``sys.path``.

Tests live under ``tests/``; packages ``dana`` / ``donna_security`` stay at repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_root_s = str(_ROOT)
if _root_s not in sys.path:
    sys.path.insert(0, _root_s)
for _sub in ("scripts", "scripts/diagnostics"):
    _p = str(_ROOT / _sub)
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "asyncio: async test body (executed via asyncio.wait_for)",
    )


@pytest.fixture(autouse=True)
def _stub_premium_logo_unless_stage899(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Avoid cross-test Tk PhotoImage / CTkImage ``pyimage`` collisions.

    Stage 8.9.9 explicitly asserts CTkImage wiring and keeps real loaders.
    """
    nodeid = getattr(request.node, "nodeid", "") or ""
    if "stage899" in nodeid or "premium_logo" in nodeid:
        return

    monkeypatch.setattr("dana.ui.logo.load_premium_logo", lambda *_a, **_k: None)
    monkeypatch.setattr("dana.ui.logo.apply_window_icon", lambda *_a, **_k: False)
    monkeypatch.setattr("dana.ui.logo.force_apply_window_icon", lambda *_a, **_k: False)
    monkeypatch.setattr("dana.ui.logo.schedule_window_icon", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "dana.ui.logo.load_premium_logo_photoimage",
        lambda *_a, **_k: None,
    )

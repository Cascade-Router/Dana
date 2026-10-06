"""scripts/launchers/launch_api_server.py installs the process hooks the old
run.py entry point used to (crash log, Windows console hiding) before it
starts uvicorn."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "launchers" / "launch_api_server.py"


def _load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("launch_api_server_under_test", LAUNCHER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_installs_crash_and_process_hooks_before_starting_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    import dana.logging
    import dana.paths

    order: list[str] = []
    monkeypatch.setattr(dana.logging, "install_fatal_crash_hooks", lambda: order.append("crash_hooks"))
    monkeypatch.setattr(dana.paths, "apply_windows_process_hardening", lambda: order.append("hardening"))

    def fake_run(app: str, **kwargs: Any) -> None:
        order.append("uvicorn")
        assert app == "dana.api.server:app"
        assert kwargs["port"] == 8123

    monkeypatch.setattr(uvicorn, "run", fake_run)
    assert _load_launcher().main(["--port", "8123"]) == 0
    assert order == ["crash_hooks", "hardening", "uvicorn"]


def test_crash_hooks_write_unhandled_exceptions_to_the_fatal_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import threading

    import dana.logging as dana_logging

    log = tmp_path / "fatal_crash.log"
    monkeypatch.setattr(dana_logging, "FATAL_CRASH_LOG_PATH", str(log))
    monkeypatch.setattr(dana_logging, "_fatal_hooks_installed", False)
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    monkeypatch.setattr(threading, "excepthook", lambda args: None)

    dana_logging.install_fatal_crash_hooks()
    try:
        raise RuntimeError("boom in main")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    assert "boom in main" in log.read_text(encoding="utf-8")

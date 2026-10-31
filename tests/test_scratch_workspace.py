"""scripts/scratch_workspace.py: the manual CAD scripts run in a temp
workspace and a scratch session, never the user's default session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dana.paths import PROJECT_ROOT
from dana.session_context import DEFAULT_SESSION_ID, get_session_id, set_session_id

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import scratch_workspace  # noqa: E402

# Every script that runs CAD builds or a ReAct loop by hand.
_CAD_SCRIPTS = (
    "run_e2e_cad.py",
    "run_kobayashi_maru.py",
    "showcase_freecad_demo.py",
    "test_api_server_smoke.py",
    "test_cad_vision_live.py",
    "test_uv_tensor.py",
)


@pytest.mark.parametrize("keep", [False, True])
def test_scratch_workspace_redirects_and_restores(monkeypatch: pytest.MonkeyPatch, keep: bool) -> None:
    from dana.plugins.freecad import engine

    real = PROJECT_ROOT / "freecad_output"
    monkeypatch.setattr(engine, "_OUTPUT_DIR", real)  # what a script sees outside pytest's own isolation
    set_session_id(DEFAULT_SESSION_ID)
    with scratch_workspace.scratch_workspace("unit", keep=keep) as session:
        root = engine._OUTPUT_DIR.parent
        assert engine._OUTPUT_DIR == root / "freecad_output" and not engine._OUTPUT_DIR.is_relative_to(real)
        assert session == get_session_id() != DEFAULT_SESSION_ID
    assert engine._OUTPUT_DIR == real
    assert get_session_id() == DEFAULT_SESSION_ID
    assert root.exists() is keep
    if keep:
        import shutil

        shutil.rmtree(root)


def test_scratch_workspace_uses_the_requested_session_id() -> None:
    with scratch_workspace.scratch_workspace("unit", session_id="debug-run") as session:
        assert session == get_session_id() == "debug-run"


@pytest.mark.parametrize("script", _CAD_SCRIPTS)
def test_cad_scripts_run_in_a_scratch_workspace(script: str) -> None:
    source = (PROJECT_ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert "scratch_workspace.scratch_workspace(" in source
    assert "scratch_workspace.add_arguments(parser" in source

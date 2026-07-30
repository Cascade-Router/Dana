"""Transactional ShadowWorkspace + fatal vs fixable REPL error classification."""

from __future__ import annotations

from pathlib import Path

import pytest

from dana.agentic_react_graph import route_after_execution
from dana.exec.shadow_workspace import (
    ShadowWorkspace,
    run_shadow_transaction,
)
from dana.graph.nodes.critic import (
    FATAL_OS_BLOCK_MSG,
    fail_closed_node,
    is_fatal_execution_error,
    is_fixable_execution_error,
    python_repl_state_patch,
)


def test_zero_division_rollback_and_critic_path(tmp_path: Path) -> None:
    """ZeroDivisionError → rollback scratch; Critic route (not fatal)."""
    dest = tmp_path / "out" / "result.txt"
    scratch_base = tmp_path / "scratch"

    def runner(ws: ShadowWorkspace) -> tuple[int, str]:
        ws.stage_write(dest, "should_not_land")
        obs = (
            "exit_code=1\nstdout:\n(empty)\nstderr:\n"
            "ZeroDivisionError: division by zero"
        )
        return 1, obs

    ws, code, obs = run_shadow_transaction(
        "sess-zdiv",
        runner,
        base_dir=scratch_base,
    )
    assert code == 1
    assert "ZeroDivisionError" in obs
    assert not dest.exists(), "rollback must not modify destinations"
    assert not (scratch_base / "sess-zdiv").exists(), "scratch cleaned on rollback"
    assert ws._rolled_back is True

    patch = python_repl_state_patch(code="print(1/0)", observation=obs)
    assert patch["execution_error"]
    assert patch.get("fatal_block") is False
    assert is_fixable_execution_error(obs)
    assert not is_fatal_execution_error(obs)
    assert (
        route_after_execution(
            {
                **patch,
                "retry_count": 0,
                "max_retries": 3,
            }
        )
        == "critic"
    )


def test_permission_error_fatal_block_halts_retries(tmp_path: Path) -> None:
    """PermissionError → fatal_block=True; bypass Critic; fail_closed path."""
    dest = tmp_path / "out" / "secret.txt"
    scratch_base = tmp_path / "scratch"

    def runner(ws: ShadowWorkspace) -> tuple[int, str]:
        ws.stage_write(dest, "staged_only")
        obs = (
            "exit_code=1\nstdout:\n(empty)\nstderr:\n"
            "PermissionError: [Errno 13] Permission denied: '/etc/shadow'"
        )
        return 1, obs

    ws, code, obs = run_shadow_transaction(
        "sess-perm",
        runner,
        base_dir=scratch_base,
    )
    assert code == 1
    assert not dest.exists()
    assert ws._rolled_back is True

    patch = python_repl_state_patch(code="open('/etc/shadow')", observation=obs)
    assert patch["fatal_block"] is True
    assert is_fatal_execution_error(obs)
    assert not is_fixable_execution_error(obs)
    assert (
        route_after_execution(
            {
                **patch,
                "retry_count": 0,
                "max_retries": 3,
            }
        )
        == "fail_closed"
    )
    # Even with retries remaining, fatal never routes to critic.
    assert (
        route_after_execution(
            {
                "execution_error": obs,
                "fatal_block": True,
                "retry_count": 0,
                "max_retries": 99,
            }
        )
        == "fail_closed"
    )

    ledger = tmp_path / "dana_security" / "patch_ledger.md"
    closed = fail_closed_node(
        {
            "execution_error": obs,
            "fatal_block": True,
            "session_id": "sess-perm",
            "critique_history": [],
            "retry_count": 0,
            "patch_ledger_path": str(ledger),
        }
    )
    assert closed.get("fatal_block") is True
    assert closed.get("halt") is True
    assert closed.get("final_raw") == FATAL_OS_BLOCK_MSG
    assert isinstance(closed.get("drafted_ticket"), dict)
    assert "Fatal OS Block" in str(closed["drafted_ticket"].get("objective") or "")
    assert ledger.is_file()
    assert "[PENDING]" in ledger.read_text(encoding="utf-8")


def test_successful_script_commits_staged_files(tmp_path: Path) -> None:
    """exit_code == 0 → commit staged files onto destinations."""
    dest = tmp_path / "out" / "ok.txt"
    scratch_base = tmp_path / "scratch"
    body = "committed payload"

    def runner(ws: ShadowWorkspace) -> tuple[int, str]:
        ws.stage_write(dest, body)
        assert not dest.exists(), "dest untouched until commit"
        assert ws.map_path(dest).is_file()
        return 0, "exit_code=0\nstdout:\nok"

    ws, code, obs = run_shadow_transaction(
        "sess-ok",
        runner,
        base_dir=scratch_base,
    )
    assert code == 0
    assert "exit_code=0" in obs
    assert dest.is_file()
    assert dest.read_text(encoding="utf-8") == body
    assert not (scratch_base / "sess-ok").exists(), "scratch cleared after commit"
    assert ws._committed is True

    patch = python_repl_state_patch(code="print('ok')", observation=obs)
    assert patch["execution_error"] is None
    assert patch.get("fatal_block") is False
    assert route_after_execution({**patch, "halt": True}) == "verifier"


def test_module_not_found_is_fatal() -> None:
    obs = "exit_code=1\nstderr:\nModuleNotFoundError: No module named 'torch'"
    assert is_fatal_execution_error(obs)
    patch = python_repl_state_patch(code="import torch", observation=obs)
    assert patch["fatal_block"] is True
    assert route_after_execution({**patch, "retry_count": 0, "max_retries": 3}) == (
        "fail_closed"
    )

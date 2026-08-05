"""Transactional file_editor staging + supervisor verify/rollback."""

from __future__ import annotations

from pathlib import Path

from dana.graph.nodes.supervisor import supervisor_node
from dana.graph.nodes.worker import run_worker
from dana.graph.state import empty_supervisor_state, empty_worker_state
from dana.paths import PROJECT_ROOT
from dana.tools.file_editor import (
    begin_staging_session,
    file_editor,
    rollback_workspace,
    transactional_file_tool,
    verify_and_commit,
)

_DIR = Path(PROJECT_ROOT) / "logs" / "file_editor_checkpoint"


def _rel(name: str) -> str:
    _DIR.mkdir(parents=True, exist_ok=True)
    return (_DIR / name).relative_to(Path(PROJECT_ROOT)).as_posix()


def test_staging_write_does_not_touch_live_until_commit() -> None:
    rel = _rel("stage_ok.py")
    live = Path(PROJECT_ROOT) / rel
    live.write_text("ORIGINAL = 1\n", encoding="utf-8")
    sid = "ckpt-stage-ok"
    tool = transactional_file_tool(sid)
    out = tool("write", rel, "ORIGINAL = 2\n")
    assert "shadow staged" in out
    assert live.read_text(encoding="utf-8") == "ORIGINAL = 1\n"

    commit = verify_and_commit(sid)
    assert commit.startswith("OK: committed")
    assert live.read_text(encoding="utf-8") == "ORIGINAL = 2\n"
    print("[PASS] staging_write_does_not_touch_live_until_commit")


def test_bad_syntax_verify_rolls_back() -> None:
    rel = _rel("stage_bad.py")
    live = Path(PROJECT_ROOT) / rel
    live.write_text("SAFE = True\n", encoding="utf-8")
    sid = "ckpt-stage-bad"
    tool = transactional_file_tool(sid)
    tool("write", rel, "def broken(:\n    pass\n")
    assert live.read_text(encoding="utf-8") == "SAFE = True\n"

    result = verify_and_commit(sid)
    assert result.startswith("ERROR: verify failed")
    assert live.read_text(encoding="utf-8") == "SAFE = True\n"
    # Session cleared after rollback.
    assert rollback_workspace(sid).startswith("OK:")
    print("[PASS] bad_syntax_verify_rolls_back")


def test_supervisor_rolls_back_failed_worker() -> None:
    rel = _rel("worker_fail.py")
    live = Path(PROJECT_ROOT) / rel
    live.write_text("KEEP = 1\n", encoding="utf-8")
    sid = "ckpt-worker-fail"
    begin_staging_session(sid)
    file_editor("write", rel, "KEEP = 99\n", staging_session=sid)
    assert live.read_text(encoding="utf-8") == "KEEP = 1\n"

    state = empty_supervisor_state("edit worker_fail.py")
    state["dag"] = [
        {
            "task_id": 1,
            "action": f"Edit {rel}",
            "dependencies": [],
            "status": "running",
            "summary": "",
            "error": "",
            "attempts": 0,
        }
    ]
    state["pending_tasks"] = [1]
    state["open_staging_sessions"] = [sid]
    state["worker_results"] = [
        {
            "task_id": 1,
            "status": "failed",
            "summary": "",
            "error": "malformed worker state",
            "staging_session_id": sid,
        }
    ]
    out = supervisor_node(state)
    assert live.read_text(encoding="utf-8") == "KEEP = 1\n"
    assert any("rolled back" in str(x).lower() for x in (out.get("checkpoint_log") or []))
    print("[PASS] supervisor_rolls_back_failed_worker")


def test_supervisor_commits_successful_worker() -> None:
    rel = _rel("worker_ok.py")
    live = Path(PROJECT_ROOT) / rel
    live.write_text("KEEP = 1\n", encoding="utf-8")

    worker = empty_worker_state(7, f"Edit {rel} to bump KEEP")
    finished = run_worker(
        worker,
        edit_content="KEEP = 2\n",
        staging_session_id="ckpt-worker-ok",
    )
    assert finished["status"] == "completed"
    assert live.read_text(encoding="utf-8") == "KEEP = 1\n", "pre-commit live must stay clean"

    state = empty_supervisor_state(f"Edit {rel}")
    state["dag"] = [
        {
            "task_id": 7,
            "action": finished["instructions"],
            "dependencies": [],
            "status": "running",
            "summary": "",
            "error": "",
            "attempts": 0,
        }
    ]
    state["pending_tasks"] = [7]
    state["open_staging_sessions"] = [str(finished.get("staging_session_id") or "")]
    state["worker_results"] = [
        {
            "task_id": 7,
            "status": finished["status"],
            "summary": finished["summary"],
            "error": "",
            "staging_session_id": finished.get("staging_session_id"),
        }
    ]
    out = supervisor_node(state)
    assert live.read_text(encoding="utf-8") == "KEEP = 2\n"
    assert any("committed" in str(x).lower() for x in (out.get("checkpoint_log") or []))
    print("[PASS] supervisor_commits_successful_worker")


if __name__ == "__main__":
    test_staging_write_does_not_touch_live_until_commit()
    test_bad_syntax_verify_rolls_back()
    test_supervisor_rolls_back_failed_worker()
    test_supervisor_commits_successful_worker()
    print("\nAll file_editor checkpoint tests passed.")

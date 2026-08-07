"""Meta-Broker + closed-loop runtime harness tests (hermetic)."""

from __future__ import annotations

from pathlib import Path

from dana.graph.builder import compile_meta_broker_graph, run_meta_broker
from dana.graph.nodes.broker import (
    broker_node,
    heuristic_split_epics,
    route_after_broker,
)
from dana.graph.runtime_harness import run_validation_harness
from dana.graph.state import empty_broker_state
from dana.paths import PROJECT_ROOT


def test_run_validation_harness_captures_subprocess(tmp_path: Path) -> None:
    ok = run_validation_harness(
        str(tmp_path),
        'python -c "print(123)"',
    )
    assert ok["success"] is True
    assert ok["exit_code"] == 0
    assert "123" in ok["stdout"]

    bad = run_validation_harness(
        str(tmp_path),
        'python -c "import sys; print(\'boom\', file=sys.stderr); sys.exit(1)"',
    )
    assert bad["success"] is False
    assert bad["exit_code"] == 1
    assert "boom" in bad["stderr"]


def test_heuristic_split_epics_multi_epic_prompt() -> None:
    prompt = """
Epic 1: Refactor auth helpers in dana/auth_helpers.py
Update login validation and token refresh.

Epic 2: Refactor storage layer in dana/storage_layer.py
Normalize path joins and add retries.

Epic 3: Add integration tests in tests/test_auth_storage.py
Cover login + storage happy path.
"""
    epics = heuristic_split_epics(prompt)
    assert len(epics) == 3
    assert epics[0]["epic_id"] == 1
    assert "auth" in epics[0]["goal"].lower()
    assert "storage" in epics[1]["goal"].lower()
    assert "test" in epics[2]["goal"].lower()


def test_broker_node_plans_and_dispatches_isolated_epic() -> None:
    prompt = """
Epic 1: Edit execution_jail/meta_broker_a.py to set A = 1
Epic 2: Edit execution_jail/meta_broker_b.py to set B = 2
"""
    state = empty_broker_state(prompt)
    out = broker_node(state)
    assert len(out.get("epics") or []) == 2
    assert out.get("broker_phase") == "await_supervisor"
    assert "meta_broker_a" in str(out.get("user_prompt") or "")
    # Isolated window: no global history, empty DAG until supervisor plans.
    assert out.get("global_conversation_history") == []
    assert out.get("dag") == []
    assert route_after_broker(out) == "supervisor"  # type: ignore[arg-type]


def test_failing_harness_triggers_repair_iteration(tmp_path: Path) -> None:
    """Opt-in repair path still works when max_repair_attempts > 0."""
    workspace = Path(PROJECT_ROOT) / "execution_jail" / "meta_broker_harness"
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "widget.py"
    target.write_text("VALUE = 0\n", encoding="utf-8")
    rel = target.relative_to(Path(PROJECT_ROOT)).as_posix()

    harness_calls: list[dict] = []

    def harness(workspace_path: str, command: str, timeout_s: float = 120.0) -> dict:
        n = len(harness_calls) + 1
        if n == 1:
            result = {
                "success": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "AssertionError: expected VALUE == 1\n  at test_widget.py:3",
            }
        else:
            result = {
                "success": True,
                "exit_code": 0,
                "stdout": "1 passed",
                "stderr": "",
            }
        harness_calls.append({"command": command, **result})
        return result

    def planner(prompt: str):
        return [
            {
                "task_id": 1,
                "action": f"Edit {rel} for: {prompt[:120]}",
                "dependencies": [],
            }
        ]

    def tool_fn(action: str, filepath: str, content: str | None = None) -> str:
        key = filepath.replace("\\", "/")
        if action == "read":
            body = target.read_text(encoding="utf-8") if target.is_file() else ""
            return f"OK: read {key}\n{body}"
        target.write_text(str(content or "VALUE = 1\n"), encoding="utf-8")
        return f"OK: write {key} (shadow staged)"

    macro = f"""
Epic 1: Edit {rel} so VALUE equals 1 and tests pass.
"""
    final = run_meta_broker(
        macro,
        planner=planner,
        tool_fn=tool_fn,
        harness_fn=harness,
        workspace_path=str(workspace),
        validation_command="python -c \"print('validate')\"",
        max_repair_attempts=3,
    )

    assert len(harness_calls) >= 2, "expected fail then pass harness cycle"
    assert harness_calls[0]["success"] is False
    assert any(c["success"] for c in harness_calls[1:])
    epics = final.get("epics") or []
    assert epics, "broker should retain epic list"
    # Repair attempt recorded on the epic.
    assert int(epics[0].get("repair_attempts") or 0) >= 1
    assert str(epics[0].get("status")) == "completed"
    assert str(final.get("status")) == "completed"
    log = " ".join(str(x) for x in (final.get("epic_log") or []))
    assert "repair" in log.lower()


def test_single_pass_fail_fast_aborts_and_rolls_back(tmp_path: Path) -> None:
    """Default max_repair_attempts=0 → ABORTED + delete unvalidated artifacts."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = workspace / "broken_module.py"

    harness_calls: list[dict] = []

    def harness(workspace_path: str, command: str, timeout_s: float = 15.0) -> dict:
        harness_calls.append({"command": command})
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": "AssertionError: boom",
            "tracked_files": ["broken_module.py"],
            "run_key": "epic-1",
        }

    def planner(prompt: str):
        return [
            {
                "task_id": 1,
                "action": "Edit broken_module.py",
                "dependencies": [],
            }
        ]

    def tool_fn(action: str, filepath: str, content: str | None = None) -> str:
        if action == "read":
            body = target.read_text(encoding="utf-8") if target.is_file() else ""
            return f"OK: read\n{body}"
        target.write_text(str(content or "x=1\n"), encoding="utf-8")
        return "OK: write broken_module.py (shadow staged)"

    macro = """
Epic 1: Write broken_module.py with a trivial function.
"""
    final = run_meta_broker(
        macro,
        planner=planner,
        tool_fn=tool_fn,
        harness_fn=harness,
        workspace_path=str(workspace),
        validation_command="python -c \"raise SystemExit(1)\"",
        max_repair_attempts=0,
    )
    assert len(harness_calls) == 1
    assert str(final.get("status")) == "ABORTED"
    log = " ".join(str(x) for x in (final.get("epic_log") or []))
    assert "ABORTED" in log or "rollback" in log.lower()
    assert not target.exists(), "unvalidated artifact should be rolled back"

def test_meta_broker_runs_epics_sequentially(tmp_path: Path) -> None:
    workspace = Path(PROJECT_ROOT) / "execution_jail" / "meta_broker_seq"
    workspace.mkdir(parents=True, exist_ok=True)
    a = workspace / "a.py"
    b = workspace / "b.py"
    a.write_text("A = 0\n", encoding="utf-8")
    b.write_text("B = 0\n", encoding="utf-8")
    rel_a = a.relative_to(Path(PROJECT_ROOT)).as_posix()
    rel_b = b.relative_to(Path(PROJECT_ROOT)).as_posix()

    seen_prompts: list[str] = []

    def planner(prompt: str):
        seen_prompts.append(prompt)
        # Pick filepath from prompt.
        path = rel_a if rel_a in prompt else rel_b if rel_b in prompt else rel_a
        return [
            {
                "task_id": 1,
                "action": f"Edit {path}",
                "dependencies": [],
            }
        ]

    def tool_fn(action: str, filepath: str, content: str | None = None) -> str:
        p = Path(PROJECT_ROOT) / filepath.replace("\\", "/")
        if action == "read":
            body = p.read_text(encoding="utf-8") if p.is_file() else ""
            return f"OK: read {filepath}\n{body}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(content or "X = 1\n"), encoding="utf-8")
        return f"OK: write {filepath} (shadow staged)"

    def harness(workspace_path: str, command: str, timeout_s: float = 120.0) -> dict:
        return {
            "success": True,
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
        }

    macro = f"""
Epic 1: Edit {rel_a} to set A = 1
Epic 2: Edit {rel_b} to set B = 2
"""
    graph = compile_meta_broker_graph(
        planner=planner,
        tool_fn=tool_fn,
        harness_fn=harness,
    )
    initial = empty_broker_state(
        macro,
        workspace_path=str(workspace),
        validation_command="python -c \"print(1)\"",
    )
    final = graph.invoke(initial, config={"recursion_limit": 80})

    assert len(final.get("epics") or []) == 2
    assert all(str(e.get("status")) == "completed" for e in final["epics"])
    assert str(final.get("status")) == "completed"
    # Isolated sequential dispatch: epic-1 goal appears before epic-2 in prompts.
    assert any(rel_a in p for p in seen_prompts)
    assert any(rel_b in p for p in seen_prompts)
    # First dispatched supervisor prompt should be epic 1 only (isolated window).
    assert rel_a in seen_prompts[0]
    assert rel_b not in seen_prompts[0]


if __name__ == "__main__":
    test_run_validation_harness_captures_subprocess(Path("."))
    test_heuristic_split_epics_multi_epic_prompt()
    test_broker_node_plans_and_dispatches_isolated_epic()
    test_failing_harness_triggers_repair_iteration(Path("."))
    test_meta_broker_runs_epics_sequentially(Path("."))
    print("\nAll meta-broker tests passed.")

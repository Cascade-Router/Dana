"""Supervisor DAG swarm — plan, isolate workers, edit files in 3 steps."""

from __future__ import annotations

from pathlib import Path

from dana.graph.nodes.supervisor import (
    heuristic_plan_dag,
    parse_dag_json,
    ready_task_ids,
    route_after_supervisor,
    supervisor_node,
)
from dana.graph.nodes.worker import build_isolated_worker, run_worker, workers_node
from dana.graph.state import empty_supervisor_state
from dana.paths import PROJECT_ROOT

_FIXTURE_DIR = Path(PROJECT_ROOT) / "logs" / "dag_supervisor_fixtures"


def _cleanup_fixtures() -> None:
    if not _FIXTURE_DIR.exists():
        return
    for p in _FIXTURE_DIR.glob("*"):
        if p.is_file():
            p.unlink()


def test_parse_and_heuristic_dag() -> None:
    raw = """
    [
      {"task_id": 1, "action": "read watchdog_graph.py", "dependencies": []},
      {"task_id": 2, "action": "refactor imports", "dependencies": [1]},
      {"task_id": 3, "action": "write summary.md", "dependencies": [2]}
    ]
    """
    dag = parse_dag_json(raw)
    assert [t["task_id"] for t in dag] == [1, 2, 3]
    assert dag[1]["dependencies"] == [1]

    prompt = (
        "1. Read logs/dag_supervisor_fixtures/alpha.py\n"
        "2. Edit logs/dag_supervisor_fixtures/beta.py to note alpha was read\n"
        "3. Write logs/dag_supervisor_fixtures/summary.md with the outcome\n"
    )
    planned = heuristic_plan_dag(prompt)
    assert len(planned) == 3
    assert planned[0]["dependencies"] == []
    assert planned[1]["dependencies"] == [1]
    assert planned[2]["dependencies"] == [2]
    print("[PASS] parse_and_heuristic_dag")


def test_worker_isolation_no_global_history() -> None:
    state = empty_supervisor_state("read logs/dag_supervisor_fixtures/alpha.py")
    state["global_conversation_history"] = [
        {"role": "user", "content": "SECRET_HISTORY_SHOULD_NOT_LEAK"},
        {"role": "assistant", "content": "prior turn"},
    ]
    state["dag"] = [
        {
            "task_id": 1,
            "action": "read logs/dag_supervisor_fixtures/alpha.py",
            "dependencies": [],
            "status": "running",
            "summary": "",
            "error": "",
            "attempts": 0,
        }
    ]
    worker = build_isolated_worker(state["dag"][0], state)
    assert worker["context_window"] == []
    blob = " ".join(
        str(m.get("content") or "") for m in worker.get("context_window") or []
    )
    assert "SECRET_HISTORY_SHOULD_NOT_LEAK" not in blob

    calls: list[tuple[str, str]] = []

    def fake_tool(action: str, filepath: str, content: str | None = None) -> str:
        calls.append((action, filepath))
        if action == "read":
            return "OK: read alpha.py (5 chars)\nhello"
        return f"OK: write {filepath}"

    def fake_outline(file_path: str) -> str:
        return f"OK: outline {file_path} lang=python symbols=1\nALPHA_MARKER = ..."

    finished = run_worker(worker, tool_fn=fake_tool, outline_fn=fake_outline)
    assert finished["status"] == "completed"
    assert "SECRET_HISTORY_SHOULD_NOT_LEAK" not in str(finished.get("context_window"))
    # Explore hops must prefer outline over full-file read_local_file.
    tools_used = [o.get("tool") for o in finished.get("tool_outputs") or []]
    assert tools_used[:1] == ["get_file_outline"]
    assert "read_local_file" not in tools_used
    assert calls == []  # file_editor/read not consulted for code explore
    print("[PASS] worker_isolation_no_global_history")


def test_three_step_file_editing_swarm() -> None:
    """Complex 3-step file editing prompt → DAG + isolated execution."""
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_fixtures()

    alpha = _FIXTURE_DIR / "alpha.py"
    beta = _FIXTURE_DIR / "beta.py"
    summary = _FIXTURE_DIR / "summary.md"
    alpha.write_text("ALPHA_MARKER = 1\n", encoding="utf-8")
    beta.write_text("BETA_MARKER = 0\n", encoding="utf-8")
    summary.write_text("", encoding="utf-8")

    prompt = (
        "Perform this multi-file codebase edit as a DAG:\n"
        "1. Read logs/dag_supervisor_fixtures/alpha.py\n"
        "2. Edit logs/dag_supervisor_fixtures/beta.py to record that alpha was inspected\n"
        "3. Write logs/dag_supervisor_fixtures/summary.md with a short completion note\n"
    )

    store: dict[str, str] = {
        "logs/dag_supervisor_fixtures/alpha.py": alpha.read_text(encoding="utf-8"),
        "logs/dag_supervisor_fixtures/beta.py": beta.read_text(encoding="utf-8"),
        "logs/dag_supervisor_fixtures/summary.md": "",
    }
    worker_contexts: list[list[dict[str, str]]] = []

    def tool_fn(action: str, filepath: str, content: str | None = None) -> str:
        key = filepath.replace("\\", "/")
        if action == "read":
            body = store.get(key, "")
            return f"OK: read {key} ({len(body)} chars)\n{body}"
        if action in {"write", "append"}:
            prev = store.get(key, "")
            store[key] = (prev + str(content or "")) if action == "append" else str(content or "")
            # Mirror onto disk for post-asserts.
            target = Path(PROJECT_ROOT) / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(store[key], encoding="utf-8")
            return f"OK: {action} {len(str(content or ''))} chars to {key}"
        return f"ERROR: bad action {action}"

    def outline_fn(file_path: str) -> str:
        key = file_path.replace("\\", "/")
        body = store.get(key, "")
        return f"OK: outline {key} lang=python symbols=1\n(module skeleton)\n{body[:40]}"

    def symbol_fn(file_path: str, symbol_name: str) -> str:
        return f"ERROR: symbol {symbol_name!r} not found in {file_path}"

    def tracking_workers(state):
        patch = workers_node(
            state,
            tool_fn=tool_fn,
            outline_fn=outline_fn,
            symbol_fn=symbol_fn,
        )
        for row in patch.get("worker_results") or []:
            worker_contexts.append(list(row.get("context_window") or []))
        return patch

    # Swap workers node via recompile with wrapper — invoke run helpers instead.
    from langgraph.graph import END, START, StateGraph

    from dana.graph.nodes.supervisor import (
        SUPERVISOR_NODE,
        WORKER_NODE,
        make_supervisor_node,
        route_after_supervisor,
        END_ROUTE,
    )
    from dana.graph.state import SupervisorState

    wf = StateGraph(SupervisorState)
    wf.add_node(SUPERVISOR_NODE, make_supervisor_node())
    wf.add_node(WORKER_NODE, tracking_workers)
    wf.add_edge(START, SUPERVISOR_NODE)
    wf.add_conditional_edges(
        SUPERVISOR_NODE,
        route_after_supervisor,
        {WORKER_NODE: WORKER_NODE, END_ROUTE: END},
    )
    wf.add_edge(WORKER_NODE, SUPERVISOR_NODE)
    app = wf.compile()

    initial = empty_supervisor_state(prompt, max_supervisor_cycles=12)
    # Poison global history — workers must ignore it.
    initial["global_conversation_history"] = [
        {"role": "user", "content": "SECRET_HISTORY_SHOULD_NOT_LEAK"}
    ]
    final = app.invoke(initial)

    dag = final.get("dag") or []
    assert len(dag) == 3, dag
    assert [t["task_id"] for t in dag] == [1, 2, 3]
    assert dag[0]["dependencies"] == []
    assert dag[1]["dependencies"] == [1]
    assert dag[2]["dependencies"] == [2]

    statuses = [str(t.get("status")) for t in dag]
    assert statuses == ["completed", "completed", "completed"], statuses
    assert len(final.get("completed_summaries") or []) == 3
    assert final.get("status") == "completed"
    assert int(final.get("supervisor_cycles") or 0) >= 3

    # Isolation: no worker context carries the secret global history.
    for ctx in worker_contexts:
        joined = " ".join(str(m.get("content") or "") for m in ctx)
        assert "SECRET_HISTORY_SHOULD_NOT_LEAK" not in joined
        assert any(m.get("role") == "system" for m in ctx)

    beta_body = Path(PROJECT_ROOT, "logs/dag_supervisor_fixtures/beta.py").read_text(
        encoding="utf-8"
    )
    summary_body = Path(
        PROJECT_ROOT, "logs/dag_supervisor_fixtures/summary.md"
    ).read_text(encoding="utf-8")
    assert "DAG-worker task" in beta_body
    assert "DAG-worker task" in summary_body
    print("[PASS] three_step_file_editing_swarm")
    print(f"final_response:\n{final.get('final_response')}")


def test_loop_guard_blocks_redispatch() -> None:
    state = empty_supervisor_state("1. Read x.py\n2. Edit y.py\n", max_supervisor_cycles=2)
    state["dag"] = heuristic_plan_dag(state["user_prompt"])
    state["pending_tasks"] = [1, 2]
    state["supervisor_cycles"] = 2
    state["last_dispatch_key"] = "1"
    state["active_task_ids"] = []
    state["worker_results"] = []
    # Force a third supervisor entry past the ceiling.
    state["supervisor_cycles"] = 2
    out = supervisor_node({**state, "max_supervisor_cycles": 2})
    # cycles becomes 3 > max 2
    assert out.get("status") == "failed"
    assert "cycle limit" in str(out.get("error") or "")
    assert route_after_supervisor({**state, **out}) == "__end__"  # type: ignore[arg-type]
    print("[PASS] loop_guard_blocks_redispatch")


def test_ready_tasks_respect_dependencies() -> None:
    state = empty_supervisor_state("demo")
    state["dag"] = parse_dag_json(
        '[{"task_id":1,"action":"a","dependencies":[]},'
        '{"task_id":2,"action":"b","dependencies":[1]}]'
    )
    state["pending_tasks"] = [1, 2]
    assert ready_task_ids(state) == [1]
    state["dag"][0]["status"] = "completed"
    state["dag"][0]["summary"] = "done"
    state["pending_tasks"] = [2]
    assert ready_task_ids(state) == [2]
    print("[PASS] ready_tasks_respect_dependencies")


if __name__ == "__main__":
    test_parse_and_heuristic_dag()
    test_worker_isolation_no_global_history()
    test_ready_tasks_respect_dependencies()
    test_loop_guard_blocks_redispatch()
    test_three_step_file_editing_swarm()
    print("\nAll DAG supervisor tests passed.")

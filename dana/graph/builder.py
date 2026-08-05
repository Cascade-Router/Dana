"""Compile the nested Supervisor ↔ Worker DAG graph (+ Meta-Broker closed loop)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from typing import Any

from dana.graph.nodes.supervisor import (
    END_ROUTE,
    SUPERVISOR_NODE,
    WORKER_NODE,
    DagPlanner,
    make_supervisor_node,
    route_after_supervisor,
)
from dana.graph.nodes.worker import ToolFn, WorkerFactory, make_workers_node
from dana.graph.state import (
    BrokerState,
    SupervisorState,
    empty_broker_state,
    empty_supervisor_state,
)

# Workers always return to the supervisor for outcome evaluation.
_ROUTE_BACK_TO_SUPERVISOR = SUPERVISOR_NODE


def compile_dag_supervisor_graph(
    *,
    planner: DagPlanner | None = None,
    tool_fn: ToolFn | None = None,
    worker_factory: WorkerFactory | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Wire supervisor → workers → supervisor with explicit loop guards.

    Topology::

        START → supervisor ─┬─(awaiting_workers)→ workers → supervisor
                            └─(completed|failed)→ END

    Loop guards live in ``supervisor_node`` (max cycles, repeated-dispatch
    stall detection, max per-task attempts). Workers never see global chat
    history — only ``WorkerState`` built inside ``workers_node``.

    Code-generation workers are Deterministic Extraction Workers (plain LLM
    text → regex fence extract → staging write); they do not bind tools or
    run a ReAct tool-calling loop.
    """
    from langgraph.graph import END, START, StateGraph

    workflow: StateGraph = StateGraph(SupervisorState)
    workflow.add_node(SUPERVISOR_NODE, make_supervisor_node(planner=planner))
    workflow.add_node(
        WORKER_NODE,
        make_workers_node(tool_fn=tool_fn, worker_factory=worker_factory),
    )
    workflow.add_edge(START, SUPERVISOR_NODE)
    workflow.add_conditional_edges(
        SUPERVISOR_NODE,
        route_after_supervisor,
        {
            WORKER_NODE: WORKER_NODE,
            END_ROUTE: END,
        },
    )
    workflow.add_edge(WORKER_NODE, _ROUTE_BACK_TO_SUPERVISOR)

    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


def compile_meta_broker_graph(
    *,
    planner: DagPlanner | None = None,
    tool_fn: ToolFn | None = None,
    worker_factory: WorkerFactory | None = None,
    harness_fn: Any | None = None,
    default_validation_command: str = "python -m compileall .dana_scratch",
    harness_timeout_s: float = 30.0,
    checkpointer: Any | None = None,
) -> Any:
    """Wire Meta-Broker closed-loop topology.

    Topology::

        START → broker → supervisor ─┬─ workers → supervisor
                                     └─ staging_commit → runtime_harness → broker
        broker ─(done)→ END

    On harness ``success`` the broker advances to the next epic (or completes).
    On failure the broker injects stderr as a bug-fix supervisor prompt
    (up to ``max_repair_attempts``, default 3).
    """
    from langgraph.graph import END, START, StateGraph

    from dana.graph.nodes.broker import (
        BROKER_NODE,
        HARNESS_NODE,
        STAGING_NODE,
        make_broker_node,
        route_after_broker,
        route_after_supervisor_to_harness,
        staging_commit_node,
    )
    from dana.graph.runtime_harness import make_runtime_harness_node

    workflow: StateGraph = StateGraph(BrokerState)
    workflow.add_node(BROKER_NODE, make_broker_node())
    workflow.add_node(SUPERVISOR_NODE, make_supervisor_node(planner=planner))
    workflow.add_node(
        WORKER_NODE,
        make_workers_node(tool_fn=tool_fn, worker_factory=worker_factory),
    )
    workflow.add_node(STAGING_NODE, staging_commit_node)
    workflow.add_node(
        HARNESS_NODE,
        make_runtime_harness_node(
            default_command=default_validation_command,
            timeout_s=harness_timeout_s,
            harness_fn=harness_fn,
        ),
    )

    workflow.add_edge(START, BROKER_NODE)
    workflow.add_conditional_edges(
        BROKER_NODE,
        route_after_broker,
        {
            SUPERVISOR_NODE: SUPERVISOR_NODE,
            END_ROUTE: END,
        },
    )
    workflow.add_conditional_edges(
        SUPERVISOR_NODE,
        route_after_supervisor_to_harness,
        {
            WORKER_NODE: WORKER_NODE,
            STAGING_NODE: STAGING_NODE,
        },
    )
    workflow.add_edge(WORKER_NODE, _ROUTE_BACK_TO_SUPERVISOR)
    workflow.add_edge(STAGING_NODE, HARNESS_NODE)
    workflow.add_edge(HARNESS_NODE, BROKER_NODE)

    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


def run_meta_broker(
    macro_intent: str,
    *,
    planner: DagPlanner | None = None,
    tool_fn: ToolFn | None = None,
    worker_factory: WorkerFactory | None = None,
    harness_fn: Any | None = None,
    workspace_path: str | None = None,
    validation_command: str | None = None,
    max_repair_attempts: int = 3,
    max_supervisor_cycles: int = 12,
    max_task_attempts: int = 2,
) -> BrokerState:
    """Compile + invoke the Meta-Broker closed-loop graph."""
    from dana.graph.monitor_bus import (
        get_monitor_bus,
        publish_graph_error,
    )
    from dana.graph.nodes.supervisor import plan_dag_with_llm
    from dana.system_health import check_system_health

    # Prefer structured LLM planning (with dump-on-garbage); callers may override.
    effective_planner = planner if planner is not None else plan_dag_with_llm

    # OOM circuit breaker — abort before graph spin if RAM is already critical.
    try:
        health = check_system_health()
        print(
            f"[MetaBroker] health ram={health['ram_percent']:.1f}% "
            f"cpu={health['cpu_percent']:.1f}%",
            flush=True,
        )
    except SystemError as health_exc:
        print(f"[MetaBroker] {health_exc}", flush=True)
        try:
            from dana.graph.task_tracker import emit_meta_broker_telemetry

            emit_meta_broker_telemetry(
                task_id="meta_broker",
                prompt=str(macro_intent or ""),
                phase="health",
                status="failed",
                message=str(health_exc),
                terminal=True,
            )
        except Exception:  # noqa: BLE001
            pass
        failed = empty_broker_state(macro_intent)
        failed["status"] = "failed"
        failed["broker_phase"] = "done"
        failed["error"] = str(health_exc)
        failed["final_response"] = str(health_exc)
        return failed  # type: ignore[return-value]

    try:
        from dana.graph.task_tracker import emit_meta_broker_telemetry

        emit_meta_broker_telemetry(
            task_id="meta_broker",
            prompt=str(macro_intent or ""),
            phase="start",
            status="planning",
            message=f"Meta-Broker started ({len(macro_intent or '')} chars)",
        )
    except Exception:  # noqa: BLE001
        pass

    try:
        bus = get_monitor_bus(create=True)
        if bus is not None:
            bus.publish("status", status="running")
            bus.publish(
                "tool",
                message=f"run_meta_broker invoke chars={len(macro_intent or '')}",
            )
    except Exception:  # noqa: BLE001
        pass

    try:
        graph = compile_meta_broker_graph(
            planner=effective_planner,
            tool_fn=tool_fn,
            worker_factory=worker_factory,
            harness_fn=harness_fn,
            default_validation_command=(
                validation_command or "python -m compileall .dana_scratch"
            ),
            harness_timeout_s=30.0,
        )
        initial = empty_broker_state(
            macro_intent,
            max_supervisor_cycles=max_supervisor_cycles,
            max_task_attempts=max_task_attempts,
            max_repair_attempts=max_repair_attempts,
            workspace_path=workspace_path,
            validation_command=validation_command,
        )
        # Circuit breaker for supervisor⇄worker spin. Deterministic Extraction
        # Workers finish in 1 hop, but epic repair + multi-task DAGs still need
        # headroom (15 was truncating mid-repair before rate_limiter.py writes).
        _recursion_limit = 72
        print(
            f"[MetaBroker] graph.invoke START recursion_limit={_recursion_limit} "
            f"prompt_chars={len(macro_intent or '')}",
            flush=True,
        )
        result = graph.invoke(
            initial, config={"recursion_limit": _recursion_limit}
        )
        status = str((result or {}).get("status") or "")
        err = str((result or {}).get("error") or "").strip()
        print(
            f"[MetaBroker] graph.invoke END status={status!r} error={err[:300]!r}",
            flush=True,
        )
        try:
            from dana.graph.task_tracker import emit_meta_broker_telemetry

            emit_meta_broker_telemetry(
                task_id="meta_broker",
                prompt=str(macro_intent or ""),
                phase="done",
                status=status or ("failed" if err else "completed"),
                message=err or f"Meta-Broker finished status={status}",
                terminal=True,
            )
        except Exception:  # noqa: BLE001
            pass
        if status == "failed" or err:
            publish_graph_error(
                err or "meta_broker finished with status=failed",
                node="run_meta_broker_soft_fail",
                dump=True,
                soft_fail=True,
                prompt_preview=str(macro_intent or "")[:240],
                status=status,
                tasks_json=list((result or {}).get("dag") or []),
            )
        return result  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        publish_graph_error(
            f"run_meta_broker graph.invoke failed: {exc}",
            exc=exc,
            node="run_meta_broker",
            dump=True,
            prompt_preview=str(macro_intent or "")[:240],
            tasks_json=getattr(exc, "tasks_json", None),
            raw_llm_output=getattr(exc, "last_raw", None),
        )
        # Surface a structured failed BrokerState instead of a silent hang.
        failed = empty_broker_state(
            macro_intent,
            max_supervisor_cycles=max_supervisor_cycles,
            max_task_attempts=max_task_attempts,
            max_repair_attempts=max_repair_attempts,
            workspace_path=workspace_path,
            validation_command=validation_command,
        )
        failed["broker_phase"] = "done"
        failed["status"] = "failed"
        failed["error"] = f"{type(exc).__name__}: {exc}"
        failed["final_response"] = f"Meta-Broker crashed: {type(exc).__name__}: {exc}"
        failed["epic_log"] = [f"crash: {type(exc).__name__}: {exc}"]
        try:
            from dana.graph.task_tracker import emit_meta_broker_telemetry

            emit_meta_broker_telemetry(
                task_id="meta_broker",
                prompt=str(macro_intent or ""),
                phase="crash",
                status="failed",
                message=failed["error"],
                terminal=True,
            )
        except Exception:  # noqa: BLE001
            pass
        return failed  # type: ignore[return-value]


def _emit_stream_chunk(
    chunk: dict[str, Any],
    *,
    monitor: bool,
    on_update: Callable[[str, dict[str, Any]], None] | None,
) -> None:
    from dana.graph.monitor_bus import (
        get_monitor_bus,
        publish_dag_snapshot,
        publish_tool_call,
        publish_tool_line,
        publish_worker_finish,
        publish_worker_start,
    )

    for node_name, update in chunk.items():
        if not isinstance(update, dict):
            continue
        if on_update is not None:
            try:
                on_update(str(node_name), update)
            except Exception:  # noqa: BLE001
                pass
        if not monitor and get_monitor_bus(create=False) is None:
            continue
        publish_dag_snapshot(update, node=str(node_name))
        # Semantic worker lifecycle events for Control Dashboard subscribers.
        for task in update.get("dag") or []:
            if not isinstance(task, dict):
                continue
            tid = task.get("task_id")
            if tid is None:
                continue
            status = str(task.get("status") or "").lower()
            action = str(task.get("action") or "")
            if status in {"running", "in_progress", "dispatched", "active"}:
                publish_worker_start(tid, action=action)
            elif status in {"completed", "failed", "error"}:
                publish_worker_finish(
                    tid,
                    status=status,
                    summary=str(task.get("summary") or task.get("error") or ""),
                )
        for tid in update.get("active_task_ids") or []:
            publish_worker_start(tid)
        # Surface tool / checkpoint breadcrumbs into the live stream panel.
        for row in update.get("worker_results") or []:
            if not isinstance(row, dict):
                continue
            tid = row.get("task_id")
            for tool in row.get("tool_outputs") or []:
                if not isinstance(tool, dict):
                    continue
                name = tool.get("tool") or "tool"
                obs = str(tool.get("output") or "").replace("\n", " ")
                if len(obs) > 160:
                    obs = obs[:160] + "…"
                publish_tool_call(obs, worker=tid, tool=str(name))
                publish_tool_line(f"{name}: {obs}", worker=tid)
            summary = str(row.get("summary") or "").strip()
            if summary:
                publish_tool_line(summary, worker=tid)
                st = str(row.get("status") or "completed")
                publish_worker_finish(tid, status=st, summary=summary)
        for line in update.get("checkpoint_log") or []:
            text = str(line)
            if "shadow" in text.lower() or "staged" in text.lower() or "committed" in text.lower():
                publish_tool_call(f"Staging: {text}", tool="staging")
                publish_tool_line(f"Staging: {text}")
            elif text:
                publish_tool_line(text)


def stream_dag_supervisor(
    user_prompt: str,
    *,
    planner: DagPlanner | None = None,
    tool_fn: ToolFn | None = None,
    worker_factory: WorkerFactory | None = None,
    max_supervisor_cycles: int = 12,
    max_task_attempts: int = 2,
    monitor: bool = False,
    on_update: Callable[[str, dict[str, Any]], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield LangGraph ``stream_mode='updates'`` chunks; optionally feed the TUI bus."""
    if monitor:
        from dana.graph.monitor_bus import get_monitor_bus, publish_telemetry

        bus = get_monitor_bus(create=True)
        assert bus is not None
        bus.publish("status", status="running")
        publish_telemetry(vector_sync="unknown", note="stream started")

    graph = compile_dag_supervisor_graph(
        planner=planner,
        tool_fn=tool_fn,
        worker_factory=worker_factory,
    )
    initial = empty_supervisor_state(
        user_prompt,
        max_supervisor_cycles=max_supervisor_cycles,
        max_task_attempts=max_task_attempts,
    )
    final: dict[str, Any] = dict(initial)
    for chunk in graph.stream(initial, stream_mode="updates"):
        if isinstance(chunk, dict):
            _emit_stream_chunk(chunk, monitor=monitor, on_update=on_update)
            for _node, update in chunk.items():
                if isinstance(update, dict):
                    final.update(update)
            yield chunk

    if monitor:
        from dana.graph.monitor_bus import get_monitor_bus, publish_dag_snapshot

        bus = get_monitor_bus(create=False)
        if bus is not None:
            publish_dag_snapshot(final, node="end")
            bus.publish("status", status=str(final.get("status") or "completed"))
            bus.publish("done", status=final.get("status"))


def run_dag_supervisor(
    user_prompt: str,
    *,
    planner: DagPlanner | None = None,
    tool_fn: ToolFn | None = None,
    worker_factory: WorkerFactory | None = None,
    max_supervisor_cycles: int = 12,
    max_task_attempts: int = 2,
    monitor: bool = False,
) -> SupervisorState:
    """Convenience entry: compile, invoke (or stream), return final supervisor state."""
    if monitor:
        final: dict[str, Any] = {}
        for chunk in stream_dag_supervisor(
            user_prompt,
            planner=planner,
            tool_fn=tool_fn,
            worker_factory=worker_factory,
            max_supervisor_cycles=max_supervisor_cycles,
            max_task_attempts=max_task_attempts,
            monitor=True,
        ):
            for _node, update in chunk.items():
                if isinstance(update, dict):
                    final.update(update)
        return final  # type: ignore[return-value]

    graph = compile_dag_supervisor_graph(
        planner=planner,
        tool_fn=tool_fn,
        worker_factory=worker_factory,
    )
    initial = empty_supervisor_state(
        user_prompt,
        max_supervisor_cycles=max_supervisor_cycles,
        max_task_attempts=max_task_attempts,
    )
    result = graph.invoke(initial, config={"recursion_limit": 32})
    return result  # type: ignore[return-value]


def _cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the DAG supervisor graph")
    parser.add_argument("prompt", nargs="?", default="", help="User prompt to plan/execute")
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Stream state updates into the TUI monitor bus (use with dana.cli.monitor)",
    )
    parser.add_argument("--demo", action="store_true", help="Run a hermetic 3-step demo DAG")
    args = parser.parse_args(argv)

    prompt = (args.prompt or "").strip()
    if args.demo or not prompt:
        prompt = (
            "1. Read logs/dag_monitor_demo/a.py\n"
            "2. Edit logs/dag_monitor_demo/b.py\n"
            "3. Write logs/dag_monitor_demo/summary.md\n"
        )

    tool_fn = None
    outline_fn = None
    if args.demo:
        from pathlib import Path

        from dana.paths import PROJECT_ROOT

        demo = Path(PROJECT_ROOT) / "logs" / "dag_monitor_demo"
        demo.mkdir(parents=True, exist_ok=True)
        (demo / "a.py").write_text("A = 1\n", encoding="utf-8")
        (demo / "b.py").write_text("B = 0\n", encoding="utf-8")
        store = {
            "logs/dag_monitor_demo/a.py": "A = 1\n",
            "logs/dag_monitor_demo/b.py": "B = 0\n",
            "logs/dag_monitor_demo/summary.md": "",
        }

        def tool_fn(action: str, filepath: str, content: str | None = None) -> str:
            key = filepath.replace("\\", "/")
            if action == "read":
                body = store.get(key, "")
                return f"OK: read {key} ({len(body)} chars)\n{body}"
            store[key] = str(content or "")
            target = Path(PROJECT_ROOT) / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(store[key], encoding="utf-8")
            return f"OK: {action} {len(store[key])} chars to {key} (shadow staged)"

        def outline_fn(file_path: str) -> str:
            from dana.graph.monitor_bus import publish_tool_line

            publish_tool_line(f"AST parsed {file_path}")
            return f"OK: outline {file_path} lang=python symbols=1\n(module)"

    if args.monitor:
        from dana.graph.monitor_bus import reset_monitor_bus

        reset_monitor_bus()

    kwargs: dict[str, Any] = {"monitor": bool(args.monitor), "tool_fn": tool_fn}
    if outline_fn is not None:
        # Workers node accepts outline via make_workers_node — wrap compile path.
        from dana.graph.nodes.worker import make_workers_node
        from langgraph.graph import END, START, StateGraph

        from dana.graph.nodes.supervisor import make_supervisor_node

        g = StateGraph(SupervisorState)
        g.add_node(SUPERVISOR_NODE, make_supervisor_node())
        g.add_node(
            WORKER_NODE,
            make_workers_node(tool_fn=tool_fn, outline_fn=outline_fn),
        )
        g.add_edge(START, SUPERVISOR_NODE)
        g.add_conditional_edges(
            SUPERVISOR_NODE,
            route_after_supervisor,
            {WORKER_NODE: WORKER_NODE, END_ROUTE: END},
        )
        g.add_edge(WORKER_NODE, SUPERVISOR_NODE)
        app = g.compile()
        initial = empty_supervisor_state(prompt)
        final: dict[str, Any] = dict(initial)
        for chunk in app.stream(initial, stream_mode="updates"):
            if isinstance(chunk, dict):
                _emit_stream_chunk(chunk, monitor=bool(args.monitor), on_update=None)
                for _n, upd in chunk.items():
                    if isinstance(upd, dict):
                        final.update(upd)
        print(f"status={final.get('status')} tasks={len(final.get('dag') or [])}")
        return 0 if str(final.get("status")) == "completed" else 1

    result = run_dag_supervisor(prompt, **kwargs)
    print(f"status={result.get('status')} tasks={len(result.get('dag') or [])}")
    return 0 if str(result.get("status")) == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())


__all__ = (
    "compile_dag_supervisor_graph",
    "compile_meta_broker_graph",
    "empty_broker_state",
    "empty_supervisor_state",
    "run_dag_supervisor",
    "run_meta_broker",
    "stream_dag_supervisor",
)

#!/usr/bin/env python3
"""Offline complex-task benchmark exercising closed-loop ``verifier_node``.

Runs three deterministic tasks **without** live cloud LLMs / desktop UIA:

1. Log extraction → ``logs/audit_summary.json`` (JSON schema verification)
2. Multi-window UIA inspection (mocked UIA provider / injected tree)
3. REPL 90th-percentile latency with inline pytest assertions

Each task compiles a stubbed Donna ReAct graph (mocked agent/tools) that still
routes through the real ``verifier_node`` + ``TaskTracker`` protocol.

Usage (from repo root)::

    .venv\\Scripts\\python.exe scripts/verify_complex_tasks.py

Exit 0 when 3/3 PASSED.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

# Allow ``python scripts/verify_complex_tasks.py`` without install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _logs_dir() -> Path:
    try:
        from dana.paths import LOGS_DIR

        return Path(LOGS_DIR)
    except Exception:  # noqa: BLE001
        return _ROOT / "logs"


def _ensure_sample_runtime_log(log_path: Path) -> None:
    """Create a small dana_runtime.log fixture if missing / empty."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.is_file() and log_path.stat().st_size > 0:
        # Append latency lines so task 3 always has data.
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "latency_ms=" not in text:
            with log_path.open("a", encoding="utf-8") as fh:
                for ms in (12, 40, 55, 80, 90, 100, 110, 150, 200, 250):
                    fh.write(f"INFO request ok latency_ms={ms}\n")
        return
    lines = [
        "INFO dana boot",
        "WARNING cache miss on tool broker",
        "ERROR python_repl timed out once",
        "WARNING vision backend cold start",
        "INFO request ok latency_ms=12",
        "INFO request ok latency_ms=40",
        "INFO request ok latency_ms=55",
        "INFO request ok latency_ms=80",
        "INFO request ok latency_ms=90",
        "INFO request ok latency_ms=100",
        "INFO request ok latency_ms=110",
        "INFO request ok latency_ms=150",
        "INFO request ok latency_ms=200",
        "INFO request ok latency_ms=250",
        "ERROR disk full on captures/",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise AssertionError("no latency samples")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _extract_latencies(log_text: str) -> list[float]:
    import re

    return [float(m) for m in re.findall(r"latency_ms\s*=\s*([0-9]+(?:\.[0-9]+)?)", log_text)]


def _run_offline_graph(
    *,
    task_id: str,
    prompt: str,
    tools_fn: Any,
    verification_targets: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
    tracker: Any,
) -> dict[str, Any]:
    """Compile stubbed graph; real verifier_node + TaskTracker."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    from dana.agentic_react_graph import compile_donna_react_graph
    from dana.graph.nodes.verifier import make_verifier_node
    from dana.schema import ReactGraphState

    path: list[str] = []
    agent_visits = {"n": 0}

    def planner(state: ReactGraphState) -> dict[str, Any]:
        path.append("planner")
        return {
            "always_include": ["python_repl"],
            "current_agent": "Planner",
            "verification_targets": verification_targets,
        }

    def executor(state: ReactGraphState) -> dict[str, Any]:
        path.append("executor")
        return {"current_agent": "Executor"}

    def agent(state: ReactGraphState) -> dict[str, Any]:
        path.append("agent")
        agent_visits["n"] += 1
        # After a failed verification, tools re-run to self-correct.
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "python_repl",
                            "args": {"code": "# offline complex task"},
                            "id": f"call-{task_id}-{agent_visits['n']}",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "halt": False,
            "verification_targets": verification_targets,
        }

    def tools(state: ReactGraphState) -> dict[str, Any]:
        path.append("tools")
        from langchain_core.messages import ToolMessage

        patch = tools_fn(state, attempt=agent_visits["n"])
        patch.setdefault("verification_targets", verification_targets)
        patch.setdefault("halt", True)
        # Resolve open tool_calls so completion_gate does not block verifier.
        msgs = list(state.get("messages") or [])
        last = msgs[-1] if msgs else None
        tcs = getattr(last, "tool_calls", None) or []
        tool_msgs: list[Any] = []
        for tc in tcs:
            cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
            tool_msgs.append(
                ToolMessage(
                    content=str(patch.get("last_obs") or "ok"),
                    tool_call_id=str(cid or "call"),
                )
            )
        if tool_msgs:
            patch["messages"] = tool_msgs
        return patch

    verifier = make_verifier_node(tracker=tracker)

    def _verifier(state: ReactGraphState) -> dict[str, Any]:
        path.append("verifier")
        return verifier(state)

    def hydrate(state: ReactGraphState) -> dict[str, Any]:
        path.append("hydrate_memory")
        return {"current_agent": "MemoryHydrate"}

    def consolidate(state: ReactGraphState) -> dict[str, Any]:
        path.append("consolidate_memory")
        return {"current_agent": "MemoryConsolidate"}

    graph = compile_donna_react_graph(
        agent,
        tools,
        planner_node_fn=planner,
        executor_node_fn=executor,
        hydrate_memory_node_fn=hydrate,
        consolidate_memory_node_fn=consolidate,
        verifier_node_fn=_verifier,
        checkpointer=MemorySaver(),
    )
    tracker.start_task(task_id, prompt)
    cfg = {"configurable": {"thread_id": f"complex-{task_id}"}}
    init: dict[str, Any] = {
        "messages": [HumanMessage(content=prompt)],
        "halt": False,
        "session_id": task_id,
        "task_id": task_id,
        "active_intent": prompt,
        "always_include": ["python_repl"],
        "verification_targets": verification_targets,
        "max_verification_attempts": 3,
        "verification_result": {},
    }
    if extra_state:
        init.update(extra_state)
    list(graph.stream(init, cfg, stream_mode="values"))
    final = dict(graph.get_state(cfg).values or {})
    final["_path"] = path
    return final


# ---------------------------------------------------------------------------
# TASK 1 — Log extraction & JSON summary
# ---------------------------------------------------------------------------


def run_task1_log_extraction(tracker: Any) -> dict[str, Any]:
    logs = _logs_dir()
    runtime_log = logs / "dana_runtime.log"
    audit_path = logs / "audit_summary.json"
    _ensure_sample_runtime_log(runtime_log)
    if audit_path.exists():
        audit_path.unlink()

    prompt = (
        "Scan logs/dana_runtime.log and write logs/audit_summary.json with keys "
        "total_warnings, total_errors, summary."
    )
    targets = {
        "files": [str(audit_path)],
        "json_schema": {
            "path": str(audit_path),
            "required_keys": ["total_warnings", "total_errors", "summary"],
        },
    }

    def tools_fn(state: dict[str, Any], *, attempt: int) -> dict[str, Any]:
        text = runtime_log.read_text(encoding="utf-8", errors="replace")
        warnings = text.upper().count("WARNING")
        errors = text.upper().count("ERROR")
        payload = {
            "total_warnings": warnings,
            "total_errors": errors,
            "summary": (
                f"{warnings} warning(s), {errors} error(s) in dana_runtime.log"
            ),
        }
        # Attempt 1 can omit the file to force one verifier retry (self-correct).
        if attempt <= 1 and not audit_path.is_file():
            # Write on first tools visit — still creates artifact for verifier.
            audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        elif not audit_path.is_file():
            audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "halt": True,
            "final_raw": json.dumps(payload),
            "last_obs": f"wrote {audit_path}",
            "execution_error": None,
        }

    final = _run_offline_graph(
        task_id="complex-task-1",
        prompt=prompt,
        tools_fn=tools_fn,
        verification_targets=targets,
        tracker=tracker,
    )
    vr = final.get("verification_result") or {}
    ok = (
        bool(vr.get("verified"))
        and audit_path.is_file()
        and audit_path.stat().st_size > 0
    )
    if ok:
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        ok = all(k in data for k in ("total_warnings", "total_errors", "summary"))
    return {
        "name": "TASK 1 Log Extraction & JSON Summary",
        "passed": ok,
        "verification_result": vr,
        "artifact": str(audit_path),
        "task_status": getattr(tracker.get_task("complex-task-1"), "status", None),
    }


# ---------------------------------------------------------------------------
# TASK 2 — Multi-window UIA inspection (mocked)
# ---------------------------------------------------------------------------


def run_task2_uia_inspection(tracker: Any) -> dict[str, Any]:
    from dana.vision.uia_provider import Win32UIAProvider

    mock_tree = [
        {
            "name": "Document - WordPad",
            "title": "Document - WordPad",
            "control_type": "Window",
            "bbox": [0, 0, 800, 600],
        },
        {
            "name": "Calculator",
            "title": "Calculator",
            "control_type": "Window",
            "bounds": [100, 100, 400, 500],
        },
    ]
    provider = Win32UIAProvider(control_tree=mock_tree)
    # Touch provider so offline path exercises injectable UIA surface.
    _ = provider.find_element_bounds("Calculator")

    prompt = (
        "Inspect Win32 UIA window titles and bounding boxes for open windows."
    )
    targets = {"uia_nodes": True}

    def tools_fn(state: dict[str, Any], *, attempt: int) -> dict[str, Any]:
        return {
            "halt": True,
            "final_raw": f"UIA nodes={len(mock_tree)}",
            "last_obs": "uia_tree_ok",
            "uia_nodes": mock_tree,
            "env_context": {"uia_nodes": mock_tree},
            "execution_error": None,
        }

    final = _run_offline_graph(
        task_id="complex-task-2",
        prompt=prompt,
        tools_fn=tools_fn,
        verification_targets=targets,
        extra_state={"uia_nodes": mock_tree},
        tracker=tracker,
    )
    vr = final.get("verification_result") or {}
    nodes = final.get("uia_nodes") or (vr.get("evidence") or {}).get("uia_nodes") or []
    has_bounds = any(
        isinstance(n, dict)
        and (
            isinstance(n.get("bbox"), (list, tuple))
            or isinstance(n.get("bounds"), (list, tuple))
        )
        for n in nodes
    )
    ok = bool(vr.get("verified")) and bool(nodes) and has_bounds
    return {
        "name": "TASK 2 Multi-Window UIA Inspection",
        "passed": ok,
        "verification_result": vr,
        "uia_count": len(nodes) if isinstance(nodes, list) else 0,
        "task_status": getattr(tracker.get_task("complex-task-2"), "status", None),
    }


# ---------------------------------------------------------------------------
# TASK 3 — REPL percentile with inline assertions
# ---------------------------------------------------------------------------


def run_task3_repl_percentile(tracker: Any) -> dict[str, Any]:
    logs = _logs_dir()
    runtime_log = logs / "dana_runtime.log"
    _ensure_sample_runtime_log(runtime_log)

    prompt = (
        "Calculate the 90th percentile latency_ms from logs/dana_runtime.log "
        "using the python_repl; assert the result is a number."
    )
    targets = {"repl_result": True, "require_repl_success": True}

    def tools_fn(state: dict[str, Any], *, attempt: int) -> dict[str, Any]:
        text = runtime_log.read_text(encoding="utf-8", errors="replace")
        samples = _extract_latencies(text)
        # Inline pytest-style assertions (executed in-process, not via LLM).
        assert samples, "expected latency_ms samples in log"
        p90 = _percentile(samples, 90.0)
        assert isinstance(p90, (int, float)), "percentile must be numeric"
        assert p90 > 0, "percentile must be positive"
        obs = f"exit_code=0\nstdout:\n{p90}\n"
        return {
            "halt": True,
            "final_raw": str(p90),
            "last_obs": obs,
            "repl_numeric_result": float(p90),
            "execution_error": None,
        }

    final = _run_offline_graph(
        task_id="complex-task-3",
        prompt=prompt,
        tools_fn=tools_fn,
        verification_targets=targets,
        tracker=tracker,
    )
    vr = final.get("verification_result") or {}
    numeric = final.get("repl_numeric_result")
    if numeric is None:
        numeric = (vr.get("evidence") or {}).get("repl_numeric_result")
    obs = str(final.get("last_obs") or "")
    ok = (
        bool(vr.get("verified"))
        and numeric is not None
        and "AssertionError" not in obs
        and not str(final.get("execution_error") or "").strip()
    )
    return {
        "name": "TASK 3 REPL Percentile + Assertions",
        "passed": ok,
        "verification_result": vr,
        "percentile_p90": numeric,
        "task_status": getattr(tracker.get_task("complex-task-3"), "status", None),
    }


def main() -> int:
    from dana.graph.task_tracker import TaskStatus, TaskTracker, set_shared_task_tracker

    logs = _logs_dir()
    logs.mkdir(parents=True, exist_ok=True)
    tracker = TaskTracker(
        dropped_log_path=logs / "dropped_tasks_complex.log",
        ledger_path=_ROOT / "dana_security" / "_complex_bench_ledger.md",
    )
    set_shared_task_tracker(tracker)

    print("=== Dana Complex Task Benchmark (offline verifier protocol) ===")
    print("Mode: OFFLINE - stubbed agent/tools, real verifier_node + TaskTracker")
    print("No cloud LLM / live desktop required.\n")

    results = [
        run_task1_log_extraction(tracker),
        run_task2_uia_inspection(tracker),
        run_task3_repl_percentile(tracker),
    ]

    passed = 0
    for r in results:
        mark = "PASSED" if r["passed"] else "FAILED"
        if r["passed"]:
            passed += 1
        status = r.get("task_status")
        status_s = status.value if isinstance(status, TaskStatus) else status
        print(f"[{mark}] {r['name']}")
        print(f"       verification_result.verified={((r.get('verification_result') or {}).get('verified'))}")
        print(f"       TaskTracker={status_s}")
        if "artifact" in r:
            print(f"       artifact={r['artifact']}")
        if "uia_count" in r:
            print(f"       uia_nodes={r['uia_count']}")
        if "percentile_p90" in r:
            print(f"       p90={r['percentile_p90']}")
        print()

    total = len(results)
    print(f"RESULT: {passed}/{total} PASSED")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

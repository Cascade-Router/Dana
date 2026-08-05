"""Supervisor node — decompose complex prompts into a DAG and dispatch workers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from dana.graph.dag_topology import validate_dag_solvability
from dana.graph.state import DagTask, SupervisorState

SUPERVISOR_NODE = "supervisor"
WORKER_NODE = "workers"
END_ROUTE = "__end__"

DEFAULT_MAX_SUPERVISOR_CYCLES = 12
DEFAULT_MAX_TASK_ATTEMPTS = 2
MAX_STALLS = 2

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_STEP_RE = re.compile(
    r"(?:^|\n)\s*(?:step\s*)?(\d+)\s*[.)\-:]\s*(.+?)(?=(?:\n\s*(?:step\s*)?\d+\s*[.)\-:])|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_FILE_TOKEN_RE = re.compile(
    r"[\w./\\-]+\.(?:py|md|txt|json|toml|yaml|yml|cfg|ini)\b",
    re.IGNORECASE,
)

DagPlanner = Callable[[str], list[DagTask]]


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("DagSupervisor", msg)
    except Exception:  # noqa: BLE001
        print(f"[DagSupervisor] {msg}", flush=True)


def _normalize_task(raw: dict[str, Any], *, fallback_id: int) -> DagTask:
    tid = raw.get("task_id", fallback_id)
    try:
        task_id = int(tid)
    except (TypeError, ValueError):
        task_id = fallback_id
    deps_raw = raw.get("dependencies") or []
    deps: list[int] = []
    if isinstance(deps_raw, list):
        for d in deps_raw:
            try:
                deps.append(int(d))
            except (TypeError, ValueError):
                continue
    action = str(raw.get("action") or raw.get("task") or "").strip()
    if not action:
        action = f"noop task {task_id}"
    tool_name = str(raw.get("tool_name") or raw.get("tool") or "").strip()
    try:
        from dana.graph.nodes.worker import with_explicit_path_passthrough

        action = with_explicit_path_passthrough(action, tool_name or "file_editor")
    except Exception:  # noqa: BLE001
        pass
    task: DagTask = {
        "task_id": task_id,
        "action": action,
        "dependencies": deps,
        "status": "pending",
        "summary": "",
        "error": "",
        "attempts": 0,
    }
    if tool_name:
        task["tool_name"] = tool_name
    return task


def parse_dag_json(text: str) -> list[DagTask]:
    """Parse a JSON DAG array from model or fixture text."""
    blob = (text or "").strip()
    if not blob:
        return []
    candidates = [blob]
    m = _JSON_ARRAY_RE.search(blob)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        tasks: list[DagTask] = []
        for i, item in enumerate(data, start=1):
            if isinstance(item, dict):
                tasks.append(_normalize_task(item, fallback_id=i))
        if tasks:
            return _validate_dag(tasks)
    return []


def heuristic_plan_dag(prompt: str) -> list[DagTask]:
    """Deterministic DAG planner for multi-step / multi-file prompts.

    Prefer numbered steps; otherwise chain file mentions in order
    (read → edit → write) so tests and offline routing stay LLM-free.
    """
    text = (prompt or "").strip()
    if not text:
        return []

    steps = _STEP_RE.findall(text)
    if len(steps) >= 2:
        tasks: list[DagTask] = []
        for idx, (_num, body) in enumerate(steps, start=1):
            action = " ".join(str(body).split())
            deps = [idx - 1] if idx > 1 else []
            tasks.append(
                {
                    "task_id": idx,
                    "action": action,
                    "dependencies": deps,
                    "status": "pending",
                    "summary": "",
                    "error": "",
                    "attempts": 0,
                }
            )
        return _validate_dag(tasks)

    files = _FILE_TOKEN_RE.findall(text)
    # Preserve order, drop duplicates.
    ordered: list[str] = []
    for f in files:
        norm = f.replace("\\", "/")
        if norm not in ordered:
            ordered.append(norm)
    if len(ordered) >= 2:
        verbs = ("read", "edit", "write")
        tasks = []
        for idx, path in enumerate(ordered[:3], start=1):
            verb = verbs[min(idx - 1, len(verbs) - 1)]
            tasks.append(
                {
                    "task_id": idx,
                    "action": f"{verb} {path}",
                    "dependencies": [idx - 1] if idx > 1 else [],
                    "status": "pending",
                    "summary": "",
                    "error": "",
                    "attempts": 0,
                }
            )
        return _validate_dag(tasks)

    # Single-node fallback — still a valid DAG.
    return _validate_dag(
        [
            {
                "task_id": 1,
                "action": text[:400],
                "dependencies": [],
                "status": "pending",
                "summary": "",
                "error": "",
                "attempts": 0,
            }
        ]
    )


def plan_dag_with_llm(
    prompt: str,
    *,
    llm_invoke: Callable[[str], str] | None = None,
    use_structured: bool = True,
) -> list[DagTask]:
    """Ask an LLM for a JSON DAG; fall back to the heuristic planner.

    When ``use_structured`` is true (default), prefer Pydantic ``DAGPlan`` via
    ``ask_ollama_structured`` (Ollama ``format`` + JSON retry middleware).
    """
    import json as _json

    from dana.graph.monitor_bus import publish_graph_error, publish_tool_line

    _log(f"LLM DAG plan BEGIN prompt_chars={len(prompt or '')}")
    print(
        f"[DagSupervisor] LLM plan PROMPT >>>\n{(prompt or '')[:2000]}\n<<<",
        flush=True,
    )
    publish_tool_line(f"supervisor LLM plan begin chars={len(prompt or '')}")

    if use_structured and llm_invoke is None:
        try:
            from dana.graph.cloud_planner import planner_mode_label, publish_planner_mode
            from dana.llm_client import ask_planner_structured
            from dana.llm_schemas import DAGPlan

            mode = publish_planner_mode(warn_missing_key=True)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are the DAG supervisor. Return ONLY JSON matching the "
                        "DAGPlan schema: {\"tasks\":[{\"task_id\":int,\"action\":str,"
                        "\"tool_name\":str,\"dependencies\":[int,...]}, ...]}. "
                        "Enforce semantic dependency order. No markdown, no prose.\n"
                        "When generating Python code for Epics, rely strictly on "
                        "Python Standard Library modules (e.g., json, math, os, sys, "
                        "deque) unless third-party packages are explicitly requested "
                        "in the prompt.\n\n"
                        "CRITICAL: You must ONLY use the exact tool names provided. "
                        "Do not invent tools. Allowed tool_name values are EXACTLY: "
                        "file_editor, get_file_outline, get_symbol_definition, "
                        "read_local_file.\n"
                        "Put the real work description in action (include file paths). "
                        "Never set action to a bare tool name like create_file.\n"
                        "Each mutating action MUST explicitly name the tool and path, e.g. "
                        "\"Use file_editor to create tests/test_rate_limiter.py ...\".\n"
                        "Use EXACT filenames from the user prompt "
                        "(e.g. rate_limiter.py — never rename to token_bucket.py).\n\n"
                        "EXAMPLE VALID DAG PLAN:\n"
                        "{\n"
                        "  \"tasks\": [\n"
                        "    {\"task_id\": 1, \"action\": \"Use file_editor to create "
                        "tests/test_rate_limiter.py with failing pytest for TokenBucket\", "
                        "\"tool_name\": \"file_editor\", \"dependencies\": []},\n"
                        "    {\"task_id\": 2, \"action\": \"Use file_editor to create "
                        "rate_limiter.py implementing TokenBucket\", "
                        "\"tool_name\": \"file_editor\", \"dependencies\": [1]}\n"
                        "  ]\n"
                        "}"
                    ),
                },
                {"role": "user", "content": f"PROMPT:\n{prompt}"},
            ]
            print(
                f"[DagSupervisor] LLM plan CALL ask_planner_structured(DAGPlan) "
                f"mode={mode or planner_mode_label()}",
                flush=True,
            )
            # Hybrid toggle may route this hop to Gemini; workers stay local.
            plan = ask_planner_structured(messages, DAGPlan, max_retries=3)
            try:
                raw_preview = _json.dumps(
                    plan.model_dump() if hasattr(plan, "model_dump") else plan,
                    ensure_ascii=False,
                )
            except Exception:  # noqa: BLE001
                raw_preview = repr(plan)
            print(
                f"[DagSupervisor] LLM plan RESPONSE >>>\n{raw_preview[:4000]}\n<<<",
                flush=True,
            )
            _log(f"LLM DAG plan RESPONSE chars={len(raw_preview)}")
            publish_tool_line(f"supervisor LLM plan response chars={len(raw_preview)}")
            planned = plan.to_dag_tasks()  # type: ignore[attr-defined]
            if planned:
                return _validate_dag(planned)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            _log(f"structured DAG plan failed ({exc}); trying legacy/heuristic")
            print(
                f"[DagSupervisor] LLM plan structured FAIL: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            try:
                from pydantic import ValidationError
            except Exception:  # noqa: BLE001
                ValidationError = ()  # type: ignore[misc, assignment]
            from dana.middleware.json_schema_retry import StructuredOutputError

            raw_out = ""
            tasks_blob: Any = None
            if isinstance(exc, StructuredOutputError):
                raw_out = str(getattr(exc, "last_raw", "") or "")
            if raw_out:
                try:
                    from dana.middleware.json_schema_retry import extract_json_payload

                    tasks_blob = _json.loads(extract_json_payload(raw_out))
                except Exception:  # noqa: BLE001
                    tasks_blob = {"raw": raw_out[:4000]}
            if isinstance(
                exc,
                (ValidationError, _json.JSONDecodeError, ValueError, StructuredOutputError),
            ):
                publish_graph_error(
                    f"supervisor DAG JSON/schema error: {exc}",
                    exc=exc if not isinstance(exc, StructuredOutputError) else None,
                    node="supervisor_plan_llm",
                    dump=True,
                    tasks_json=tasks_blob,
                    raw_llm_output=raw_out or None,
                )

    if llm_invoke is not None:
        instruction = (
            "Analyze the user prompt and return ONLY a JSON array of sub-tasks "
            "for a file/codebase DAG. Each item must be "
            '{"task_id": int, "action": str, "tool_name": str, '
            '"dependencies": [int, ...]}. '
            "CRITICAL: tool_name must be exactly one of "
            "file_editor, get_file_outline, get_symbol_definition, read_local_file. "
            "Do not invent tools like create_file. "
            "EXAMPLE: "
            '[{"task_id":1,"action":"Use file_editor to create '
            'x/test_rate_limiter.py with failing pytest",'
            '"tool_name":"file_editor","dependencies":[]},'
            '{"task_id":2,"action":"Use file_editor to create '
            'x/rate_limiter.py implementing TokenBucket",'
            '"tool_name":"file_editor","dependencies":[1]}]. '
            "Enforce semantic order via dependencies. No prose.\n\n"
            f"PROMPT:\n{prompt}"
        )
        try:
            print(
                f"[DagSupervisor] LLM plan CALL llm_invoke chars={len(instruction)}",
                flush=True,
            )
            raw = llm_invoke(instruction)
            raw_s = raw if isinstance(raw, str) else str(raw)
            print(
                f"[DagSupervisor] LLM plan RESPONSE >>>\n{raw_s[:4000]}\n<<<",
                flush=True,
            )
            _log(f"LLM DAG plan legacy RESPONSE chars={len(raw_s)}")
            publish_tool_line(f"supervisor LLM legacy response chars={len(raw_s)}")
            planned = parse_dag_json(raw_s)
            if planned:
                try:
                    validate_dag_solvability(planned)
                except ValueError as topo_exc:
                    publish_graph_error(
                        f"supervisor DAG topology error: {topo_exc}",
                        exc=topo_exc,
                        node="supervisor_plan_llm_legacy",
                        dump=True,
                        tasks_json=planned,
                        raw_llm_output=raw_s,
                    )
                    raise
                return planned
        except Exception as exc:  # noqa: BLE001
            _log(f"LLM DAG plan failed ({exc}); using heuristic")
            print(
                f"[DagSupervisor] LLM plan legacy FAIL: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            publish_graph_error(
                f"supervisor LLM invoke/parse failed: {exc}",
                exc=exc,
                node="supervisor_plan_llm_legacy",
                dump=True,
                tasks_json=None,
                raw_llm_output=locals().get("raw_s"),
            )
    _log("LLM DAG plan falling back to heuristic_plan_dag")
    return heuristic_plan_dag(prompt)


def _validate_dag(tasks: list[DagTask]) -> list[DagTask]:
    """Drop self-deps / unknown deps; keep stable task_id order."""
    ids = {int(t["task_id"]) for t in tasks}
    cleaned: list[DagTask] = []
    for t in sorted(tasks, key=lambda x: int(x["task_id"])):
        deps = [
            d
            for d in (t.get("dependencies") or [])
            if int(d) in ids and int(d) != int(t["task_id"])
        ]
        row: DagTask = {
            "task_id": int(t["task_id"]),
            "action": str(t.get("action") or ""),
            "dependencies": deps,
            "status": str(t.get("status") or "pending"),  # type: ignore[typeddict-item]
            "summary": str(t.get("summary") or ""),
            "error": str(t.get("error") or ""),
            "attempts": int(t.get("attempts") or 0),
        }
        tool_name = str(t.get("tool_name") or "").strip()
        if tool_name:
            row["tool_name"] = tool_name
        cleaned.append(row)
    return cleaned


def _task_map(state: SupervisorState) -> dict[int, DagTask]:
    return {int(t["task_id"]): t for t in (state.get("dag") or [])}


def _deps_satisfied(task: DagTask, by_id: dict[int, DagTask]) -> bool:
    for dep in task.get("dependencies") or []:
        parent = by_id.get(int(dep))
        if parent is None:
            return False
        if str(parent.get("status") or "") != "completed":
            return False
        # Semantic gate: empty summary on a dependency blocks advancement.
        if not str(parent.get("summary") or "").strip():
            return False
    return True


def ready_task_ids(state: SupervisorState) -> list[int]:
    """Tasks whose dependencies completed successfully and are still pending."""
    by_id = _task_map(state)
    ready: list[int] = []
    for tid in state.get("pending_tasks") or []:
        task = by_id.get(int(tid))
        if task is None:
            continue
        status = str(task.get("status") or "pending")
        if status not in {"pending", "ready", "failed"}:
            continue
        if status == "failed" and int(task.get("attempts") or 0) >= int(
            state.get("max_task_attempts") or DEFAULT_MAX_TASK_ATTEMPTS
        ):
            continue
        if _deps_satisfied(task, by_id):
            ready.append(int(tid))
    return ready


def _checkpoint_worker_result(
    result: dict[str, Any],
    *,
    ok: bool,
) -> tuple[bool, str, str]:
    """verify_and_commit on success; rollback_workspace on failure/malformed.

    Returns ``(accepted, checkpoint_message, error_extra)``.
    """
    from dana.tools.file_editor import rollback_workspace, verify_and_commit

    sid = str(result.get("staging_session_id") or "").strip()
    if not sid:
        return ok, "", ""

    malformed = (not ok) or (not str(result.get("summary") or "").strip())
    if malformed or str(result.get("status") or "") != "completed":
        msg = rollback_workspace(sid)
        _log(f"rollback_workspace session={sid}: {msg}")
        return False, msg, "staging rolled back"
    msg = verify_and_commit(sid)
    _log(f"verify_and_commit session={sid}: {msg}")
    if str(msg).startswith("ERROR:"):
        # verify_and_commit already rolled back on syntax failure.
        return False, msg, str(msg)
    return True, msg, ""


def _absorb_worker_results(state: SupervisorState) -> dict[str, Any]:
    """Merge worker_results into the DAG before deciding the next dispatch."""
    dag = [dict(t) for t in (state.get("dag") or [])]
    by_id = {int(t["task_id"]): t for t in dag}
    completed = list(state.get("completed_summaries") or [])
    pending = [int(x) for x in (state.get("pending_tasks") or [])]
    open_sessions = [
        str(s) for s in (state.get("open_staging_sessions") or []) if str(s).strip()
    ]
    checkpoint_log = list(state.get("checkpoint_log") or [])

    for result in state.get("worker_results") or []:
        if not isinstance(result, dict):
            continue
        tid = int(result.get("task_id") or -1)
        task = by_id.get(tid)
        if task is None:
            continue
        ok = str(result.get("status") or "") == "completed"
        summary = str(result.get("summary") or "").strip()
        err = str(result.get("error") or "").strip()
        attempts = int(task.get("attempts") or 0) + 1
        task["attempts"] = attempts

        accepted, ck_msg, ck_err = _checkpoint_worker_result(result, ok=ok and bool(summary))
        if ck_msg:
            checkpoint_log.append(ck_msg)
        sid = str(result.get("staging_session_id") or "").strip()
        if sid and sid in open_sessions:
            open_sessions.remove(sid)

        if accepted and summary:
            task["status"] = "completed"
            task["summary"] = summary
            task["error"] = ""
            completed.append({"task_id": tid, "summary": summary})
            if tid in pending:
                pending.remove(tid)
        else:
            task["status"] = "failed"
            task["error"] = err or ck_err or "worker returned no summary"
            task["summary"] = ""
            max_a = int(state.get("max_task_attempts") or DEFAULT_MAX_TASK_ATTEMPTS)
            if attempts >= max_a and tid in pending:
                pending.remove(tid)

    return {
        "dag": dag,
        "completed_summaries": completed,
        "pending_tasks": pending,
        "worker_results": [],
        "active_task_ids": [],
        "open_staging_sessions": open_sessions,
        "checkpoint_log": checkpoint_log,
    }


def _all_terminal(dag: list[dict[str, Any]]) -> bool:
    if not dag:
        return False
    for t in dag:
        if str(t.get("status") or "") not in {"completed", "failed", "blocked"}:
            return False
    return True


def _finalize(dag: list[dict[str, Any]], completed: list[dict[str, Any]]) -> str:
    lines = ["Supervisor DAG complete."]
    for t in sorted(dag, key=lambda x: int(x.get("task_id") or 0)):
        tid = t.get("task_id")
        st = t.get("status")
        summary = (t.get("summary") or t.get("error") or "").strip()
        lines.append(f"- task {tid} [{st}]: {summary[:240]}")
    if not completed:
        lines.append("(no successful worker summaries)")
    return "\n".join(lines)


def _rollback_open_sessions(
    session_ids: list[str] | None,
    checkpoint_log: list[str] | None = None,
) -> list[str]:
    """Best-effort rollback of any leftover staging buffers."""
    from dana.tools.file_editor import rollback_workspace

    log = list(checkpoint_log or [])
    for sid in list(session_ids or []):
        sid_s = str(sid or "").strip()
        if not sid_s:
            continue
        msg = rollback_workspace(sid_s)
        _log(f"rollback_workspace (supervisor halt) session={sid_s}: {msg}")
        log.append(msg)
    return log


def supervisor_node(
    state: SupervisorState,
    *,
    planner: DagPlanner | None = None,
) -> dict[str, Any]:
    """Plan / evaluate / dispatch — never advances without dependency outcomes."""
    cycles = int(state.get("supervisor_cycles") or 0) + 1
    max_cycles = int(state.get("max_supervisor_cycles") or DEFAULT_MAX_SUPERVISOR_CYCLES)
    patch: dict[str, Any] = {"supervisor_cycles": cycles}

    # Absorb any worker payloads from the previous hop first.
    if state.get("worker_results"):
        absorbed = _absorb_worker_results(state)
        patch.update(absorbed)
        # Work from the absorbed view for readiness checks.
        merged: SupervisorState = {**state, **absorbed}  # type: ignore[misc]
    else:
        merged = state

    dag = [dict(t) for t in (merged.get("dag") or [])]

    # Step 1 — build the DAG once, then continue into dispatch in-node.
    completed_summaries = list(merged.get("completed_summaries") or [])
    pending_tasks = list(merged.get("pending_tasks") or [])
    if not dag:
        plan_fn = planner or heuristic_plan_dag
        planned = plan_fn(str(merged.get("user_prompt") or ""))
        if not planned:
            return {
                **patch,
                "status": "failed",
                "error": "supervisor failed to produce a DAG",
                "final_response": "ERROR: empty DAG",
                "active_task_ids": [],
            }
        dag = [dict(t) for t in planned]
        pending_tasks = [int(t["task_id"]) for t in dag]
        completed_summaries = []
        _log(f"planned DAG with {len(dag)} tasks: {pending_tasks}")

    # Loop guard — hard cycle ceiling.
    if cycles > max_cycles:
        _log(f"loop guard: supervisor_cycles={cycles} > max={max_cycles}")
        ck_log = _rollback_open_sessions(
            list(merged.get("open_staging_sessions") or []),
            list(merged.get("checkpoint_log") or []),
        )
        return {
            **patch,
            "dag": dag,
            "status": "failed",
            "error": f"supervisor cycle limit ({max_cycles}) exceeded",
            "final_response": _finalize(dag, completed_summaries),
            "active_task_ids": [],
            "pending_tasks": pending_tasks,
            "open_staging_sessions": [],
            "checkpoint_log": ck_log,
        }

    view: SupervisorState = {
        **merged,  # type: ignore[misc]
        "dag": dag,  # type: ignore[typeddict-item]
        "pending_tasks": pending_tasks,
        "completed_summaries": completed_summaries,
    }
    ready = ready_task_ids(view)

    if not ready:
        if _all_terminal(dag) or not (view.get("pending_tasks") or []):
            status = "completed"
            if any(str(t.get("status")) == "failed" for t in dag) and not any(
                str(t.get("status")) == "completed" for t in dag
            ):
                status = "failed"
            return {
                **patch,
                "dag": dag,
                "status": status,
                "final_response": _finalize(
                    dag, list(view.get("completed_summaries") or [])
                ),
                "active_task_ids": [],
                "pending_tasks": list(view.get("pending_tasks") or []),
            }
        # Pending remain but none are ready (failed/blocked parents) — halt.
        ck_log = _rollback_open_sessions(
            list(view.get("open_staging_sessions") or []),
            list(view.get("checkpoint_log") or []),
        )
        return {
            **patch,
            "dag": dag,
            "stall_count": int(merged.get("stall_count") or 0) + 1,
            "status": "failed",
            "error": "supervisor stalled: no ready tasks (dependency gate)",
            "final_response": _finalize(
                dag, list(view.get("completed_summaries") or [])
            ),
            "active_task_ids": [],
            "pending_tasks": list(view.get("pending_tasks") or []),
            "open_staging_sessions": [],
            "checkpoint_log": ck_log,
        }

    dispatch_key = ",".join(str(i) for i in ready)
    last_key = str(merged.get("last_dispatch_key") or "")
    stall = int(merged.get("stall_count") or 0)
    if dispatch_key == last_key and not state.get("worker_results"):
        stall += 1
        if stall > MAX_STALLS:
            _log(f"loop guard: repeated dispatch {dispatch_key!r}")
            return {
                **patch,
                "dag": dag,
                "stall_count": stall,
                "status": "failed",
                "error": "infinite re-dispatch prevented",
                "final_response": _finalize(
                    dag, list(view.get("completed_summaries") or [])
                ),
                "active_task_ids": [],
            }
    else:
        stall = 0

    # Mark ready tasks running for the worker hop.
    for t in dag:
        if int(t["task_id"]) in ready:
            t["status"] = "running"

    _log(f"dispatching workers for tasks {ready}")
    return {
        **patch,
        "dag": dag,
        "status": "awaiting_workers",
        "active_task_ids": ready,
        "last_dispatch_key": dispatch_key,
        "stall_count": stall,
        "pending_tasks": list(view.get("pending_tasks") or []),
        "completed_summaries": list(view.get("completed_summaries") or []),
    }


def route_after_supervisor(state: SupervisorState) -> str:
    """Conditional edge: workers while dispatching; else END."""
    status = str(state.get("status") or "")
    if status == "awaiting_workers" and (state.get("active_task_ids") or []):
        return WORKER_NODE
    return END_ROUTE


def make_supervisor_node(
    planner: DagPlanner | None = None,
) -> Callable[[SupervisorState], dict[str, Any]]:
    def _node(state: SupervisorState) -> dict[str, Any]:
        import json
        from json import JSONDecodeError

        from dana.graph.monitor_bus import publish_graph_error

        try:
            from pydantic import ValidationError
        except Exception:  # noqa: BLE001
            ValidationError = ()  # type: ignore[misc, assignment]

        try:
            result = supervisor_node(state, planner=planner)
        except (ValidationError, JSONDecodeError, json.JSONDecodeError, Exception) as exc:
            msg = f"supervisor_node crashed: {type(exc).__name__}: {exc}"
            _log(msg)
            print(f"[DagSupervisor] {msg}", flush=True)
            try:
                publish_graph_error(msg, exc=exc, node="supervisor", dump=True)
            except Exception:  # noqa: BLE001
                pass
            return {
                "status": "failed",
                "error": msg,
                "final_response": msg,
                "active_task_ids": [],
            }
        if str(result.get("status") or "") == "failed":
            err = str(
                result.get("error") or result.get("final_response") or "supervisor failed"
            )
            print(f"[DagSupervisor] STATUS=failed: {err}", flush=True)
            try:
                # Soft: broker may still repair via harness; do not emit done yet.
                publish_graph_error(
                    err,
                    node="supervisor",
                    dump=True,
                    terminal=False,
                    soft_fail=True,
                )
            except Exception:  # noqa: BLE001
                pass
        return result

    return _node


__all__ = (
    "DEFAULT_MAX_SUPERVISOR_CYCLES",
    "DEFAULT_MAX_TASK_ATTEMPTS",
    "END_ROUTE",
    "SUPERVISOR_NODE",
    "WORKER_NODE",
    "heuristic_plan_dag",
    "make_supervisor_node",
    "parse_dag_json",
    "plan_dag_with_llm",
    "ready_task_ids",
    "rollback_workspace",
    "route_after_supervisor",
    "supervisor_node",
    "validate_dag_solvability",
    "verify_and_commit",
)


def verify_and_commit(session_id: str) -> str:
    """Supervisor-facing commit gate (syntax verify then persist)."""
    from dana.tools.file_editor import verify_and_commit as _vac

    return _vac(session_id)


def rollback_workspace(session_id: str) -> str:
    """Supervisor-facing staging discard (live workspace untouched)."""
    from dana.tools.file_editor import rollback_workspace as _rb

    return _rb(session_id)


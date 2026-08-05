"""Meta-Broker node — multi-epic decomposition + closed-loop repair routing."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from dana.graph.state import BrokerState, Epic, empty_supervisor_state
from dana.system_health import llm_lock

BROKER_NODE = "broker"
SUPERVISOR_NODE = "supervisor"
STAGING_NODE = "staging_commit"
HARNESS_NODE = "runtime_harness"
END_ROUTE = "__end__"

DEFAULT_MAX_REPAIR_ATTEMPTS = 3

_EPIC_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:epic|phase|milestone)\s*(\d+)\s*[:.)\-]\s*(.+?)(?="
    r"(?:\n\s*(?:epic|phase|milestone)\s*\d+\s*[:.)\-])|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_NUMBERED_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*(\d+)\s*[.)]\s+(.+?)(?=(?:\n\s*\d+\s*[.)]\s+)|\Z)",
    re.DOTALL,
)


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("MetaBroker", msg)
    except Exception:  # noqa: BLE001
        print(f"[MetaBroker] {msg}", flush=True)


_FILE_TOKEN_RE = re.compile(
    r"([\w./\\-]+\.(?:py|pyi|md|txt|json))\b",
    re.I,
)


def _collect_files_from_text(*blobs: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for blob in blobs:
        for m in _FILE_TOKEN_RE.finditer(blob or ""):
            rel = m.group(1).replace("\\", "/")
            key = rel.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(rel)
    return found


def _snapshot_epic_artifacts(
    epic: dict[str, Any],
    *,
    workspace: str | None = None,
) -> list[dict[str, str]]:
    """Read on-disk files mentioned by an epic (paths + capped content)."""
    from pathlib import Path

    from dana.paths import PROJECT_ROOT

    root = Path(workspace or PROJECT_ROOT).resolve()
    paths = _collect_files_from_text(
        str(epic.get("goal") or ""),
        str(epic.get("title") or ""),
        str(epic.get("validation_command") or ""),
    )
    out: list[dict[str, str]] = []
    for rel in paths:
        path = root / rel
        if not path.is_file():
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(body) > 8000:
            body = body[:8000] + "\n...[truncated]..."
        out.append({"path": rel, "content": body, "epic_id": str(epic.get("epic_id"))})
    return out


def _format_completed_epic_artifacts(artifacts: list[dict[str, Any]]) -> str:
    """Accumulate ALL prior completed epic files for Deep Artifact Context."""
    if not artifacts:
        return ""
    blocks: list[str] = ["### Completed Epic Artifacts"]
    # De-dupe by path, keeping the latest snapshot of each file.
    latest: dict[str, dict[str, Any]] = {}
    for row in artifacts:
        path = str(row.get("path") or "").strip()
        if not path:
            continue
        latest[path] = row
    for path, row in latest.items():
        body = str(row.get("content") or "")
        eid = row.get("epic_id")
        blocks.append(f"#### `{path}` (from epic {eid})")
        blocks.append("```python")
        blocks.append(body.rstrip())
        blocks.append("```")
    return "\n".join(blocks)


def _prompt_with_completed_artifacts(
    goal: str,
    artifacts: list[dict[str, Any]],
) -> str:
    goal_s = (goal or "").strip()
    parts: list[str] = []
    try:
        from dana.graph.artifact_manifest import format_manifest_contract_block

        contract = format_manifest_contract_block()
        if contract:
            parts.append(contract)
    except Exception:  # noqa: BLE001
        pass
    block = _format_completed_epic_artifacts(artifacts)
    if block:
        parts.append(block)
        parts.append(
            "Use the Completed Epic Artifacts above as ground truth. "
            "Do not reinvent conflicting APIs."
        )
    parts.append(f"CURRENT EPIC GOAL:\n{goal_s}")
    return "\n\n".join(p for p in parts if p)


def _infer_validation_command(goal: str, title: str = "") -> str:
    """Derive a targeted validation command from epic text (never global pytest)."""
    text = f"{title}\n{goal}".replace("\\", "/")
    low = text.lower()
    # Explicit pytest path (preferred).
    m = re.search(
        r"(tests/[\w./\-]*test_[\w./\-]+\.py)",
        text,
        re.IGNORECASE,
    )
    if m:
        return f"python -m pytest {m.group(1)} -q"
    m = re.search(r"\b(test_[\w.\-]+\.py)\b", text, re.IGNORECASE)
    if m and ("pytest" in low or "test suite" in low or "write a pytest" in low):
        rel = m.group(1)
        if not rel.lower().startswith("tests/"):
            rel = f"tests/{rel}"
        return f"python -m pytest {rel} -q"
    # Tk / animation scripts — compile only (running Tk can hang headless).
    m = re.search(r"\b([\w./\-]*popup_animation\.py)\b", text, re.IGNORECASE)
    if m or ("tkinter" in low and "animation" in low):
        path = m.group(1) if m else "popup_animation.py"
        return f"python -m py_compile {path}"
    # Generic single-module epic — compile the named .py even without verbs.
    m = re.search(r"\b([\w./\-]+\.py)\b", text)
    if m:
        rel = m.group(1).replace("\\", "/")
        base = rel.rsplit("/", 1)[-1].lower()
        if base.startswith("test_") or rel.lower().startswith("tests/"):
            pytest_rel = rel if rel.lower().startswith("tests/") else f"tests/{base}"
            return f"python -m pytest {pytest_rel} -q"
        return f"python -m py_compile {rel}"
    return ""


def _normalize_epic(raw: dict[str, Any], *, fallback_id: int) -> Epic:
    eid = raw.get("epic_id", fallback_id)
    try:
        epic_id = int(eid)
    except (TypeError, ValueError):
        epic_id = fallback_id
    title = str(raw.get("title") or raw.get("name") or f"Epic {epic_id}").strip()
    goal = str(raw.get("goal") or raw.get("prompt") or raw.get("action") or "").strip()
    if not goal:
        goal = title
    epic: Epic = {
        "epic_id": epic_id,
        "title": title,
        "goal": goal,
        "status": "pending",
        "repair_attempts": int(raw.get("repair_attempts") or 0),
    }
    cmd = str(raw.get("validation_command") or "").strip()
    if not cmd:
        cmd = _infer_validation_command(goal, title)
    if cmd:
        # Block global pytest even if the LLM emits it.
        lowered = cmd.lower().replace("\\", "/").strip()
        if lowered in {
            "pytest",
            "pytest -q",
            "python -m pytest",
            "python -m pytest -q",
        }:
            cmd = _infer_validation_command(goal, title) or (
                "python -m compileall .dana_scratch"
            )
        epic["validation_command"] = cmd
    if raw.get("workspace_path"):
        epic["workspace_path"] = str(raw["workspace_path"])
    return epic


def split_epics_with_llm(macro_intent: str) -> list[Epic]:
    """Optional LLM epic decomposition (hybrid cloud or local Ollama).

    Structured ``Epic N:`` / numbered macros stay heuristic (hermetic / offline).
    Unstructured macros use ``ask_planner_structured`` (cloud when Hybrid is on
    and a key is present; otherwise local Ollama). Workers / harness are never
    invoked here — planning only.
    """
    text = str(macro_intent or "").strip()
    if not text:
        return []
    # Prefer explicit structure without spending an LLM turn.
    heuristic = heuristic_split_epics(text)
    if len(heuristic) >= 2:
        return heuristic
    try:
        from dana.graph.cloud_planner import planner_mode_label, publish_planner_mode
        from dana.llm_client import ask_planner_structured
        from dana.llm_schemas import EpicPlan

        mode = publish_planner_mode(warn_missing_key=True)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Meta-Broker. Decompose the macro intent into "
                    "sequential epics. Return ONLY JSON matching EpicPlan: "
                    '{"epics":[{"epic_id":int,"title":str,"goal":str,'
                    '"validation_command":str},...]}. '
                    "Each goal must be self-contained. Prefer 1–4 epics. "
                    "For each Epic, you MUST provide a highly specific "
                    "validation_command to test ONLY that epic's work. "
                    "Examples: `python -m pytest tests/test_rate_limiter.py -q` "
                    "or `python -m py_compile popup_animation.py` or "
                    "`python popup_animation.py`. "
                    "NEVER use a bare global `pytest` / `python -m pytest -q` "
                    "over the whole repository. "
                    "When generating Python code for Epics, rely strictly on "
                    "Python Standard Library modules (e.g., json, math, os, sys, "
                    "deque) unless third-party packages are explicitly requested "
                    "in the prompt. No markdown, no prose."
                ),
            },
            {"role": "user", "content": f"MACRO INTENT:\n{text}"},
        ]
        print(
            f"[MetaBroker] epic plan CALL ask_planner_structured "
            f"mode={mode or planner_mode_label()}",
            flush=True,
        )
        with llm_lock:
            plan = ask_planner_structured(messages, EpicPlan, max_retries=2)
        rows = plan.to_epics()  # type: ignore[attr-defined]
        epics = [
            _normalize_epic(dict(r), fallback_id=i)
            for i, r in enumerate(rows or [], start=1)
        ]
        if epics:
            _log(f"LLM epic plan ok count={len(epics)} mode={mode}")
            return epics
    except Exception as exc:  # noqa: BLE001
        _log(f"LLM epic plan failed ({exc}); using heuristic")
        print(
            f"[MetaBroker] epic plan FAIL: {type(exc).__name__}: {exc}",
            flush=True,
        )
    return heuristic


def heuristic_split_epics(macro_intent: str) -> list[Epic]:
    """Split a macro prompt into sequential epics (LLM-free / hermetic).

    Preference order:
      1. Explicit ``Epic N:`` / ``Phase N:`` headers
      2. Numbered top-level blocks (``1. …``) when ≥2 blocks mention distinct goals
      3. Paragraph chunks separated by blank lines (when ≥2)
      4. Single epic wrapping the whole macro intent
    """
    text = str(macro_intent or "").strip()
    if not text:
        return []

    # Inline "Epic N:" markers on one line (common /broker prompts).
    inline_parts = [
        p.strip()
        for p in re.split(
            r"(?=(?:Epic|Phase|Milestone)\s*\d+\s*[:.)\-])",
            text,
            flags=re.IGNORECASE,
        )
        if p.strip()
    ]
    inline_epics = [
        p
        for p in inline_parts
        if re.match(r"(?:Epic|Phase|Milestone)\s*\d+\s*[:.)\-]", p, re.I)
    ]
    if len(inline_epics) >= 2:
        epics: list[Epic] = []
        for i, body in enumerate(inline_epics, start=1):
            m = re.match(
                r"(?:Epic|Phase|Milestone)\s*(\d+)\s*[:.)\-]\s*(.+)",
                body,
                re.I | re.DOTALL,
            )
            if not m:
                continue
            try:
                eid = int(m.group(1))
            except ValueError:
                eid = i
            goal = str(m.group(2) or "").strip()
            title = goal.split("\n", 1)[0].strip()[:80] or f"Epic {eid}"
            epics.append(
                _normalize_epic(
                    {"epic_id": eid, "title": title, "goal": goal},
                    fallback_id=i,
                )
            )
        if len(epics) >= 2:
            return epics

    headers = list(_EPIC_HEADER_RE.finditer(text))
    if len(headers) >= 2:
        epics: list[Epic] = []
        for i, m in enumerate(headers, start=1):
            try:
                eid = int(m.group(1))
            except ValueError:
                eid = i
            body = str(m.group(2) or "").strip()
            title = body.split("\n", 1)[0].strip()[:80] or f"Epic {eid}"
            epics.append(
                _normalize_epic(
                    {"epic_id": eid, "title": title, "goal": body},
                    fallback_id=i,
                )
            )
        return epics

    numbered = list(_NUMBERED_BLOCK_RE.finditer(text))
    if len(numbered) >= 2:
        # Treat as epics only when blocks look like high-level goals (not micro-steps).
        # Heuristic: each block mentions a file OR is ≥40 chars.
        file_re = re.compile(r"[\w./\\-]+\.(?:py|md|txt|json|toml)\b", re.I)
        rich = [
            m
            for m in numbered
            if file_re.search(m.group(2) or "") or len((m.group(2) or "").strip()) >= 40
        ]
        if len(rich) >= 2:
            epics = []
            for i, m in enumerate(rich, start=1):
                body = str(m.group(2) or "").strip()
                title = body.split("\n", 1)[0].strip()[:80]
                epics.append(
                    _normalize_epic(
                        {"epic_id": i, "title": title, "goal": body},
                        fallback_id=i,
                    )
                )
            return epics

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) >= 2:
        return [
            _normalize_epic(
                {
                    "epic_id": i,
                    "title": p.split("\n", 1)[0].strip()[:80],
                    "goal": p,
                },
                fallback_id=i,
            )
            for i, p in enumerate(paragraphs, start=1)
        ]

    return [
        _normalize_epic(
            {"epic_id": 1, "title": "Primary epic", "goal": text},
            fallback_id=1,
        )
    ]


def _fresh_supervisor_fields(prompt: str, state: BrokerState) -> dict[str, Any]:
    """Isolated supervisor window for the active epic (no prior DAG / history)."""
    fresh = empty_supervisor_state(
        prompt,
        max_supervisor_cycles=int(state.get("max_supervisor_cycles") or 12),
        max_task_attempts=int(state.get("max_task_attempts") or 2),
    )
    return dict(fresh)


def _active_epic(epics: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    if 0 <= idx < len(epics):
        return epics[idx]
    return None


def _inject_repair_prompt(epic: dict[str, Any], feedback: dict[str, Any]) -> str:
    stderr = str(feedback.get("stderr") or "").strip()
    stdout = str(feedback.get("stdout") or "").strip()
    command = str(feedback.get("command") or "").strip()
    exit_code = feedback.get("exit_code")
    trace = stderr or stdout or "(no output captured)"
    if len(trace) > 4000:
        trace = trace[:4000] + "\n...[truncated]..."
    triage = str(feedback.get("repair_triage") or "").strip().upper()
    target = str(feedback.get("repair_target_filepath") or "").strip()
    attempts = int(
        feedback.get("repair_attempts") or epic.get("repair_attempts") or 0
    )
    triage_line = ""
    if triage == "COLLECTION" and target:
        triage_line = (
            f"\nExit-2 Collection Failure → TARGET FILEPATH: {target}\n"
            "This is a syntax/import error, NOT a logic failure. "
            f"Fix imports, typos, or indentation in `{target}` "
            "(and create missing modules named by the traceback).\n"
        )
    elif triage in {"TEST", "CODE"} and target:
        blame = (
            "the test logic is flawed — fix the TEST file only"
            if triage == "TEST"
            else "the implementation is wrong — fix the CODE/impl file only"
        )
        triage_line = (
            f"\nBidirectional Repair Triage: {triage} → TARGET FILEPATH: {target}\n"
            f"({blame}). Use file_editor to repair `{target}`.\n"
            "Do not destroy a valid counterpart file to satisfy a bad assertion.\n"
        )
    return (
        f"REPAIR_ATTEMPTS: {attempts}\n"
        f"BUG FIX for epic {epic.get('epic_id')}: {epic.get('title')}\n"
        f"Original goal:\n{epic.get('goal')}\n"
        f"{triage_line}\n"
        f"Validation command `{command}` failed with exit_code={exit_code}.\n"
        f"Captured stderr / stack trace:\n```\n{trace}\n```\n"
        "Repair the failing code so the validation command passes. "
        "Keep changes minimal and scoped to this epic."
    )


def _gc_between_epics(label: str = "") -> None:
    """Eagerly dump Python AST/string residue between epic boundaries."""
    import gc

    try:
        n = gc.collect()
        _log(f"gc.collect() after epic{(' ' + label) if label else ''} freed={n}")
    except Exception:  # noqa: BLE001
        try:
            gc.collect()
        except Exception:  # noqa: BLE001
            pass


def _update_manifest_after_epic(
    artifacts: list[dict[str, Any]],
    *,
    epic_id: Any = None,
    workspace: str | None = None,
) -> None:
    try:
        from dana.graph.artifact_manifest import update_manifest_from_epic_artifacts

        update_manifest_from_epic_artifacts(
            artifacts, epic_id=epic_id, workspace=workspace
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"manifest update skipped ({exc})")


def broker_node(state: BrokerState) -> dict[str, Any]:
    """Plan epics, dispatch isolated supervisors, advance or repair from harness."""
    import json
    from json import JSONDecodeError

    from dana.graph.monitor_bus import publish_graph_error

    try:
        from pydantic import ValidationError
    except Exception:  # noqa: BLE001
        ValidationError = ()  # type: ignore[misc, assignment]

    try:
        return _broker_node_impl(state)
    except (ValidationError, JSONDecodeError, json.JSONDecodeError, Exception) as exc:
        msg = f"broker_node crashed: {type(exc).__name__}: {exc}"
        _log(msg)
        print(f"[MetaBroker] {msg}", flush=True)
        try:
            publish_graph_error(msg, exc=exc, node="broker", dump=True)
        except Exception:  # noqa: BLE001
            pass
        return {
            "broker_phase": "done",
            "status": "failed",
            "error": msg,
            "final_response": msg,
            "epic_log": list(state.get("epic_log") or []) + [msg],
        }


def _broker_node_impl(state: BrokerState) -> dict[str, Any]:
    """Inner Meta-Broker state machine (exceptions surfaced by ``broker_node``)."""
    macro = str(state.get("macro_intent") or state.get("user_prompt") or "").strip()
    epics: list[dict[str, Any]] = [dict(e) for e in (state.get("epics") or [])]
    idx = int(state.get("active_epic_index") or 0)
    max_repairs = int(state.get("max_repair_attempts") or DEFAULT_MAX_REPAIR_ATTEMPTS)
    phase = str(state.get("broker_phase") or "plan")
    feedback = dict(state.get("runtime_feedback") or {})
    epic_log = list(state.get("epic_log") or [])
    completed_artifacts: list[dict[str, Any]] = [
        dict(a) for a in (state.get("completed_epic_artifacts") or []) if isinstance(a, dict)
    ]

    def _track(msg: str, *, epic_title: str = "", status: str = "", terminal: bool = False) -> None:
        try:
            from dana.graph.task_tracker import emit_meta_broker_telemetry

            emit_meta_broker_telemetry(
                task_id="meta_broker",
                prompt=macro,
                phase=phase,
                status=status or str(state.get("status") or ""),
                message=msg,
                epic_title=epic_title,
                terminal=terminal,
            )
        except Exception:  # noqa: BLE001
            pass

        # Forward to parent GUI when Meta-Broker runs in an isolated process.
        try:
            from dana.graph.meta_broker_process import child_queue_put

            child_queue_put(
                {
                    "type": "telemetry",
                    "message": msg,
                    "phase": phase,
                    "status": status or str(state.get("status") or ""),
                    "epic_title": epic_title,
                    "terminal": terminal,
                }
            )
        except Exception:  # noqa: BLE001
            pass

    # --- Plan ---------------------------------------------------------------
    if not epics:
        # Hybrid cloud (when enabled + key) or local Ollama; heuristic fallback.
        epics = [dict(e) for e in split_epics_with_llm(macro)]
        if not epics:
            _track(
                "Meta-Broker could not derive epics",
                status="failed",
                terminal=True,
            )
            return {
                "epics": [],
                "broker_phase": "done",
                "status": "failed",
                "error": "broker could not derive epics from macro_intent",
                "final_response": "No epics planned.",
            }
        epic_log.append(f"planned {len(epics)} epic(s)")
        _log(f"planned {len(epics)} epic(s) from macro intent")
        _track(f"Planned {len(epics)} epic(s)", status="planning")
        idx = 0
        phase = "dispatch_epic"

    # --- Feedback: success → advance; failure → repair (≤3) ---------------
    if phase == "feedback" and feedback:
        epic = _active_epic(epics, idx)
        if epic is None:
            return {
                "epics": epics,
                "broker_phase": "done",
                "status": "failed",
                "error": "active_epic_index out of range",
                "epic_log": epic_log,
                "completed_epic_artifacts": completed_artifacts,
            }
        if bool(feedback.get("success")):
            epic["status"] = "completed"
            epics[idx] = epic
            # Deep Artifact Context: accumulate ALL completed epic files.
            new_arts = _snapshot_epic_artifacts(
                epic,
                workspace=str(
                    state.get("workspace_path")
                    or feedback.get("workspace_path")
                    or ""
                )
                or None,
            )
            if new_arts:
                completed_artifacts.extend(new_arts)
                epic_log.append(
                    f"epic {epic.get('epic_id')} artifacts stored: "
                    + ", ".join(a["path"] for a in new_arts)
                )
            epic_log.append(f"epic {epic.get('epic_id')} validated OK")
            _log(f"epic {epic.get('epic_id')} passed harness")
            _track(
                f"Epic {epic.get('epic_id')} validated OK",
                epic_title=str(epic.get("title") or ""),
                status="planning",
            )
            # Contract schema + memory reclaim between sequential epics.
            _update_manifest_after_epic(
                new_arts,
                epic_id=epic.get("epic_id"),
                workspace=str(state.get("workspace_path") or "") or None,
            )
            _gc_between_epics(str(epic.get("epic_id") or ""))
            idx += 1
            if idx >= len(epics):
                titles = ", ".join(str(e.get("title") or e.get("epic_id")) for e in epics)
                _track(
                    f"All {len(epics)} epic(s) completed",
                    status="completed",
                    terminal=True,
                )
                _gc_between_epics("all-done")
                return {
                    "epics": epics,
                    "active_epic_index": idx,
                    "broker_phase": "done",
                    "status": "completed",
                    "runtime_feedback": feedback,
                    "epic_log": epic_log,
                    "completed_epic_artifacts": completed_artifacts,
                    "final_response": f"All {len(epics)} epic(s) completed: {titles}",
                    "user_prompt": "",
                    "dag": [],
                    "pending_tasks": [],
                    "active_task_ids": [],
                    "worker_results": [],
                }
            phase = "dispatch_epic"
        else:
            # repair_attempts is incremented in runtime_harness on each failure.
            attempts = int(
                feedback.get("repair_attempts")
                or epic.get("repair_attempts")
                or 0
            )
            epic["repair_attempts"] = attempts
            # Log immediately so hangs in prompt assembly are visible.
            _log(f"repair iteration {attempts} for epic {epic.get('epic_id')}")
            # Allow attempts == max_repairs as the final repair (escalation window).
            if attempts > max_repairs:
                epic["status"] = "failed"
                epics[idx] = epic
                epic_log.append(
                    f"epic {epic.get('epic_id')} failed after {attempts} repair(s)"
                )
                _log(f"epic {epic.get('epic_id')} exhausted repairs")
                _track(
                    f"Epic {epic.get('epic_id')} failed after {attempts} repair(s)",
                    epic_title=str(epic.get("title") or ""),
                    status="failed",
                    terminal=True,
                )
                _gc_between_epics(f"fail-{epic.get('epic_id')}")
                return {
                    "epics": epics,
                    "active_epic_index": idx,
                    "broker_phase": "done",
                    "status": "failed",
                    "error": (
                        f"epic {epic.get('epic_id')} failed validation after "
                        f"{attempts} repair attempt(s)"
                    ),
                    "runtime_feedback": feedback,
                    "epic_log": epic_log,
                    "completed_epic_artifacts": completed_artifacts,
                    "final_response": (
                        f"Epic {epic.get('epic_id')} failed:\n"
                        f"{str(feedback.get('stderr') or '')[:500]}"
                    ),
                }
            epic["status"] = "repairing"
            epics[idx] = epic
            repair_prompt = _inject_repair_prompt(epic, feedback)
            # Carry prior epic artifacts into repair window as well.
            repair_prompt = _prompt_with_completed_artifacts(
                repair_prompt, completed_artifacts
            )
            triage = str(feedback.get("repair_triage") or "").strip().upper()
            target = str(feedback.get("repair_target_filepath") or "").strip()
            if triage:
                epic_log.append(
                    f"epic {epic.get('epic_id')} Bidirectional Repair Triage="
                    f"{triage} target={target or '(none)'}"
                )
            if feedback.get("collection_failure"):
                epic_log.append(
                    f"epic {epic.get('epic_id')} Exit-2 collection failure → "
                    f"repair {target or 'epic file'}"
                )
            epic_log.append(
                f"epic {epic.get('epic_id')} repair iteration {attempts}/{max_repairs}"
            )
            patch = _fresh_supervisor_fields(repair_prompt, state)
            return {
                **patch,
                "epics": epics,
                "active_epic_index": idx,
                "broker_phase": "repair",
                "status": "planning",
                "runtime_feedback": feedback,
                "epic_log": epic_log,
                "completed_epic_artifacts": completed_artifacts,
                "macro_intent": macro,
                "max_repair_attempts": max_repairs,
            }

    # --- Dispatch next / current epic --------------------------------------
    epic = _active_epic(epics, idx)
    if epic is None:
        return {
            "epics": epics,
            "active_epic_index": idx,
            "broker_phase": "done",
            "status": "completed",
            "epic_log": epic_log,
            "completed_epic_artifacts": completed_artifacts,
            "final_response": "No remaining epics.",
        }

    if str(epic.get("status") or "") in {"pending", "active", "repairing"} or phase in {
        "plan",
        "dispatch_epic",
        "advance",
    }:
        if str(epic.get("status") or "") == "pending":
            epic["status"] = "active"
            epics[idx] = epic
        goal = str(epic.get("goal") or "").strip()
        # Deep Artifact Context: Epic N sees ALL prior completed epic files.
        prompt = _prompt_with_completed_artifacts(goal, completed_artifacts)
        patch = _fresh_supervisor_fields(prompt, state)
        epic_log.append(f"dispatch epic {epic.get('epic_id')}: {epic.get('title')}")
        _track(
            f"Starting Epic {epic.get('epic_id')}: {epic.get('title')}",
            epic_title=str(epic.get("title") or ""),
            status="planning",
        )
        if completed_artifacts:
            epic_log.append(
                f"dispatch epic {epic.get('epic_id')} with "
                f"{len({a.get('path') for a in completed_artifacts})} prior artifact(s)"
            )
        _log(f"dispatch epic {epic.get('epic_id')} → supervisor")
        _gc_between_epics(f"dispatch-{epic.get('epic_id')}")
        return {
            **patch,
            "epics": epics,
            "active_epic_index": idx,
            "broker_phase": "await_supervisor",
            "status": "planning",
            "runtime_feedback": {},
            "epic_log": epic_log,
            "completed_epic_artifacts": completed_artifacts,
            "macro_intent": macro,
            "max_repair_attempts": max_repairs,
        }

    return {
        "epics": epics,
        "active_epic_index": idx,
        "broker_phase": "done",
        "status": "failed",
        "error": f"broker stuck in phase={phase!r}",
        "epic_log": epic_log,
        "completed_epic_artifacts": completed_artifacts,
    }


def route_after_broker(state: BrokerState) -> str:
    """Broker → Supervisor while work remains; else END."""
    phase = str(state.get("broker_phase") or "")
    status = str(state.get("status") or "")
    if phase == "done" or status in {"completed", "failed"}:
        # Only end when broker explicitly finished (not mid-supervisor planning).
        if phase == "done":
            return END_ROUTE
    if phase in {"await_supervisor", "repair"}:
        return SUPERVISOR_NODE
    if status == "planning" and (state.get("user_prompt") or "").strip():
        return SUPERVISOR_NODE
    return END_ROUTE


def staging_commit_node(state: BrokerState) -> dict[str, Any]:
    """Finalize any remaining staging sessions before the runtime harness."""
    from dana.tools.file_editor import rollback_workspace, verify_and_commit

    open_sessions = [
        str(s) for s in (state.get("open_staging_sessions") or []) if str(s).strip()
    ]
    checkpoint_log = list(state.get("checkpoint_log") or [])
    remaining: list[str] = []
    ok = str(state.get("status") or "") == "completed"
    for sid in open_sessions:
        if ok:
            msg = verify_and_commit(sid)
        else:
            msg = rollback_workspace(sid)
        checkpoint_log.append(msg)
        if str(msg).startswith("ERROR:"):
            remaining.append(sid)
    return {
        "open_staging_sessions": remaining,
        "checkpoint_log": checkpoint_log,
        "broker_phase": "validate",
    }


def route_after_supervisor_to_harness(state: BrokerState) -> str:
    """Supervisor → workers while dispatching; else staging → harness path."""
    status = str(state.get("status") or "")
    if status == "awaiting_workers" and (state.get("active_task_ids") or []):
        return "workers"
    return STAGING_NODE


def make_broker_node() -> Callable[[BrokerState], dict[str, Any]]:
    def _node(state: BrokerState) -> dict[str, Any]:
        # broker_node already traps ValidationError / JSONDecodeError / Exception.
        return broker_node(state)

    return _node


__all__ = (
    "BROKER_NODE",
    "DEFAULT_MAX_REPAIR_ATTEMPTS",
    "END_ROUTE",
    "HARNESS_NODE",
    "STAGING_NODE",
    "SUPERVISOR_NODE",
    "broker_node",
    "heuristic_split_epics",
    "make_broker_node",
    "route_after_broker",
    "route_after_supervisor_to_harness",
    "split_epics_with_llm",
    "staging_commit_node",
)

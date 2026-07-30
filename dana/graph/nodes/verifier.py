"""Closed-loop verification node (Generator-Critic physical evidence gate).

After tool / sub-agent execution, the corridor lands here before END.
Heuristics check filesystem artifacts, JSON schemas, UIA trees, and optional
process liveness. Inject ``verify_fn`` for unit tests / offline benches.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dana.schema import ReactGraphState

logger = logging.getLogger(__name__)

MAX_VERIFICATION_ATTEMPTS = 3

# ``(state) -> {verified: bool, evidence: Any}`` — attempts are owned by the node.
VerifyFn = Callable[[ReactGraphState], dict[str, Any]]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _resolve_targets(state: ReactGraphState | dict[str, Any]) -> dict[str, Any]:
    """Pull verification_targets from top-level state or env_context."""
    st = state or {}
    direct = st.get("verification_targets")
    if isinstance(direct, dict) and direct:
        return dict(direct)
    env = _as_dict(st.get("env_context"))
    nested = env.get("verification_targets")
    if isinstance(nested, dict) and nested:
        return dict(nested)
    return {}


def _check_file_nonempty(path: str | Path) -> tuple[bool, str]:
    p = Path(path)
    if not p.is_file():
        return False, f"file missing: {p}"
    try:
        size = p.stat().st_size
    except OSError as exc:
        return False, f"file unreadable: {p} ({exc})"
    if size <= 0:
        return False, f"file empty: {p}"
    return True, f"file ok ({size} bytes): {p}"


def _check_json_schema(
    path: str | Path,
    required_keys: list[str] | tuple[str, ...] | None = None,
) -> tuple[bool, str, Any]:
    p = Path(path)
    ok, msg = _check_file_nonempty(p)
    if not ok:
        return False, msg, None
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"invalid JSON at {p}: {exc}", None
    if data in (None, "", {}, []):
        return False, f"JSON empty at {p}", data
    keys = list(required_keys or [])
    if keys:
        if not isinstance(data, dict):
            return False, f"JSON root must be object at {p}", data
        missing = [k for k in keys if k not in data]
        if missing:
            return False, f"JSON missing keys {missing} at {p}", data
    return True, f"JSON schema ok at {p}", data


def _uia_nodes_from_state(state: ReactGraphState | dict[str, Any]) -> list[Any]:
    st = state or {}
    for key in ("uia_nodes", "uia_tree", "uia_controls"):
        nodes = st.get(key)
        if isinstance(nodes, list) and nodes:
            return list(nodes)
    env = _as_dict(st.get("env_context"))
    for key in ("uia_nodes", "uia_tree", "uia_controls"):
        nodes = env.get(key)
        if isinstance(nodes, list) and nodes:
            return list(nodes)
    vr = _as_dict(st.get("verification_result"))
    ev = vr.get("evidence")
    if isinstance(ev, dict):
        nodes = ev.get("uia_nodes")
        if isinstance(nodes, list) and nodes:
            return list(nodes)
    return []


def _node_has_bounds(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    for key in ("bbox", "bounds", "bounds_norm", "rect", "rectangle"):
        val = node.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)) and len(val) >= 4:
            return True
        if isinstance(val, dict) and any(
            k in val for k in ("left", "top", "right", "bottom", "x", "y", "width", "height")
        ):
            return True
    return False


def _check_process_active(
    name: str,
    *,
    process_checker: Callable[[str], bool] | None = None,
) -> tuple[bool, str]:
    target = str(name or "").strip()
    if not target:
        return False, "process name empty"
    if process_checker is not None:
        try:
            alive = bool(process_checker(target))
        except Exception as exc:  # noqa: BLE001
            return False, f"process check error: {exc}"
        return (
            (True, f"process active: {target}")
            if alive
            else (False, f"process not active: {target}")
        )
    # Offline-friendly default: treat explicit state flag as authority when set.
    return False, f"no process_checker for: {target}"


def default_physical_evidence_check(
    state: ReactGraphState | dict[str, Any],
    *,
    process_checker: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Heuristic / offline-friendly evidence scan.

    When ``verification_targets`` are present, all declared checks must pass.
    Otherwise soft-pass on a healthy halt (preserves corridors without targets).
    """
    st = state or {}
    targets = _resolve_targets(st)
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    evidence: dict[str, Any] = {"checks": checks}

    if not targets:
        # Soft success path: successful tool halt with spoken / obs payload.
        err = st.get("execution_error")
        healthy = not (err is not None and str(err).strip())
        spoken = str(st.get("final_raw") or st.get("last_obs") or "").strip()
        if bool(st.get("halt")) and healthy and spoken:
            # Also accept inline UIA payloads even without explicit targets.
            nodes = _uia_nodes_from_state(st)
            if nodes:
                evidence["uia_nodes"] = nodes
                evidence["mode"] = "soft_halt_with_uia"
            else:
                evidence["mode"] = "soft_halt"
            return {"verified": True, "evidence": evidence}
        failures.append("no verification_targets and no healthy halt evidence")
        evidence["failures"] = failures
        evidence["mode"] = "soft_fail"
        return {"verified": False, "evidence": evidence}

    # --- file exists / non-empty ---
    files = targets.get("files") or targets.get("paths") or []
    if isinstance(files, (str, Path)):
        files = [files]
    for item in files:
        if isinstance(item, dict):
            path = item.get("path") or item.get("file") or ""
            require_nonempty = bool(item.get("non_empty", True))
        else:
            path = item
            require_nonempty = True
        ok, msg = _check_file_nonempty(path) if require_nonempty else (
            (True, f"file exists: {path}")
            if Path(path).is_file()
            else (False, f"file missing: {path}")
        )
        checks.append({"kind": "file", "ok": ok, "detail": msg})
        if not ok:
            failures.append(msg)

    # --- JSON schema ---
    json_specs = targets.get("json_schema") or targets.get("json") or []
    if isinstance(json_specs, dict):
        json_specs = [json_specs]
    for spec in json_specs:
        if not isinstance(spec, dict):
            continue
        path = spec.get("path") or spec.get("file") or ""
        required = spec.get("required_keys") or spec.get("keys") or []
        ok, msg, data = _check_json_schema(path, required_keys=list(required))
        checks.append({"kind": "json_schema", "ok": ok, "detail": msg})
        if ok and data is not None:
            evidence["json_data"] = data
        if not ok:
            failures.append(msg)

    # --- UIA tree returned with bounds ---
    want_uia = bool(
        targets.get("uia")
        or targets.get("uia_tree")
        or targets.get("uia_nodes")
        or targets.get("require_uia")
    )
    if want_uia:
        nodes = _uia_nodes_from_state(st)
        # Allow targets to inject expected nodes directly.
        injected = targets.get("expected_uia_nodes")
        if isinstance(injected, list) and injected and not nodes:
            nodes = list(injected)
        has_bounds = any(_node_has_bounds(n) for n in nodes)
        ok = bool(nodes) and has_bounds
        msg = (
            f"UIA nodes={len(nodes)} bounds={has_bounds}"
            if ok
            else "UIA tree missing or no bounding boxes"
        )
        checks.append({"kind": "uia", "ok": ok, "detail": msg})
        evidence["uia_nodes"] = nodes
        if not ok:
            failures.append(msg)

    # --- process active ---
    proc = targets.get("process") or targets.get("process_name")
    if proc:
        name = proc.get("name") if isinstance(proc, dict) else str(proc)
        checker = process_checker
        if checker is None and isinstance(proc, dict) and callable(proc.get("checker")):
            checker = proc["checker"]
        # State flag override for offline benches.
        if checker is None and isinstance(st.get("process_active"), bool):
            checker = lambda _n, flag=bool(st.get("process_active")): flag  # noqa: E731
        ok, msg = _check_process_active(str(name or ""), process_checker=checker)
        checks.append({"kind": "process", "ok": ok, "detail": msg})
        if not ok:
            failures.append(msg)

    # --- REPL / numerical result ---
    if targets.get("repl_result") or targets.get("require_repl_success"):
        obs = str(st.get("last_obs") or st.get("final_raw") or "")
        err = st.get("execution_error")
        has_err = err is not None and str(err).strip()
        assertion_fail = "AssertionError" in obs
        # Prefer an explicit numeric payload when provided.
        numeric = st.get("repl_numeric_result")
        if numeric is None:
            env = _as_dict(st.get("env_context"))
            numeric = env.get("repl_numeric_result")
        ok = (not has_err) and (not assertion_fail) and (
            numeric is not None or ("exit_code=0" in obs) or bool(obs.strip())
        )
        msg = (
            f"repl ok numeric={numeric!r}"
            if ok
            else f"repl failed err={err!r} assertion={assertion_fail}"
        )
        checks.append({"kind": "repl", "ok": ok, "detail": msg})
        if numeric is not None:
            evidence["repl_numeric_result"] = numeric
        if not ok:
            failures.append(msg)

    verified = not failures
    evidence["failures"] = failures
    evidence["mode"] = "targeted"
    return {"verified": verified, "evidence": evidence}


def make_verifier_node(
    verify_fn: VerifyFn | None = None,
    *,
    tracker: Any | None = None,
    max_attempts: int = MAX_VERIFICATION_ATTEMPTS,
    process_checker: Callable[[str], bool] | None = None,
) -> Callable[[ReactGraphState], dict[str, Any]]:
    """Build ``verifier_node`` with optional injectable evidence checker / tracker."""

    def _default(state: ReactGraphState) -> dict[str, Any]:
        return default_physical_evidence_check(state, process_checker=process_checker)

    check = verify_fn or _default
    limit = max(1, int(max_attempts))

    def verifier_node(state: ReactGraphState) -> dict[str, Any]:
        prev = _as_dict(state.get("verification_result"))
        attempts = int(prev.get("attempts") or 0) + 1
        try:
            raw = check(state) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("verify_fn raised: %s", exc)
            raw = {"verified": False, "evidence": {"error": str(exc)}}
        verified = bool(raw.get("verified"))
        evidence = raw.get("evidence")
        result = {
            "verified": verified,
            "evidence": evidence,
            "attempts": attempts,
        }
        patch: dict[str, Any] = {
            "verification_result": result,
            "current_agent": "Verifier",
        }

        tid = str(
            state.get("task_id") or state.get("session_id") or ""
        ).strip()
        try:
            from dana.graph.task_tracker import TaskStatus, get_shared_task_tracker

            tr = tracker if tracker is not None else get_shared_task_tracker()
            if tid:
                if verified:
                    tr.update_status(
                        tid,
                        TaskStatus.COMPLETED,
                        metadata={"verification": result},
                    )
                elif attempts >= limit:
                    tr.update_status(
                        tid,
                        TaskStatus.FAILED,
                        metadata={"verification": result},
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("TaskTracker update skipped: %s", exc)

        if verified:
            patch["halt"] = True
            patch["pending_synthesis"] = False
        else:
            patch["halt"] = False
            patch["pending_synthesis"] = True
            patch["last_obs"] = (
                f"VERIFICATION_FAILED (attempt {attempts}/{limit}): {evidence!r}"
            )
            # Keep final_raw informative for self-correction without claiming done.
            if attempts >= limit:
                patch["final_raw"] = (
                    f"Verification exhausted after {attempts} attempts: {evidence!r}"
                )
        return patch

    return verifier_node


verifier_node = make_verifier_node()


def route_after_verifier(state: ReactGraphState | dict[str, Any]) -> str:
    """verified → END (consolidate); else agent retry or fail_closed."""
    from langgraph.graph import END

    st = state or {}
    vr = _as_dict(st.get("verification_result"))
    if bool(vr.get("verified")):
        return END
    attempts = int(vr.get("attempts") or 0)
    max_a = st.get("max_verification_attempts")
    limit = int(max_a) if max_a is not None else MAX_VERIFICATION_ATTEMPTS
    if attempts < max(1, limit):
        return "agent"
    return "fail_closed"


__all__ = (
    "MAX_VERIFICATION_ATTEMPTS",
    "VerifyFn",
    "default_physical_evidence_check",
    "make_verifier_node",
    "route_after_verifier",
    "verifier_node",
)

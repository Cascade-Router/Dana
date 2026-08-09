"""Titan Repair — offline self-healing swarm over CAMGRASPER/tracker/bug_tracker.json.

Flow (reuses Tool Forge security stages):
  1. Read open bugs from the Autonomous Bug Tracker ledger.
  2. ``dana_coder`` analyzes the traceback and drafts a Python patch for the
     crashed file (unified-diff style body embedded in JSON).
  3. AST Gatekeeper + Security Reviewer must APPROVE the patch body.
  4. Approved patches are written to CAMGRASPER/tracker/pending_patches/ for human review
     — core source is NEVER hot-patched.

Triggered via ``dispatch_titan_repair`` (verbal) or ``run_titan_repair()`` on a
schedule / background thread.
"""

from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from dana.bug_tracker import mark_bug_status, open_bugs
from dana.paths import PENDING_PATCHES_DIR, PROJECT_ROOT
from dana.swarm.tool_forge_graph import (
    DEFAULT_MODEL,
    MAX_FORGE_REVISIONS,
    _chat_ollama,
    _extract_json,
    _llm_content,
    analyze_tool_ast,
    security_reviewer_agent,
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)

REPAIR_CODER_SYSTEM = """
You are Dana's Titan Repair coder. Given a crash ledger entry (user query +
exception/traceback), draft a MINIMAL Python patch for the specific source file
that crashed.

Return ONLY JSON:
{
  "target_file": "relative/path/under/repo.py",
  "summary": "one-line description of the fix",
  "code": "<full replacement module OR a self-contained patched function block>"
}

Hard rules for ``code``:
- Prefer the smallest correct fix; do not rewrite unrelated modules.
- No Tier-3 imports (os/sys/subprocess/shutil/socket/ctypes/pickle/…).
- No native open()/eval()/exec()/__import__.
- File I/O must use dana.tools.sandbox_io helpers when reading docs/.
- target_file MUST be a repo-relative .py path that appears in the traceback
  (or a clearly related helper under dana/).
""".strip()


class TitanRepairState(TypedDict):
    query: str
    bug: dict[str, Any]
    target_file: str
    code: str
    summary: str
    lint_errors: str
    security_feedback: str
    security_review: dict[str, Any]
    status: str
    feedback: str
    revisions: int
    history: list[dict[str, Any]]
    patch_path: str
    bugs_processed: int
    patches_written: int


def _safe_rel_py(path: str) -> str | None:
    raw = (path or "").replace("\\", "/").strip().lstrip("./")
    if not raw or ".." in raw.split("/") or not raw.endswith(".py"):
        return None
    candidate = (PROJECT_ROOT / raw).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    return raw


def _guess_target_from_traceback(tb: str) -> str:
    """Pick the deepest CAMGRASPER/dana frame from a traceback string."""
    hits: list[str] = []
    for line in (tb or "").splitlines():
        m = re.search(
            r'File "([^"]+\.py)"',
            line,
        )
        if not m:
            continue
        p = m.group(1).replace("\\", "/")
        if "CAMGRASPER" in p or "/dana/" in p or p.startswith("dana/") or "/dana/" in p or p.startswith("dana/"):
            # Prefer repo-relative tail.
            if "CAMGRASPER/" in p:
                p = p.split("CAMGRASPER/", 1)[1]
            elif "dana/" in p:
                p = "dana/" + p.split("dana/", 1)[1]
            elif "dana/" in p:
                # Legacy package path in older tracebacks → current package.
                p = "dana/" + p.split("dana/", 1)[1]
            safe = _safe_rel_py(p)
            if safe:
                hits.append(safe)
    return hits[-1] if hits else "dana/agentic.py"


def repair_coder(state: TitanRepairState, *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    bug = dict(state.get("bug") or {})
    revisions = int(state.get("revisions") or 0)
    lint_errors = (state.get("lint_errors") or "").strip()
    security_feedback = (state.get("security_feedback") or "").strip()
    guessed = _guess_target_from_traceback(str(bug.get("traceback") or ""))

    user = (
        f"USER QUERY:\n{bug.get('user_query') or state.get('query') or ''}\n\n"
        f"ERROR:\n{bug.get('error') or ''}\n\n"
        f"TRACEBACK:\n{bug.get('traceback') or '(none)'}\n\n"
        f"Suggested target_file: {guessed}\n"
    )
    if lint_errors:
        user += f"\nFATAL AST LINT ERRORS (fix first):\n{lint_errors}\n"
    if security_feedback:
        user += f"\nSECURITY REMEDIATION (must apply):\n{security_feedback}\n"

    try:
        from dana.logging import log as _log

        _log(
            "TitanRepair",
            f"tool_forge_node=repair_coder bug={bug.get('id')!r} revision={revisions}",
        )
    except Exception:
        pass

    try:
        llm = _chat_ollama(model=model, temperature=0.1)
        raw = _llm_content(
            llm.invoke(
                [
                    {"role": "system", "content": REPAIR_CODER_SYSTEM},
                    {"role": "user", "content": user},
                ]
            )
        )
        data = _extract_json(raw) or {}
        target = _safe_rel_py(str(data.get("target_file") or guessed)) or guessed
        code = textwrap.dedent(str(data.get("code") or "")).strip()
        summary = str(data.get("summary") or f"Repair for {bug.get('id')}").strip()
        if not code:
            return {
                "status": "error",
                "feedback": "Repair coder returned empty code.",
                "revisions": revisions,
            }
        history = list(state.get("history") or [])
        history.append(
            {
                "stage": "repair_coder",
                "revision": revisions,
                "target_file": target,
                "status": "drafting",
            }
        )
        return {
            "target_file": target,
            "code": code,
            "summary": summary,
            "lint_errors": "",
            "status": "drafting",
            "history": history,
            "revisions": revisions,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "feedback": f"Repair coder failed: {exc}",
            "revisions": revisions,
        }


def repair_ast_gatekeeper(state: TitanRepairState) -> dict[str, Any]:
    """Reuse Tool Forge ``analyze_tool_ast`` on the drafted patch body."""
    from dana.swarm.tool_forge_graph import ast_gatekeeper_forge

    # ast_gatekeeper_forge expects ToolForgeState fields; map compatibly.
    mapped = {
        "code": state.get("code") or "",
        "tool_name": state.get("target_file") or "repair_patch",
        "revisions": state.get("revisions") or 0,
        "history": list(state.get("history") or []),
        "lint_errors": "",
        "status": "",
    }
    result = ast_gatekeeper_forge(mapped)  # type: ignore[arg-type]
    return {
        "lint_errors": result.get("lint_errors") or "",
        "status": result.get("status") or "",
        "revisions": result.get("revisions") or 0,
        "history": result.get("history") or mapped["history"],
    }


def write_pending_patch(state: TitanRepairState) -> dict[str, Any]:
    """Persist an approved patch under tracker/pending_patches/ (no core hot-patch)."""
    PENDING_PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    bug = dict(state.get("bug") or {})
    bug_id = str(bug.get("id") or "unknown")
    target = _safe_rel_py(state.get("target_file") or "") or "dana/unknown.py"
    code = textwrap.dedent(state.get("code") or "").strip()
    summary = (state.get("summary") or "").strip()

    # Final AST re-check before disk write.
    fatal = analyze_tool_ast(code)
    if fatal:
        return {
            "status": "LINT_FAIL",
            "lint_errors": "\n".join(fatal),
            "feedback": "Pending-patch write aborted: AST re-check failed",
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_id = re.sub(r"[^A-Za-z0-9_\-]+", "_", bug_id)[:64]
    path = PENDING_PATCHES_DIR / f"{stamp}_{safe_id}.json"
    payload = {
        "bug_id": bug_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_file": target,
        "summary": summary,
        "user_query": bug.get("user_query"),
        "error": bug.get("error"),
        "status": "pending_human_review",
        "code": code,
        "security_review": dict(state.get("security_review") or {}),
        "note": (
            "Do NOT apply automatically. Review and merge manually — "
            "Titan Repair never hot-patches core source."
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mark_bug_status(
        bug_id,
        "patched_pending_review",
        extra={"patch_path": str(path), "target_file": target},
    )
    history = list(state.get("history") or [])
    history.append(
        {
            "stage": "write_pending_patch",
            "bug_id": bug_id,
            "path": str(path),
            "status": "pending_human_review",
        }
    )
    try:
        from dana.logging import log as _log

        _log("TitanRepair", f"pending patch written → {path}")
    except Exception:
        pass
    return {
        "status": "patched_pending_review",
        "patch_path": str(path),
        "feedback": f"OK: pending patch for `{target}` → {path.name}",
        "history": history,
        "patches_written": int(state.get("patches_written") or 0) + 1,
    }


def _route_after_ast(
    state: TitanRepairState,
) -> Literal["security_reviewer", "repair_coder", "terminal_failure"]:
    status = (state.get("status") or "").strip().upper()
    revisions = int(state.get("revisions") or 0)
    if status == "LINT_OK":
        return "security_reviewer"
    if status == "LINT_FAIL" and revisions < MAX_FORGE_REVISIONS:
        return "repair_coder"
    return "terminal_failure"


def _route_after_security(
    state: TitanRepairState,
) -> Literal["write_pending_patch", "repair_coder", "terminal_failure"]:
    status = (state.get("status") or "").strip().upper()
    revisions = int(state.get("revisions") or 0)
    if status == "APPROVED":
        return "write_pending_patch"
    if status == "SEC_REJECTED" and revisions < MAX_FORGE_REVISIONS:
        return "repair_coder"
    return "terminal_failure"


def terminal_failure_repair(state: TitanRepairState) -> dict[str, Any]:
    bug = dict(state.get("bug") or {})
    detail = (
        state.get("lint_errors")
        or state.get("security_feedback")
        or state.get("feedback")
        or "Titan Repair aborted"
    )
    if bug.get("id"):
        mark_bug_status(str(bug["id"]), "repair_failed", extra={"detail": str(detail)[:500]})
    return {"status": "error", "feedback": str(detail)}


def build_titan_repair_graph(*, model: str = DEFAULT_MODEL):
    def _coder(state: TitanRepairState) -> dict[str, Any]:
        return repair_coder(state, model=model)

    def _sec(state: TitanRepairState) -> dict[str, Any]:
        # Map onto Tool Forge security reviewer state shape.
        forge_state = {
            "code": state.get("code") or "",
            "query": state.get("query") or "",
            "tool_name": state.get("target_file") or "repair_patch",
            "revisions": state.get("revisions") or 0,
            "history": list(state.get("history") or []),
            "lint_errors": "",
            "security_feedback": "",
            "security_review": {},
            "feedback": "",
            "status": "",
            "loaded_tool": "",
        }
        return security_reviewer_agent(forge_state, model=model)  # type: ignore[arg-type]

    graph = StateGraph(TitanRepairState)
    graph.add_node("repair_coder", _coder)
    graph.add_node("ast_gatekeeper", repair_ast_gatekeeper)
    graph.add_node("security_reviewer", _sec)
    graph.add_node("write_pending_patch", write_pending_patch)
    graph.add_node("terminal_failure", terminal_failure_repair)

    graph.add_edge(START, "repair_coder")
    graph.add_edge("repair_coder", "ast_gatekeeper")
    graph.add_conditional_edges(
        "ast_gatekeeper",
        _route_after_ast,
        {
            "security_reviewer": "security_reviewer",
            "repair_coder": "repair_coder",
            "terminal_failure": "terminal_failure",
        },
    )
    graph.add_conditional_edges(
        "security_reviewer",
        _route_after_security,
        {
            "write_pending_patch": "write_pending_patch",
            "repair_coder": "repair_coder",
            "terminal_failure": "terminal_failure",
        },
    )
    graph.add_edge("write_pending_patch", END)
    graph.add_edge("terminal_failure", END)
    return graph.compile()


def repair_one_bug(
    bug: dict[str, Any],
    *,
    query: str = "",
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    app = build_titan_repair_graph(model=model)
    initial: TitanRepairState = {
        "query": query or str(bug.get("user_query") or ""),
        "bug": bug,
        "target_file": "",
        "code": "",
        "summary": "",
        "lint_errors": "",
        "security_feedback": "",
        "security_review": {},
        "status": "pending",
        "feedback": "",
        "revisions": 0,
        "history": [],
        "patch_path": "",
        "bugs_processed": 0,
        "patches_written": 0,
    }
    return dict(app.invoke(initial))


def run_titan_repair(
    *,
    query: str = "",
    model: str = DEFAULT_MODEL,
    max_bugs: int = 5,
) -> str:
    """Process PENDING bugs; draft patches under tracker/pending_patches/ (no hot-patch)."""
    bugs = open_bugs()
    if not bugs:
        return "No PENDING bugs in the tracker."

    processed = 0
    written = 0
    review_prompts: list[str] = []
    notes: list[str] = []
    for bug in bugs[: max(1, max_bugs)]:
        processed += 1
        bug_label = str(bug.get("id") or bug.get("error") or f"bug#{processed}")[:80]
        try:
            result = repair_one_bug(bug, query=query, model=model)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{bug_label}: failed ({exc})")
            mark_bug_status(str(bug.get("id") or ""), "repair_failed", extra={"detail": str(exc)})
            continue
        status = str(result.get("status") or "")
        if status == "patched_pending_review":
            written += 1
            patch_name = Path(str(result.get("patch_path") or "")).name or "pending_patches"
            review_prompts.append(
                f"I have drafted a fix for {bug_label} and placed it in "
                f"CAMGRASPER/tracker/pending_patches/{patch_name}. Would you like to review it?"
            )
            notes.append(result.get("feedback") or f"{bug_label}: patch pending")
        else:
            notes.append(
                f"{bug_label}: {status} — {result.get('feedback') or result.get('lint_errors')}"
            )

    if review_prompts:
        # Primary user-facing notification (no automatic apply).
        return " ".join(review_prompts[:3])

    summary = (
        f"Processed {processed} PENDING bug(s); wrote {written} draft(s) "
        f"under CAMGRASPER/tracker/pending_patches/ (not applied to source)."
    )
    if notes:
        summary += " " + " | ".join(notes[:3])
    return summary

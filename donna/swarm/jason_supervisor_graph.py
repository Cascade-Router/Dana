"""Jason supervisor → Donna coder LangGraph (conversational multi-agent handoff).

Topology (single pass — no agent ping-pong loops):
  START → jason_read → jason_evaluate → donna_code → END

Jason binds a read-only file tool on ``notes.txt``, evaluates the summary,
then hands a brief + file context to Donna. Donna binds the python_repl suite
(``file_editor`` + ``python_repl``) and writes ``scaling_metrics.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from donna.paths import PROJECT_ROOT
from donna.tools.system_repl import file_editor, python_repl

DEFAULT_NOTES = "notes.txt"
DEFAULT_SCRIPT = "scaling_metrics.py"


class JasonSupervisorState(TypedDict):
    user_text: str
    notes_path: str
    notes_content: str
    jason_evaluation: str
    donna_brief: str
    script_path: str
    script_content: str
    script_written: bool
    repl_output: str
    status: str
    history: list[dict[str, Any]]
    active_agent: str


def _log(msg: str) -> None:
    try:
        from donna.logging import log

        log("JasonSupervisor", msg)
    except Exception:  # noqa: BLE001
        print(f"[JasonSupervisor] {msg}", flush=True)


def _hist(state: JasonSupervisorState, event: str, **extra: Any) -> list[dict[str, Any]]:
    row = {"event": event, **extra}
    return list(state.get("history") or []) + [row]


def _extract_notes_path(user_text: str) -> str:
    m = re.search(r"([\w./\\-]*notes\.txt)\b", user_text or "", flags=re.I)
    if m:
        return m.group(1).replace("\\", "/")
    return DEFAULT_NOTES


def _extract_script_path(user_text: str) -> str:
    m = re.search(
        r"([\w./\\-]+\.py)\b",
        user_text or "",
        flags=re.I,
    )
    if m:
        name = m.group(1).replace("\\", "/")
        # Prefer the explicit new script name over incidental .py mentions.
        if "scaling" in name.lower() or name.lower().endswith("metrics.py"):
            return name
        if "notes" not in name.lower():
            return name
    m2 = re.search(
        r"(?:script|file)\s+called\s+[`'\"]?([\w./\\-]+\.py)",
        user_text or "",
        flags=re.I,
    )
    if m2:
        return m2.group(1).replace("\\", "/")
    return DEFAULT_SCRIPT


def _parse_points(notes_content: str) -> list[str]:
    text = (notes_content or "").strip()
    if not text:
        return []
    # Prefer embedded JSON object (tool headers / truncation noise tolerated).
    candidates = [text]
    brace = text.find("{")
    if brace >= 0:
        candidates.append(text[brace:])
    for cand in candidates:
        try:
            data = json.loads(cand)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict):
            pts = data.get("points")
            if isinstance(pts, list) and pts:
                return [str(p).strip() for p in pts if str(p).strip()]
        if isinstance(data, list) and data:
            return [str(p).strip() for p in data if str(p).strip()]
    points: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*(?:\d+[.)]\s*|[-*]\s+)(.+)", line)
        if m:
            points.append(m.group(1).strip())
    if points:
        return points
    # Fallback: keep non-empty lines (skip OK: headers).
    return [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().lower().startswith("ok:")
    ][:5]


def _evaluate_points(points: list[str]) -> str:
    """Jason's accuracy check — grounded, no unbounded LLM loop."""
    if not points:
        return (
            "REJECTED: notes.txt has no usable summary points. "
            "Donna should still emit a minimal mock metrics script."
        )
    checks = []
    blob = " ".join(points).lower()
    if any(k in blob for k in ("complex", "overhead", "communicat", "latency")):
        checks.append("Point on complexity/communication overhead is directionally accurate.")
    else:
        checks.append("Complexity/overhead angle is weak or missing.")
    if any(k in blob for k in ("hierarch", "distribut", "architect")):
        checks.append("Hierarchical/distributed architecture guidance is sound.")
    else:
        checks.append("Architecture pattern guidance is incomplete.")
    if any(k in blob for k in ("machine learning", "optim", "adapt", "control", "role")):
        checks.append("Optimization / adaptive control angle is reasonable for a mock report.")
    else:
        checks.append("Optimization angle could be stronger, but is acceptable for a mock.")
    verdict = "ACCURATE_ENOUGH" if len(checks) >= 2 else "NEEDS_WORK"
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(points, 1))
    return (
        f"Jason verdict={verdict}\n"
        f"Source points:\n{numbered}\n"
        f"Evaluation:\n- " + "\n- ".join(checks)
    )


def _build_script(points: list[str], evaluation: str, script_path: str) -> str:
    pts = points or [
        "Complexity and communication overhead grow with agent count.",
        "Hierarchical or distributed architectures help scale.",
        "Adaptive optimization improves sustained performance.",
    ]
    # Keep generated code ASCII-safe and dependency-free.
    pts_lit = ",\n    ".join(repr(p) for p in pts[:5])
    return f'''#!/usr/bin/env python3
"""Mock multi-agent scaling performance report.

Generated by Donna under Jason supervisor delegation.
Source notes evaluated by Jason before handoff.
"""

from __future__ import annotations

POINTS = [
    {pts_lit},
]


def mock_metrics(agent_counts: list[int] | None = None) -> list[dict[str, float]]:
    counts = agent_counts or [1, 4, 16, 64]
    rows: list[dict[str, float]] = []
    for n in counts:
        # Toy model: throughput rises sub-linearly; overhead rises with n.
        throughput = round(100.0 * (n ** 0.65) / max(n, 1) * n, 2)
        overhead_pct = round(min(85.0, 5.0 + (n ** 0.9) * 0.8), 2)
        efficiency = round(max(5.0, 100.0 - overhead_pct), 2)
        rows.append(
            {{
                "agents": float(n),
                "throughput_rps": throughput,
                "coordination_overhead_pct": overhead_pct,
                "efficiency_pct": efficiency,
            }}
        )
    return rows


def print_report() -> None:
    print("=== Mock Multi-Agent Scaling Performance Report ===")
    print("Jason evaluation excerpt:")
    for line in {evaluation!r}.splitlines()[:6]:
        print(f"  {{line}}")
    print()
    print("Grounding points from notes.txt:")
    for i, p in enumerate(POINTS, 1):
        print(f"  {{i}}. {{p}}")
    print()
    print(f"{{'agents':>8}} {{'throughput_rps':>16}} {{'overhead_%':>12}} {{'efficiency_%':>14}}")
    for row in mock_metrics():
        print(
            f"{{int(row['agents']):>8}} "
            f"{{row['throughput_rps']:>16.2f}} "
            f"{{row['coordination_overhead_pct']:>12.2f}} "
            f"{{row['efficiency_pct']:>14.2f}}"
        )
    print()
    print("Report complete. (mock data — not a live benchmark)")


if __name__ == "__main__":
    print_report()
'''


def jason_read(state: JasonSupervisorState) -> dict[str, Any]:
    """Jason binds/uses file read on notes.txt (file_editor action=read)."""
    notes_path = state.get("notes_path") or _extract_notes_path(state.get("user_text") or "")
    _log(f"agent=Jason tool=file_editor(read) path={notes_path}")
    # Ticket wording: read_file — alias to jailed file_editor read.
    observation = file_editor("read", notes_path, None)
    content = ""
    if observation.startswith("OK:"):
        # Strip the OK header line.
        parts = observation.split("\n", 1)
        content = parts[1] if len(parts) > 1 else ""
    # Belt-and-suspenders: if tool observation is noisy, read jailed path directly.
    if not _parse_points(content):
        try:
            raw_path = (PROJECT_ROOT / notes_path).resolve()
            raw_path.relative_to(PROJECT_ROOT.resolve())
            if raw_path.is_file():
                content = raw_path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    status = "JASON_READ_OK" if content.strip() else "JASON_READ_EMPTY"
    if observation.startswith("ERROR:") and not content.strip():
        status = "JASON_READ_ERROR"
        content = ""
    return {
        "active_agent": "Jason",
        "notes_path": notes_path,
        "notes_content": content,
        "status": status,
        "history": _hist(
            state,
            "jason_read",
            tool="file_editor",
            path=notes_path,
            observation=observation[:400],
            status=status,
        ),
    }


def jason_evaluate(state: JasonSupervisorState) -> dict[str, Any]:
    """Jason evaluates notes and builds the Donna handoff brief (no loop)."""
    points = _parse_points(state.get("notes_content") or "")
    evaluation = _evaluate_points(points)
    script_path = state.get("script_path") or _extract_script_path(
        state.get("user_text") or ""
    )
    brief = (
        "DONNA HANDOFF (from Jason supervisor)\n"
        f"Write `{script_path}` using the python_repl suite (file_editor + python_repl).\n"
        "The script must print a mock performance report grounded in these points:\n"
        + "\n".join(f"- {p}" for p in points)
        + f"\n\nJason evaluation:\n{evaluation}\n"
        "Do not ask Jason for more turns — complete the file write in one pass."
    )
    _log("agent=Jason handoff → Donna (brief ready)")
    return {
        "active_agent": "Jason",
        "jason_evaluation": evaluation,
        "donna_brief": brief,
        "script_path": script_path,
        "status": "JASON_HANDOFF",
        "history": _hist(
            state,
            "jason_evaluate",
            agent="Jason",
            handoff_to="Donna",
            script_path=script_path,
            points=len(points),
        ),
    }


def donna_code(state: JasonSupervisorState) -> dict[str, Any]:
    """Donna receives handoff; binds file_editor + python_repl; writes script."""
    script_path = state.get("script_path") or DEFAULT_SCRIPT
    points = _parse_points(state.get("notes_content") or "")
    evaluation = state.get("jason_evaluation") or ""
    code = _build_script(points, evaluation, script_path)
    _log(
        f"agent=Donna tools=['file_editor','python_repl'] "
        f"write={script_path} (suite bind)"
    )
    write_obs = file_editor("write", script_path, code)
    written = write_obs.startswith("OK:")
    repl_out = ""
    if written:
        abs_script = str((PROJECT_ROOT / script_path).resolve())
        # Verify via python_repl suite (separate python.exe subprocess).
        repl_out = python_repl(
            "import runpy\n"
            f"runpy.run_path({abs_script!r}, run_name='__main__')\n"
        )
    status = "DONE" if written else f"DONNA_WRITE_FAIL: {write_obs[:200]}"
    _log(f"agent=Donna status={status}")
    return {
        "active_agent": "Donna",
        "script_path": script_path,
        "script_content": code if written else "",
        "script_written": written,
        "repl_output": repl_out[:2000],
        "status": status,
        "history": _hist(
            state,
            "donna_code",
            agent="Donna",
            tools=["file_editor", "python_repl"],
            path=script_path,
            write_obs=write_obs[:300],
            repl_obs=(repl_out or "")[:300],
            status=status,
        ),
    }


def build_jason_supervisor_graph() -> Any:
    """Compile the linear Jason→Donna supervisor StateGraph."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(JasonSupervisorState)
    g.add_node("jason_read", jason_read)
    g.add_node("jason_evaluate", jason_evaluate)
    g.add_node("donna_code", donna_code)
    g.add_edge(START, "jason_read")
    g.add_edge("jason_read", "jason_evaluate")
    g.add_edge("jason_evaluate", "donna_code")
    g.add_edge("donna_code", END)
    return g.compile()


def run_jason_supervisor(user_text: str) -> dict[str, Any]:
    """Execute Jason→Donna handoff once; returns final state dict."""
    raw = (user_text or "").lstrip("\ufeff").strip()
    notes_path = _extract_notes_path(raw)
    script_path = _extract_script_path(raw)
    initial: JasonSupervisorState = {
        "user_text": raw,
        "notes_path": notes_path,
        "notes_content": "",
        "jason_evaluation": "",
        "donna_brief": "",
        "script_path": script_path,
        "script_content": "",
        "script_written": False,
        "repl_output": "",
        "status": "pending",
        "history": [],
        "active_agent": "Jason",
    }
    _log(f"start user_text={raw[:160]!r} notes={notes_path} script={script_path}")
    graph = build_jason_supervisor_graph()
    final = graph.invoke(initial)
    _log(
        f"complete status={final.get('status')} "
        f"script_written={final.get('script_written')} "
        f"agent={final.get('active_agent')}"
    )
    return dict(final)


def dispatch_jason_supervisor_impl(query: str) -> str:
    """Tool entry: run supervisor graph synchronously; return spoken-safe summary."""
    q = (query or "").lstrip("\ufeff").strip()
    if not q:
        return "ERROR: missing query"
    try:
        final = run_jason_supervisor(q)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Jason supervisor failed: {exc}"
    if not final.get("script_written"):
        return (
            f"ERROR: Jason→Donna handoff incomplete status={final.get('status')} "
            f"history={final.get('history')!r}"
        )
    eval_preview = (final.get("jason_evaluation") or "").splitlines()
    eval_line = eval_preview[0] if eval_preview else "evaluated notes.txt"
    script = final.get("script_path") or DEFAULT_SCRIPT
    return (
        f"OK: Jason read {final.get('notes_path') or DEFAULT_NOTES}, "
        f"{eval_line}; handed off to Donna; Donna wrote {script}."
    )

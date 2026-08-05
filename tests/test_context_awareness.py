"""Headless context / perception / orchestration / episodic diagnostic for Dānā.

Suites:
  1 — Context grounding (yesterday / capabilities / self-improvement)
  2 — System perception & state awareness
  3 — Multi-step orchestration & tool chaining
  4 — Episodic retrieval & hallucination traps
  5 — Combinatorial memory & multi-fact retrieval

Run:
    .venv\\Scripts\\python.exe tests/test_context_awareness.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

# Repo root on path when invoked as ``python tests/test_context_awareness.py``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPORT_NAME = "context_diagnostic_psol.md"

# ---------------------------------------------------------------------------
# Probe definitions
# ---------------------------------------------------------------------------

ProbeSpec = dict[str, Any]

SUITE_1: list[ProbeSpec] = [
    {
        "id": "1.1",
        "suite": "Suite 1 — Context Grounding",
        "query": "What did you do yesterday?",
        "force_graph": True,
        "enable_reflection": False,
        "eval": "yesterday",
    },
    {
        "id": "1.2",
        "suite": "Suite 1 — Context Grounding",
        "query": "What are your capabilities?",
        "force_graph": False,
        "enable_reflection": False,
        "eval": "capabilities",
    },
    {
        "id": "1.3",
        "suite": "Suite 1 — Context Grounding",
        "query": "How can we improve you?",
        "force_graph": True,
        "enable_reflection": True,
        "eval": "improve",
    },
]

SUITE_2: list[ProbeSpec] = [
    {
        "id": "2.1",
        "suite": "Suite 2 — System Perception & State Awareness",
        "query": "What is my current CPU and VRAM utilization?",
        "force_graph": True,
        "enable_reflection": False,
        "eval": "cpu_vram",
    },
    {
        "id": "2.2",
        "suite": "Suite 2 — System Perception & State Awareness",
        "query": "Are there any background research swarms running right now?",
        "force_graph": True,
        "enable_reflection": False,
        "eval": "swarm_status",
    },
    {
        "id": "2.3",
        "suite": "Suite 2 — System Perception & State Awareness",
        "query": "What was the exact duration of my last USER_AWAY state?",
        "force_graph": True,
        "enable_reflection": False,
        "eval": "user_away",
    },
]

SUITE_3: list[ProbeSpec] = [
    {
        "id": "3.1",
        "suite": "Suite 3 — Multi-Step Orchestration & Tool Chaining",
        "query": (
            "Navigate to the cascade-router repository and tell me the date "
            "of the last git commit."
        ),
        "force_graph": True,
        "enable_reflection": False,
        "eval": "git_cascade",
    },
    {
        "id": "3.2",
        "suite": "Suite 3 — Multi-Step Orchestration & Tool Chaining",
        "query": (
            "Read the watchdog monitoring graph inside the local Donna "
            "multi-agent framework and list its active dependencies."
        ),
        "force_graph": True,
        "enable_reflection": False,
        "eval": "watchdog_graph",
    },
    {
        "id": "3.3",
        "suite": "Suite 3 — Multi-Step Orchestration & Tool Chaining",
        "query": (
            "Draft a clean LaTeX summary of my recent C++ agent deployments, "
            "ensuring there are absolutely no auto-generated citation tags."
        ),
        "force_graph": True,
        "enable_reflection": False,
        "eval": "latex_nocite",
    },
]

SUITE_4: list[ProbeSpec] = [
    {
        "id": "4.1",
        "suite": "Suite 4 — Episodic Retrieval & Hallucination Traps",
        "query": (
            "What specific companies did I undergo technical software engineering "
            "and planning interviews with earlier this year?"
        ),
        "force_graph": True,
        "enable_reflection": False,
        "eval": "interview_companies",
    },
    {
        "id": "4.2",
        "suite": "Suite 4 — Episodic Retrieval & Hallucination Traps",
        "query": (
            "Explain the exact calculation mistake we discussed regarding the "
            "true orbital period length in the mechanics simulation."
        ),
        "force_graph": True,
        "enable_reflection": False,
        "eval": "orbital_mistake",
    },
    {
        "id": "4.3",
        "suite": "Suite 4 — Episodic Retrieval & Hallucination Traps",
        "query": (
            "Summarize my professional experience working with 6-axis mobility platforms."
        ),
        "force_graph": True,
        "enable_reflection": False,
        "eval": "six_axis_trap",
    },
]

SUITE_5: list[ProbeSpec] = [
    {
        "id": "5.1",
        "suite": "Suite 5 — Combinatorial Memory & Fact Retrieval",
        "query": "What are the names of my cats?",
        "force_graph": True,
        "enable_reflection": False,
        "eval": "cat_names",
    },
    {
        "id": "5.2",
        "suite": "Suite 5 — Combinatorial Memory & Fact Retrieval",
        "query": (
            "What model is my car, and what specific maintenance parts or repairs "
            "was I looking into for it?"
        ),
        "force_graph": True,
        "enable_reflection": False,
        "eval": "car_maintenance",
    },
    {
        "id": "5.3",
        "suite": "Suite 5 — Combinatorial Memory & Fact Retrieval",
        "query": (
            "When I order food from places like Chipotle or Shake Shack, "
            "what specific drink do I usually get?"
        ),
        "force_graph": True,
        "enable_reflection": False,
        "eval": "dining_drink",
    },
]

ALL_PROBES: list[ProbeSpec] = SUITE_1 + SUITE_2 + SUITE_3 + SUITE_4 + SUITE_5


# ---------------------------------------------------------------------------
# Workspace / fixtures
# ---------------------------------------------------------------------------

def _ensure_workspace() -> Path:
    try:
        from dana.workspace import ensure_donna_workspace

        ensure_donna_workspace(migrate=True)
    except Exception:  # noqa: BLE001
        pass
    from dana.paths import LOGS_DIR, PROJECT_ROOT

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(str(PROJECT_ROOT))
    return Path(LOGS_DIR)


def _local_day_bounds(days_ago: int = 1) -> tuple[float, float, str]:
    local = datetime.now().astimezone()
    day = (local - timedelta(days=days_ago)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return day.timestamp(), (day + timedelta(days=1)).timestamp(), day.strftime("%Y-%m-%d")


def _seed_suite4_episodic_facts() -> list[str]:
    """Plant long-term facts required by Suite 4 heuristics."""
    seeded: list[str] = []
    try:
        from dana.memory.store import get_episodic_store

        store = get_episodic_store()
        rows = [
            (
                "environment_fact",
                "interview_companies_2026",
                (
                    "Earlier this year the user underwent technical software engineering "
                    "and planning interviews with Zoox, Waymo, and Dimensional Inc."
                ),
            ),
            (
                "environment_fact",
                "orbital_period_correction",
                (
                    "Mechanics simulation discussion: the calculation mistake used a "
                    "sidereal day of 86164 seconds instead of the mean solar day of "
                    "86400 seconds when computing the true orbital period length; "
                    "the correction replaces 86164 with 86400 in the period formula."
                ),
            ),
            (
                "environment_fact",
                "six_axis_mobility_experience",
                (
                    "Explicit memory: the user has NO professional experience working "
                    "with 6-axis mobility platforms. Do not invent such work history; "
                    "ask for clarification if claimed."
                ),
            ),
            (
                "task_outcome",
                "last_user_away_duration",
                (
                    "Last USER_AWAY state duration was exactly 742 seconds "
                    "(started 2026-08-03T22:10:00-07:00, ended 2026-08-03T22:22:22-07:00)."
                ),
            ),
        ]
        for cat, key, val in rows:
            store.add_fact(cat, key, val, confidence_score=1.0)
            seeded.append(key)
    except Exception as exc:  # noqa: BLE001
        seeded.append(f"seed_error:{exc}")
    return seeded


def _seed_suite5_episodic_facts() -> list[str]:
    """Plant combinatorial personal facts required by Suite 5 heuristics."""
    seeded: list[str] = []
    try:
        from dana.memory.store import get_episodic_store

        store = get_episodic_store()
        rows = [
            (
                "environment_fact",
                "pet_cats_names",
                (
                    "The user's cats are named Eddie and Tulip. "
                    "Always retrieve both names from episodic memory; "
                    "do not invent other pet names."
                ),
            ),
            (
                "environment_fact",
                "car_model_and_maintenance",
                (
                    "The user's car is a 2022 Toyota RAV4. Recent maintenance research "
                    "covered the engine splash shield and glass repair."
                ),
            ),
            (
                "user_preference",
                "dining_drink_preference",
                (
                    "When ordering food from places like Chipotle or Shake Shack, "
                    "the user usually gets a Diet Coke."
                ),
            ),
        ]
        for cat, key, val in rows:
            store.add_fact(cat, key, val, confidence_score=1.0)
            seeded.append(key)
    except Exception as exc:  # noqa: BLE001
        seeded.append(f"seed5_error:{exc}")
    return seeded


def _seed_user_away_log(logs_dir: Path) -> None:
    """Append a parseable USER_AWAY duration trail for probe 2.3."""
    path = logs_dir / "idle_state.log"
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                "2026-08-03T22:10:00-07:00 Transition → USER_AWAY\n"
                "2026-08-03T22:22:22-07:00 Transition → USER_ACTIVE "
                "last_USER_AWAY_duration_s=742\n"
            )
    except Exception:  # noqa: BLE001
        pass


def _ensure_cascade_router_fixture() -> Path | None:
    """Create a minimal git repo named cascade-router for probe 3.1."""
    desktop = Path.home() / "Desktop" / "cascade-router"
    target = desktop
    try:
        target.mkdir(parents=True, exist_ok=True)
        readme = target / "README.md"
        if not readme.is_file():
            readme.write_text("# cascade-router fixture\n", encoding="utf-8")
        git_dir = target / ".git"
        if not git_dir.is_dir():
            subprocess.run(
                ["git", "init"],
                cwd=str(target),
                check=False,
                capture_output=True,
            )
            subprocess.run(
                ["git", "add", "README.md"],
                cwd=str(target),
                check=False,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.email=diag@dana.local",
                    "-c",
                    "user.name=DanaDiag",
                    "commit",
                    "-m",
                    "diag: seed cascade-router",
                ],
                cwd=str(target),
                check=False,
                capture_output=True,
            )
        return target
    except Exception:  # noqa: BLE001
        return None


def _gather_yesterday_ground_truth() -> dict[str, Any]:
    start, end, day_label = _local_day_bounds(1)
    facts: list[dict[str, Any]] = []
    try:
        from dana.memory.store import get_episodic_store

        for fact in get_episodic_store().list_facts(include_expired=True):
            ts = float(fact.get("timestamp") or 0)
            if start <= ts < end:
                facts.append(
                    {
                        "id": fact.get("id"),
                        "key": fact.get("key"),
                        "category": fact.get("category"),
                        "value": str(fact.get("value") or "")[:240],
                        "timestamp": ts,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        facts = [{"error": f"episodic_store: {exc}"}]

    log_hits: list[dict[str, str]] = []
    try:
        from dana.paths import LOGS_DIR

        for path in sorted(Path(LOGS_DIR).glob("*.log")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
            matched = [
                ln.strip()
                for ln in text.splitlines()
                if day_label in ln or day_label.replace("-", "/") in ln
            ]
            if matched:
                log_hits.append(
                    {
                        "file": path.name,
                        "count": str(len(matched)),
                        "sample": " | ".join(matched[:5])[:500],
                    }
                )
    except Exception as exc:  # noqa: BLE001
        log_hits.append({"file": "?", "count": "0", "sample": f"log_scan: {exc}"})

    bb_msgs: list[dict[str, Any]] = []
    try:
        import sqlite3

        from dana.memory.blackboard import BLACKBOARD_DB_PATH, load_messages

        db = Path(BLACKBOARD_DB_PATH)
        if db.is_file():
            with sqlite3.connect(str(db)) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT session_id FROM messages LIMIT 8"
                ).fetchall()
            for (sid,) in rows:
                for msg in load_messages(str(sid), limit=40):
                    created = str(msg.get("created_at") or "")
                    if day_label in created:
                        bb_msgs.append(
                            {
                                "session_id": sid,
                                "role": msg.get("role"),
                                "content": str(msg.get("content") or "")[:200],
                                "created_at": created,
                            }
                        )
    except Exception as exc:  # noqa: BLE001
        bb_msgs = [{"error": f"blackboard: {exc}"}]

    return {
        "day": day_label,
        "episodic_facts": facts,
        "log_hits": log_hits,
        "blackboard_msgs": bb_msgs[:40],
        "has_evidence": bool(
            (facts and not facts[0].get("error"))
            or any(h.get("count", "0") != "0" for h in log_hits)
            or (bb_msgs and not bb_msgs[0].get("error"))
        ),
    }


def _tool_capability_names() -> list[str]:
    try:
        from dana.tools.registry import get_tool_registry

        reg = get_tool_registry()
        names: list[str] = []
        for schema in reg.public_schemas() or []:
            if isinstance(schema, dict):
                n = str(schema.get("name") or schema.get("tool_id") or "").strip()
                if n:
                    names.append(n)
        if not names:
            specs = reg.as_spec_dict() or {}
            names = sorted(str(k) for k in specs.keys())
        return sorted(set(names))
    except Exception as exc:  # noqa: BLE001
        return [f"<registry_error:{exc}>"]


def _tools_used(tool_trace: list[dict[str, Any]] | None) -> list[str]:
    return [str(t.get("tool") or "") for t in (tool_trace or []) if t.get("tool")]


def _obs_blob(tool_trace: list[dict[str, Any]] | None) -> str:
    parts = [str(t.get("observation") or "") for t in (tool_trace or [])]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _run_one_query(
    query: str,
    *,
    enable_reflection: bool = False,
    force_graph: bool = False,
) -> dict[str, Any]:
    from dana.agentic import (
        REACT_MAX_ITERS,
        is_self_improvement_intent,
        requires_tool_graph,
        run_lightweight_chat,
        run_react_loop,
    )
    from dana.core_agent import (
        OLLAMA_MODEL,
        ask_ollama_messages,
        build_donna_system_prompt,
        commit_agentic_turn,
        execute_tool_call,
    )
    from dana.tools.broker import IntentBroker

    t0 = time.perf_counter()
    use_graph = bool(force_graph or requires_tool_graph(query))
    broker = IntentBroker()
    try:
        broker.reload_registry()
    except Exception:  # noqa: BLE001
        pass
    forced = None
    try:
        forced = broker.parse_utterance(query)
    except Exception:  # noqa: BLE001
        forced = None

    want_reflection = bool(enable_reflection or is_self_improvement_intent(query))

    if use_graph:
        system = build_donna_system_prompt([], user_text=query)
        result = run_react_loop(
            user_text=query,
            system_prompt=system,
            execute_fn=execute_tool_call,
            max_iters=max(REACT_MAX_ITERS, 5),
            broker=broker,
            enable_reflection=want_reflection,
            forced_tool=forced,
            tts_callback=None,
            model=OLLAMA_MODEL,
        )
        try:
            commit_agentic_turn(system, query, result.final_text or "")
        except Exception:  # noqa: BLE001
            pass
        route = "react_graph"
    else:
        result = run_lightweight_chat(
            user_text=query,
            ask_fn=ask_ollama_messages,
            model=OLLAMA_MODEL,
            use_chat_memory=True,
        )
        route = "lightweight_chat"

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "query": query,
        "route": route,
        "final_text": str(getattr(result, "final_text", "") or ""),
        "iterations": int(getattr(result, "iterations", 0) or 0),
        "tool_trace": list(getattr(result, "tool_trace", None) or []),
        "had_errors": bool(getattr(result, "had_errors", False)),
        "reflection": getattr(result, "reflection", None),
        "elapsed_ms": elapsed_ms,
        "requires_tool_graph": bool(requires_tool_graph(query)),
    }


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def _pass_result(verdict: str = "pass", **extra: Any) -> dict[str, Any]:
    out = {"verdict": verdict, "flaw": None, "flaws": [], "status": "PASS"}
    out.update(extra)
    return out


def _fail_result(verdict: str, *flaws: str, **extra: Any) -> dict[str, Any]:
    fl = [f for f in flaws if f]
    out = {
        "verdict": verdict,
        "flaw": fl[0] if fl else None,
        "flaws": fl[1:],
        "status": "FAIL",
    }
    out.update(extra)
    return out


def _eval_yesterday(
    answer: str,
    ground: dict[str, Any],
    *,
    route: str = "",
    tool_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tools = _tools_used(tool_trace)
    used = "list_activity_for_day" in tools
    if used and route == "react_graph":
        return _pass_result("retrieved", used_day_index=True)
    if ground.get("has_evidence"):
        return _fail_result(
            "retrieval_failure",
            "Yesterday evidence exists but list_activity_for_day was not used.",
            used_day_index=used,
        )
    return _pass_result("honest_gap", used_day_index=used)


def _eval_capabilities(answer: str, tool_names: list[str]) -> dict[str, Any]:
    low = (answer or "").lower()
    stack_ok = bool(re.search(r"\b(?:langgraph|ollama)\b", low) or re.search(r"\breact\b", low))
    cats = sum(
        1
        for k in (
            "desktop",
            "code execution",
            "file",
            "research",
            "perception",
            "memory",
            "vault",
        )
        if k in low
    )
    if stack_ok and cats >= 2:
        return _pass_result("grounded", registry_size=len(tool_names))
    return _fail_result(
        "partial",
        "Weak tool-registry / agentic-stack grounding in capability answer.",
        registry_size=len(tool_names),
    )


def _eval_improve(answer: str, reflection: Any) -> dict[str, Any]:
    low = (answer or "").lower()
    ok = (
        any(k in low for k in ("langgraph", "react", "ollama"))
        and any(k in low for k in ("memory", "episodic", "vault", "retrieval"))
        and any(k in low for k in ("gap", "fail", "limit", "weak", "missing", "blind", "hallucin"))
        and reflection is not None
    )
    if ok:
        return _pass_result("strong", has_reflection_payload=True)
    flaws = []
    if reflection is None:
        flaws.append("Reflection payload absent.")
    if not any(k in low for k in ("langgraph", "react", "ollama")):
        flaws.append("Missing framework/architecture awareness.")
    if not any(k in low for k in ("memory", "episodic", "vault")):
        flaws.append("Missing memory retention critique.")
    return _fail_result("partial", *flaws, has_reflection_payload=reflection is not None)


def _eval_cpu_vram(answer: str, tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    tools = _tools_used(tool_trace)
    obs = _obs_blob(tool_trace).lower() + "\n" + (answer or "").lower()
    telemetry_tools = {
        "get_system_telemetry",
        "execute_powershell",
        "shell_execute",
        "run_terminal_command",
        "execute_command",
        "python_repl",
        "execute_python_script",
        "read_system_architecture",
    }
    used = bool(set(tools) & telemetry_tools)
    # Live metrics: percent-like numbers that are not all zeros.
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", obs)]
    has_nonzero = any(n > 0.0 for n in nums)
    has_vram_or_gpu = any(k in obs for k in ("vram", "gpu", "cuda", "nvidia", "dedicated"))
    has_cpu = "cpu" in obs
    if used and has_cpu and (has_nonzero or has_vram_or_gpu):
        return _pass_result("pass", tools=tools, metrics=nums[:8])
    if not used:
        return _fail_result(
            "hallucination_or_refusal",
            "Did not call telemetry/shell diagnostics tools for CPU/VRAM.",
            tools=tools,
        )
    return _fail_result(
        "partial",
        "Tools ran but live non-zero CPU/VRAM metrics were not clearly returned.",
        tools=tools,
    )


def _eval_swarm_status(answer: str, tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    tools = _tools_used(tool_trace)
    allowed = {
        "get_sandbox_job_status",
        "dispatch_research_swarm",
        "list_todo_basket",
        "python_repl",
        "shell_execute",
        "execute_powershell",
        "run_terminal_command",
        "list_activity_for_day",
    }
    used = bool(set(tools) & allowed) or any(
        "sandbox" in t or "swarm" in t or "watchdog" in t for t in tools
    )
    low = (answer or "").lower()
    guessed = (not used) and any(
        k in low for k in ("yes", "no", "running", "none", "active", "swarm")
    )
    if used:
        return _pass_result("pass", tools=tools)
    if guessed:
        return _fail_result(
            "guessed_without_tools",
            "Guessed swarm status without get_sandbox_job_status / state checks.",
            tools=tools,
        )
    return _fail_result(
        "partial",
        "No swarm/sandbox state tool was executed.",
        tools=tools,
    )


def _eval_user_away(answer: str, tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    tools = _tools_used(tool_trace)
    obs = _obs_blob(tool_trace).lower()
    low = (answer or "").lower()
    blob = obs + "\n" + low
    used = bool(tools) and any(
        t in tools
        for t in (
            "parse_idle_log_duration",
            "list_activity_for_day",
            "read_local_file",
            "shell_execute",
            "run_terminal_command",
            "execute_powershell",
            "python_repl",
            "read_vault_memory",
            "file_editor",
        )
    )
    # Duration parse: seconds or mm:ss / human minutes.
    has_duration = bool(
        re.search(r"\b742\b", blob)
        or re.search(r"\b\d+\s*(?:seconds?|secs?|minutes?|mins?)\b", blob)
        or re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", blob)
    )
    mentions_away = "user_away" in blob or "away" in blob
    if used and has_duration and mentions_away:
        return _pass_result("pass", tools=tools)
    if not used:
        return _fail_result(
            "retrieval_failure",
            "Did not query idle/state logs for last USER_AWAY duration.",
            tools=tools,
        )
    return _fail_result(
        "partial",
        "Queried state but did not parse an exact USER_AWAY duration.",
        tools=tools,
    )


def _eval_git_cascade(answer: str, tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    tools = _tools_used(tool_trace)
    obs = _obs_blob(tool_trace)
    blob = (obs + "\n" + (answer or "")).lower()
    shellish = bool(
        set(tools)
        & {
            "run_terminal_command",
            "shell_execute",
            "execute_powershell",
            "execute_command",
            "python_repl",
        }
    )
    touched_repo = "cascade-router" in blob or "cascade_router" in blob
    git_used = "git" in blob
    date_like = bool(
        re.search(
            r"\b(?:20\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            blob,
            flags=re.I,
        )
    )
    if shellish and git_used and date_like:
        return _pass_result("pass", tools=tools, touched_repo=touched_repo)
    flaws = []
    if not shellish:
        flaws.append("No shell/git tool execution for cascade-router commit date.")
    if not git_used:
        flaws.append("Git was not invoked / observed in the trajectory.")
    if not date_like:
        flaws.append("Could not parse a commit date from tool output / answer.")
    return _fail_result("partial", *flaws, tools=tools)


def _eval_watchdog_graph(answer: str, tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    tools = _tools_used(tool_trace)
    obs = _obs_blob(tool_trace)
    blob = (obs + "\n" + (answer or "")).lower()
    read_tools = bool(
        set(tools)
        & {
            "read_local_file",
            "file_editor",
            "shell_execute",
            "run_terminal_command",
            "execute_powershell",
            "python_repl",
            "read_system_architecture",
            "search_vault",
        }
    )
    hit_watchdog = "watchdog" in blob
    # Dependency-like tokens from dana.swarm.watchdog_graph / graph nodes.
    deps = [
        w
        for w in (
            "langgraph",
            "titan",
            "experience",
            "sqlite",
            "supervisor",
            "dispatcher",
            "node",
            "import",
            "dependency",
            "compile",
            "state",
        )
        if w in blob
    ]
    if read_tools and hit_watchdog and len(deps) >= 2:
        return _pass_result("pass", tools=tools, deps=deps)
    return _fail_result(
        "partial",
        "Did not locate/read watchdog graph and extract active dependencies.",
        tools=tools,
        deps=deps,
    )


def _eval_latex_nocite(answer: str, tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    text = answer or ""
    # Also accept LaTeX living in tool observations (file writes).
    blob = text + "\n" + _obs_blob(tool_trace)
    latexish = bool(
        re.search(r"\\begin\{|\\section|\\textbf|\\itemize|\\documentclass", blob)
        or ("\\item" in blob)
    )
    cites = re.findall(r"\\cite[a-zA-Z]*\b|\\bibliography\b|\\citep\b", blob)
    if latexish and not cites:
        return _pass_result("pass", tools=_tools_used(tool_trace))
    if cites:
        return _fail_result(
            "constraint_violation",
            f"LaTeX contained forbidden citation tags: {cites[:6]}",
        )
    return _fail_result(
        "partial",
        "Response was not valid LaTeX (missing document/structure commands).",
    )


def _eval_interview_companies(answer: str, tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    low = (answer or "").lower()
    obs = _obs_blob(tool_trace).lower()
    blob = low + "\n" + obs
    needed = ("zoox", "waymo", "dimensional")
    hits = [c for c in needed if c in blob]
    if len(hits) == 3:
        return _pass_result("pass", companies=hits, tools=_tools_used(tool_trace))
    if hits:
        return _fail_result(
            "partial",
            f"Only partial company recall: {hits} (need Zoox, Waymo, Dimensional Inc.).",
            companies=hits,
        )
    return _fail_result(
        "retrieval_failure",
        "Failed to retrieve Zoox / Waymo / Dimensional Inc. from episodic memory.",
        tools=_tools_used(tool_trace),
    )


def _eval_orbital_mistake(answer: str, tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    blob = ((answer or "") + "\n" + _obs_blob(tool_trace)).lower()
    # Specific timing correction, not generic orbital mechanics.
    specific = (
        ("86164" in blob and "86400" in blob)
        or ("sidereal" in blob and "solar" in blob)
        or ("86164" in blob)
    )
    generic_only = (
        not specific
        and any(k in blob for k in ("kepler", "2*pi", "sqrt", "gm", "newton"))
    )
    if specific:
        return _pass_result("pass", tools=_tools_used(tool_trace))
    if generic_only:
        return _fail_result(
            "generic_physics",
            "Gave generic orbital formulas instead of the discussed 86164→86400 correction.",
        )
    return _fail_result(
        "retrieval_failure",
        "Did not extract the episodic orbital-period calculation mistake.",
        tools=_tools_used(tool_trace),
    )


def _eval_six_axis_trap(answer: str, tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    low = (answer or "").lower()
    refuse = any(
        k in low
        for k in (
            "no professional",
            "no experience",
            "don't have",
            "do not have",
            "don't recall",
            "do not recall",
            "no record",
            "not in my memory",
            "no memory",
            "never worked",
            "haven't worked",
            "have not worked",
            "clarify",
            "clarification",
            "i don't see",
            "no evidence",
        )
    )
    # Hallucination: invents roles/projects on 6-axis platforms.
    invents = bool(
        re.search(
            r"\b(i (worked|designed|built|deployed|led)|your experience includes|"
            r"you worked|you designed|years? of experience)\b",
            low,
        )
    ) and ("6-axis" in low or "six-axis" in low or "6 axis" in low)
    if refuse and not invents:
        return _pass_result("pass", tools=_tools_used(tool_trace))
    if invents:
        return _fail_result(
            "hallucination",
            "Hallucinated 6-axis mobility experience instead of refusing.",
        )
    return _fail_result(
        "partial",
        "Did not clearly refuse/push back on nonexistent 6-axis mobility experience.",
    )


def _eval_cat_names(answer: str, tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    blob = ((answer or "") + "\n" + _obs_blob(tool_trace)).lower()
    has_eddie = "eddie" in blob
    has_tulip = "tulip" in blob
    denies = any(
        k in blob
        for k in (
            "don't know",
            "do not know",
            "no record",
            "no memory",
            "not sure",
            "don't recall",
            "do not recall",
        )
    )
    # Generic guess without the seeded names.
    generic_guess = (not has_eddie or not has_tulip) and any(
        k in blob for k in ("fluffy", "mittens", "whiskers", "shadow", "luna", "milo")
    )
    if has_eddie and has_tulip and not generic_guess:
        return _pass_result("pass", names=["eddie", "tulip"], tools=_tools_used(tool_trace))
    if denies and not (has_eddie and has_tulip):
        return _fail_result(
            "retrieval_failure",
            "Claimed not to know cat names instead of retrieving Eddie and Tulip.",
        )
    if generic_guess:
        return _fail_result(
            "hallucination",
            "Guessed generic pet names instead of Eddie and Tulip.",
        )
    missing = [n for n, ok in (("Eddie", has_eddie), ("Tulip", has_tulip)) if not ok]
    return _fail_result(
        "partial",
        f"Missing cat name(s) from episodic memory: {', '.join(missing)}.",
    )


def _eval_car_maintenance(
    answer: str, tool_trace: list[dict[str, Any]] | None
) -> dict[str, Any]:
    blob = ((answer or "") + "\n" + _obs_blob(tool_trace)).lower()
    has_model = ("rav4" in blob or "rav 4" in blob) and (
        "2022" in blob or "toyota" in blob
    )
    has_splash = "splash shield" in blob or "engine splash" in blob
    has_glass = "glass repair" in blob or (
        "glass" in blob and "repair" in blob
    )
    if has_model and has_splash and has_glass:
        return _pass_result(
            "pass",
            model="2022 Toyota RAV4",
            parts=["engine splash shield", "glass repair"],
            tools=_tools_used(tool_trace),
        )
    missing = []
    if not has_model:
        missing.append("2022 Toyota RAV4")
    if not has_splash:
        missing.append("engine splash shield")
    if not has_glass:
        missing.append("glass repair")
    return _fail_result(
        "retrieval_failure",
        "Failed to retrieve car model / maintenance facts: " + ", ".join(missing) + ".",
    )


def _eval_dining_drink(
    answer: str, tool_trace: list[dict[str, Any]] | None
) -> dict[str, Any]:
    blob = ((answer or "") + "\n" + _obs_blob(tool_trace)).lower()
    has_diet_coke = "diet coke" in blob or "dietcola" in blob.replace(" ", "")
    wrong_drink = any(
        k in blob
        for k in ("sprite", "pepsi", "dr pepper", "lemonade", "iced tea", "water only")
    ) and not has_diet_coke
    if has_diet_coke:
        return _pass_result("pass", drink="Diet Coke", tools=_tools_used(tool_trace))
    if wrong_drink:
        return _fail_result(
            "hallucination",
            "Invented a drink other than Diet Coke from dining preferences.",
        )
    return _fail_result(
        "retrieval_failure",
        "Did not retrieve Diet Coke from episodic dining preferences.",
    )


def _dispatch_eval(
    probe: ProbeSpec,
    turn: dict[str, Any],
    *,
    ground: dict[str, Any],
    tool_names: list[str],
) -> dict[str, Any]:
    kind = str(probe.get("eval") or "")
    answer = str(turn.get("final_text") or "")
    trace = list(turn.get("tool_trace") or [])
    route = str(turn.get("route") or "")
    if kind == "yesterday":
        return _eval_yesterday(answer, ground, route=route, tool_trace=trace)
    if kind == "capabilities":
        return _eval_capabilities(answer, tool_names)
    if kind == "improve":
        return _eval_improve(answer, turn.get("reflection"))
    if kind == "cpu_vram":
        return _eval_cpu_vram(answer, trace)
    if kind == "swarm_status":
        return _eval_swarm_status(answer, trace)
    if kind == "user_away":
        return _eval_user_away(answer, trace)
    if kind == "git_cascade":
        return _eval_git_cascade(answer, trace)
    if kind == "watchdog_graph":
        return _eval_watchdog_graph(answer, trace)
    if kind == "latex_nocite":
        return _eval_latex_nocite(answer, trace)
    if kind == "interview_companies":
        return _eval_interview_companies(answer, trace)
    if kind == "orbital_mistake":
        return _eval_orbital_mistake(answer, trace)
    if kind == "six_axis_trap":
        return _eval_six_axis_trap(answer, trace)
    if kind == "cat_names":
        return _eval_cat_names(answer, trace)
    if kind == "car_maintenance":
        return _eval_car_maintenance(answer, trace)
    if kind == "dining_drink":
        return _eval_dining_drink(answer, trace)
    return _fail_result("unknown", f"No evaluator for {kind!r}")


# ---------------------------------------------------------------------------
# PSOL report
# ---------------------------------------------------------------------------

def _llm_psol_synthesis(
    turns: list[dict[str, Any]],
    evals: dict[str, Any],
    ground: dict[str, Any],
) -> str:
    try:
        from dana.core_agent import OLLAMA_MODEL, ask_ollama_messages
    except Exception:
        return ""

    payload = {
        "turns": [
            {
                "id": t.get("id"),
                "suite": t.get("suite"),
                "query": t.get("query"),
                "route": t.get("route"),
                "answer": (t.get("final_text") or "")[:900],
                "tools": _tools_used(list(t.get("tool_trace") or []))[:12],
                "elapsed_ms": t.get("elapsed_ms"),
                "iterations": t.get("iterations"),
            }
            for t in turns
        ],
        "evals": {
            k: {
                "verdict": v.get("verdict"),
                "status": v.get("status"),
                "flaws": ([v.get("flaw")] if v.get("flaw") else []) + list(v.get("flaws") or []),
            }
            for k, v in evals.items()
        },
        "yesterday_evidence": {
            "day": ground.get("day"),
            "has_evidence": ground.get("has_evidence"),
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a ruthless architecture auditor for Dānā. Using ONLY the JSON, "
                "write exactly four markdown sections: ## Problem, ## Solution, "
                "## Outcome, ## Lessons. Cover perception, orchestration, episodic "
                "retrieval, and hallucination traps. No preamble."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)[:14000]},
    ]
    try:
        return str(ask_ollama_messages(messages, model=OLLAMA_MODEL) or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"(LLM PSOL synthesis failed: {exc})"


def _render_psol(
    *,
    turns: list[dict[str, Any]],
    evals: dict[str, Any],
    ground: dict[str, Any],
    tool_names: list[str],
    llm_psol: str,
    seeded: list[str],
) -> str:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    passed = sum(1 for e in evals.values() if e.get("status") == "PASS")
    total = len(evals)
    flaws: list[str] = []
    for key, ev in evals.items():
        if ev.get("status") == "PASS":
            continue
        if ev.get("flaw"):
            flaws.append(f"[{key}] {ev['flaw']}")
        for f in ev.get("flaws") or []:
            flaws.append(f"[{key}] {f}")
    if not flaws:
        flaws.append("No high-confidence heuristic flaws flagged (still review raw turns).")

    body: list[str] = [
        "# Dānā Context Awareness — Aggregated PSOL Diagnostic",
        "",
        f"_Generated: {now}_",
        f"_Score: {passed}/{total} probes PASS_",
        "",
        "## Problem",
        "",
        "Expanded headless suites (perception, orchestration, episodic traps) "
        "still surface the following failures / gaps:",
        "",
    ]
    for f in flaws:
        body.append(f"- {f}")
    body.extend(
        [
            "",
            f"- Suite 4 episodic seeds applied: `{', '.join(seeded)}`.",
            f"- Yesterday evidence day `{ground.get('day')}` "
            f"has_evidence=`{ground.get('has_evidence')}`.",
            f"- Tool registry size: `{len(tool_names)}`.",
            "",
            "## Solution",
            "",
            "1. **Live telemetry tools** — bind PowerShell/GPU queries for CPU/VRAM asks; "
            "forbid metric invention on lightweight chat.",
            "2. **Swarm/state bus** — force `get_sandbox_job_status` (or equivalent) for "
            "background swarm questions.",
            "3. **Idle ledger** — persist USER_AWAY start/end timestamps and expose a "
            "duration helper readable by the tool graph.",
            "4. **Repo navigation** — resolve named repositories (e.g. cascade-router) "
            "before `git log` and parse commit dates from stdout.",
            "5. **Graph literacy** — teach watchdog_graph path + dependency extraction "
            "without dumping full files into context.",
            "6. **Constraint following** — negative constraints (no `\\cite`) must be "
            "checked before FINAL.",
            "7. **Episodic grounding** — interview / orbital / mobility facts must come "
            "from SQLite matches; hallucination traps must refuse.",
            "8. **Combinatorial episodic retrieval** — multi-fact personal asks "
            "(pets, car maintenance, dining drinks) must combine distinct "
            "`episodic_facts` rows without inventing fillers.",
            "",
            "## Outcome",
            "",
        ]
    )

    current_suite = ""
    for t in turns:
        suite = str(t.get("suite") or "")
        if suite != current_suite:
            current_suite = suite
            body.append(f"### {suite}")
            body.append("")
        tid = str(t.get("id"))
        ev = evals.get(tid, {})
        body.append(
            f"#### Probe `{tid}` — `{ev.get('status', '?')}` / `{ev.get('verdict', 'n/a')}` "
            f"— route `{t.get('route')}` ({t.get('elapsed_ms')} ms, iters={t.get('iterations')})"
        )
        body.append("")
        body.append(f"**User:** {t.get('query')}")
        body.append("")
        body.append("**Dana:**")
        body.append("")
        body.append(t.get("final_text") or "_(empty)_")
        body.append("")
        tools = _tools_used(list(t.get("tool_trace") or []))
        if tools:
            body.append(f"**Tools:** {', '.join(tools)}")
            body.append("")
        if ev.get("flaw") or ev.get("flaws"):
            body.append("**Flaws:**")
            if ev.get("flaw"):
                body.append(f"- {ev['flaw']}")
            for f in ev.get("flaws") or []:
                body.append(f"- {f}")
            body.append("")
        body.append(
            f"**Routing:** requires_tool_graph=`{t.get('requires_tool_graph')}` "
            f"forced_route=`{t.get('route')}`"
        )
        body.append("")

    body.extend(
        [
            "### Suite scoreboard",
            "",
            "| Probe | Suite | Status | Verdict | Route | ms | Tools |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for t in turns:
        tid = str(t.get("id"))
        ev = evals.get(tid, {})
        tools = ", ".join(_tools_used(list(t.get("tool_trace") or []))[:6]) or "—"
        body.append(
            f"| `{tid}` | {t.get('suite', '')} | {ev.get('status')} | "
            f"`{ev.get('verdict')}` | `{t.get('route')}` | {t.get('elapsed_ms')} | {tools} |"
        )
    body.extend(
        [
            "",
            "## Lessons",
            "",
            "- Perception questions without bound telemetry tools become confident fiction.",
            "- Orchestration probes fail when the agent summarizes instead of chaining "
            "shell → parse → answer.",
            "- Episodic traps require both seeded SQLite facts and a refusal policy for "
            "absent career history.",
            "- Combinatorial personal facts (cats / car / drinks) fail when SQLite "
            "priority is skipped or only one of several seeded rows is hydrated.",
            "- Aggregated PSOL should drive nightly regressions across all five suites, "
            "not only Suite 1 chat probes.",
            "",
        ]
    )
    if llm_psol.strip():
        body.extend(
            [
                "---",
                "",
                "## LLM Auditor PSOL (secondary)",
                "",
                llm_psol.strip(),
                "",
            ]
        )
    body.append(f"_Registry sample:_ `{', '.join(tool_names[:24])}`")
    body.append("")
    return "\n".join(body)


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic() -> Path:
    logs_dir = _ensure_workspace()
    report_path = logs_dir / REPORT_NAME

    print("=== Dana context awareness diagnostic (Suites 1–5, headless) ===")
    print("Loading core_agent / Ollama bindings...")
    import dana.core_agent as _agent  # noqa: F401

    try:
        from dana.agentic import clear_chat_memory

        clear_chat_memory()
    except Exception:  # noqa: BLE001
        pass

    seeded = _seed_suite4_episodic_facts() + _seed_suite5_episodic_facts()
    _seed_user_away_log(logs_dir)
    cascade = _ensure_cascade_router_fixture()
    print(f"Episodic seeds: {seeded}")
    print(f"cascade-router fixture: {cascade}")

    tool_names = _tool_capability_names()
    print(f"Tool registry: {len(tool_names)} public tool(s)")
    ground = _gather_yesterday_ground_truth()

    turns: list[dict[str, Any]] = []
    evals: dict[str, Any] = {}

    for probe in ALL_PROBES:
        pid = str(probe["id"])
        query = str(probe["query"])
        print(f"\n--- Inject [{pid}] {probe.get('suite')} ---\n{query}")
        row = _run_one_query(
            query,
            enable_reflection=bool(probe.get("enable_reflection")),
            force_graph=bool(probe.get("force_graph")),
        )
        row["id"] = pid
        row["suite"] = probe.get("suite")
        turns.append(row)
        preview = (row.get("final_text") or "").replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        print(
            f"route={row['route']} graph_flag={row.get('requires_tool_graph')} "
            f"ms={row['elapsed_ms']} tools={_tools_used(row.get('tool_trace'))} → {preview}"
        )
        ev = _dispatch_eval(probe, row, ground=ground, tool_names=tool_names)
        evals[pid] = ev

    print("\n=== Heuristic summary ===")
    for probe in ALL_PROBES:
        pid = str(probe["id"])
        ev = evals[pid]
        print(
            f"  [{pid}] {ev.get('status')} verdict={ev.get('verdict')} "
            f"flaws={[ev.get('flaw')] + list(ev.get('flaws') or []) if ev.get('status') == 'FAIL' else '-'}"
        )

    print("\nSynthesizing aggregated PSOL…")
    llm_psol = _llm_psol_synthesis(turns, evals, ground)
    report = _render_psol(
        turns=turns,
        evals=evals,
        ground=ground,
        tool_names=tool_names,
        llm_psol=llm_psol,
        seeded=seeded,
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {report_path}")
    return report_path


def test_context_awareness_diagnostic_generates_psol() -> None:
    """Pytest entry: skip if Ollama unreachable; otherwise require PSOL artifact."""
    import urllib.request

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)  # noqa: S310
    except Exception:
        import pytest

        pytest.skip("Ollama not reachable — live context diagnostic skipped")

    path = run_diagnostic()
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for section in ("## Problem", "## Solution", "## Outcome", "## Lessons"):
        assert section in text, f"missing PSOL section {section}"
    assert (
        "Suite 2" in text
        and "Suite 3" in text
        and "Suite 4" in text
        and "Suite 5" in text
    )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    try:
        path = run_diagnostic()
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: diagnostic failed: {exc}", file=sys.stderr)
        return 1
    print("\n======== PSOL REPORT ========\n")
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

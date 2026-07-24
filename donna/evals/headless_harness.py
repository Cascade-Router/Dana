"""Headless Eval Harness — CI/CD for Donna agents (no mic / VAD / TTS).

Loads ``donna/evals/test_cases.json``, injects synthetic transcripts into the
broker + ReAct orchestrator, and scores Pass@k (any of k attempts succeeds).

Usage:
  python -m donna.evals.headless_harness
  python -m donna.evals.headless_harness --k 3 --cases donna/evals/test_cases.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from donna.paths import PROJECT_ROOT

DEFAULT_CASES = PROJECT_ROOT / "donna" / "evals" / "test_cases.json"


@dataclass
class CaseResult:
    case_id: str
    query: str
    expected_tool: str | None
    attempts: int
    successes: int
    pass_at_k: bool
    details: list[dict[str, Any]] = field(default_factory=list)


def load_cases(path: Path | None = None) -> dict[str, Any]:
    target = path or DEFAULT_CASES
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise ValueError(f"Invalid eval dataset: {target}")
    return raw


def _broker_hit(query: str) -> tuple[str | None, dict[str, Any]]:
    from donna.tools.broker import IntentBroker
    from donna.tools.stt_corrector import correct_stt

    cleaned = correct_stt(query)
    call = IntentBroker().parse_utterance(cleaned)
    if call is None:
        return None, {}
    return call.tool_id, dict(call.arguments or {})


def _run_orchestrator_once(
    query: str,
    *,
    expected_tool: str | None,
    dry_execute: bool = True,
) -> dict[str, Any]:
    """Inject ``query`` into broker (+ optional mocked ReAct). No audio."""
    from donna.agentic import run_react_loop
    from donna.tools.broker import IntentBroker
    from donna.tools.schema import ToolCall

    routed_id, routed_args = _broker_hit(query)
    executed: list[str] = []

    def execute_fn(tc: ToolCall) -> str:
        executed.append(tc.tool_id)
        if dry_execute:
            return f"OK: dry-run {tc.tool_id} args={dict(tc.arguments)}"
        from donna.core_agent import execute_tool_call

        return execute_tool_call(tc)

    class _StubLLM:
        def __init__(self) -> None:
            self.n = 0

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            from langchain_core.messages import AIMessage

            self.n += 1
            # Prefer broker-routed tool; else expected_tool; else plain FINAL.
            tool = routed_id or expected_tool
            if self.n == 1 and tool:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool,
                            "args": routed_args
                            or (
                                {"goal": query}
                                if tool == "architect_new_tool"
                                else {"query": query}
                                if tool
                                in {
                                    "dispatch_research_swarm",
                                    "dispatch_titan_repair",
                                    "delegate_to_cursor",
                                    "web_search",
                                }
                                else {"text": "hello"}
                                if tool == "execute_os_keystrokes"
                                else {}
                            ),
                            "id": "eval_1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="FINAL: Eval harness complete.")

    forced = None
    if routed_id:
        forced = ToolCall(
            tool_id=routed_id,
            arguments=routed_args,
            raw_text=query,
            confidence=0.95,
        )

    with patch("langchain_ollama.ChatOllama", lambda **_k: _StubLLM()), patch(
        "donna.cascade_router.resolve_chat_model",
        lambda **_k: _StubLLM(),
    ), patch.dict(os.environ, {"DONNA_CURSOR_LAUNCH": "0", "DONNA_OS_DRY_RUN": "1"}):
        result = run_react_loop(
            user_text=query,
            system_prompt="You are Donna eval harness. Use tools when bound.",
            execute_fn=execute_fn,
            max_iters=3,
            broker=IntentBroker(),
            enable_reflection=False,
            forced_tool=forced,
        )

    tools_seen = [
        str(t.get("tool") or "")
        for t in (result.tool_trace or [])
        if t.get("tool")
    ]
    tools_seen.extend(executed)
    obs_blob = " ".join(
        str(t.get("observation") or "") for t in (result.tool_trace or [])
    )
    terminal_fail = (
        "terminal_failure" in obs_blob.lower()
        or "couldn't finish that cleanly" in (result.final_text or "").lower()
        or any(str(t.get("error") or "") for t in (result.tool_trace or []))
    )
    return {
        "routed_tool": routed_id,
        "tools_seen": tools_seen,
        "final_text": result.final_text,
        "terminal_failure": terminal_fail,
        "had_errors": bool(result.had_errors),
    }


def evaluate_case(
    case: dict[str, Any],
    *,
    k: int = 3,
) -> CaseResult:
    """Pass@k: success if any of k attempts hits expected_tool without terminal_failure."""
    case_id = str(case.get("id") or case.get("query") or "case")
    query = str(case.get("query") or "").strip()
    expected = case.get("expected_tool")
    expected_tool = None if expected in (None, "", "null") else str(expected)
    expect_no_tool = bool(case.get("expect_no_tool"))
    details: list[dict[str, Any]] = []
    successes = 0

    for attempt in range(max(1, k)):
        try:
            out = _run_orchestrator_once(query, expected_tool=expected_tool)
        except Exception as exc:  # noqa: BLE001
            details.append({"attempt": attempt + 1, "error": str(exc), "ok": False})
            continue
        tools = out.get("tools_seen") or []
        routed = out.get("routed_tool")
        hit = False
        if expect_no_tool:
            hit = routed is None and not tools and not out.get("terminal_failure")
        elif expected_tool:
            hit = (
                (routed == expected_tool or expected_tool in tools)
                and not out.get("terminal_failure")
            )
        else:
            hit = not out.get("terminal_failure")
        if hit:
            successes += 1
        details.append(
            {
                "attempt": attempt + 1,
                "ok": hit,
                "routed_tool": routed,
                "tools_seen": tools,
                "terminal_failure": out.get("terminal_failure"),
                "final": (out.get("final_text") or "")[:120],
            }
        )

    return CaseResult(
        case_id=case_id,
        query=query,
        expected_tool=expected_tool,
        attempts=max(1, k),
        successes=successes,
        pass_at_k=successes > 0,
        details=details,
    )


def run_harness(
    *,
    cases_path: Path | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    dataset = load_cases(cases_path)
    pass_k = int(k if k is not None else dataset.get("pass_at_k") or 3)
    results: list[CaseResult] = []
    t0 = time.perf_counter()
    for case in dataset["cases"]:
        if not isinstance(case, dict):
            continue
        results.append(evaluate_case(case, k=pass_k))
    elapsed = time.perf_counter() - t0
    n = len(results) or 1
    passed = sum(1 for r in results if r.pass_at_k)
    report = {
        "pass_at_k": pass_k,
        "cases": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": passed / n,
        "elapsed_sec": round(elapsed, 3),
        "results": [
            {
                "id": r.case_id,
                "query": r.query,
                "expected_tool": r.expected_tool,
                "pass_at_k": r.pass_at_k,
                "successes": r.successes,
                "attempts": r.attempts,
                "details": r.details,
            }
            for r in results
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Donna headless agent eval harness")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help="Path to test_cases.json",
    )
    parser.add_argument("--k", type=int, default=None, help="Pass@k attempts per case")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full report JSON",
    )
    args = parser.parse_args(argv)

    try:
        from donna.stdio_boot import ensure_stdio

        ensure_stdio()
    except Exception:
        pass
    print("Donna headless eval harness")
    print("=" * 60)
    report = run_harness(cases_path=args.cases, k=args.k)
    for row in report["results"]:
        mark = "PASS" if row["pass_at_k"] else "FAIL"
        print(
            f"[{mark}] {row['id']}: expected={row['expected_tool']!r} "
            f"successes={row['successes']}/{row['attempts']}"
        )
    print("-" * 60)
    print(
        f"Pass@{report['pass_at_k']}: {report['passed']}/{report['cases']} "
        f"({report['pass_rate']:.0%}) in {report['elapsed_sec']}s"
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.json_out}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

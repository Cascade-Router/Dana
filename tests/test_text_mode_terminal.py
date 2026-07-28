"""Temporary headless ReAct probe: LLM + run_terminal_command (no mic/TTS).

Query: list current directory and say if any Python files exist.
"""

from __future__ import annotations

import json
import os
import re
import sys

from dana.agentic import REACT_MAX_ITERS, run_react_loop
from dana.os_automation import run_terminal_command
from dana.tools.broker import IntentBroker
from dana.tools.schema import ToolCall
from dana.prompts.spatial_synthesis import build_agent_system_prompt

QUERY = (
    "Can you list the files in my current directory and tell me if there "
    "are any Python files?"
)

WINDOW_CMDS = ("dir", "ls", "Get-ChildItem", "gci", "python")


def execute_tool_call(tc: ToolCall) -> str:
    if tc.tool_id == "run_terminal_command":
        command = str(tc.arguments.get("command") or "").strip()
        if not command:
            return "ERROR: missing command"
        result = run_terminal_command(command)
        if str(result).upper().startswith("ERROR"):
            return str(result)
        return f"OK: run_terminal_command output=\n{result}"
    return f"ERROR: unsupported tool {tc.tool_id} (terminal probe only)"


def _looks_like_raw_dump(final: str, observation: str) -> bool:
    """Heuristic: FINAL copies large chunks of dir listing instead of summarizing."""
    text = (final or "").strip()
    if not text:
        return True
    # Long multi-line dumps with many .py paths look like raw terminal read-aloud.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 8:
        return True
    if len(text.split()) > 60:
        return True
    obs = observation or ""
    if obs and len(text) > 80:
        # Identical chunk of 40+ chars from observation → likely dump.
        for chunk in re.findall(r".{40,}", obs):
            if chunk in text:
                return True
            if chunk[:40] in text and text.count(".py") >= 5:
                return True
    return False


def main() -> int:
    try:
        import dana.core_agent as _agent  # noqa: F401 — ensure agent package loads (Ollama host)
    except ImportError as exc:
        print(f"ERROR: could not import dana.core_agent: {exc}", file=sys.stderr)
        return 1

    broker = IntentBroker()
    print("=== Broker fast-path ===")
    intent = broker.parse_utterance(QUERY)
    if intent:
        print(f"  tool_id={intent.tool_id!r}  args={intent.arguments!r}")
    else:
        print("  (none — LLM chooses)")

    system = build_agent_system_prompt(
        spatial_block="vis=screen|ui=idle|dom=none@center|scene=[]|intent=",
        labels_csv="",
        profile_summary="{}",
        reply_lang="en",
    )
    assert "OS Automation Rules" in system
    assert "run_terminal_command" in system

    print("\n=== Running ReAct loop ===")
    print(f"Query: {QUERY!r}\n")

    result = run_react_loop(
        user_text=QUERY,
        system_prompt=system,
        execute_fn=execute_tool_call,
        max_iters=REACT_MAX_ITERS,
        broker=broker,
        enable_reflection=False,
    )

    print("=== Tool trace ===")
    print(json.dumps(result.tool_trace, indent=2, ensure_ascii=False))

    print("\n=== Final response ===")
    print(result.final_text)
    print(f"\n(iterations={result.iterations}, reply_lang={result.reply_lang})")

    term_steps = [t for t in result.tool_trace if t.get("tool") == "run_terminal_command"]
    cmds = [str((t.get("args") or {}).get("command") or "") for t in term_steps]
    obs_blob = "\n".join(str(t.get("observation") or "") for t in term_steps)
    final = result.final_text or ""

    print("\n=== Analysis ===")
    called = bool(term_steps)
    print(f"1. Called run_terminal_command? {'YES' if called else 'NO'}")
    if called:
        for i, c in enumerate(cmds, 1):
            print(f"   command[{i}]: {c!r}")

    valid_win = any(
        any(tok.lower() in c.lower() for tok in WINDOW_CMDS) for c in cmds
    )
    print(
        f"2. Valid Windows listing command (dir/ls/Get-ChildItem/...)? "
        f"{'YES' if valid_win else 'NO / N/A'}"
    )

    mentions_python = bool(re.search(r"\.?py(thon)?\b", final, re.I))
    dumped = _looks_like_raw_dump(final, obs_blob)
    fallback = bool(
        re.search(
            r"couldn't finish|please ask me again|| ",
            final,
            re.I,
        )
    )
    any_success_obs = any(
        not str(t.get("observation") or "").upper().startswith("ERROR")
        for t in term_steps
    )
    natural = (
        called
        and any_success_obs
        and not dumped
        and not fallback
        and len(final.split()) <= 60
        and "TOOL:" not in final
        and "Observation:" not in final
    )
    print(f"3. Natural summary (not raw dump)? {'YES' if natural else 'NO'}")
    print(f"   mentions python/.py? {'YES' if mentions_python else 'NO'}")
    print(f"   word_count={len(final.split())}  dump_heuristic={dumped}  fallback={fallback}")
    print(f"   successful terminal observation? {'YES' if any_success_obs else 'NO'}")

    # Windows cmd.exe does not provide POSIX `ls` / `ls -l`.
    posix_only = any(re.fullmatch(r"\s*ls(\s+-l)?\s*", c or "") for c in cmds)
    if posix_only and os.name == "nt":
        print("   note: used POSIX `ls` on Windows cmd — expected to fail without PowerShell wrap")

    if not called:
        print("\nVERDICT: FAIL — agent never called run_terminal_command")
        return 1
    if not any_success_obs:
        print(
            "\nVERDICT: FAIL — terminal tool called but command never succeeded "
            "(likely wrong shell syntax for this OS)"
        )
        return 2
    if dumped or fallback or not natural:
        print(
            "\nVERDICT: FAIL — no usable spoken summary; "
            "prompt guardrails / OS command guidance may need tightening"
        )
        return 3
    print("\nVERDICT: PASS — tool used with a listing command and a spoken summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())

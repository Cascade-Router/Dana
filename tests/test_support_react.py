"""Test-only helper: script LangChain bind_tools turns from legacy TOOL:/FINAL: lines.

Production no longer parses TOOL:/JSON Initiative text. Headless tests still
author scenarios in that compact form; this adapter turns them into AIMessage
tool_calls for ``run_react_loop`` (native path only).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage

from dana.tools.broker import IntentBroker, get_broker

ScriptFn = Callable[[list[dict[str, str]]], str]
ScriptSource = Sequence[str] | ScriptFn | list[str]

_TOOL_HEAD_RE = re.compile(
    r"^\s*(?:TOOL|Tool|tool|Action|ACTION)\s*[:：]\s*(.+?)\s*$",
    re.DOTALL,
)


def _lc_messages_to_dicts(messages: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages or []:
        role = getattr(m, "type", None) or getattr(m, "role", None)
        content = str(getattr(m, "content", "") or "")
        if role in ("human", "user"):
            out.append({"role": "user", "content": content})
        elif role in ("ai", "assistant"):
            out.append({"role": "assistant", "content": content})
        elif role == "system":
            out.append({"role": "system", "content": content})
        elif role == "tool":
            out.append({"role": "user", "content": f"Observation: {content}"})
        else:
            out.append({"role": "user", "content": content})
    return out


def _tool_call_dict(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "name": name,
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


def script_line_to_aimessage(
    raw: str,
    *,
    broker: IntentBroker | None = None,
    call_id: str = "call_test",
) -> AIMessage:
    """Map one legacy script line / JSON blob to an AIMessage for bind_tools."""
    broker = broker or get_broker()
    text = (raw or "").strip()
    if not text:
        return AIMessage(content="")

    # Explicit FINAL → conversational content (prefix stripped later by loop).
    if re.match(r"^\s*(?:FINAL|Final|final)\s*[:：]", text):
        return AIMessage(content=text)

    # JSON tool object (legacy test scripts only).
    if "{" in text and ("tool" in text or "tool_id" in text):
        try:
            start = text.find("{")
            end = text.rfind("}")
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                name = str(data.get("tool") or data.get("tool_id") or "").strip()
                args = data.get("args")
                if args is None:
                    args = data.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                if name:
                    return AIMessage(
                        content="",
                        tool_calls=[_tool_call_dict(name, args, call_id)],
                    )
        except Exception:
            pass

    # TOOL: id(k=v) via IntentBroker structured parse (STT router parser — not ReAct).
    m = _TOOL_HEAD_RE.match(text)
    payload = (m.group(1).strip() if m else text)
    candidates = [payload, f"tool: {payload}", text]
    for cand in candidates:
        try:
            structured = broker.parse_structured(cand)
        except Exception:
            structured = None
        if structured is None:
            continue
        return AIMessage(
            content="",
            tool_calls=[
                _tool_call_dict(
                    structured.tool_id,
                    dict(structured.arguments or {}),
                    call_id,
                )
            ],
        )

    # Bare prose → final spoken answer.
    return AIMessage(content=text)


def patch_scripted_llm(
    monkeypatch: Any,
    script: ScriptSource,
    *,
    broker: IntentBroker | None = None,
) -> None:
    """Monkeypatch ChatOllama.bind_tools to play back ``script`` as AIMessages."""
    broker = broker or get_broker()
    state = {"i": 0, "n": 0}

    class _Bound:
        def invoke(self, messages):  # noqa: ANN001
            state["n"] += 1
            call_id = f"call_{state['n']}"
            if callable(script):
                raw = script(_lc_messages_to_dicts(messages))
            else:
                if state["i"] >= len(script):
                    return AIMessage(content="FINAL: Done.")
                raw = script[state["i"]]
                state["i"] += 1
            return script_line_to_aimessage(str(raw or ""), broker=broker, call_id=call_id)

    class _FakeLLM:
        def bind_tools(self, _tools, **_kwargs):  # noqa: ANN001
            return _Bound()

    monkeypatch.setattr("langchain_ollama.ChatOllama", lambda **_kwargs: _FakeLLM())

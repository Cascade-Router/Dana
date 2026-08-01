"""OS Execution Worker — isolates PowerShell / system commands from the MoA ReAct loop."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from dana.schema import ReactGraphState

logger = logging.getLogger(__name__)

OS_WORKER_NODE = "os_worker"
OS_WORKER_SYSTEM_PROMPT = (
    "You are the OS Execution Worker. Your ONLY job is to execute system "
    "commands to answer the user's query and return the exact output. Do not converse."
)

# Narrow system/file/code corridor for OS actuators (not vision / ticket / swarm).
_OS_WORKER_INTENT_RE = re.compile(
    r"("
    r"\b(?:terminal|shell|powershell|pwsh|cmd|command\s*prompt)\b|"
    r"\b(?:execute_powershell|shell_execute|run_terminal_command)\b|"
    r"\b(?:get-netadapter|get-process|get-service|get-childitem)\b|"
    r"\bnetwork\s+adapters?\b|"
    r"\b(?:run|use|execute|open)\s+(?:(?:a|the|my)\s+)?"
    r"(?:powershell|pwsh|cmd|command\s*prompt|terminal|shell)\b|"
    r"\b(?:system\s+command|os\s+command|subprocess)\b"
    r")",
    re.IGNORECASE,
)

_OS_TOOL_IDS = frozenset(
    {"execute_powershell", "shell_execute", "run_terminal_command"}
)

# Vision / ticket / swarm must stay on the default ReAct agent corridor.
_OS_WORKER_EXCLUDE_RE = re.compile(
    r"("
    r"\b(?:what\s+you\s+see|on\s+my\s+screen|describe\s+(?:the\s+)?screen)\b|"
    r"\b(?:analyze_visual_context|ocr_with_region)\b|"
    r"\b(?:draft_cursor_prompt|dispatch_research_swarm|dispatch_watchdog)\b|"
    r"\b(?:research\s+swarm|watchdog)\b"
    r")",
    re.IGNORECASE,
)

OsExecuteFn = Callable[[str], str]
OsLLMFactory = Callable[[], Any]


def _extract_user_text(state: ReactGraphState | dict[str, Any]) -> str:
    messages = state.get("messages") or []
    for msg in reversed(list(messages)):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if role in {"human", "user"} and isinstance(content, str) and content.strip():
            return content.split("\n\nVisual Context:", 1)[0].strip()
        if isinstance(msg, dict):
            if msg.get("role") == "user" and str(msg.get("content") or "").strip():
                return str(msg["content"]).split("\n\nVisual Context:", 1)[0].strip()
    return str(state.get("active_intent") or "").strip()


def _powershell_hint(text: str) -> bool:
    """Reuse broker PowerShell cue when available; keep a local fallback."""
    blob = text or ""
    try:
        from dana.tools.broker import _POWERSHELL_HINT_RE

        if bool(_POWERSHELL_HINT_RE.search(blob)):
            return True
    except Exception:  # noqa: BLE001
        pass
    return bool(_OS_WORKER_INTENT_RE.search(blob))


def _planned_os_tools_only(state: ReactGraphState | dict[str, Any]) -> bool:
    plan = state.get("execution_plan") or {}
    required = [
        str(t).strip()
        for t in (plan.get("required_tools") or [])
        if str(t).strip()
    ]
    always = [
        str(t).strip() for t in (state.get("always_include") or []) if str(t).strip()
    ]
    tools = required or always
    if not tools:
        return False
    return bool(set(tools) & _OS_TOOL_IDS) and set(tools) <= _OS_TOOL_IDS


def should_route_to_os_worker(state: ReactGraphState | dict[str, Any]) -> bool:
    """True when the turn should bypass MoA and use the OS worker.

    Routes on PowerShell hint regex, narrow system/file/code OS cues, or a
    planner bind limited to OS actuators. Vision / ticket / swarm stay on agent.
    """
    text = _extract_user_text(state)
    if not text.strip():
        return False
    if _OS_WORKER_EXCLUDE_RE.search(text):
        return False
    if _powershell_hint(text):
        return True
    if _OS_WORKER_INTENT_RE.search(text):
        return True
    return _planned_os_tools_only(state)


def route_after_executor(state: ReactGraphState | dict[str, Any]) -> str:
    """LangGraph conditional edge: ``os_worker`` or default ``agent``."""
    if should_route_to_os_worker(state):
        return OS_WORKER_NODE
    return "agent"


def _default_execute_powershell(command: str) -> str:
    from dana.tools.powershell import execute_powershell

    return execute_powershell(command)


def _heuristic_powershell_command(user_text: str) -> str:
    """Best-effort command when no LLM is available (offline / test fallback)."""
    raw = (user_text or "").strip()
    if not raw:
        return "Write-Output ''"

    # Prefer an explicit PowerShell cmdlet pasted in the utterance.
    m = re.search(
        r"((?:Get|Set|Write|Select|ConvertTo|Format)-[A-Za-z][\w-]*.*)$",
        raw,
        flags=re.I | re.M,
    )
    if m:
        return m.group(1).strip()

    if re.search(r"network\s+adapters?", raw, flags=re.I):
        return (
            "Get-NetAdapter | Select-Object Name, Status, MacAddress, LinkSpeed | "
            "Format-Table -AutoSize | Out-String"
        )

    # Strip common English wrappers; leftover may already be a script.
    stripped = re.sub(
        r"^(?:please\s+)?(?:use|run|execute)\s+(?:(?:a|the|my)\s+)?"
        r"(?:powershell|pwsh|cmd|terminal|shell)\s+(?:to\s+)?",
        "",
        raw,
        flags=re.I,
    ).strip()
    if stripped and stripped != raw and len(stripped) < 400:
        # Still English — wrap as Write-Output so the actuator returns something.
        if not re.search(r"^(?:Get|Set|Write|Select)-", stripped, flags=re.I):
            safe = stripped.replace("'", "''")[:200]
            return f"Write-Output '{safe}'"
        return stripped

    safe = raw.replace("'", "''")[:200]
    return f"Write-Output '{safe}'"


def _make_execute_powershell_tool(execute_fn: OsExecuteFn) -> Any:
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _Args(BaseModel):
        command: str = Field(..., description="PowerShell script text to execute")

    def _run(command: str) -> str:
        return execute_fn(command)

    _run.__name__ = "execute_powershell"
    return StructuredTool.from_function(
        func=_run,
        name="execute_powershell",
        description=(
            "Run a Windows PowerShell command (-NoProfile -NonInteractive) "
            "and return stdout/stderr/returncode."
        ),
        args_schema=_Args,
    )


def _iter_tool_calls(message: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tc in getattr(message, "tool_calls", None) or []:
        if isinstance(tc, dict):
            name = str(tc.get("name") or "").strip()
            args = dict(tc.get("args") or tc.get("arguments") or {})
            call_id = str(tc.get("id") or "").strip()
        else:
            name = str(getattr(tc, "name", "") or "").strip()
            args = dict(getattr(tc, "args", None) or {})
            call_id = str(getattr(tc, "id", "") or "").strip()
        if name:
            out.append({"name": name, "args": args, "id": call_id})
    return out


def _run_powershell_calls(
    tool_calls: list[dict[str, Any]],
    execute_fn: OsExecuteFn,
) -> str:
    chunks: list[str] = []
    for tc in tool_calls:
        name = tc.get("name") or ""
        if name != "execute_powershell":
            chunks.append(f"ERROR: OS worker refuses unbound tool `{name}`")
            continue
        args = tc.get("args") or {}
        command = str(args.get("command") or "").strip()
        chunks.append(execute_fn(command))
    return "\n".join(chunks).strip()


def make_os_worker_node(
    llm: Any | None = None,
    *,
    execute_powershell_fn: OsExecuteFn | None = None,
    llm_factory: OsLLMFactory | None = None,
    tools: list[Any] | None = None,
) -> Callable[[ReactGraphState | dict[str, Any]], dict[str, Any]]:
    """Build ``os_worker_node`` with injectable LLM / PowerShell actuator.

    Binds **only** ``execute_powershell`` — no vision, memory, or swarm tools.
    When the LLM is missing or fails, calls ``execute_powershell`` directly
    (offline-safe hermetic path for tests / Ollama-down).
    """

    ps_fn = execute_powershell_fn or _default_execute_powershell

    def os_worker_node(state: ReactGraphState | dict[str, Any]) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        try:
            from dana.ui.status_bus import emit_state_change

            emit_state_change("executing", tool="execute_powershell")
        except Exception:  # noqa: BLE001
            pass

        user_text = _extract_user_text(state)
        bound_tools = tools
        if bound_tools is None:
            bound_tools = [_make_execute_powershell_tool(ps_fn)]

        active_llm = llm
        if active_llm is None and llm_factory is not None:
            try:
                active_llm = llm_factory()
            except Exception as exc:  # noqa: BLE001
                logger.debug("os_worker llm_factory failed: %s", exc)
                active_llm = None

        if active_llm is not None:
            try:
                bound = active_llm.bind_tools(bound_tools, strict=True)
            except TypeError:
                bound = active_llm.bind_tools(bound_tools)
            except Exception as exc:  # noqa: BLE001
                logger.debug("os_worker bind_tools failed: %s", exc)
                bound = None

            if bound is not None:
                try:
                    messages = [
                        SystemMessage(content=OS_WORKER_SYSTEM_PROMPT),
                        HumanMessage(content=user_text or ""),
                    ]
                    # Prefer prior human turns if already present.
                    prior = list(state.get("messages") or [])
                    if prior:
                        messages = [SystemMessage(content=OS_WORKER_SYSTEM_PROMPT)]
                        messages.extend(prior)

                    ai_msg = bound.invoke(messages)
                    calls = _iter_tool_calls(ai_msg)
                    if calls:
                        obs = _run_powershell_calls(calls, ps_fn)
                        return {
                            "messages": [ai_msg, AIMessage(content=obs)],
                            "final_raw": obs,
                            "last_obs": obs,
                            "halt": True,
                            "current_agent": "OS_Worker",
                            "pending_synthesis": False,
                        }
                    content = str(getattr(ai_msg, "content", "") or "").strip()
                    # Worker must not converse — if no tool call, fall through offline.
                    if content:
                        logger.debug(
                            "os_worker LLM returned text without tool_calls; "
                            "using offline PowerShell fallback"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("os_worker LLM invoke failed: %s", exc)

        # Offline-safe fallback: call execute_powershell directly (no MoA / vision).
        command = _heuristic_powershell_command(user_text)
        obs = ps_fn(command)
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute_powershell",
                            "args": {"command": command},
                            "id": "os-worker-offline",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content=obs),
            ],
            "final_raw": obs,
            "last_obs": obs,
            "halt": True,
            "current_agent": "OS_Worker",
            "pending_synthesis": False,
        }

    return os_worker_node


os_worker_node = make_os_worker_node()


__all__ = (
    "OS_WORKER_NODE",
    "OS_WORKER_SYSTEM_PROMPT",
    "make_os_worker_node",
    "os_worker_node",
    "route_after_executor",
    "should_route_to_os_worker",
)

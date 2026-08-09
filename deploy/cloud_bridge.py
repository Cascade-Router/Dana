"""Stage 9.2 — Cloud / Hugging Face runtime bridge for Dānā LangGraph.

Applies cloud-safe environment defaults, mocks desktop/vision actuators, and
runs a single text turn through ``run_react_loop`` (same corridor as Stage 8.10
silent inject → LangGraph).
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

_LOCK = threading.Lock()
_BOOTSTRAPPED = False
_HISTORY: list[dict[str, str]] = []
_HISTORY_MAX = 12

# Tools that require a local desktop / webcam — never crash the Space.
_CLOUD_MOCK_TOOLS = frozenset(
    {
        "analyze_visual_context",
        "ocr_with_region",
        "navigate_and_click",
        "press_key",
        "type_stealth_text",
        "shell_execute",
        "execute_powershell",
        "python_repl",
        "file_editor",
        "dispatch_jason_supervisor",
        "evaluate_slide_and_type",
    }
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_cloud_mode() -> bool:
    return (os.environ.get("DANA_CLOUD") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def apply_cloud_mode() -> None:
    """Idempotent env + sys.path bootstrap for Hugging Face / headless hosts."""
    global _BOOTSTRAPPED
    with _LOCK:
        # Repo root on path (Space may launch from deploy/).
        root = str(project_root())
        if root not in sys.path:
            sys.path.insert(0, root)

        os.environ.setdefault("DANA_CLOUD", "1")
        os.environ.setdefault("DANA_HITL_AUTO_APPROVE", "1")
        os.environ.setdefault("DANA_HITL_REQUIRE_GUI", "0")
        os.environ.setdefault("DANA_OS_DRY_RUN", "1")
        os.environ.setdefault("DANA_GHOST_SKIP_HOTKEY", "1")
        # Avoid interactive vault prompts on Spaces.
        if not (os.environ.get("DANA_VAULT_KEY") or "").strip():
            os.environ.setdefault("DANA_VAULT_KEY", "hf-space-ephemeral")

        # Prefer OpenAI-compatible cloud LLM when keys are present.
        if (os.environ.get("OPENAI_API_KEY") or "").strip() or (
            os.environ.get("GROQ_API_KEY") or ""
        ).strip():
            os.environ.setdefault("DANA_CASCADE_EXTERNAL", "1")
            os.environ.setdefault("DANA_HF_CLOUD_LLM", "1")

        if _BOOTSTRAPPED:
            return
        _BOOTSTRAPPED = True

        try:
            from dana.agentic import set_dana_mode
            from dana.core_agent import set_engine_engaged

            set_dana_mode("developer", as_voice=False)
            set_engine_engaged(True)
        except Exception as exc:  # noqa: BLE001
            print(f"[cloud_bridge] WARNING: mode/engine bootstrap: {exc}", flush=True)


def use_cloud_llm() -> bool:
    flag = (os.environ.get("DANA_HF_CLOUD_LLM") or "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool(
        (os.environ.get("OPENAI_API_KEY") or "").strip()
        or (os.environ.get("GROQ_API_KEY") or "").strip()
        or (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        or (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    )


def _build_cloud_chat_model(*, temperature: float = 0.2) -> Any:
    """OpenAI-compatible Chat model for Spaces without local Ollama."""
    from langchain_openai import ChatOpenAI

    model = (
        (os.environ.get("DANA_CLOUD_MODEL") or "").strip()
        or (os.environ.get("DANA_CASCADE_MODEL") or "").strip()
        or "gpt-4o-mini"
    )
    kwargs: dict[str, Any] = {"model": model, "temperature": temperature}

    base = (
        (os.environ.get("OPENAI_BASE_URL") or "").strip()
        or (os.environ.get("DANA_OPENAI_BASE_URL") or "").strip()
    )
    # Groq OpenAI-compatible endpoint when only GROQ_API_KEY is set.
    if not base and (os.environ.get("GROQ_API_KEY") or "").strip():
        base = "https://api.groq.com/openai/v1"
        if not (os.environ.get("OPENAI_API_KEY") or "").strip():
            os.environ["OPENAI_API_KEY"] = os.environ["GROQ_API_KEY"]
        if model == "gpt-4o-mini":
            kwargs["model"] = (
                (os.environ.get("DANA_CLOUD_MODEL") or "").strip()
                or "llama-3.3-70b-versatile"
            )
    # DeepSeek OpenAI-compatible endpoint.
    if not base and (os.environ.get("DEEPSEEK_API_KEY") or "").strip():
        base = "https://api.deepseek.com/v1"
        if not (os.environ.get("OPENAI_API_KEY") or "").strip():
            os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
        if model == "gpt-4o-mini":
            kwargs["model"] = (
                (os.environ.get("DANA_CLOUD_MODEL") or "").strip() or "deepseek-chat"
            )

    if base:
        kwargs["base_url"] = base
    return ChatOpenAI(**kwargs)


def _install_cloud_llm_patches() -> Callable[[], None]:
    """Patch cascade + Ollama preflight for cloud LLM. Returns restore fn."""
    import dana.agentic as ag
    import dana.cascade_router as cr

    orig_resolve = cr.resolve_chat_model
    orig_reachable = ag.ollama_service_reachable

    def _resolve_cloud(query: str = "", **kwargs: Any) -> Any:  # noqa: ARG001
        temp = float(kwargs.get("temperature") or 0.2)
        return _build_cloud_chat_model(temperature=temp)

    def _reachable_cloud(*_a: Any, **_k: Any) -> bool:
        return True

    cr.resolve_chat_model = _resolve_cloud  # type: ignore[assignment]
    ag.ollama_service_reachable = _reachable_cloud  # type: ignore[assignment]

    def _restore() -> None:
        cr.resolve_chat_model = orig_resolve  # type: ignore[assignment]
        ag.ollama_service_reachable = orig_reachable  # type: ignore[assignment]

    return _restore


def _cloud_execute(tool_call: Any) -> str:
    """Execute tools with desktop/vision mocked; ledger write still allowed."""
    tid = str(getattr(tool_call, "tool_id", "") or "").strip()
    args = dict(getattr(tool_call, "arguments", None) or {})

    if tid in _CLOUD_MOCK_TOOLS:
        return (
            f"[CLOUD] Tool `{tid}` is mocked on Hugging Face "
            f"(no local desktop/webcam). Args={args!r}"
        )

    if tid == "draft_cursor_prompt":
        try:
            from dana.tools.general.draft_cursor_prompt import draft_cursor_prompt

            return str(
                draft_cursor_prompt(
                    objective=str(args.get("objective") or ""),
                    context=str(args.get("context") or ""),
                )
            )
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: draft_cursor_prompt failed: {type(exc).__name__}: {exc}"

    # Prefer production tool executor when importable; else soft stub.
    try:
        from dana.core_agent import execute_tool_call

        return str(execute_tool_call(tool_call))
    except Exception as exc:  # noqa: BLE001
        return f"[CLOUD] execute `{tid}` skipped ({type(exc).__name__}: {exc})"


def _system_prompt(user_text: str) -> str:
    try:
        from dana.core_agent import build_dana_system_prompt

        return build_dana_system_prompt([], user_text=user_text)
    except Exception:  # noqa: BLE001
        return (
            "You are Dānā, a local-first cybernetic copilot running in "
            "cloud/Hugging Face mode. Prefer tools when asked. Vision and OS "
            "actuators are mocked — explain limitations briefly when hit."
        )


def run_text_command(message: str, *, history: list | None = None) -> str:
    """Inject ``message`` into LangGraph (Stage 8.10 parity) and return final text.

    ``history`` is accepted for Gradio ChatInterface compatibility but the Astro
    REST client sends a single prompt; we keep a small process-local prior window.
    """
    apply_cloud_mode()
    text = (message or "").strip()
    if not text:
        return "Type a command for Dānā."

    # Optional Gradio chat tuples → ignore; Astro uses Interface text→text.
    _ = history

    restore: Callable[[], None] | None = None
    try:
        from dana.agentic import run_react_loop
        from dana.tools.broker import IntentBroker, get_broker

        if use_cloud_llm():
            restore = _install_cloud_llm_patches()

        broker = get_broker()
        forced = None
        try:
            forced = IntentBroker().parse_utterance(text)
        except Exception:  # noqa: BLE001
            forced = None

        with _LOCK:
            prior = list(_HISTORY[-_HISTORY_MAX:])

        result = run_react_loop(
            user_text=text,
            system_prompt=_system_prompt(text),
            execute_fn=_cloud_execute,
            max_iters=6,
            broker=broker,
            enable_reflection=False,
            prior_messages=prior,
            on_tool_start=None,
            visual_context="[CLOUD] No local vision frame.",
            model=os.environ.get("DANA_LOCAL_MODEL") or "qwen2.5-coder:7b",
            forced_tool=forced,
            tts_callback=None,
        )
        answer = str(getattr(result, "final_text", "") or "").strip()
        if not answer:
            answer = "Dānā returned an empty response."

        with _LOCK:
            _HISTORY.append({"role": "user", "content": text})
            _HISTORY.append({"role": "assistant", "content": answer})
            if len(_HISTORY) > _HISTORY_MAX * 2:
                del _HISTORY[: len(_HISTORY) - _HISTORY_MAX * 2]

        return answer
    except Exception as exc:  # noqa: BLE001
        return (
            f"Dānā cloud error: {type(exc).__name__}: {exc}. "
            "If this is a cold boot, wait a moment and retry. "
            "Ensure Space Secrets include a LLM API key "
            "(OPENAI_API_KEY / GROQ_API_KEY / DEEPSEEK_API_KEY) "
            "or a reachable OLLAMA_URL."
        )
    finally:
        if restore is not None:
            try:
                restore()
            except Exception:  # noqa: BLE001
                pass


def clear_history() -> None:
    with _LOCK:
        _HISTORY.clear()

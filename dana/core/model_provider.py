"""Hybrid ModelProvider — local Ollama with optional encrypted cloud fallback."""

from __future__ import annotations

import os
from typing import Any, Literal

from dana.core.openai_tool_bridge import build_multimodal_messages, complete_openai_with_tools
from dana.tools.schema import openai_tool_calls_to_ir

ProviderKind = Literal["local", "cloud", "auto"]

# Providers whose tool-calling / vision schema is not OpenAI-wire-compatible.
# Gemini and Anthropic each use their own function-calling and image payload
# shapes; bridging them is out of scope for the OpenAI tool-calling bridge.
_NON_OPENAI_SCHEMA_PROVIDERS = frozenset({"gemini", "google", "anthropic"})

_DEFAULT_LOCAL_MODEL = "qwen2.5-coder:7b"
_COMPLEXITY_REJECT = "REJECT: Task too complex for local model"


def ensure_dotenv_loaded() -> None:
    try:
        from dana.graph.cloud_planner import ensure_dotenv_loaded as _load

        _load()
    except Exception:  # noqa: BLE001
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:  # noqa: BLE001
            pass


def cloud_fallback_enabled() -> bool:
    ensure_dotenv_loaded()
    raw = (os.environ.get("DANA_ALLOW_CLOUD_FALLBACK") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def force_local() -> bool:
    return (os.environ.get("DANA_FORCE_LOCAL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def local_model_name() -> str:
    ensure_dotenv_loaded()
    return (
        (os.environ.get("DANA_LOCAL_MODEL") or "").strip()
        or (os.environ.get("OLLAMA_MODEL") or "").strip()
        or _DEFAULT_LOCAL_MODEL
    )


def cloud_provider_name() -> str:
    ensure_dotenv_loaded()
    return (
        (os.environ.get("DANA_CLOUD_PROVIDER") or "").strip().lower()
        or "gemini"
    )


def complexity_reject_marker() -> str:
    return _COMPLEXITY_REJECT


def is_complexity_reject(text: str) -> bool:
    s = str(text or "").strip()
    if not s.upper().startswith("REJECT:"):
        return False
    low = s.lower()
    return (
        "too complex for local" in low
        or "task too complex" in low
        or s.startswith(_COMPLEXITY_REJECT)
    )


class ModelProvider:
    """Unified chat completion for Spec Compiler / Meta-Broker planning."""

    def __init__(
        self,
        *,
        local_model: str | None = None,
        prefer: ProviderKind = "auto",
    ) -> None:
        self.local_model = (local_model or local_model_name()).strip()
        self.prefer = prefer
        self.last_provider: str = "none"
        self.last_error: str = ""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int = 512,
        temperature: float = 0.1,
        allow_cloud: bool | None = None,
        response_mime_type: str | None = "text/plain",
    ) -> str:
        """Run a chat completion; optionally fall back to cloud."""
        allow = cloud_fallback_enabled() if allow_cloud is None else bool(allow_cloud)
        if force_local():
            allow = False

        use_cloud_first = self.prefer == "cloud" and allow
        if use_cloud_first:
            try:
                return self._complete_cloud(
                    messages,
                    num_predict=num_predict,
                    temperature=temperature,
                    response_mime_type=response_mime_type,
                )
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"cloud: {exc}"
                # Fall through to local.

        try:
            text = self._complete_local(
                messages,
                num_predict=num_predict,
                temperature=temperature,
            )
            self.last_provider = "local"
            return text
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"local: {exc}"
            if not allow:
                raise
            return self._complete_cloud(
                messages,
                num_predict=num_predict,
                temperature=temperature,
                response_mime_type=response_mime_type,
            )

    def complete_with_complexity_fallback(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Local first; if response is a complexity REJECT and cloud fallback
        is enabled, re-issue the request on the cloud provider.
        """
        local_out = self.complete(
            messages,
            num_predict=num_predict,
            temperature=temperature,
            allow_cloud=False,
        )
        if not is_complexity_reject(local_out):
            return local_out
        if not cloud_fallback_enabled() or force_local():
            return local_out
        print(
            "[ModelProvider] local complexity REJECT → cloud fallback",
            flush=True,
        )
        cloud_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "The local model rejected this as too complex. "
                    "Produce a precise /broker multi-epic specification now. "
                    "Do not REJECT unless truly impossible with stdlib + MCP tools."
                ),
            }
        ]
        try:
            out = self._complete_cloud(
                cloud_messages,
                num_predict=max(num_predict, 768),
                temperature=temperature,
                response_mime_type="text/plain",
            )
            return out
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"cloud_fallback: {exc}"
            return local_out

    def _complete_local(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int,
        temperature: float,
    ) -> str:
        from dana.core.agent_loop import ask_ollama_messages
        from dana.system_health import llm_lock

        with llm_lock:
            raw = ask_ollama_messages(
                messages,
                model=self.local_model,
                num_predict=num_predict,
                temperature=temperature,
            )
        self.last_provider = "local"
        return str(raw or "").strip()

    def _complete_cloud(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int,
        temperature: float,
        response_mime_type: str | None,
    ) -> str:
        provider = cloud_provider_name()
        if provider in {"gemini", "google"}:
            from dana.graph.cloud_planner import ask_gemini_text

            text = ask_gemini_text(
                messages,
                temperature=temperature,
                max_output_tokens=int(num_predict),
                response_mime_type=response_mime_type,
            )
            self.last_provider = "cloud:gemini"
            return str(text or "").strip()

        # OpenAI-compatible path (OpenAI / Groq / Anthropic via compatible gateway).
        return self._complete_openai_compatible(
            messages,
            num_predict=num_predict,
            temperature=temperature,
            provider=provider,
        )

    def _resolve_openai_endpoint(self, provider: str) -> tuple[str, str, str]:
        """Return ``(api_key, base_url, model)`` for an OpenAI-wire-compatible provider.

        Shared by the plain-text ``_complete_openai_compatible`` path and the
        tool-calling / vision bridge below — one place that knows how each
        provider's env vars map to a key/base/model triple. ``"ollama"``
        targets the local Ollama daemon's own OpenAI-compatible
        ``/v1/chat/completions`` surface (distinct from the native
        ``/api/chat`` path used by ``_complete_local``), which needs no real
        API key.
        """
        ensure_dotenv_loaded()
        if provider == "groq":
            key = (os.environ.get("GROQ_API_KEY") or "").strip()
            base = (
                (os.environ.get("GROQ_API_BASE") or "").strip()
                or "https://api.groq.com/openai/v1"
            )
            model = (
                (os.environ.get("DANA_GROQ_MODEL") or "").strip()
                or "llama-3.3-70b-versatile"
            )
        elif provider == "ollama":
            key = (os.environ.get("OLLAMA_API_KEY") or "").strip() or "ollama"
            base = (
                (os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
                + "/v1"
            )
            model = (os.environ.get("DANA_OPENAI_TOOLS_MODEL") or "").strip() or self.local_model
        else:
            key = (os.environ.get("OPENAI_API_KEY") or "").strip()
            base = (
                (os.environ.get("OPENAI_API_BASE") or "").strip()
                or "https://api.openai.com/v1"
            )
            model = (
                (os.environ.get("DANA_OPENAI_MODEL") or "").strip()
                or "gpt-4o-mini"
            )
        if not key:
            raise RuntimeError(f"No API key configured for cloud provider={provider}")
        return key, base, model

    def _complete_openai_compatible(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int,
        temperature: float,
        provider: str,
    ) -> str:
        ensure_dotenv_loaded()
        if provider == "anthropic":
            # Prefer Anthropic Messages API via env-compatible OpenAI proxy if set;
            # otherwise use raw Anthropic endpoint.
            key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not configured")
            return self._complete_anthropic(
                messages,
                num_predict=num_predict,
                temperature=temperature,
                api_key=key,
            )

        key, base, model = self._resolve_openai_endpoint(provider)
        raw = complete_openai_with_tools(
            messages,
            api_key=key,
            base_url=base,
            model=model,
            num_predict=num_predict,
            temperature=temperature,
        )
        self.last_provider = f"cloud:{provider}"
        return str(raw.get("content") or "").strip()

    def complete_with_tool_calls(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        provider: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        num_predict: int = 1024,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """OpenAI-schema tool-calling turn against a cloud or local-Ollama endpoint.

        Returns ``{"content": str, "tool_calls": list[ToolCall], "provider": str}``
        — ``tool_calls`` is already Dana's native IR (see
        ``dana.tools.schema.openai_tool_calls_to_ir``), so callers can hand
        results straight to the existing broker/dispatch path
        (``dana.core.agent_loop.execute_tool_call``) with no OpenAI-shape
        parsing of their own. Raises ``NotImplementedError`` for providers
        that don't speak the OpenAI tools schema (Gemini, Anthropic).
        """
        resolved_provider = (provider or cloud_provider_name()).strip().lower()
        if resolved_provider in _NON_OPENAI_SCHEMA_PROVIDERS:
            raise NotImplementedError(
                f"OpenAI tool-calling bridge does not support provider={resolved_provider!r} "
                "(uses a non-OpenAI tool schema)"
            )
        key, base, model = self._resolve_openai_endpoint(resolved_provider)
        raw = complete_openai_with_tools(
            messages,
            api_key=key,
            base_url=base,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            num_predict=num_predict,
            temperature=temperature,
        )
        self.last_provider = f"cloud:{resolved_provider}"
        return {
            "content": str(raw.get("content") or "").strip(),
            "tool_calls": openai_tool_calls_to_ir(
                raw.get("tool_calls"), raw_text=str(raw.get("content") or "")
            ),
            "provider": self.last_provider,
        }

    def complete_vision(
        self,
        prompt: str,
        image_b64: str,
        *,
        mime_type: str = "image/png",
        provider: str | None = None,
        num_predict: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Describe/analyze ``image_b64`` via an OpenAI-vision-compatible model.

        ``provider="ollama"`` (the default when cloud fallback is off) hits
        the local Ollama VLM (e.g. Qwen2.5-VL) over its OpenAI-compatible
        surface at zero cost/egress; any other resolved provider goes to the
        matching cloud OpenAI-wire endpoint (GPT-4o-class on OpenAI/Groq).
        Raises ``NotImplementedError`` for Gemini/Anthropic, whose image
        payload shapes are not OpenAI-compatible.
        """
        resolved_provider = (provider or cloud_provider_name()).strip().lower()
        if resolved_provider in _NON_OPENAI_SCHEMA_PROVIDERS:
            raise NotImplementedError(
                f"complete_vision does not support provider={resolved_provider!r} "
                "(uses a non-OpenAI image payload schema)"
            )
        key, base, model = self._resolve_openai_endpoint(resolved_provider)
        messages = build_multimodal_messages(prompt, image_b64=image_b64, mime_type=mime_type)
        raw = complete_openai_with_tools(
            messages,
            api_key=key,
            base_url=base,
            model=model,
            num_predict=num_predict,
            temperature=temperature,
        )
        self.last_provider = f"cloud:{resolved_provider}"
        return str(raw.get("content") or "").strip()

    def _complete_anthropic(
        self,
        messages: list[dict[str, str]],
        *,
        num_predict: int,
        temperature: float,
        api_key: str,
    ) -> str:
        import json
        import urllib.request

        system = ""
        converted: list[dict[str, Any]] = []
        for m in messages:
            role = str(m.get("role") or "user")
            content = str(m.get("content") or "")
            if role == "system":
                system = (system + "\n" + content).strip()
                continue
            converted.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": content,
                }
            )
        model = (
            (os.environ.get("DANA_ANTHROPIC_MODEL") or "").strip()
            or "claude-3-5-haiku-latest"
        )
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": int(num_predict),
            "temperature": float(temperature),
            "messages": converted or [{"role": "user", "content": "Hello"}],
        }
        if system:
            payload["system"] = system
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        parts = body.get("content") or []
        texts = [
            str(p.get("text") or "")
            for p in parts
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        self.last_provider = "cloud:anthropic"
        return "\n".join(texts).strip()


def get_default_provider() -> ModelProvider:
    return ModelProvider()


__all__ = (
    "ModelProvider",
    "cloud_fallback_enabled",
    "complexity_reject_marker",
    "force_local",
    "get_default_provider",
    "is_complexity_reject",
    "local_model_name",
)

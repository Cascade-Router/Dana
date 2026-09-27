"""Model Registry — transparent, user-observable catalog of every LLM Dana
knows about, across every provider, with live active/inactive detection.

Distinct from ``dana.core.llm_router``/``routing_config.yaml``: that system
is an opt-in runtime fallback chain scoped to "what to try, in order, for one
tool-calling turn". This module is the full catalog behind the frontend's
Model Registry control panel — every known model, whether or not it's
currently usable, and *why* (a detected key, a missing one, an un-pulled
Ollama model, ...). Nothing here changes which model a turn actually uses.

Data lives in ``dana/data/models_registry.json`` (see ``dana.paths.
MODELS_REGISTRY_PATH``), refreshed by ``scripts/sync_model_catalog.py``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from dana.paths import MODEL_PREFERENCES_PATH, MODELS_REGISTRY_PATH

_OLLAMA_TAGS_TIMEOUT_S = 2.0


class ProviderType(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class PricingTier:
    prompt_per_1m: float = 0.0
    completion_per_1m: float = 0.0


@dataclass(frozen=True)
class RateLimitConfig:
    tpm: int | None = None
    rpm: int | None = None
    rpd: int | None = None


@dataclass
class ModelMetadata:
    id: str
    name: str
    context_window: int
    max_output_tokens: int
    pricing: PricingTier
    rate_limits: RateLimitConfig
    modalities: list[str]
    supports_tool_calling: bool
    supports_thought_signature: bool
    is_deprecated: bool
    # Runtime status — computed by ModelRegistryService, not part of the catalog file.
    is_active: bool = False
    status_reason: str = "not evaluated"
    # Circuit-breaker overlay — set when llm_router.py reported a live 429/404/
    # 503/etc. for this model since it was last cleared (see ModelRegistryService.
    # report_runtime_error/clear_runtime_error). Independent of is_active: a key
    # being present is still true, so is_active is left alone — this is "the
    # credentials work, but the last real call to this model failed".
    has_runtime_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "pricing": asdict(self.pricing),
            "rate_limits": asdict(self.rate_limits),
            "modalities": list(self.modalities),
            "supports_tool_calling": self.supports_tool_calling,
            "supports_thought_signature": self.supports_thought_signature,
            "is_deprecated": self.is_deprecated,
            "is_active": self.is_active,
            "status_reason": self.status_reason,
            "has_runtime_error": self.has_runtime_error,
        }


@dataclass
class ProviderDomain:
    provider: ProviderType
    requires_api_key: bool
    api_key_env: str | None
    models: list[ModelMetadata] = field(default_factory=list)
    is_available: bool = False
    status_reason: str = "not evaluated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.value,
            "requires_api_key": self.requires_api_key,
            "api_key_env": self.api_key_env,
            "is_available": self.is_available,
            "status_reason": self.status_reason,
            "models": [m.to_dict() for m in self.models],
        }


def _model_from_json(raw: dict[str, Any]) -> ModelMetadata:
    pricing_raw = raw.get("pricing") or {}
    limits_raw = raw.get("rate_limits") or {}
    return ModelMetadata(
        id=str(raw["id"]),
        name=str(raw.get("name") or raw["id"]),
        context_window=int(raw.get("context_window") or 0),
        max_output_tokens=int(raw.get("max_output_tokens") or 0),
        pricing=PricingTier(
            prompt_per_1m=float(pricing_raw.get("prompt_per_1m") or 0.0),
            completion_per_1m=float(pricing_raw.get("completion_per_1m") or 0.0),
        ),
        rate_limits=RateLimitConfig(
            tpm=limits_raw.get("tpm"),
            rpm=limits_raw.get("rpm"),
            rpd=limits_raw.get("rpd"),
        ),
        modalities=list(raw.get("modalities") or ["text"]),
        supports_tool_calling=bool(raw.get("supports_tool_calling", False)),
        supports_thought_signature=bool(raw.get("supports_thought_signature", False)),
        is_deprecated=bool(raw.get("is_deprecated", False)),
    )


class ModelRegistryService:
    """Loads the catalog, cross-references it against the live environment
    (API keys, downloaded Ollama models) and the user's saved preferences,
    and produces the matrix the control panel renders.

    Every provider recognized here reuses the SAME env var names
    ``dana/api/system.py``'s ``_SENSITIVE_VARS`` allowlist already exposes to
    the Environment Viewer, so "key detected" here and "configured" there
    never disagree.
    """

    _API_KEY_ENV_BY_PROVIDER: dict[ProviderType, str] = {
        ProviderType.OPENAI: "OPENAI_API_KEY",
        ProviderType.GEMINI: "GEMINI_API_KEY",
        ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
        ProviderType.GROQ: "GROQ_API_KEY",
        ProviderType.DEEPSEEK: "DEEPSEEK_API_KEY",
    }

    def __init__(
        self,
        catalog_path: Any = MODELS_REGISTRY_PATH,
        preferences_path: Any = MODEL_PREFERENCES_PATH,
    ) -> None:
        self._catalog_path = catalog_path
        self._preferences_path = preferences_path
        # Circuit-breaker state: model_id -> {"timestamp", "status_code", "message"}.
        # In-memory only (a runtime error is about THIS process's live session,
        # not a fact worth persisting across restarts) — guarded by a lock since
        # llm_router.py reports into it from whichever thread is running a turn
        # while the API's GET/POST handlers read it from the request thread.
        self._runtime_errors_lock = threading.Lock()
        self.runtime_errors: dict[str, dict[str, Any]] = {}

    # -- circuit breaker (runtime errors) -----------------------------------

    def report_runtime_error(self, model_id: str, status_code: int, message: str) -> None:
        with self._runtime_errors_lock:
            self.runtime_errors[model_id] = {
                "timestamp": time.time(),
                "status_code": status_code,
                "message": message,
            }

    def clear_runtime_error(self, model_id: str) -> None:
        with self._runtime_errors_lock:
            self.runtime_errors.pop(model_id, None)

    # -- catalog -----------------------------------------------------------

    def _load_catalog(self) -> dict[ProviderType, ProviderDomain]:
        raw = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        domains: dict[ProviderType, ProviderDomain] = {}
        for provider_id, provider_raw in (raw.get("providers") or {}).items():
            try:
                provider = ProviderType(provider_id)
            except ValueError:
                continue
            domains[provider] = ProviderDomain(
                provider=provider,
                requires_api_key=bool(provider_raw.get("requires_api_key", True)),
                api_key_env=provider_raw.get("api_key_env"),
                models=[_model_from_json(m) for m in provider_raw.get("models") or []],
            )
        return domains

    # -- preferences ---------------------------------------------------------

    def load_preferences(self) -> dict[str, Any]:
        """``{"order": [model_id, ...], "disabled": [model_id, ...]}`` — missing
        file or unreadable content is a supported "use catalog defaults" state,
        never an error."""
        if not self._preferences_path.exists():
            return {"order": [], "disabled": []}
        try:
            data = json.loads(self._preferences_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"order": [], "disabled": []}
        return {
            "order": list(data.get("order") or []),
            "disabled": list(data.get("disabled") or []),
        }

    def save_preferences(self, order: list[str], disabled: list[str]) -> dict[str, Any]:
        prefs = {"order": list(order), "disabled": list(disabled)}
        self._preferences_path.parent.mkdir(parents=True, exist_ok=True)
        self._preferences_path.write_text(json.dumps(prefs, indent=2) + "\n", encoding="utf-8")
        return prefs

    # -- live environment probes --------------------------------------------

    def _has_key(self, env_var: str | None) -> bool:
        if not env_var:
            return False
        return bool((os.environ.get(env_var) or "").strip())

    def _query_ollama_tags(self) -> tuple[set[str], str | None]:
        """Returns ``(downloaded_model_names, error)`` — ``error`` is set (and
        the set is empty) when Ollama isn't reachable, which is a normal,
        expected state (Ollama isn't required to run Dana), never raised."""
        base = (os.environ.get("OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
        try:
            with urllib.request.urlopen(f"{base}/api/tags", timeout=_OLLAMA_TAGS_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            return set(), f"Ollama not reachable at {base} ({exc})"
        names = {str(m.get("name")) for m in data.get("models") or [] if m.get("name")}
        return names, None

    # -- matrix --------------------------------------------------------------

    def get_matrix(self) -> dict[str, Any]:
        domains = self._load_catalog()
        prefs = self.load_preferences()
        disabled = set(prefs["disabled"])
        order = prefs["order"]
        with self._runtime_errors_lock:
            runtime_errors = dict(self.runtime_errors)

        ollama_tags, ollama_error = (set(), None)
        if ProviderType.OLLAMA in domains:
            ollama_tags, ollama_error = self._query_ollama_tags()

        for provider, domain in domains.items():
            if provider is ProviderType.OLLAMA:
                domain.is_available = ollama_error is None
                domain.status_reason = ollama_error or "Ollama reachable"
            else:
                has_key = self._has_key(domain.api_key_env)
                domain.is_available = has_key
                domain.status_reason = "Key detected" if has_key else f"Missing {domain.api_key_env}"

            for model in domain.models:
                user_disabled = model.id in disabled
                if model.is_deprecated:
                    model.is_active = False
                    model.status_reason = "Deprecated"
                elif provider is ProviderType.OLLAMA:
                    pulled = model.id in ollama_tags
                    if ollama_error:
                        model.is_active = False
                        model.status_reason = ollama_error
                    elif not pulled:
                        model.is_active = False
                        model.status_reason = "Model not pulled in Ollama"
                    elif user_disabled:
                        model.is_active = False
                        model.status_reason = "Disabled by user"
                    else:
                        model.is_active = True
                        model.status_reason = "Downloaded"
                elif not domain.is_available:
                    model.is_active = False
                    model.status_reason = domain.status_reason
                elif user_disabled:
                    model.is_active = False
                    model.status_reason = "Disabled by user"
                else:
                    model.is_active = True
                    model.status_reason = "Key detected"

                # Circuit-breaker overlay: a live runtime failure always wins
                # the DISPLAYED reason (so the HITL sees the exact 429/404/503
                # that just happened, not the generic static check) without
                # touching is_active — the key still exists, this is "it's
                # configured but the last real call to it failed".
                error = runtime_errors.get(model.id)
                if error is not None:
                    model.has_runtime_error = True
                    model.status_reason = f"Runtime Error {error['status_code']}: {error['message']}"
                else:
                    model.has_runtime_error = False

            if order:
                rank = {model_id: i for i, model_id in enumerate(order)}
                domain.models.sort(key=lambda m: rank.get(m.id, len(order)))

        return {
            "providers": [domains[p].to_dict() for p in ProviderType if p in domains],
            "preferences": prefs,
        }


# Process-wide default instance — shared by dana.api.models (reads the matrix,
# handles the clear-error endpoint) and dana.core.llm_router (reports fleet
# failures into it) so both sides observe the SAME circuit-breaker state
# rather than each holding an independent, disconnected ModelRegistryService.
_default_service: ModelRegistryService | None = None


def get_registry_service() -> ModelRegistryService:
    global _default_service
    if _default_service is None:
        _default_service = ModelRegistryService()
    return _default_service


__all__ = (
    "ProviderType",
    "PricingTier",
    "RateLimitConfig",
    "ModelMetadata",
    "ProviderDomain",
    "ModelRegistryService",
    "get_registry_service",
)

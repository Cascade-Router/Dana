"""Model Registry tests (hermetic, no real network): active/inactive
detection by env key presence, Ollama /api/tags cross-referencing, and
preference save/load round-tripping — a temp catalog file + temp
preferences file per test, same "patch the path" convention
tests/test_llm_router.py already uses for routing_config.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dana.core.model_registry import ModelRegistryService

_CATALOG: dict = {
    "version": 1,
    "providers": {
        "deepseek": {
            "requires_api_key": True,
            "api_key_env": "DEEPSEEK_API_KEY",
            "models": [
                {
                    "id": "deepseek-chat",
                    "name": "DeepSeek Chat",
                    "context_window": 64000,
                    "max_output_tokens": 8192,
                    "pricing": {"prompt_per_1m": 0.28, "completion_per_1m": 0.42},
                    "rate_limits": {"tpm": None, "rpm": None, "rpd": None},
                    "modalities": ["text"],
                    "supports_tool_calling": True,
                    "supports_thought_signature": False,
                    "is_deprecated": False,
                },
                {
                    "id": "deepseek-old",
                    "name": "DeepSeek Old",
                    "context_window": 8000,
                    "max_output_tokens": 4096,
                    "pricing": {"prompt_per_1m": 0.0, "completion_per_1m": 0.0},
                    "rate_limits": {"tpm": None, "rpm": None, "rpd": None},
                    "modalities": ["text"],
                    "supports_tool_calling": False,
                    "supports_thought_signature": False,
                    "is_deprecated": True,
                },
            ],
        },
        "ollama": {
            "requires_api_key": False,
            "api_key_env": None,
            "models": [
                {
                    "id": "qwen2.5-coder:14b",
                    "name": "Qwen 2.5 Coder 14B",
                    "context_window": 32768,
                    "max_output_tokens": 8192,
                    "pricing": {"prompt_per_1m": 0.0, "completion_per_1m": 0.0},
                    "rate_limits": {"tpm": None, "rpm": None, "rpd": None},
                    "modalities": ["text"],
                    "supports_tool_calling": True,
                    "supports_thought_signature": False,
                    "is_deprecated": False,
                },
                {
                    "id": "llama3.1:8b",
                    "name": "Llama 3.1 8B",
                    "context_window": 131072,
                    "max_output_tokens": 8192,
                    "pricing": {"prompt_per_1m": 0.0, "completion_per_1m": 0.0},
                    "rate_limits": {"tpm": None, "rpm": None, "rpd": None},
                    "modalities": ["text"],
                    "supports_tool_calling": True,
                    "supports_thought_signature": False,
                    "is_deprecated": False,
                },
            ],
        },
    },
}


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    path = tmp_path / "models_registry.json"
    path.write_text(json.dumps(_CATALOG), encoding="utf-8")
    return path


@pytest.fixture
def prefs_path(tmp_path: Path) -> Path:
    return tmp_path / "model_preferences.json"


def _service(catalog_path: Path, prefs_path: Path) -> ModelRegistryService:
    return ModelRegistryService(catalog_path=catalog_path, preferences_path=prefs_path)


def _find(matrix: dict, provider: str, model_id: str) -> dict:
    domain = next(p for p in matrix["providers"] if p["provider"] == provider)
    return next(m for m in domain["models"] if m["id"] == model_id)


def test_missing_api_key_marks_models_inactive(
    monkeypatch: pytest.MonkeyPatch, catalog_path: Path, prefs_path: Path
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        ModelRegistryService, "_query_ollama_tags", lambda self: (set(), "Ollama not reachable (test)")
    )
    matrix = _service(catalog_path, prefs_path).get_matrix()

    deepseek_domain = next(p for p in matrix["providers"] if p["provider"] == "deepseek")
    assert deepseek_domain["is_available"] is False
    assert "DEEPSEEK_API_KEY" in deepseek_domain["status_reason"]

    model = _find(matrix, "deepseek", "deepseek-chat")
    assert model["is_active"] is False


def test_present_api_key_marks_non_deprecated_models_active(
    monkeypatch: pytest.MonkeyPatch, catalog_path: Path, prefs_path: Path
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setattr(
        ModelRegistryService, "_query_ollama_tags", lambda self: (set(), "Ollama not reachable (test)")
    )
    matrix = _service(catalog_path, prefs_path).get_matrix()

    active_model = _find(matrix, "deepseek", "deepseek-chat")
    assert active_model["is_active"] is True
    assert active_model["status_reason"] == "Key detected"

    deprecated_model = _find(matrix, "deepseek", "deepseek-old")
    assert deprecated_model["is_active"] is False
    assert deprecated_model["status_reason"] == "Deprecated"


def test_ollama_active_only_for_downloaded_tags(catalog_path: Path, prefs_path: Path) -> None:
    service = _service(catalog_path, prefs_path)
    service._query_ollama_tags = lambda: ({"qwen2.5-coder:14b"}, None)  # type: ignore[method-assign]
    matrix = service.get_matrix()

    pulled = _find(matrix, "ollama", "qwen2.5-coder:14b")
    assert pulled["is_active"] is True
    assert pulled["status_reason"] == "Downloaded"

    not_pulled = _find(matrix, "ollama", "llama3.1:8b")
    assert not_pulled["is_active"] is False
    assert not_pulled["status_reason"] == "Model not pulled in Ollama"


def test_save_and_load_preferences_round_trip(catalog_path: Path, prefs_path: Path) -> None:
    service = _service(catalog_path, prefs_path)
    service.save_preferences(order=["deepseek-chat", "deepseek-old"], disabled=["deepseek-old"])

    reloaded = _service(catalog_path, prefs_path)
    prefs = reloaded.load_preferences()
    assert prefs == {"order": ["deepseek-chat", "deepseek-old"], "disabled": ["deepseek-old"]}


def test_disabled_model_is_inactive_even_with_key(
    monkeypatch: pytest.MonkeyPatch, catalog_path: Path, prefs_path: Path
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    service = _service(catalog_path, prefs_path)
    service._query_ollama_tags = lambda: (set(), "Ollama not reachable (test)")  # type: ignore[method-assign]
    service.save_preferences(order=[], disabled=["deepseek-chat"])

    matrix = service.get_matrix()
    model = _find(matrix, "deepseek", "deepseek-chat")
    assert model["is_active"] is False
    assert model["status_reason"] == "Disabled by user"


def test_missing_preferences_file_defaults_to_catalog_order(catalog_path: Path, tmp_path: Path) -> None:
    service = _service(catalog_path, tmp_path / "does_not_exist.json")
    assert service.load_preferences() == {"order": [], "disabled": []}


# --------------------------------------------------------------------------
# Circuit breaker — runtime error tracking (report/clear/matrix overlay)
# --------------------------------------------------------------------------


def test_report_runtime_error_overlays_status_reason_but_keeps_is_active(
    monkeypatch: pytest.MonkeyPatch, catalog_path: Path, prefs_path: Path
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    service = _service(catalog_path, prefs_path)
    service._query_ollama_tags = lambda: (set(), "Ollama not reachable (test)")  # type: ignore[method-assign]

    service.report_runtime_error("deepseek-chat", 429, "Quota exhausted")
    matrix = service.get_matrix()

    model = _find(matrix, "deepseek", "deepseek-chat")
    assert model["is_active"] is True  # the key still exists — circuit breaker doesn't flip this
    assert model["has_runtime_error"] is True
    assert model["status_reason"] == "Runtime Error 429: Quota exhausted"


def test_clear_runtime_error_restores_the_normal_status_reason(
    monkeypatch: pytest.MonkeyPatch, catalog_path: Path, prefs_path: Path
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    service = _service(catalog_path, prefs_path)
    service._query_ollama_tags = lambda: (set(), "Ollama not reachable (test)")  # type: ignore[method-assign]

    service.report_runtime_error("deepseek-chat", 429, "Quota exhausted")
    service.clear_runtime_error("deepseek-chat")
    matrix = service.get_matrix()

    model = _find(matrix, "deepseek", "deepseek-chat")
    assert model["has_runtime_error"] is False
    assert model["status_reason"] == "Key detected"


def test_clear_runtime_error_on_a_model_with_no_error_is_a_no_op(catalog_path: Path, prefs_path: Path) -> None:
    service = _service(catalog_path, prefs_path)
    service.clear_runtime_error("deepseek-chat")  # must not raise
    assert "deepseek-chat" not in service.runtime_errors


def test_models_without_a_reported_error_are_unaffected(
    monkeypatch: pytest.MonkeyPatch, catalog_path: Path, prefs_path: Path
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    service = _service(catalog_path, prefs_path)
    service._query_ollama_tags = lambda: (set(), "Ollama not reachable (test)")  # type: ignore[method-assign]

    service.report_runtime_error("deepseek-chat", 429, "Quota exhausted")
    matrix = service.get_matrix()

    other_model = _find(matrix, "deepseek", "deepseek-old")
    assert other_model["has_runtime_error"] is False
    assert other_model["status_reason"] == "Deprecated"

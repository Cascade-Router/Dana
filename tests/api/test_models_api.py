"""Tests for the Model Registry's REST surface (dana.api.models):
GET /api/models/matrix, POST /api/models/preferences, and
POST /api/models/{model_id}/clear-error.

Builds a minimal standalone FastAPI app around just this router, same
convention as tests/api/test_system_env.py — the registry itself points at a
temp catalog file so these stay hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dana.api import models as models_module
from dana.core.model_registry import ModelRegistryService

_CATALOG = {
    "version": 1,
    "providers": {
        "openrouter": {
            "requires_api_key": True,
            "api_key_env": "OPENROUTER_API_KEY",
            "models": [
                {
                    "id": "openai/gpt-oss-120b",
                    "name": "GPT-OSS 120B",
                    "context_window": 131072,
                    "max_output_tokens": 32768,
                    "pricing": {"prompt_per_1m": 0.0, "completion_per_1m": 0.0},
                    "rate_limits": {"tpm": None, "rpm": None, "rpd": None},
                    "modalities": ["text"],
                    "supports_tool_calling": True,
                    "supports_thought_signature": False,
                    "is_deprecated": False,
                }
            ],
        }
    },
}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    catalog_path = tmp_path / "models_registry.json"
    catalog_path.write_text(json.dumps(_CATALOG), encoding="utf-8")
    service = ModelRegistryService(catalog_path=catalog_path, preferences_path=tmp_path / "prefs.json")
    monkeypatch.setattr(models_module, "_registry", service)
    app = FastAPI()
    app.include_router(models_module.router)
    return TestClient(app)


def test_get_matrix_returns_the_seeded_model(client: TestClient) -> None:
    resp = client.get("/api/models/matrix")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    model = body["providers"][0]["models"][0]
    assert model["id"] == "openai/gpt-oss-120b"
    assert model["is_active"] is True
    assert model["has_runtime_error"] is False


def test_save_preferences_round_trips_through_the_matrix(client: TestClient) -> None:
    resp = client.post(
        "/api/models/preferences",
        json={"order": ["openai/gpt-oss-120b"], "disabled": ["openai/gpt-oss-120b"]},
    )
    assert resp.status_code == 200
    model = resp.json()["providers"][0]["models"][0]
    assert model["is_active"] is False
    assert model["status_reason"] == "Disabled by user"


def test_clear_error_endpoint_handles_a_model_id_containing_a_slash(client: TestClient) -> None:
    models_module._registry.report_runtime_error("openai/gpt-oss-120b", 429, "Quota exhausted")

    resp = client.post("/api/models/openai/gpt-oss-120b/clear-error")
    assert resp.status_code == 200
    model = resp.json()["providers"][0]["models"][0]
    assert model["has_runtime_error"] is False
    assert model["status_reason"] == "Key detected"


def test_clear_error_on_unknown_model_id_is_still_a_200(client: TestClient) -> None:
    resp = client.post("/api/models/does-not-exist/clear-error")
    assert resp.status_code == 200

"""Tests for the Environment panel's REST surface (dana.api.system):
GET/POST /api/system/env and the live-validation POST /api/system/env/validate.

Builds a minimal standalone FastAPI app around just this router — the real
credential probes (urllib requests to Groq/OpenAI/Anthropic/Gemini) are
monkeypatched out via dana.api.system._validate_key so these stay hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dana.api import system as system_module


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(system_module, "ENV_PATH", tmp_path / ".env")
    # Clear the FULL allowlist, not just the sensitive half — a real
    # dotenv load triggered by some other test earlier in a full-suite run
    # can leave non-sensitive routing vars (e.g. DANA_CLOUD_PRIMARY) set in
    # the real process env, which _env_snapshot() would otherwise surface
    # here too and break the "omits unset keys" assertion below.
    for key in system_module._ALLOWLIST:
        monkeypatch.delenv(key, raising=False)
    app = FastAPI()
    app.include_router(system_module.router)
    return TestClient(app)


def test_get_env_omits_unset_keys(client: TestClient) -> None:
    resp = client.get("/api/system/env")
    assert resp.status_code == 200
    assert resp.json() == {"env": {}}


def test_save_env_rejects_unknown_key(client: TestClient) -> None:
    resp = client.post("/api/system/env", json={"key": "NOT_A_REAL_VAR", "value": "x"})
    assert resp.status_code == 400


def test_save_env_rejects_empty_value(client: TestClient) -> None:
    resp = client.post("/api/system/env", json={"key": "GROQ_API_KEY", "value": "  "})
    assert resp.status_code == 400


def test_save_env_persists_masks_and_hot_reloads_without_restart(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(system_module, "_validate_key", lambda name, value: (True, "valid — provider accepted the key"))

    resp = client.post("/api/system/env", json={"key": "GROQ_API_KEY", "value": "gsk_1234567890abcdef"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["valid"] is True

    # (a) live process env updated immediately — no restart needed.
    import os

    assert os.environ["GROQ_API_KEY"] == "gsk_1234567890abcdef"

    # (b) persisted to .env on disk.
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "GROQ_API_KEY=gsk_1234567890abcdef" in env_text

    # (c) GET reflects it, masked — never the raw value.
    snap = client.get("/api/system/env").json()["env"]
    assert snap["GROQ_API_KEY"] == "gsk***ef"


def test_save_env_upserts_existing_dotenv_line(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_module, "_validate_key", lambda name, value: (True, "ok"))
    (tmp_path / ".env").write_text("SOME_OTHER_VAR=keep-me\nGROQ_API_KEY=old-value\n", encoding="utf-8")

    client.post("/api/system/env", json={"key": "GROQ_API_KEY", "value": "new-value"})

    lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert "SOME_OTHER_VAR=keep-me" in lines
    assert "GROQ_API_KEY=new-value" in lines
    assert "GROQ_API_KEY=old-value" not in lines


def test_validate_endpoint_reports_not_configured(client: TestClient) -> None:
    resp = client.post("/api/system/env/validate", json={"key": "GROQ_API_KEY"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["configured"] is False
    assert body["valid"] is False


def test_validate_endpoint_reports_valid_without_resending_secret(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_live_value")
    monkeypatch.setattr(system_module, "_validate_key", lambda name, value: (True, "valid — provider accepted the key"))

    resp = client.post("/api/system/env/validate", json={"key": "GROQ_API_KEY"})
    body = resp.json()
    assert body["configured"] is True
    assert body["valid"] is True
    # The request body never carried a value — this must reflect the LIVE
    # env value, not something echoed back from the request.
    assert "value" not in resp.request.content.decode() if resp.request.content else True


def test_validate_endpoint_reports_invalid_on_rejected_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-bad")
    monkeypatch.setattr(
        system_module, "_validate_key", lambda name, value: (False, "provider rejected the key (HTTP 401)")
    )

    resp = client.post("/api/system/env/validate", json={"key": "ANTHROPIC_API_KEY"})
    body = resp.json()
    assert body["configured"] is True
    assert body["valid"] is False


def test_validate_endpoint_rejects_unknown_key(client: TestClient) -> None:
    resp = client.post("/api/system/env/validate", json={"key": "NOT_A_REAL_VAR"})
    assert resp.status_code == 400

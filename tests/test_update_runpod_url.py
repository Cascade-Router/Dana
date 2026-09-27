"""Offline tests for scripts/update_runpod_url.py — fake runpod module, temp .env."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import update_runpod_url as mod


def _pod(pod_id: str, *, status: str = "RUNNING", gpus: int = 1, ports: str = "11434/http,22/tcp", name: str = "vlm"):
    return {"id": pod_id, "name": name, "desiredStatus": status, "gpuCount": gpus, "ports": ports,
            "machine": {"gpuDisplayName": "RTX 4090"}}


def test_proxy_url_format():
    assert mod.proxy_url("lugryg9mduv0xo") == "https://lugryg9mduv0xo-11434.proxy.runpod.net"


def test_select_pod_picks_single_running_gpu_pod_exposing_ollama():
    pods = [
        _pod("stopped1", status="EXITED"),
        _pod("cpuonly1", gpus=0),
        _pod("noollama", ports="8888/http,22/tcp"),
        _pod("target01"),
    ]
    assert mod.select_pod(pods)["id"] == "target01"


def test_select_pod_refuses_to_guess_between_multiple_matches():
    with pytest.raises(SystemExit, match="2 pods matched"):
        mod.select_pod([_pod("a1"), _pod("b2")])


def test_select_pod_reports_when_nothing_matches():
    with pytest.raises(SystemExit, match="No pod matched"):
        mod.select_pod([_pod("off", status="EXITED")])


def test_select_pod_explicit_id_must_be_running():
    pods = [_pod("a1"), _pod("off", status="EXITED")]
    assert mod.select_pod(pods, pod_id="a1")["id"] == "a1"
    with pytest.raises(SystemExit, match="not RUNNING"):
        mod.select_pod(pods, pod_id="off")


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "RUNPOD_API_KEY=rp_test\n"
        "DANA_LOCAL_VISION_MODEL=qwen2.5vl:7b\n"
        "OLLAMA_URL              =https://oldpod-11434.proxy.runpod.net\n"
        "# keep this comment\n"
        "OLLAMA_BASE_URL=https://oldpod-11434.proxy.runpod.net\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_runpod(monkeypatch: pytest.MonkeyPatch):
    fake = types.ModuleType("runpod")
    fake.api_key = None
    fake.pods = [_pod("newpod123"), _pod("off", status="EXITED")]
    fake.get_pods = lambda: fake.pods
    monkeypatch.setitem(sys.modules, "runpod", fake)
    return fake


def test_main_rewrites_both_keys_and_preserves_rest(env_file, fake_runpod, monkeypatch):
    monkeypatch.setattr(mod, "verify_ollama", lambda url: (True, "Ollama is running"))
    assert mod.main(["--env-file", str(env_file)]) == 0
    text = env_file.read_text(encoding="utf-8")
    assert fake_runpod.api_key == "rp_test"
    assert "OLLAMA_URL=https://newpod123-11434.proxy.runpod.net\n" in text
    assert "OLLAMA_BASE_URL=https://newpod123-11434.proxy.runpod.net\n" in text
    assert "oldpod" not in text
    assert "DANA_LOCAL_VISION_MODEL=qwen2.5vl:7b" in text
    assert "# keep this comment" in text


def test_main_dry_run_leaves_env_untouched(env_file, fake_runpod, monkeypatch):
    monkeypatch.setattr(mod, "verify_ollama", lambda url: (True, "Ollama is running"))
    before = env_file.read_text(encoding="utf-8")
    assert mod.main(["--env-file", str(env_file), "--dry-run"]) == 0
    assert env_file.read_text(encoding="utf-8") == before


def test_main_failed_health_check_leaves_env_untouched(env_file, fake_runpod, monkeypatch):
    monkeypatch.setattr(mod, "verify_ollama", lambda url: (False, "HTTP 404 ''"))
    before = env_file.read_text(encoding="utf-8")
    assert mod.main(["--env-file", str(env_file)]) == 1
    assert env_file.read_text(encoding="utf-8") == before


def test_main_requires_api_key(tmp_path, fake_runpod, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("OLLAMA_URL=http://localhost:11434\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="RUNPOD_API_KEY is not set"):
        mod.main(["--env-file", str(path)])

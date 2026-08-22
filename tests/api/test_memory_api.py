"""Tests for the /api/memory REST endpoints backing the frontend's
MemoryPlugin — GET/POST must read and write straight through
dana.plugins.memory.core_memory's own helpers, with no separate storage
logic of their own. Every test redirects CORE_MEMORY_PATH to a throwaway
temp file (see the autouse `_memory_file` fixture) — none of these ever
touch the real on-disk agent_workspace.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dana.api import server as server_module
from dana.plugins.memory import core_memory


@pytest.fixture(autouse=True)
def _memory_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "data" / "core_memory.json"
    monkeypatch.setattr(core_memory, "CORE_MEMORY_PATH", path)
    return path


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_module.app)


def test_get_memory_returns_empty_dict_when_no_file_yet(client: TestClient) -> None:
    resp = client.get("/api/memory")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "memory": {}}


def test_get_memory_returns_existing_contents(client: TestClient) -> None:
    core_memory.write_core_memory("user_preferences", "prefers metric units")
    resp = client.get("/api/memory")
    assert resp.json() == {"ok": True, "memory": {"user_preferences": "prefers metric units"}}


def test_post_memory_writes_through_to_core_memory_module(client: TestClient) -> None:
    resp = client.post("/api/memory", json={"active_project": "60x40x20mm enclosure"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "memory": {"active_project": "60x40x20mm enclosure"}}
    # Reading directly through the underlying module proves the API took
    # the SAME on-disk path the agent's own update_core_memory tool uses —
    # not a separate file/serialization of its own.
    assert core_memory.read_core_memory() == {"active_project": "60x40x20mm enclosure"}


def test_post_memory_then_get_round_trips(client: TestClient) -> None:
    client.post("/api/memory", json={"user_preferences": "likes dark mode"})
    resp = client.get("/api/memory")
    assert resp.json()["memory"] == {"user_preferences": "likes dark mode"}


def test_post_memory_is_a_full_overwrite_not_a_merge(client: TestClient) -> None:
    """Unlike the agent's own update_core_memory tool (single-section
    read-modify-write), the REST POST must fully replace the file — a
    section present on disk but absent from the posted dict is dropped,
    since the frontend always sends its complete edited state back."""
    core_memory.write_core_memory("user_preferences", "prefers metric units")
    core_memory.write_core_memory("active_project", "60x40x20mm enclosure")

    resp = client.post("/api/memory", json={"active_project": "revised spec"})

    assert resp.json()["memory"] == {"active_project": "revised spec"}
    assert core_memory.read_core_memory() == {"active_project": "revised spec"}


def test_post_memory_with_empty_dict_clears_everything(client: TestClient) -> None:
    core_memory.write_core_memory("user_preferences", "prefers metric units")
    resp = client.post("/api/memory", json={})
    assert resp.json() == {"ok": True, "memory": {}}
    assert core_memory.read_core_memory() == {}


def test_post_memory_creates_parent_directories_on_first_use(client: TestClient, _memory_file: Path) -> None:
    assert not _memory_file.parent.exists()
    client.post("/api/memory", json={"active_project": "new"})
    assert _memory_file.is_file()


def test_post_memory_rejects_non_string_values(client: TestClient) -> None:
    """FastAPI's dict[str, str] body annotation enforces the shape at the
    schema layer — a nested object/number value is a validation error, not
    something that reaches core_memory at all."""
    resp = client.post("/api/memory", json={"bad_section": {"nested": "object"}})
    assert resp.status_code == 422
    assert core_memory.read_core_memory() == {}

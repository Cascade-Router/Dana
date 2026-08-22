"""Tests for dana.api.workspace — the read-only REST API backing the
frontend's Workspace Explorer plugin. Every test redirects the sandbox
root to a throwaway temp directory (see the autouse `_sandbox` fixture) —
none of these ever touch the real AGENT_WORKSPACE_DIR on disk.

Path traversal is tested via the URL-ENCODED form (``..%2f``) — a literal
``../`` in a request URL gets collapsed by standard URL normalization
before it ever reaches the ASGI app (confirmed: it 404s as "no matching
route", not because the endpoint rejected it), so the encoded form is the
one that actually exercises resolve_sandboxed_path's rejection, matching
the real attack vector a client would have to use to get a literal ``..``
segment past normalization.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dana.api import server as server_module
from dana.plugins.os import file_system


@pytest.fixture(autouse=True)
def _sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "agent_workspace"
    # exist_ok=True: tests/conftest.py's global _isolate_os_tools_sandbox
    # autouse fixture already creates this same tmp_path/agent_workspace
    # directory first — tolerate it already existing rather than raising.
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(file_system, "_SANDBOX_ROOT", root)
    return root


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_module.app)


@pytest.fixture(autouse=True)
def _mounts_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Dynamic Workspace Mounting's on-disk registry, redirected to a
    throwaway file — none of these tests may ever touch the real
    AGENT_WORKSPACE_DIR/data/mounts.json."""
    from dana.api import workspace as workspace_module

    mounts_path = tmp_path / "mounts.json"
    monkeypatch.setattr(workspace_module, "_MOUNTS_PATH", mounts_path)
    return mounts_path


# --------------------------------------------------------------------------
# GET /api/workspace/tree
# --------------------------------------------------------------------------


def test_tree_on_empty_sandbox(client: TestClient, _sandbox: Path) -> None:
    resp = client.get("/api/workspace/tree")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["tree"] == {"name": "workspace", "path": "", "type": "directory", "children": []}


def test_tree_lists_files_and_nested_directories(client: TestClient, _sandbox: Path) -> None:
    (_sandbox / "notes.txt").write_text("hello")
    sub = _sandbox / "sub"
    sub.mkdir()
    (sub / "nested.py").write_text("print(1)")

    resp = client.get("/api/workspace/tree")
    assert resp.status_code == 200
    tree = resp.json()["tree"]

    names = {child["name"] for child in tree["children"]}
    assert names == {"notes.txt", "sub"}

    sub_node = next(c for c in tree["children"] if c["name"] == "sub")
    assert sub_node["type"] == "directory"
    assert sub_node["path"] == "sub"
    assert sub_node["children"] == [{"name": "nested.py", "path": "sub/nested.py", "type": "file", "size": 8}]


def test_tree_sorts_directories_before_files_alphabetically(client: TestClient, _sandbox: Path) -> None:
    (_sandbox / "z_file.txt").write_text("z")
    (_sandbox / "a_file.txt").write_text("a")
    (_sandbox / "b_dir").mkdir()

    resp = client.get("/api/workspace/tree")
    names_in_order = [c["name"] for c in resp.json()["tree"]["children"]]
    assert names_in_order == ["b_dir", "a_file.txt", "z_file.txt"]


def test_tree_reports_file_size(client: TestClient, _sandbox: Path) -> None:
    (_sandbox / "sized.txt").write_text("0123456789")
    resp = client.get("/api/workspace/tree")
    node = resp.json()["tree"]["children"][0]
    assert node["size"] == 10


# --------------------------------------------------------------------------
# GET /api/workspace/file/{file_path:path}
# --------------------------------------------------------------------------


def test_get_file_returns_text_content(client: TestClient, _sandbox: Path) -> None:
    (_sandbox / "hello.txt").write_text("hello world")
    resp = client.get("/api/workspace/file/hello.txt")
    assert resp.status_code == 200
    assert resp.text == "hello world"
    assert resp.headers["content-type"].startswith("text/plain")


def test_get_file_guesses_python_mime_type(client: TestClient, _sandbox: Path) -> None:
    (_sandbox / "script.py").write_text("print('hi')")
    resp = client.get("/api/workspace/file/script.py")
    assert resp.status_code == 200
    assert "python" in resp.headers["content-type"]


def test_get_file_guesses_image_mime_type(client: TestClient, _sandbox: Path) -> None:
    (_sandbox / "pixel.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    resp = client.get("/api/workspace/file/pixel.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_get_file_nested_path(client: TestClient, _sandbox: Path) -> None:
    sub = _sandbox / "a" / "b"
    sub.mkdir(parents=True)
    (sub / "deep.txt").write_text("deep content")
    resp = client.get("/api/workspace/file/a/b/deep.txt")
    assert resp.status_code == 200
    assert resp.text == "deep content"


def test_get_file_missing_returns_404(client: TestClient, _sandbox: Path) -> None:
    resp = client.get("/api/workspace/file/does_not_exist.txt")
    assert resp.status_code == 404


def test_get_file_on_a_directory_returns_400(client: TestClient, _sandbox: Path) -> None:
    (_sandbox / "adir").mkdir()
    resp = client.get("/api/workspace/file/adir")
    assert resp.status_code == 400


# --------------------------------------------------------------------------
# Path traversal rejection — the crucial security requirement.
# --------------------------------------------------------------------------


def test_get_file_rejects_encoded_parent_traversal(client: TestClient, _sandbox: Path, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret, outside the sandbox")

    resp = client.get("/api/workspace/file/..%2fsecret.txt")

    assert resp.status_code == 400
    assert "outside the sandbox" in resp.json()["detail"]


def test_get_file_rejects_deeply_nested_encoded_traversal(client: TestClient, _sandbox: Path) -> None:
    resp = client.get("/api/workspace/file/a%2fb%2f..%2f..%2f..%2fetc%2fpasswd")
    assert resp.status_code == 400
    assert "outside the sandbox" in resp.json()["detail"]


def test_get_file_rejects_windows_absolute_path(client: TestClient, _sandbox: Path) -> None:
    resp = client.get("/api/workspace/file/C:%2fWindows%2fSystem32")
    assert resp.status_code == 400


def test_tree_endpoint_never_reveals_content_outside_sandbox(
    client: TestClient, _sandbox: Path, tmp_path: Path
) -> None:
    """Sanity check: a sibling directory next to the sandbox root must
    never appear in the tree, confirming the walk never escapes root."""
    sibling = tmp_path / "not_the_sandbox"
    sibling.mkdir()
    (sibling / "outside.txt").write_text("should never be listed")

    resp = client.get("/api/workspace/tree")
    tree_json = resp.json()
    assert "not_the_sandbox" not in str(tree_json)
    assert "outside.txt" not in str(tree_json)


# --------------------------------------------------------------------------
# Dynamic Workspace Mounting — GET /api/workspace/mounts, POST /api/workspace/mount
# --------------------------------------------------------------------------


def test_mounts_endpoint_starts_empty(client: TestClient) -> None:
    resp = client.get("/api/workspace/mounts")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "mounted_directories": []}


def test_mount_endpoint_registers_an_existing_absolute_directory(client: TestClient, tmp_path: Path) -> None:
    external = tmp_path / "external_project"
    external.mkdir()

    resp = client.post("/api/workspace/mount", json={"path": str(external)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert str(external.resolve()) in body["mounted_directories"]

    listed = client.get("/api/workspace/mounts").json()
    assert str(external.resolve()) in listed["mounted_directories"]


def test_mount_endpoint_rejects_relative_path(client: TestClient) -> None:
    resp = client.post("/api/workspace/mount", json={"path": "relative/dir"})
    assert resp.status_code == 400
    assert "absolute" in resp.json()["detail"]


def test_mount_endpoint_rejects_nonexistent_directory(client: TestClient, tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    resp = client.post("/api/workspace/mount", json={"path": str(missing)})
    assert resp.status_code == 400


def test_mount_endpoint_rejects_a_file_path(client: TestClient, tmp_path: Path) -> None:
    a_file = tmp_path / "not_a_directory.txt"
    a_file.write_text("x")
    resp = client.post("/api/workspace/mount", json={"path": str(a_file)})
    assert resp.status_code == 400


def test_mount_endpoint_is_idempotent(client: TestClient, tmp_path: Path) -> None:
    external = tmp_path / "external_project"
    external.mkdir()

    first = client.post("/api/workspace/mount", json={"path": str(external)}).json()
    second = client.post("/api/workspace/mount", json={"path": str(external)}).json()

    assert first["mounted_directories"] == second["mounted_directories"]
    assert second["mounted_directories"].count(str(external.resolve())) == 1

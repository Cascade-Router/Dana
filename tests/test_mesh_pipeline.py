"""Headless integration test for the CAD mesh pipeline: real FreeCAD engine
call -> tessellated .glb on disk -> REST download endpoint -> valid glTF
binary bytes a browser's GLTFLoader could actually parse. Exists so a
regression in any one hop (a session-path miscomputation, export_mesh_stl's
STL->GLB conversion, dana.api.cad's media_type/filename resolution) is
caught by ``pytest`` alone, with no manual click-through of the Tauri/React
UI required.

Naming note: the tool_ids a chat turn actually dispatches are
``create_freecad_cylinder``/``create_freecad_helix``/``perform_freecad_boolean``
(see ``dana.plugins.freecad.engine._execute_ir_tool``'s own call sites) —
but those are IR/dispatch identifiers, not the importable Python callables.
The real functions this module exercises directly are
``engine.create_cylinder``, ``engine.create_helix``, and
``engine.apply_boolean``, each of which passes that exact tool_id straight
through to ``_execute_ir_tool`` internally.

Requires a real ``FreeCADCmd`` binary on PATH (or ``DANA_FREECADCMD_PATH``) —
this drives ``dana.plugins.freecad.engine`` directly, not
``dana.platform.mock``'s stand-in, so the whole suite is skipped when one
isn't found rather than silently testing nothing real. Each test spawns
2-4 real FreeCADCmd subprocesses, so this file runs in the ten-seconds-to-
low-minutes range, not milliseconds — run it on its own (see the bottom of
this docstring) rather than as part of every quick ``pytest`` loop.

Run just this file::

    pytest tests/test_mesh_pipeline.py -v

Exclude it from a routine full-suite run (it's marked ``e2e``, same marker
``tests/test_e2e_lifecycle.py`` uses)::

    pytest -m "not e2e"
"""

from __future__ import annotations

import json
import struct
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dana.api import cad as cad_module
from dana.api import server as server_module
from dana.plugins.freecad import engine
from dana.session_context import DEFAULT_SESSION_ID, set_session_id

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        engine.detect_freecadcmd() is None,
        reason="real FreeCADCmd not found (PATH or DANA_FREECADCMD_PATH) — this test drives the real engine",
    ),
]

_GLTF_MAGIC = b"glTF"


def _assert_valid_glb(path: Path) -> bytes:
    """Non-zero-byte, well-formed glTF-Binary: real magic/version, and the
    header's own declared total length matches the file actually on disk
    (the exact invariant a torn/partial write — the class of bug that
    produced the frontend's "Invalid typed array length" crash — would
    violate)."""
    assert path.is_file(), f"expected a mesh file at {path}, found none"
    data = path.read_bytes()
    assert len(data) > 12, f"{path} is empty/truncated ({len(data)} bytes)"
    magic = data[:4]
    version, total_length = struct.unpack_from("<II", data, 4)
    assert magic == _GLTF_MAGIC, f"{path} does not start with the glTF binary magic (got {magic!r})"
    assert version == 2, f"{path}: unexpected glTF binary container version {version}"
    assert total_length == len(data), (
        f"{path}: header declares length {total_length} but the file is {len(data)} bytes "
        "(a torn/partial write)"
    )
    return data


def _call(fn, /, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Every dana.plugins.freecad.engine.* tool wrapper returns a JSON
    string envelope (``{"ok": ..., ...}``), never a dict directly."""
    result = json.loads(fn(*args, **kwargs))
    assert result.get("ok"), f"{fn.__name__} failed: {result.get('error')}"
    return result


@pytest.fixture(autouse=True)
def _isolated_cad_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirects engine.py's and cad.py's own output/export base
    directories to a throwaway tmp_path for THIS test only (same
    "monkeypatch the module-level Path constant" convention
    tests/conftest.py's other autouse fixtures already use for
    dana.api.sessions/dana.plugins.os.file_system) — every file this test
    produces still lands under the same ``sessions/<session_id>/`` shape
    ``freecad_output/``/``exports/`` always use, just rooted under
    tmp_path instead of the real repo tree, so a failed run never leaves
    real geometry files behind and repeated runs never accumulate garbage
    in the actual ``freecad_output/`` directory on disk.

    Also forces ``DANA_HEADLESS=true`` (mirrors
    ``scripts/launchers/launch_api_server.py``'s own default) so
    ``_auto_show`` never tries to pop open a real FreeCAD GUI window
    mid-test, and clears ``DANA_OS_DRY_RUN`` so a stray ambient env var
    can't silently turn every engine call below into a no-op dry run.
    """
    monkeypatch.setattr(engine, "_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(engine, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(cad_module, "_FREECAD_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(cad_module, "_FREECAD_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setenv("DANA_HEADLESS", "true")
    monkeypatch.delenv("DANA_OS_DRY_RUN", raising=False)


@pytest.fixture
def session_id() -> str:
    sid = f"test-{uuid.uuid4().hex[:12]}"
    set_session_id(sid)
    yield sid
    set_session_id(DEFAULT_SESSION_ID)


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_module.app)


def _download(client: TestClient, session_id: str, filename: str):
    return client.get(f"/api/cad/artifacts/{filename}/download", params={"session_id": session_id})


def test_create_cylinder_produces_downloadable_glb(session_id: str, client: TestClient) -> None:
    created = _call(engine.create_cylinder, radius=12.0, height=30.0, name="PipelineCylinder")
    session_path = Path(created["path"])
    assert session_path.is_file(), "create_cylinder must save the session .FCStd document"
    object_name = created["name"]

    exported = _call(engine.export_mesh_stl, str(session_path), name=object_name, target_object=object_name)
    glb_path = Path(exported["path"])
    _assert_valid_glb(glb_path)

    resp = _download(client, session_id, glb_path.name)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "model/gltf-binary"
    assert resp.content == glb_path.read_bytes()


def test_create_helix_produces_downloadable_glb(session_id: str, client: TestClient) -> None:
    created = _call(
        engine.create_helix, coil_radius=15.0, pitch=5.0, height=20.0, pipe_radius=2.0, name="PipelineHelix"
    )
    session_path = Path(created["path"])
    assert session_path.is_file()
    object_name = created["name"]

    exported = _call(engine.export_mesh_stl, str(session_path), name=object_name, target_object=object_name)
    glb_path = Path(exported["path"])
    _assert_valid_glb(glb_path)

    resp = _download(client, session_id, glb_path.name)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "model/gltf-binary"
    assert resp.content == glb_path.read_bytes()


def test_apply_boolean_cut_produces_downloadable_glb(session_id: str, client: TestClient) -> None:
    base = _call(engine.create_box, 60.0, 40.0, 20.0, name="PipelineBase")
    tool = _call(
        engine.create_cylinder, radius=8.0, height=40.0, name="PipelineBore", placement=(30.0, 20.0, -10.0)
    )

    cut = _call(
        engine.apply_boolean, "cut", base_object=base["name"], tool_object=tool["name"], name="PipelineCutResult"
    )
    session_path = Path(cut["path"])
    assert session_path.is_file()
    object_name = cut["name"]

    exported = _call(engine.export_mesh_stl, str(session_path), name=object_name, target_object=object_name)
    glb_path = Path(exported["path"])
    _assert_valid_glb(glb_path)

    resp = _download(client, session_id, glb_path.name)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "model/gltf-binary"
    assert resp.content == glb_path.read_bytes()

"""Deliverables bar backend: blueprint PDF, BOM CSV and the replay macro are
listed and downloadable through dana.api.cad's session-scoped route, and
nothing outside a session's own directories ever is."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dana.api import artifacts_registry
from dana.api import cad as cad_module
from dana.api import server as server_module
from dana.plugins.freecad import py_export
from dana.plugins.freecad.call_log import CadCallLog

SID = "deliverables-1"


@pytest.fixture(autouse=True)
def _dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cad_module, "_FREECAD_OUTPUT_DIR", tmp_path / "freecad_output")
    monkeypatch.setattr(cad_module, "_FREECAD_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(cad_module, "_AGENT_EXPORT_DIR", tmp_path / "agent_exports")
    monkeypatch.setattr(py_export, "_EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(artifacts_registry, "_artifacts", [])


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_module.app)


def _download(client: TestClient, filename: str, session_id: str = SID):
    return client.get(f"/api/cad/artifacts/{filename}/download", params={"session_id": session_id})


def test_blueprint_bom_and_macro_are_listed_and_downloadable(client: TestClient, tmp_path: Path) -> None:
    exports = tmp_path / "exports" / "sessions" / SID
    exports.mkdir(parents=True)
    (exports / "Plate.pdf").write_bytes(b"%PDF-1.7")
    (exports / "dana_session_macro.py").write_text("import FreeCAD\n", encoding="utf-8")
    (exports / "legacy.FCMacro").write_text("import FreeCAD\n", encoding="utf-8")
    bom_dir = tmp_path / "agent_exports" / "sessions" / SID
    bom_dir.mkdir(parents=True)
    (bom_dir / "Kit_bom.csv").write_text("Part Name\n", encoding="utf-8")

    listed = {a["filename"] for a in client.get("/api/cad/artifacts", params={"session_id": SID}).json()["artifacts"]}
    assert {"Plate.pdf", "dana_session_macro.py", "legacy.FCMacro", "Kit_bom.csv"} <= listed

    pdf = _download(client, "Plate.pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    csv = _download(client, "Kit_bom.csv")
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    assert _download(client, "legacy.FCMacro").status_code == 200


def test_another_sessions_deliverables_are_not_reachable(client: TestClient, tmp_path: Path) -> None:
    other = tmp_path / "exports" / "sessions" / "someone-else"
    other.mkdir(parents=True)
    (other / "secret.pdf").write_bytes(b"%PDF")
    assert _download(client, "secret.pdf").status_code == 404


@pytest.mark.parametrize(
    "filename",
    ["..%2F..%2F.env", "..%5C..%5Csecret.py", "sessions%2Fsomeone-else%2Fsecret.pdf", "notes.txt", ".env"],
)
def test_traversal_and_unlisted_extensions_are_rejected(client: TestClient, tmp_path: Path, filename: str) -> None:
    (tmp_path / ".env").write_text("KEY=1", encoding="utf-8")
    assert _download(client, filename).status_code in (400, 404)


def test_export_python_script_returns_a_download_url(client: TestClient) -> None:
    log = CadCallLog()
    log.record("create_freecad_box", {"name": "Plate", "length": 10, "width": 5, "height": 2}, ok=True,
               result={"name": "Plate", "dimensions": {"length": 10, "width": 5, "height": 2}})

    with client.websocket_connect(f"/ws/chat?session_id={SID}") as ws:
        ws.receive_json()  # ready
        session = next(s for s in server_module._active_sessions.values() if s["session_id"] == SID)
        session["call_log"] = log
        ws.send_json({"type": "export_python_script"})
        event = ws.receive_json()

    assert event["type"] == "python_script_exported"
    assert Path(event["path"]).parent.name == SID  # session-scoped, and "path" kept for old clients
    assert event["filename"] == "dana_session_macro.py"
    assert event["download_url"] == f"/api/cad/artifacts/dana_session_macro.py/download?session_id={SID}"
    resp = client.get(event["download_url"])
    assert resp.status_code == 200 and "Plate" in resp.text

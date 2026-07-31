"""Phase 2B — dual-slot blue-green rotator + health-check rollbacks (offline)."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_unpack_and_atomic_pointer_switch(tmp_path: Path) -> None:
    """Unpack into inactive slot, then atomically switch slot_a ↔ slot_b."""
    from dana.updater.slot_manager import SLOT_A, SLOT_B, SlotManager

    base = tmp_path / "engine_slots"
    sm = SlotManager(base_dir=base, initial_version="0.1.0")

    assert sm.active_slot_name() == SLOT_A
    assert sm.inactive_slot_name() == SLOT_B
    assert (base / "active_slot").read_text(encoding="utf-8").strip() == SLOT_A
    assert sm.get_active_version() == "0.1.0"
    assert "Slot A" in sm.display_active()

    pkg = _zip_bytes(
        {
            "engine.py": b"print('v0.2.0')\n",
            "VERSION": b"0.2.0\n",
        }
    )
    pkg_path = tmp_path / "dana-0.2.0.zip"
    pkg_path.write_bytes(pkg)

    candidate = sm.unpack_into_inactive(pkg_path, version="0.2.0")
    assert candidate == sm.slot_path(SLOT_B)
    assert (candidate / "engine.py").is_file()
    assert (candidate / "VERSION").read_text(encoding="utf-8").strip() == "0.2.0"
    # Active pointer untouched during unpack.
    assert sm.active_slot_name() == SLOT_A
    assert sm.get_active_version() == "0.1.0"

    switched = sm.switch_active(SLOT_B, version="0.2.0")
    assert switched == SLOT_B
    assert sm.active_slot_name() == SLOT_B
    assert sm.inactive_slot_name() == SLOT_A
    assert (base / "active_slot").read_text(encoding="utf-8").strip() == SLOT_B
    active_json = json.loads((base / "active.json").read_text(encoding="utf-8"))
    assert active_json["active_slot"] == SLOT_B
    assert sm.get_active_version() == "0.2.0"
    assert "Slot B" in sm.display_active()

    meta = sm.metadata()
    assert meta["active_slot"] == SLOT_B
    assert meta["active_version"] == "0.2.0"
    assert any(h.get("event") == "switch" for h in meta["history"])

    # Round-trip back to slot_a.
    pkg2 = _zip_bytes({"engine.py": b"print('v0.3.0')\n", "VERSION": b"0.3.0\n"})
    sm.unpack_into_inactive(pkg2, version="0.3.0")
    sm.switch_active(SLOT_A, version="0.3.0")
    assert sm.active_slot_name() == SLOT_A
    assert sm.get_active_version() == "0.3.0"


def test_failing_update_rolls_back_active_untouched(tmp_path: Path) -> None:
    """Corrupted / failing health check → wipe candidate, leave active healthy."""
    from dana.updater.health_check import run_slot_health_check
    from dana.updater.slot_manager import SLOT_A, SLOT_B, SlotManager

    base = tmp_path / "engine_slots"
    fail_log = tmp_path / "update_failures.log"
    sm = SlotManager(base_dir=base, initial_version="0.1.0")

    # Seed a healthy marker in the active slot.
    (sm.active_slot_path() / "HEALTHY").write_text("ok\n", encoding="utf-8")
    active_before = sm.active_slot_name()
    version_before = sm.get_active_version()
    active_marker = (sm.active_slot_path() / "HEALTHY").read_text(encoding="utf-8")

    bad = _zip_bytes({"engine.py": b"broken\n", "CORRUPT": b"1\n", "VERSION": b"9.9.9\n"})
    candidate = sm.unpack_into_inactive(bad, version="9.9.9")
    # Active pointer still slot_a; candidate landed in inactive slot_b.
    assert sm.active_slot_name() == SLOT_A
    assert sm.inactive_slot_name() == SLOT_B
    assert candidate == sm.slot_path(SLOT_B)

    restarts: list[str] = []

    def verify_fail(_path: Path) -> bool:
        return False

    def ipc() -> dict:
        restarts.append("hot_restart")
        return {"ok": True}

    result = run_slot_health_check(
        candidate,
        verify_fn=verify_fail,
        ipc_client=ipc,
        slot_manager=sm,
        failure_log=fail_log,
        version="9.9.9",
    )
    assert result.ok is False
    assert result.wiped is True
    assert result.switched is False
    assert result.status == "failed"
    # Active slot untouched.
    assert sm.active_slot_name() == active_before == SLOT_A
    assert sm.get_active_version() == version_before
    assert (sm.active_slot_path() / "HEALTHY").read_text(encoding="utf-8") == active_marker
    # Candidate wiped (no CORRUPT sentinel left).
    assert not (sm.slot_path(SLOT_B) / "CORRUPT").exists()
    assert not (sm.slot_path(SLOT_B) / "engine.py").exists()
    # Failure logged; hot_restart never fired.
    assert fail_log.is_file()
    assert "ROLLBACK" in fail_log.read_text(encoding="utf-8")
    assert restarts == []


def test_health_check_promotes_and_hot_restarts(tmp_path: Path) -> None:
    from dana.updater.health_check import run_slot_health_check
    from dana.updater.slot_manager import SLOT_B, SlotManager

    base = tmp_path / "engine_slots"
    sm = SlotManager(base_dir=base, initial_version="0.1.0")
    pkg = _zip_bytes({"engine.py": b"ok\n", "VERSION": b"0.2.0\n"})
    candidate = sm.unpack_into_inactive(pkg, version="0.2.0")

    calls: list[str] = []

    class FakeClient:
        def hot_restart(self, **_kwargs):  # noqa: ANN003
            calls.append("hot_restart")
            return {"ok": True, "swap": True}

    result = run_slot_health_check(
        candidate,
        verify_fn=lambda p: (p / "VERSION").is_file(),
        ipc_client=FakeClient(),
        slot_manager=sm,
        failure_log=tmp_path / "failures.log",
        version="0.2.0",
    )
    assert result.ok is True
    assert result.switched is True
    assert sm.active_slot_name() == SLOT_B
    assert sm.get_active_version() == "0.2.0"
    assert calls == ["hot_restart"]


def test_ota_download_promotes_via_blue_green(tmp_path: Path) -> None:
    """OTAManifestManager download → unpack → health check (injectable)."""
    from dana.updater.manifest import OTAManifestManager

    payload = _zip_bytes({"engine.py": b"v2\n", "VERSION": b"2.0.0\n"})
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "version": "2.0.0",
        "sha256": digest,
        "package_url": "mem://pkg.zip",
    }
    urls = {
        "mem://latest.json": json.dumps(manifest).encode("utf-8"),
        "mem://pkg.zip": payload,
    }
    restarts: list[str] = []

    mgr = OTAManifestManager(
        manifest_url="mem://latest.json",
        staging_dir=tmp_path / "staging",
        fetch_fn=lambda u: urls[u],
        local_version="1.0.0",
        auto_update_mode="manual",
        slots_dir=tmp_path / "engine_slots",
        verify_fn=lambda p: (p / "VERSION").is_file(),
        ipc_client=lambda: restarts.append("hr") or {"ok": True},
        failure_log=tmp_path / "failures.log",
        auto_promote=False,
    )
    mgr.fetch_manifest()
    mgr.download_and_stage()
    st = mgr.state()
    assert st.staged_version == "2.0.0"
    assert "Slot A" in st.active_slot_label

    result = mgr.hot_apply()
    assert result["ok"] is True
    assert result["version"] == "2.0.0"
    assert result["active_slot"] == "slot_b"
    assert restarts == ["hr"]
    st2 = mgr.state()
    assert st2.staging_health == "healthy"
    assert st2.local_version == "2.0.0"
    assert "Slot B" in st2.active_slot_label


def test_ota_corrupt_promote_leaves_active(tmp_path: Path) -> None:
    from dana.updater.manifest import OTAManifestManager

    payload = _zip_bytes({"CORRUPT": b"1\n", "VERSION": b"9.0.0\n"})
    digest = hashlib.sha256(payload).hexdigest()
    urls = {
        "mem://latest.json": json.dumps(
            {
                "version": "9.0.0",
                "sha256": digest,
                "package_url": "mem://pkg.zip",
            }
        ).encode("utf-8"),
        "mem://pkg.zip": payload,
    }
    mgr = OTAManifestManager(
        manifest_url="mem://latest.json",
        staging_dir=tmp_path / "staging",
        fetch_fn=lambda u: urls[u],
        local_version="0.1.0",
        auto_update_mode="manual",
        slots_dir=tmp_path / "engine_slots",
        verify_fn=lambda _p: False,
        failure_log=tmp_path / "failures.log",
        auto_promote=False,
    )
    mgr.fetch_manifest()
    mgr.download_and_stage()
    with pytest.raises(RuntimeError):
        mgr.hot_apply()
    st = mgr.state()
    assert st.staging_health == "failed"
    assert st.active_slot == "slot_a"
    assert st.local_version == "0.1.0"
    assert (tmp_path / "failures.log").is_file()

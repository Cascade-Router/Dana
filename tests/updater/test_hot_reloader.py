"""Phase 1 — DynamicToolReloader + OTAManifestManager (offline)."""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_importlib_reload_updates_tool_definition_in_memory(tmp_path: Path) -> None:
    """``importlib.reload()`` updates tool callable in a registry dict without exceptions."""
    tools_root = tmp_path / "dana" / "tools" / "general"
    tools_root.mkdir(parents=True)
    (tools_root.parent / "__init__.py").write_text("", encoding="utf-8")
    (tools_root / "__init__.py").write_text("", encoding="utf-8")
    module_path = tools_root / "demo_hot_tool.py"
    module_path.write_text(
        textwrap.dedent(
            """
            def demo_hot_tool(text: str = "") -> str:
                return "v1:" + str(text)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    # Point reloader at tmp tools root (parent of general/).
    from dana.tools.hot_reloader import DynamicToolReloader

    registry: dict = {
        "demo_hot_tool": SimpleNamespace(callable=None, name="demo_hot_tool"),
    }
    reloader = DynamicToolReloader(
        tools_root=tools_root.parent,
        registry=registry,
    )
    reloaded = reloader.reload_tools(force=True, paths=[module_path], registry=registry)
    assert "dana.tools.general.demo_hot_tool" in reloaded
    assert callable(registry["demo_hot_tool"].callable)
    assert registry["demo_hot_tool"].callable("x") == "v1:x"

    # Mutate on disk and reload again — in-memory definition must update.
    module_path.write_text(
        textwrap.dedent(
            """
            def demo_hot_tool(text: str = "") -> str:
                return "v2:" + str(text)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    reloaded2 = reloader.reload_tools(force=True, paths=[module_path], registry=registry)
    assert "dana.tools.general.demo_hot_tool" in reloaded2
    assert registry["demo_hot_tool"].callable("x") == "v2:x"


def test_hot_reloader_skips_security_critical_modules(tmp_path: Path) -> None:
    from dana.tools.hot_reloader import DynamicToolReloader

    tools_root = tmp_path / "tools"
    tools_root.mkdir()
    guards = tools_root / "guards.py"
    guards.write_text("X = 1\n", encoding="utf-8")
    reloader = DynamicToolReloader(tools_root=tools_root, registry={})
    assert reloader.is_reload_safe("dana.tools.guards") is False
    result = reloader.reload_detailed(force=True, paths=[guards], registry={})
    assert "dana.tools.guards" in result.skipped
    assert result.reloaded == []


def test_checksum_validation_catches_corrupted_download(tmp_path: Path) -> None:
    from dana.updater.manifest import OTAManifestManager, validate_sha256

    good = b"healthy-package-bytes"
    digest = hashlib.sha256(good).hexdigest()
    assert validate_sha256(good, digest) is True
    assert validate_sha256(b"corrupted-package-bytes", digest) is False

    manifest = {
        "version": "9.9.9",
        "sha256": digest,
        "package_url": "https://example.test/pkg.bin",
    }
    urls = {
        "https://example.test/latest.json": json.dumps(manifest).encode("utf-8"),
        "https://example.test/pkg.bin": b"corrupted-package-bytes",
    }

    def fetch(url: str) -> bytes:
        return urls[url]

    mgr = OTAManifestManager(
        manifest_url="https://example.test/latest.json",
        staging_dir=tmp_path / "staging",
        fetch_fn=fetch,
        local_version="0.1.0",
        auto_update_mode="manual",
    )
    info = mgr.fetch_manifest()
    assert info.version == "9.9.9"
    assert mgr.is_update_available() is True

    with pytest.raises(ValueError, match="SHA256"):
        mgr.download_and_stage()
    assert mgr.state().staged_version == ""
    assert mgr.state().status == "checksum_failed"


def test_manifest_stages_valid_package_and_status_pill(tmp_path: Path) -> None:
    from dana.updater.manifest import OTAManifestManager

    payload = b"ok-patch-body"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "version": "1.2.3",
        "sha256": digest,
        "package_url": "mem://pkg.bin",
    }
    urls = {
        "mem://latest.json": json.dumps(manifest).encode("utf-8"),
        "mem://pkg.bin": payload,
    }
    mgr = OTAManifestManager(
        manifest_url="mem://latest.json",
        staging_dir=tmp_path / "staging",
        fetch_fn=lambda u: urls[u],
        local_version="1.0.0",
        auto_update_mode="manual",
    )
    mgr.fetch_manifest()
    dest = mgr.download_and_stage()
    assert dest.is_file()
    st = mgr.state()
    assert st.staged_version == "1.2.3"
    assert st.status_pill() == "[UPDATE READY: v1.2.3]"
    assert st.update_available is True


def test_get_local_version_reads_package() -> None:
    from dana.updater.manifest import get_local_version

    ver = get_local_version()
    assert isinstance(ver, str) and ver
    # Must be packaging-comparable.
    from packaging.version import Version

    Version(ver.lstrip("vV"))

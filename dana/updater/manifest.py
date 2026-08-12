"""OTA version manifest manager — fetch, compare, checksum, stage patches."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from packaging.version import InvalidVersion, Version

FetchFn = Callable[[str], bytes]
AutoUpdateMode = Literal["silent", "manual"]

DEFAULT_MANIFEST_URL = os.environ.get(
    "DANA_OTA_MANIFEST_URL",
    "https://github.com/dana-ai/dana/releases/latest/download/latest.json",
)


def dana_home() -> Path:
    """``~/.dana`` (Windows: ``%USERPROFILE%\\.dana``)."""
    override = os.environ.get("DANA_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".dana").resolve()


def default_staging_dir() -> Path:
    return dana_home() / "updates" / "staging"


def get_local_version() -> str:
    """Resolve installed Dānā version from package metadata or sources."""
    try:
        import dana as _dana

        ver = getattr(_dana, "__version__", None)
        if ver:
            return str(ver).strip()
    except Exception:  # noqa: BLE001
        pass

    try:
        from importlib.metadata import version

        return str(version("dana")).strip()
    except Exception:  # noqa: BLE001 — PackageNotFoundError or others
        pass

    # Fallback: pyproject.toml next to the dana package.
    try:
        root = Path(__file__).resolve().parents[2]
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            text = pyproject.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("version") and "=" in stripped:
                    _, _, rhs = stripped.partition("=")
                    return rhs.strip().strip("\"'")
    except Exception:  # noqa: BLE001
        pass
    return "0.0.0"


def _default_fetch(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "dana-ota/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return resp.read()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_sha256(data: bytes, expected: str) -> bool:
    """Return True when ``data`` matches the expected hex digest (case-insensitive)."""
    exp = (expected or "").strip().lower()
    if not exp:
        return False
    return sha256_bytes(data) == exp


@dataclass
class ManifestInfo:
    version: str
    sha256: str = ""
    package_url: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ManifestInfo:
        version = str(
            payload.get("version")
            or payload.get("tag_name")
            or payload.get("name")
            or ""
        ).strip().lstrip("vV")
        sha = str(
            payload.get("sha256")
            or payload.get("checksum")
            or payload.get("digest")
            or ""
        ).strip()
        if sha.lower().startswith("sha256:"):
            sha = sha.split(":", 1)[1].strip()
        pkg = str(
            payload.get("package_url")
            or payload.get("url")
            or payload.get("download_url")
            or payload.get("browser_download_url")
            or ""
        ).strip()
        return cls(version=version, sha256=sha, package_url=pkg, raw=dict(payload))


@dataclass
class OTAState:
    """Headless-safe snapshot for GUI / HUD bindings."""

    local_version: str = "0.0.0"
    remote_version: str = ""
    update_available: bool = False
    staged_version: str = ""
    staged_path: str = ""
    auto_update_mode: AutoUpdateMode = "manual"
    status: str = "idle"
    last_error: str = ""
    # Phase 2B — blue-green slot state.
    active_slot: str = ""
    active_slot_label: str = ""
    staging_health: str = "idle"

    def status_pill(self) -> str:
        if self.staging_health == "checking":
            return "[STAGING: HEALTH CHECK…]"
        if self.staging_health == "failed":
            return "[STAGING: FAILED — ROLLED BACK]"
        if self.staging_health == "healthy" and self.active_slot_label:
            return f"[ACTIVE: {self.active_slot_label}]"
        if self.staged_version:
            return f"[UPDATE READY: v{self.staged_version.lstrip('vV')}]"
        if self.update_available and self.remote_version:
            return f"[UPDATE AVAILABLE: v{self.remote_version.lstrip('vV')}]"
        if self.active_slot_label:
            return f"[ACTIVE: {self.active_slot_label}]"
        return "[UP TO DATE]"


class OTAManifestManager:
    """Fetch remote ``latest.json``, compare versions, stage checksum-verified patches."""

    def __init__(
        self,
        *,
        manifest_url: str | None = None,
        staging_dir: Path | str | None = None,
        fetch_fn: FetchFn | None = None,
        local_version: str | None = None,
        auto_update_mode: AutoUpdateMode | None = None,
        slot_manager: Any | None = None,
        slots_dir: Path | str | None = None,
        verify_fn: Callable[[Path], bool] | None = None,
        ipc_client: Any | None = None,
        failure_log: Path | str | None = None,
        auto_promote: bool | None = None,
    ) -> None:
        self.manifest_url = (manifest_url or DEFAULT_MANIFEST_URL).strip()
        self.staging_dir = Path(staging_dir) if staging_dir else default_staging_dir()
        self._fetch = fetch_fn or _default_fetch
        self._local_version = (
            str(local_version).strip() if local_version is not None else get_local_version()
        )
        mode = auto_update_mode
        if mode is None:
            mode = _load_auto_update_mode()
        self._auto_update_mode: AutoUpdateMode = (
            "silent" if str(mode).lower() == "silent" else "manual"
        )
        self._lock = threading.RLock()
        self._remote: ManifestInfo | None = None
        self._staged_version = ""
        self._staged_path: Path | None = None
        self._status = "idle"
        self._last_error = ""
        self._staging_health = "idle"
        self._verify_fn = verify_fn
        self._ipc_client = ipc_client
        self._failure_log = Path(failure_log) if failure_log else None
        self._auto_promote = auto_promote
        self._slot_manager = slot_manager
        self._slots_dir = Path(slots_dir) if slots_dir else None
        self._discover_staged()

    # --- mode / state -----------------------------------------------------

    @property
    def auto_update_mode(self) -> AutoUpdateMode:
        return self._auto_update_mode

    def set_auto_update_mode(self, mode: str) -> AutoUpdateMode:
        normalized: AutoUpdateMode = (
            "silent" if str(mode).strip().lower() == "silent" else "manual"
        )
        with self._lock:
            self._auto_update_mode = normalized
        _persist_auto_update_mode(normalized)
        return normalized

    def _get_slot_manager(self) -> Any:
        if self._slot_manager is not None:
            return self._slot_manager
        from dana.updater.slot_manager import SlotManager

        kwargs: dict[str, Any] = {"initial_version": self._local_version}
        if self._slots_dir is not None:
            kwargs["base_dir"] = self._slots_dir
        self._slot_manager = SlotManager(**kwargs)
        return self._slot_manager

    def state(self) -> OTAState:
        with self._lock:
            remote_ver = self._remote.version if self._remote else ""
            available = False
            if remote_ver:
                available = is_newer_version(remote_ver, self._local_version)
            active_slot = ""
            active_label = ""
            try:
                sm = self._get_slot_manager()
                active_slot = sm.active_slot_name()
                active_label = sm.display_active()
            except Exception:  # noqa: BLE001
                pass
            return OTAState(
                local_version=self._local_version,
                remote_version=remote_ver,
                update_available=available,
                staged_version=self._staged_version,
                staged_path=str(self._staged_path or ""),
                auto_update_mode=self._auto_update_mode,
                status=self._status,
                last_error=self._last_error,
                active_slot=active_slot,
                active_slot_label=active_label,
                staging_health=self._staging_health,
            )

    # --- fetch / compare --------------------------------------------------

    def fetch_manifest(self) -> ManifestInfo:
        with self._lock:
            self._status = "checking"
            self._last_error = ""
        try:
            raw = self._fetch(self.manifest_url)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("manifest root must be a JSON object")
            info = ManifestInfo.from_dict(payload)
            if not info.version:
                raise ValueError("manifest missing version")
            with self._lock:
                self._remote = info
                self._status = "manifest_ok"
            if self._auto_update_mode == "silent" and is_newer_version(
                info.version, self._local_version
            ):
                try:
                    self.download_and_stage()
                    # Silent mode: auto-promote via blue-green health gate.
                    if self._auto_promote is not False:
                        self.promote_staged_update()
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._last_error = f"{type(exc).__name__}: {exc}"
            return info
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._status = "error"
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def is_update_available(self) -> bool:
        info = self._remote
        if info is None:
            try:
                info = self.fetch_manifest()
            except Exception:  # noqa: BLE001
                return False
        return is_newer_version(info.version, self._local_version)

    # --- download / checksum / stage --------------------------------------

    def download_and_stage(self, *, package_url: str | None = None) -> Path:
        """Download package bytes, validate SHA256, write under staging/."""
        with self._lock:
            info = self._remote
            if info is None:
                raise RuntimeError("No remote manifest loaded; call fetch_manifest() first")
            url = (package_url or info.package_url or "").strip()
            expected = (info.sha256 or "").strip()
            version = info.version
            self._status = "downloading"
            self._last_error = ""
        if not url:
            raise ValueError("manifest missing package_url")
        if not expected:
            raise ValueError("manifest missing sha256")

        data = self._fetch(url)
        if not validate_sha256(data, expected):
            with self._lock:
                self._status = "checksum_failed"
                self._last_error = "SHA256 mismatch — refusing to stage corrupted package"
            raise ValueError("SHA256 mismatch — refusing to stage corrupted package")

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        dest = self.staging_dir / f"dana-{version}.bin"
        dest.write_bytes(data)
        meta = {
            "version": version,
            "sha256": expected.lower(),
            "package_url": url,
            "path": str(dest),
        }
        meta_path = self.staging_dir / "staged.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        # Optional zip extraction into staging/extracted/ when payload is a zip.
        extract_dir = self.staging_dir / "extracted" / version
        if _looks_like_zip(data):
            if extract_dir.exists():
                _rm_tree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(extract_dir)

        with self._lock:
            self._staged_version = version
            self._staged_path = dest
            self._status = "staged"
            self._staging_health = "idle"
        return dest

    def promote_staged_update(self) -> dict[str, Any]:
        """Unpack staged OTA into inactive slot → health-check → switch or rollback."""
        from dana.updater.health_check import run_slot_health_check

        with self._lock:
            staged_ver = self._staged_version
            staged_path = self._staged_path
            if not staged_ver or staged_path is None or not Path(staged_path).is_file():
                raise RuntimeError("No staged update ready to promote")
            self._status = "promoting"
            self._staging_health = "checking"
            self._last_error = ""

        sm = self._get_slot_manager()
        try:
            candidate = sm.unpack_into_inactive(Path(staged_path), version=staged_ver)
            hc = run_slot_health_check(
                candidate,
                verify_fn=self._verify_fn,
                ipc_client=self._ipc_client,
                slot_manager=sm,
                failure_log=self._failure_log,
                version=staged_ver,
            )
            payload = hc.to_dict() if hasattr(hc, "to_dict") else dict(hc)
            if payload.get("ok"):
                with self._lock:
                    self._local_version = staged_ver
                    self._status = "applied"
                    self._staging_health = "healthy"
                    self._last_error = ""
                    # Clear staged marker after successful promote.
                    self._staged_version = ""
                    self._staged_path = None
                return {
                    "ok": True,
                    "version": staged_ver,
                    "blue_green": payload,
                    "active_slot": sm.active_slot_name(),
                    "active_label": sm.display_active(),
                }
            err = str(payload.get("error") or "health check failed")
            with self._lock:
                self._status = "promote_failed"
                self._staging_health = "failed"
                self._last_error = err
            return {
                "ok": False,
                "version": staged_ver,
                "blue_green": payload,
                "error": err,
                "active_slot": sm.active_slot_name(),
                "active_label": sm.display_active(),
            }
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._status = "promote_failed"
                self._staging_health = "failed"
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def hot_apply(self) -> dict[str, Any]:
        """Promote staged update via blue-green gate, then reload tool modules."""
        with self._lock:
            staged_ver = self._staged_version
            staged_path = self._staged_path
            if not staged_ver or staged_path is None or not Path(staged_path).is_file():
                raise RuntimeError("No staged update ready to hot-apply")
            self._status = "applying"

        # Phase 2B — dual-slot promote + health-check rollback.
        bg = self.promote_staged_update()
        if not bg.get("ok"):
            err = str(bg.get("error") or "blue-green promote failed")
            with self._lock:
                self._status = "apply_failed"
                self._last_error = err
            raise RuntimeError(err)

        reloaded: list[str] = []
        try:
            extract_root = self.staging_dir / "extracted" / staged_ver
            if extract_root.is_dir():
                _overlay_tool_modules(extract_root)

            from dana.tools.hot_reloader import DynamicToolReloader

            reloader = DynamicToolReloader()
            reloaded = reloader.reload_tools(force=True)
            with self._lock:
                self._local_version = staged_ver
                self._status = "applied"
                self._staging_health = "healthy"
                self._last_error = ""
            return {
                "ok": True,
                "version": staged_ver,
                "reloaded": reloaded,
                "blue_green": bg.get("blue_green") or bg,
                "active_slot": bg.get("active_slot"),
                "active_label": bg.get("active_label"),
            }
        except Exception as exc:  # noqa: BLE001
            # Slot already switched — surface tool-reload failure without undoing promote.
            with self._lock:
                self._status = "applied"
                self._last_error = f"tools reload: {type(exc).__name__}: {exc}"
            return {
                "ok": True,
                "version": staged_ver,
                "reloaded": reloaded,
                "blue_green": bg.get("blue_green") or bg,
                "active_slot": bg.get("active_slot"),
                "active_label": bg.get("active_label"),
                "reload_error": f"{type(exc).__name__}: {exc}",
            }

    def _discover_staged(self) -> None:
        meta_path = self.staging_dir / "staged.json"
        if not meta_path.is_file():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ver = str(meta.get("version") or "").strip()
            path = Path(str(meta.get("path") or ""))
            if ver and path.is_file():
                expected = str(meta.get("sha256") or "").strip()
                if expected and not validate_sha256(path.read_bytes(), expected):
                    return
                self._staged_version = ver
                self._staged_path = path
                self._status = "staged"
        except Exception:  # noqa: BLE001
            return


# --- helpers --------------------------------------------------------------


def is_newer_version(remote: str, local: str) -> bool:
    try:
        return Version(_norm_ver(remote)) > Version(_norm_ver(local))
    except InvalidVersion:
        return _norm_ver(remote) != _norm_ver(local) and bool(remote)


def _norm_ver(value: str) -> str:
    return str(value or "0").strip().lstrip("vV") or "0"


def _looks_like_zip(data: bytes) -> bool:
    return len(data) >= 4 and data[:2] == b"PK"


def _rm_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _overlay_tool_modules(extract_root: Path) -> None:
    """Copy staged ``dana/tools/**/*.py`` overlays into the live package (safe subset)."""
    import shutil

    from dana.paths import PROJECT_ROOT

    tools_src = extract_root / "dana" / "tools"
    if not tools_src.is_dir():
        # Flat tools/ layout also accepted.
        tools_src = extract_root / "tools"
    if not tools_src.is_dir():
        return
    dest_root = PROJECT_ROOT / "dana" / "tools"
    skip_names = {
        "guards.py",
        "vault.py",
        "broker.py",
        "registry.py",
        "schema.py",
    }
    for path in tools_src.rglob("*.py"):
        if path.name in skip_names:
            continue
        rel = path.relative_to(tools_src)
        # Never overlay security plugin enforcers.
        parts = {p.lower() for p in rel.parts}
        if "file_jail_enforcer.py" in parts or "guards.py" in parts:
            continue
        target = dest_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _load_auto_update_mode() -> AutoUpdateMode:
    try:
        from dana.settings import load_dana_settings

        raw = str(load_dana_settings().get("auto_update_mode") or "manual").strip().lower()
        return "silent" if raw == "silent" else "manual"
    except Exception:  # noqa: BLE001
        return "manual"


def _persist_auto_update_mode(mode: AutoUpdateMode) -> None:
    try:
        from dana.settings import SETTINGS_PATH, load_dana_settings

        cfg = load_dana_settings(force_reload=True)
        cfg["auto_update_mode"] = mode
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except Exception:  # noqa: BLE001
        return


_manager_singleton: OTAManifestManager | None = None
_manager_lock = threading.Lock()


def get_ota_manager(*, reset: bool = False, **kwargs: Any) -> OTAManifestManager:
    """Process-wide OTAManifestManager singleton."""
    global _manager_singleton
    with _manager_lock:
        if _manager_singleton is None or reset or kwargs:
            _manager_singleton = OTAManifestManager(**kwargs)
        return _manager_singleton

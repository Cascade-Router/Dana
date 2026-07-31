"""Dual-slot (blue-green) engine directory rotator.

Windows-friendly active pointer: plain ``active_slot`` text file (``slot_a`` /
``slot_b``) plus optional ``active.json``. Symlink ``active`` is best-effort
when the OS allows it without elevation.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SlotName = Literal["slot_a", "slot_b"]
SLOT_A: SlotName = "slot_a"
SLOT_B: SlotName = "slot_b"
VALID_SLOTS: frozenset[str] = frozenset({SLOT_A, SLOT_B})


def dana_home() -> Path:
    override = os.environ.get("DANA_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".dana").resolve()


def default_slots_dir() -> Path:
    return dana_home() / "engine_slots"


@dataclass
class SlotInfo:
    name: SlotName
    path: Path
    version: str = ""


@dataclass
class SlotMetadata:
    active_slot: SlotName = SLOT_A
    active_version: str = ""
    slots: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_slot": self.active_slot,
            "active_version": self.active_version,
            "slots": self.slots,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SlotMetadata:
        active = str(raw.get("active_slot") or SLOT_A).strip().lower()
        if active not in VALID_SLOTS:
            active = SLOT_A
        return cls(
            active_slot=active,  # type: ignore[arg-type]
            active_version=str(raw.get("active_version") or "").strip(),
            slots=dict(raw.get("slots") or {}),
            history=list(raw.get("history") or []),
        )


class SlotManager:
    """Blue-green rotator over ``slot_a`` / ``slot_b`` under an injectable base dir."""

    def __init__(
        self,
        base_dir: Path | str | None = None,
        *,
        initial_version: str = "0.0.0",
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else default_slots_dir()
        self._lock = threading.RLock()
        self._initial_version = str(initial_version or "0.0.0").strip() or "0.0.0"
        self._ensure_layout()

    # --- paths ------------------------------------------------------------

    @property
    def active_slot_file(self) -> Path:
        return self.base_dir / "active_slot"

    @property
    def active_json_file(self) -> Path:
        return self.base_dir / "active.json"

    @property
    def metadata_file(self) -> Path:
        return self.base_dir / "metadata.json"

    @property
    def active_link(self) -> Path:
        """Optional symlink / junction named ``active`` (best-effort)."""
        return self.base_dir / "active"

    def slot_path(self, name: str) -> Path:
        key = self._normalize_slot(name)
        return self.base_dir / key

    # --- queries ----------------------------------------------------------

    def active_slot_name(self) -> SlotName:
        with self._lock:
            return self._read_pointer()

    def inactive_slot_name(self) -> SlotName:
        active = self.active_slot_name()
        return SLOT_B if active == SLOT_A else SLOT_A

    def active_slot_path(self) -> Path:
        return self.slot_path(self.active_slot_name())

    def inactive_slot_path(self) -> Path:
        return self.slot_path(self.inactive_slot_name())

    def get_active_version(self) -> str:
        with self._lock:
            meta = self._load_metadata()
            return meta.active_version or self._initial_version

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return self._load_metadata().to_dict()

    def display_active(self) -> str:
        """Human label like ``Slot A (v0.1.0)``."""
        name = self.active_slot_name()
        ver = self.get_active_version()
        label = "Slot A" if name == SLOT_A else "Slot B"
        return f"{label} (v{ver.lstrip('vV')})"

    # --- mutate -----------------------------------------------------------

    def unpack_into_inactive(
        self,
        package: Path | str | bytes,
        *,
        version: str,
    ) -> Path:
        """Wipe the inactive slot and unpack ``package`` into it. Returns slot path."""
        with self._lock:
            inactive = self.inactive_slot_name()
            dest = self.slot_path(inactive)
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)

            if isinstance(package, (bytes, bytearray)):
                data = bytes(package)
                self._extract_bytes(data, dest)
            else:
                pkg = Path(package)
                if not pkg.is_file():
                    raise FileNotFoundError(f"OTA package not found: {pkg}")
                data = pkg.read_bytes()
                if _looks_like_zip(data):
                    with zipfile.ZipFile(pkg) as zf:
                        zf.extractall(dest)
                else:
                    # Opaque blob — store as-is for health probes.
                    (dest / "package.bin").write_bytes(data)
                    (dest / "VERSION").write_text(
                        str(version).strip().lstrip("vV") + "\n",
                        encoding="utf-8",
                    )

            ver = str(version).strip().lstrip("vV")
            (dest / "VERSION").write_text(ver + "\n", encoding="utf-8")
            meta = self._load_metadata()
            meta.slots[inactive] = {
                "path": str(dest),
                "version": ver,
                "staged_at": time.time(),
            }
            meta.history.append(
                {
                    "ts": time.time(),
                    "event": "unpacked",
                    "slot": inactive,
                    "version": ver,
                }
            )
            self._save_metadata(meta)
            return dest

    def switch_active(self, to_slot: str, *, version: str | None = None) -> SlotName:
        """Atomically point the active slot at ``to_slot``."""
        with self._lock:
            target = self._normalize_slot(to_slot)
            prev = self._read_pointer()
            ver = (
                str(version).strip().lstrip("vV")
                if version is not None
                else self._slot_version(target)
            )
            self._write_pointer(target)
            meta = self._load_metadata()
            meta.active_slot = target
            if ver:
                meta.active_version = ver
            meta.slots.setdefault(target, {})
            meta.slots[target]["path"] = str(self.slot_path(target))
            if ver:
                meta.slots[target]["version"] = ver
            meta.history.append(
                {
                    "ts": time.time(),
                    "event": "switch",
                    "from": prev,
                    "to": target,
                    "version": ver,
                    "result": "promoted",
                }
            )
            self._save_metadata(meta)
            self._try_update_symlink(target)
            return target

    def wipe_slot(self, name: str) -> None:
        with self._lock:
            key = self._normalize_slot(name)
            if key == self._read_pointer():
                raise RuntimeError("Refusing to wipe the active slot")
            path = self.slot_path(key)
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
            meta = self._load_metadata()
            meta.slots[key] = {"path": str(path), "version": ""}
            meta.history.append(
                {
                    "ts": time.time(),
                    "event": "wipe",
                    "slot": key,
                    "result": "aborted",
                }
            )
            self._save_metadata(meta)

    # --- internals --------------------------------------------------------

    def _ensure_layout(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for name in (SLOT_A, SLOT_B):
            self.slot_path(name).mkdir(parents=True, exist_ok=True)
        if not self.active_slot_file.is_file() and not self.active_json_file.is_file():
            self._write_pointer(SLOT_A)
            meta = SlotMetadata(
                active_slot=SLOT_A,
                active_version=self._initial_version,
                slots={
                    SLOT_A: {
                        "path": str(self.slot_path(SLOT_A)),
                        "version": self._initial_version,
                    },
                    SLOT_B: {"path": str(self.slot_path(SLOT_B)), "version": ""},
                },
                history=[
                    {
                        "ts": time.time(),
                        "event": "init",
                        "slot": SLOT_A,
                        "version": self._initial_version,
                    }
                ],
            )
            self._save_metadata(meta)
            # Seed active slot VERSION marker.
            ver_file = self.slot_path(SLOT_A) / "VERSION"
            if not ver_file.is_file():
                ver_file.write_text(self._initial_version + "\n", encoding="utf-8")
            self._try_update_symlink(SLOT_A)
        elif not self.metadata_file.is_file():
            active = self._read_pointer()
            meta = SlotMetadata(
                active_slot=active,
                active_version=self._initial_version,
                slots={
                    SLOT_A: {
                        "path": str(self.slot_path(SLOT_A)),
                        "version": self._initial_version if active == SLOT_A else "",
                    },
                    SLOT_B: {
                        "path": str(self.slot_path(SLOT_B)),
                        "version": self._initial_version if active == SLOT_B else "",
                    },
                },
            )
            self._save_metadata(meta)

    def _normalize_slot(self, name: str) -> SlotName:
        key = str(name or "").strip().lower()
        if key in {"a", "slot-a", "slota"}:
            key = SLOT_A
        elif key in {"b", "slot-b", "slotb"}:
            key = SLOT_B
        if key not in VALID_SLOTS:
            raise ValueError(f"Invalid slot name: {name!r}")
        return key  # type: ignore[return-value]

    def _read_pointer(self) -> SlotName:
        # Prefer active_slot text file (portable on Windows).
        if self.active_slot_file.is_file():
            raw = self.active_slot_file.read_text(encoding="utf-8").strip().lower()
            if raw in VALID_SLOTS:
                return raw  # type: ignore[return-value]
        if self.active_json_file.is_file():
            try:
                payload = json.loads(self.active_json_file.read_text(encoding="utf-8"))
                raw = str(payload.get("active_slot") or "").strip().lower()
                if raw in VALID_SLOTS:
                    return raw  # type: ignore[return-value]
            except Exception:  # noqa: BLE001
                pass
        if self.metadata_file.is_file():
            try:
                meta = self._load_metadata()
                if meta.active_slot in VALID_SLOTS:
                    return meta.active_slot
            except Exception:  # noqa: BLE001
                pass
        return SLOT_A

    def _write_pointer(self, slot: SlotName) -> None:
        """Atomic write of active_slot + active.json."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # Text pointer — write temp then replace.
        self._atomic_write_text(self.active_slot_file, slot + "\n")
        payload = json.dumps({"active_slot": slot}, indent=2) + "\n"
        self._atomic_write_text(self.active_json_file, payload)

    def _atomic_write_text(self, path: Path, text: str) -> None:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(self.base_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            Path(tmp_name).replace(path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            # Fallback non-atomic.
            path.write_text(text, encoding="utf-8")

    def _try_update_symlink(self, slot: SlotName) -> None:
        """Best-effort ``active`` → slot directory link (skip on Windows without privilege)."""
        link = self.active_link
        target = self.slot_path(slot)
        try:
            if link.is_symlink() or link.exists():
                if link.is_dir() and not link.is_symlink():
                    # Never wipe a real directory named active.
                    return
                link.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            try:
                if link.exists() or link.is_symlink():
                    link.unlink()
            except OSError:
                return
        except OSError:
            return
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            # Windows may require admin / Developer Mode — ignore.
            return

    def _load_metadata(self) -> SlotMetadata:
        if not self.metadata_file.is_file():
            return SlotMetadata()
        try:
            raw = json.loads(self.metadata_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return SlotMetadata.from_dict(raw)
        except Exception:  # noqa: BLE001
            pass
        return SlotMetadata()

    def _save_metadata(self, meta: SlotMetadata) -> None:
        self._atomic_write_text(
            self.metadata_file,
            json.dumps(meta.to_dict(), indent=2) + "\n",
        )

    def _slot_version(self, name: SlotName) -> str:
        meta = self._load_metadata()
        slot_meta = meta.slots.get(name) or {}
        ver = str(slot_meta.get("version") or "").strip()
        if ver:
            return ver
        ver_file = self.slot_path(name) / "VERSION"
        if ver_file.is_file():
            return ver_file.read_text(encoding="utf-8").strip().lstrip("vV")
        return ""

    @staticmethod
    def _extract_bytes(data: bytes, dest: Path) -> None:
        if _looks_like_zip(data):
            tmp = dest / ".pkg.zip"
            tmp.write_bytes(data)
            try:
                with zipfile.ZipFile(tmp) as zf:
                    zf.extractall(dest)
            finally:
                tmp.unlink(missing_ok=True)  # type: ignore[call-arg]
        else:
            (dest / "package.bin").write_bytes(data)


def _looks_like_zip(data: bytes) -> bool:
    return len(data) >= 4 and data[:2] == b"PK"


_manager_singleton: SlotManager | None = None
_manager_lock = threading.Lock()


def get_slot_manager(*, reset: bool = False, **kwargs: Any) -> SlotManager:
    global _manager_singleton
    with _manager_lock:
        if _manager_singleton is None or reset or kwargs:
            _manager_singleton = SlotManager(**kwargs)
        return _manager_singleton

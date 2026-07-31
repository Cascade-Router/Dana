"""Dynamic tool reloader — importlib.reload user tool modules at runtime."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

# Security-critical / forge-adjacent modules — never hot-reload these.
_SKIP_MODULE_SUFFIXES: frozenset[str] = frozenset(
    {
        "dana.tools.guards",
        "dana.tools.vault",
        "dana.tools.broker",
        "dana.tools.registry",
        "dana.tools.schema",
        "dana.tools.plugins.file_jail_enforcer",
        "dana.tools.plugins.kill_watchdog",
    }
)

_SKIP_FILE_NAMES: frozenset[str] = frozenset(
    {
        "guards.py",
        "vault.py",
        "broker.py",
        "registry.py",
        "schema.py",
        "file_jail_enforcer.py",
        "kill_watchdog.py",
        "hot_reloader.py",
        "__init__.py",
    }
)

# Prefer reloading user / general / dynamic tool modules.
_PREFERRED_PREFIXES: tuple[str, ...] = (
    "dana.tools.general.",
    "dana.tools.custom.",
    "dana.tools.dynamic.",
    "dana.tools.plugins.",
)


@dataclass
class ReloadResult:
    reloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    registry_swaps: list[str] = field(default_factory=list)


class DynamicToolReloader:
    """Scan ``dana/tools/``, reload modified modules, swap registry callables."""

    def __init__(
        self,
        *,
        tools_root: Path | str | None = None,
        registry: MutableMapping[str, Any] | None = None,
    ) -> None:
        if tools_root is not None:
            self.tools_root = Path(tools_root)
        else:
            self.tools_root = Path(__file__).resolve().parent
        self._mtimes: dict[str, float] = {}
        self._lock = threading.RLock()
        self._registry_override = registry
        self._seed_mtimes()

    def _seed_mtimes(self) -> None:
        for path in self._iter_tool_files():
            try:
                self._mtimes[str(path)] = path.stat().st_mtime
            except OSError:
                continue

    def _iter_tool_files(self) -> list[Path]:
        if not self.tools_root.is_dir():
            return []
        out: list[Path] = []
        for path in sorted(self.tools_root.rglob("*.py")):
            if path.name in _SKIP_FILE_NAMES:
                continue
            # Skip caches / private trees.
            if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
                continue
            out.append(path)
        return out

    def module_name_for(self, path: Path) -> str | None:
        try:
            rel = path.resolve().relative_to(self.tools_root.resolve())
        except ValueError:
            return None
        parts = list(rel.with_suffix("").parts)
        if not parts:
            return None
        return "dana.tools." + ".".join(parts)

    def is_reload_safe(self, module_name: str) -> bool:
        name = str(module_name or "").strip()
        if not name or name in _SKIP_MODULE_SUFFIXES:
            return False
        for skipped in _SKIP_MODULE_SUFFIXES:
            if name == skipped or name.startswith(skipped + "."):
                return False
        # Prefer user/general/dynamic/plugin modules; allow other non-critical tools/*.py
        # at the tools root except skipped names.
        if name.startswith(_PREFERRED_PREFIXES):
            return True
        if name.startswith("dana.tools.") and name.count(".") == 2:
            # e.g. dana.tools.audio_switcher — OK unless skipped above
            return True
        return False

    def scan_modified(self) -> list[Path]:
        """Return tool module paths that are new or mtime-changed since last scan."""
        changed: list[Path] = []
        with self._lock:
            for path in self._iter_tool_files():
                key = str(path)
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                prev = self._mtimes.get(key)
                if prev is None or mtime > prev:
                    changed.append(path)
        return changed

    def reload_tools(
        self,
        *,
        force: bool = False,
        paths: Iterable[Path | str] | None = None,
        registry: MutableMapping[str, Any] | None = None,
    ) -> list[str]:
        """Reload modified (or forced) tool modules and swap registry callables.

        Returns the list of successfully reloaded module names.
        """
        result = self.reload_detailed(force=force, paths=paths, registry=registry)
        return list(result.reloaded)

    def reload_detailed(
        self,
        *,
        force: bool = False,
        paths: Iterable[Path | str] | None = None,
        registry: MutableMapping[str, Any] | None = None,
    ) -> ReloadResult:
        result = ReloadResult()
        if paths is not None:
            candidates = [Path(p) for p in paths]
        elif force:
            candidates = self._iter_tool_files()
        else:
            candidates = self.scan_modified()

        reg = registry if registry is not None else self._resolve_registry()

        with self._lock:
            for path in candidates:
                module_name = self.module_name_for(path)
                if module_name is None:
                    result.skipped.append(str(path))
                    continue
                if not self.is_reload_safe(module_name):
                    result.skipped.append(module_name)
                    continue
                try:
                    reloaded_mod = self._reload_module(module_name, path)
                    swaps = self._swap_registry_callables(reg, module_name, reloaded_mod)
                    result.reloaded.append(module_name)
                    result.registry_swaps.extend(swaps)
                    try:
                        self._mtimes[str(path)] = path.stat().st_mtime
                    except OSError:
                        self._mtimes[str(path)] = time.time()
                except Exception as exc:  # noqa: BLE001
                    result.errors[module_name] = f"{type(exc).__name__}: {exc}"
        return result

    def _reload_module(self, module_name: str, path: Path) -> Any:
        """Reload a tool module from disk (compile/exec — avoids stale import caches)."""
        importlib.invalidate_caches()
        _bump_mtime(path)

        # Prefer importlib.reload when the module was normally imported from this path.
        existing = sys.modules.get(module_name)
        if existing is not None:
            try:
                found = importlib.util.find_spec(module_name)
                origin = getattr(found, "origin", None) if found is not None else None
                if (
                    origin
                    and origin not in ("", "built-in")
                    and Path(origin).resolve() == path.resolve()
                ):
                    return importlib.reload(existing)
            except (ImportError, ModuleNotFoundError, ValueError, OSError):
                pass

        source = path.read_text(encoding="utf-8")
        code = compile(source, str(path), "exec")
        module = existing if existing is not None else ModuleType(module_name)
        parent_name, _, _ = module_name.rpartition(".")
        if parent_name:
            if parent_name not in sys.modules:
                try:
                    importlib.import_module(parent_name)
                except Exception:  # noqa: BLE001
                    pass
            module.__package__ = parent_name
        module.__file__ = str(path)
        module.__name__ = module_name
        sys.modules[module_name] = module
        exec(code, module.__dict__)  # noqa: S102 — intentional hot-reload of local tool source
        return module

    def _resolve_registry(self) -> MutableMapping[str, Any]:
        if self._registry_override is not None:
            return self._registry_override
        try:
            from dana.tools.registry import get_tool_registry

            return get_tool_registry().tools  # type: ignore[return-value]
        except Exception:  # noqa: BLE001
            return {}

    def _swap_registry_callables(
        self,
        registry: MutableMapping[str, Any],
        module_name: str,
        module: Any,
    ) -> list[str]:
        """Swap in-memory callables on a registry dict without full app restart."""
        swapped: list[str] = []
        if not isinstance(registry, Mapping):
            return swapped

        stem = module_name.rsplit(".", 1)[-1]
        candidates: list[tuple[str, Callable[..., Any]]] = []

        primary = getattr(module, stem, None)
        if callable(primary):
            candidates.append((stem, primary))
        # Also expose any TOOL_NAME / run entrypoints.
        for attr in ("run", "execute", "main", "TOOL_NAME"):
            obj = getattr(module, attr, None)
            if attr == "TOOL_NAME" and isinstance(obj, str):
                fn = getattr(module, obj, None)
                if callable(fn):
                    candidates.append((obj, fn))
                continue
            if callable(obj):
                candidates.append((stem if attr != "TOOL_NAME" else stem, obj))

        # LangChain @tool wrappers — unwrap .func when present.
        normalized: list[tuple[str, Callable[..., Any]]] = []
        for name, fn in candidates:
            if hasattr(fn, "func") and callable(getattr(fn, "func", None)):
                normalized.append((name, fn.func))
            else:
                normalized.append((name, fn))

        for tool_id, fn in normalized:
            entry = registry.get(tool_id)
            if entry is None:
                # Allow plain dict registries used in unit tests.
                if tool_id in registry or not hasattr(registry, "get"):
                    registry[tool_id] = fn
                    swapped.append(tool_id)
                continue
            if hasattr(entry, "callable"):
                entry.callable = fn
                swapped.append(tool_id)
            elif isinstance(entry, dict):
                entry["callable"] = fn
                swapped.append(tool_id)
            else:
                registry[tool_id] = fn
                swapped.append(tool_id)
        return swapped


def _bump_mtime(path: Path) -> None:
    """Ensure ``path`` mtime advances so SourceFileLoader re-reads source."""
    try:
        st = path.stat()
        os.utime(path, (st.st_atime, st.st_mtime + 1.0))
    except OSError:
        pass

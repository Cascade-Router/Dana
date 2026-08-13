"""Zero-touch plugin auto-discovery — scan ``dana/plugins/*/manifest.json``.

Each plugin subfolder ships a ``manifest.json`` (tool schemas + an entry
point module) and its own engine module. ``load_all_plugins()`` imports
each entry point via ``importlib``, resolves every declared tool id to a
callable on that module, and returns ``(ToolSpec, callable)`` pairs ready
to register into ``dana.tools.registry.ToolRegistry`` — no edits to
``broker.py``/``agent_loop.py`` needed to add a new plugin, just drop a
folder here and restart.

A single broken plugin (bad manifest, missing entry point, import error)
is skipped with a logged warning rather than taking down the whole tool
registry — see ``load_all_plugins``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

from dana.tools.schema import ToolParameterSpec, ToolSpec

PLUGINS_DIR = Path(__file__).resolve().parent

_cached_plugins: list[tuple[ToolSpec, Callable[..., Any]]] | None = None  # reassigned


class PluginLoadError(RuntimeError):
    """Raised for one plugin's manifest/entry-point failure; caught per-plugin."""


def _manifest_to_tool_spec(tool_def: dict[str, Any]) -> ToolSpec:
    params = tuple(
        ToolParameterSpec(
            name=str(p["name"]),
            type=str(p.get("type", "string")),
            required=bool(p.get("required", True)),
            enum=tuple(str(x) for x in (p.get("enum") or [])),
            description_en=str(p.get("description_en") or ""),
            description_fa=str(p.get("description_fa") or ""),
        )
        for p in (tool_def.get("parameters") or [])
    )
    return ToolSpec(
        id=str(tool_def["id"]),
        description_en=str(tool_def.get("description_en") or ""),
        description_fa=str(tool_def.get("description_fa") or ""),
        parameters=params,
        aliases_en={k: tuple(v) for k, v in (tool_def.get("aliases_en") or {}).items()},
        aliases_fa={k: tuple(v) for k, v in (tool_def.get("aliases_fa") or {}).items()},
    )


def _load_entry_module(plugin_dir: Path, entry_point: str, plugin_name: str) -> Any:
    entry_path = plugin_dir / entry_point
    if not entry_path.is_file():
        raise PluginLoadError(f"entry point not found: {entry_path}")
    module_name = f"dana.plugins.{plugin_name}.{entry_path.stem}"
    spec = importlib.util.spec_from_file_location(
        module_name, entry_path, submodule_search_locations=[str(plugin_dir)]
    )
    if spec is None or spec.loader is None:
        raise PluginLoadError(f"could not build import spec for {entry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(module_name, None)
        raise PluginLoadError(f"entry point {entry_path} raised on import: {exc}") from exc
    return module


def load_plugin(plugin_dir: Path) -> list[tuple[ToolSpec, Callable[..., Any]]]:
    """Load one plugin folder's manifest + entry point; raises ``PluginLoadError``."""
    manifest_path = plugin_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PluginLoadError(f"no manifest.json in {plugin_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PluginLoadError(f"invalid manifest.json in {plugin_dir}: {exc}") from exc

    plugin_name = str(manifest.get("name") or plugin_dir.name)
    entry_point = str(manifest.get("entry_point") or "engine.py")
    module = _load_entry_module(plugin_dir, entry_point, plugin_name)

    tools: list[tuple[ToolSpec, Callable[..., Any]]] = []
    for tool_def in manifest.get("tools") or []:
        tool_id = str(tool_def.get("id") or "").strip()
        if not tool_id:
            continue
        func_name = str(tool_def.get("function") or tool_id)
        func = getattr(module, func_name, None)
        if not callable(func):
            raise PluginLoadError(
                f"plugin {plugin_name!r}: entry point has no callable "
                f"{func_name!r} for tool {tool_id!r}"
            )
        tools.append((_manifest_to_tool_spec(tool_def), func))
    return tools


def discover_plugin_dirs(root: Path | None = None) -> list[Path]:
    """Every subfolder of ``root`` (default ``PLUGINS_DIR``) with a manifest.json."""
    base = root or PLUGINS_DIR
    return sorted(p.parent for p in base.glob("*/manifest.json"))


def load_all_plugins(
    *, force_refresh: bool = False, root: Path | None = None
) -> list[tuple[ToolSpec, Callable[..., Any]]]:
    """Scan every ``dana/plugins/*/`` subfolder and load its declared tools.

    Cached after the first default-root scan (``force_refresh=True`` or a
    non-default ``root`` bypasses the cache) since this re-imports every
    plugin's entry point module — cheap after the first call thanks to
    ``sys.modules``, but no reason to re-read every manifest.json on disk
    each time a broker is constructed.
    """
    global _cached_plugins
    if _cached_plugins is not None and not force_refresh and root is None:
        return _cached_plugins

    all_tools: list[tuple[ToolSpec, Callable[..., Any]]] = []
    for plugin_dir in discover_plugin_dirs(root):
        try:
            all_tools.extend(load_plugin(plugin_dir))
        except PluginLoadError as exc:
            print(f"[plugin_manager] WARNING: skipping plugin {plugin_dir.name!r}: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[plugin_manager] WARNING: unexpected error loading plugin "
                f"{plugin_dir.name!r}: {exc}",
                flush=True,
            )

    if root is None:
        _cached_plugins = all_tools
    return all_tools


__all__ = (
    "PluginLoadError",
    "discover_plugin_dirs",
    "load_all_plugins",
    "load_plugin",
)

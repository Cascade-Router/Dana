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
            items_type=str(p.get("items_type") or ""),
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
        # Safe-by-default: absent/false means HITL-gated — see ToolSpec.read_only's
        # own docstring for why a manifest plugin must opt OUT of gating, not in.
        read_only=bool(tool_def.get("read_only", False)),
    )


def _load_entry_module(
    plugin_dir: Path, entry_point: str, plugin_name: str, *, force_refresh: bool = False
) -> Any:
    """Import one plugin's entry-point file as ``dana.plugins.<plugin_name>.<stem>``.

    Reuses an already-loaded module for that dotted name from ``sys.modules``
    unless ``force_refresh`` is set. This is the fix for a real split-brain
    bug: ``load_all_plugins`` and ``load_all_plugins_grouped`` each keep
    their OWN cache (``_cached_plugins``/``_cached_plugins_grouped``), so
    whichever one happens to run its "first call" LATER in the process
    (e.g. the ``check_plugin_registry`` tool driving ``load_all_plugins``
    for the first time, well after ``react_dispatch.refresh_plugin_tools()``
    already populated ``load_all_plugins_grouped`` at import time) used to
    unconditionally re-exec every plugin's entry point via
    ``importlib.util.spec_from_file_location`` and overwrite
    ``sys.modules[module_name]`` with a brand-new module object — silently
    orphaning every function reference the OTHER cache had already bound
    (``dana.core.react_dispatch.TOOL_HANDLERS``'s manifest-plugin handlers
    included). Confirmed live: a test patching
    ``dana.plugins.freecad.engine._run_freecad_script`` after
    ``check_plugin_registry`` had fired once silently patched an orphaned
    module twin, and the REAL FreeCADCmd binary ran instead of the mock.
    Explicit ``force_refresh=True`` (a genuine hot-reload request) still
    re-execs, matching ``refresh_plugin_tools()``'s documented "safe to call
    again later" contract.
    """
    entry_path = plugin_dir / entry_point
    if not entry_path.is_file():
        raise PluginLoadError(f"entry point not found: {entry_path}")
    module_name = f"dana.plugins.{plugin_name}.{entry_path.stem}"
    if not force_refresh:
        existing = sys.modules.get(module_name)
        if existing is not None:
            return existing
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


def _load_plugin_full(
    plugin_dir: Path, *, force_refresh: bool = False
) -> tuple[str, list[tuple[ToolSpec, Callable[..., Any]]]]:
    """Shared body for ``load_plugin``/``load_all_plugins_grouped`` — loads
    the manifest + entry point exactly once and also returns the plugin's
    declared capability DOMAIN (``manifest["domain"]``, falling back to the
    plugin's own name), which ``dana.core.react_dispatch``'s generic plugin
    dispatch wiring groups this plugin's tools under in
    ``_CAPABILITY_TOOL_IDS`` — the piece that makes "no react_dispatch.py
    edits needed per plugin" actually true. Kept private/split out from
    ``load_plugin`` specifically so a caller that needs the domain too
    never re-parses the manifest or re-execs the entry module a second time.
    """
    manifest_path = plugin_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PluginLoadError(f"no manifest.json in {plugin_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PluginLoadError(f"invalid manifest.json in {plugin_dir}: {exc}") from exc

    plugin_name = str(manifest.get("name") or plugin_dir.name)
    domain = str(manifest.get("domain") or plugin_name)
    entry_point = str(manifest.get("entry_point") or "engine.py")
    module = _load_entry_module(plugin_dir, entry_point, plugin_name, force_refresh=force_refresh)

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
    return domain, tools


def load_plugin(plugin_dir: Path, *, force_refresh: bool = False) -> list[tuple[ToolSpec, Callable[..., Any]]]:
    """Load one plugin folder's manifest + entry point; raises ``PluginLoadError``."""
    _domain, tools = _load_plugin_full(plugin_dir, force_refresh=force_refresh)
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
            all_tools.extend(load_plugin(plugin_dir, force_refresh=force_refresh))
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


_cached_plugins_grouped: dict[str, list[tuple[ToolSpec, Callable[..., Any]]]] | None = None


def load_all_plugins_grouped(
    *, force_refresh: bool = False, root: Path | None = None
) -> dict[str, list[tuple[ToolSpec, Callable[..., Any]]]]:
    """Every discovered plugin's tools, grouped by capability DOMAIN (each
    manifest's own ``"domain"`` field) — the entry point
    ``dana.core.react_dispatch.refresh_plugin_tools`` merges into
    ``TOOL_HANDLERS``/``_CAPABILITY_TOOL_IDS``/this turn's ``tools=`` schema,
    so a new plugin folder needs zero edits there to become reachable by
    the live ReAct loop. Cached separately from ``load_all_plugins`` (that
    one's own cache/shape is untouched, still used by
    ``plugin_registry_view``'s introspection) — the two caches' own result
    SHAPES stay independent, but ``_load_entry_module`` reuses whichever
    module ``sys.modules`` already holds for a given plugin unless
    ``force_refresh`` is set, so a "first call" to this function and a
    later "first call" to ``load_all_plugins`` (or vice versa) share the
    same module instance instead of each re-execing its own throwaway
    twin — see ``_load_entry_module``'s docstring for the bug this fixes.
    """
    global _cached_plugins_grouped
    if _cached_plugins_grouped is not None and not force_refresh and root is None:
        return _cached_plugins_grouped

    grouped: dict[str, list[tuple[ToolSpec, Callable[..., Any]]]] = {}
    for plugin_dir in discover_plugin_dirs(root):
        try:
            domain, tools = _load_plugin_full(plugin_dir, force_refresh=force_refresh)
        except PluginLoadError as exc:
            print(f"[plugin_manager] WARNING: skipping plugin {plugin_dir.name!r}: {exc}", flush=True)
            continue
        except Exception as exc:  # noqa: BLE001
            print(
                f"[plugin_manager] WARNING: unexpected error loading plugin "
                f"{plugin_dir.name!r}: {exc}",
                flush=True,
            )
            continue
        grouped.setdefault(domain, []).extend(tools)

    if root is None:
        _cached_plugins_grouped = grouped
    return grouped


__all__ = (
    "PluginLoadError",
    "discover_plugin_dirs",
    "load_all_plugins",
    "load_all_plugins_grouped",
    "load_plugin",
)

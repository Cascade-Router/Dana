"""Semantic Tool Registry — O(1) execution lookup + local vector RAG retrieval.

``ToolRegistry.tools[name]`` is a plain dict (O(1) hot-path dispatch).
Schemas/descriptions are embedded into a local vector index so the agentic loop
can inject only the top-K relevant tool schemas into the LLM context.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dana.paths import (
    CUSTOM_TOOLS_DIR,
    GENERAL_TOOLS_DIR,
    REPO_CUSTOM_TOOLS_DIR,
    SECURITY_POLICY_PATH,
    TOOLS_JSON,
)
from dana.tools.schema import ToolSpec, load_tool_registry

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.I)
_EMBED_DIM = 384
_DEFAULT_TOP_K = 6
# Sources that live under CAMGRASPER/custom_tools and are wiped on context reset.
_CUSTOM_TOOL_SOURCES = frozenset({"forge", "custom", "dynamic"})


@dataclass
class RegisteredTool:
    """Runtime tool entry: schema + optional callable for O(1) execution."""

    name: str
    spec: ToolSpec
    description: str
    schema_text: str
    callable: Callable[..., Any] | None = None
    source: str = "tools.json"  # tools.json | forge | custom | general | dynamic
    metadata: dict[str, Any] = field(default_factory=dict)
    ephemeral: bool = False

    @property
    def is_ephemeral(self) -> bool:
        if self.ephemeral:
            return True
        return bool(self.metadata.get("ephemeral"))


def _custom_tools_roots() -> list[Path]:
    """Resolved directories treated as custom/ephemeral tool storage.

    Canonical root is ``CUSTOM_TOOLS_DIR`` only. Legacy mirrors
    (``dana/tools/custom``, root ``generated_tools``) are scanned for wipe
    cleanup if they still exist, but forge writes never go there.
    """
    roots: list[Path] = []
    try:
        roots.append(CUSTOM_TOOLS_DIR.resolve())
    except OSError:
        roots.append(CUSTOM_TOOLS_DIR)
    # Cleanup-only fallbacks for leftover pre-migration files.
    for candidate in (
        REPO_CUSTOM_TOOLS_DIR,
        CUSTOM_TOOLS_DIR.parent / "generated_tools",
    ):
        try:
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved not in roots:
            roots.append(resolved)
    return roots


def cleanup_ephemeral_tools(*, archive: bool = True) -> list[str]:
    """On clean shutdown: unregister + archive/delete session-forged tools.

    Returns the list of tool ids cleaned. Never touches ``source=general`` tools.
    """
    from dana.paths import CUSTOM_TOOLS_ARCHIVE_DIR

    registry = get_tool_registry()
    cleaned: list[str] = []
    roots = {p.resolve() for p in _custom_tools_roots()}
    with registry._lock:
        targets = [
            entry
            for entry in list(registry.tools.values())
            if entry.source != "general"
            and (entry.source in _CUSTOM_TOOL_SOURCES or entry.is_ephemeral)
        ]
    for entry in targets:
        path_str = str((entry.metadata or {}).get("path") or "")
        path = Path(path_str) if path_str else (CUSTOM_TOOLS_DIR / f"{entry.name}.py")
        try:
            parent = path.parent.resolve() if path.is_file() else None
            if path.is_file() and parent in roots:
                if archive:
                    CUSTOM_TOOLS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
                    dest = CUSTOM_TOOLS_ARCHIVE_DIR / path.name
                    if dest.exists():
                        stamp = time.strftime("%Y%m%dT%H%M%S")
                        dest = CUSTOM_TOOLS_ARCHIVE_DIR / f"{path.stem}_{stamp}{path.suffix}"
                    path.replace(dest)
                else:
                    path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        registry.unregister(entry.name)
        cleaned.append(entry.name)
    return cleaned


def wipe_custom_tools(*, reason: str = "context_wipe") -> list[str]:
    """Context-wipe failsafe: delete custom ``.py`` tools, unregister, clear sys.modules.

    Keeps ``__init__.py``. Does **not** touch ``dana/tools/general/``.
    Returns the list of tool module stems removed from disk / registry.
    """
    import sys

    from dana.paths import ensure_workspace_on_syspath

    ensure_workspace_on_syspath()
    registry = get_tool_registry()
    wiped: list[str] = []
    roots = _custom_tools_roots()

    # 1) Delete .py files on disk (except __init__.py).
    for root in roots:
        try:
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.py")):
                if path.name == "__init__.py":
                    continue
                stem = path.stem
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    continue
                wiped.append(stem)
        except OSError:
            continue

    # 2) Unregister forge/custom/ephemeral entries from the live registry.
    def _path_is_custom(path_str: str) -> bool:
        norm = path_str.replace("\\", "/")
        return any(
            marker in norm
            for marker in (
                "/custom_tools/",
                "/generated_tools/",
                "/tools/custom/",
            )
        )

    with registry._lock:
        names = [
            entry.name
            for entry in list(registry.tools.values())
            if entry.source in _CUSTOM_TOOL_SOURCES
            or entry.is_ephemeral
            or _path_is_custom(str((entry.metadata or {}).get("path") or ""))
        ]
    for name in names:
        if name == "publish_tool_to_general":
            continue
        registry.unregister(name)
        if name not in wiped:
            wiped.append(name)

    # 3) Drop cached import modules.
    prefixes = (
        "custom_tools.",
        "generated_tools.",
        "dana.tools.custom.",
    )
    doomed = [
        mod
        for mod in list(sys.modules)
        if mod in ("custom_tools", "generated_tools", "dana.tools.custom")
        or any(mod.startswith(p) for p in prefixes)
    ]
    for mod in doomed:
        sys.modules.pop(mod, None)

    unique = sorted(set(wiped))
    try:
        from dana.logging import log

        log(
            "ToolRegistry",
            f"Custom tools wiped ({reason}): removed {len(unique)} tool(s) "
            f"{unique!r} under {CUSTOM_TOOLS_DIR}",
        )
    except Exception:  # noqa: BLE001
        print(
            f"[ToolRegistry] Custom tools wiped ({reason}): {unique!r}",
            flush=True,
        )
    return unique


def _dynamic_source_available(tool_id: str) -> bool:
    """True when a dynamic tool has executable source on disk."""
    name = str(tool_id or "").strip()
    if not name:
        return False
    try:
        from dana_security import load_dynamic_source

        if load_dynamic_source(name):
            return True
    except Exception:  # noqa: BLE001
        pass
    for root in (GENERAL_TOOLS_DIR, CUSTOM_TOOLS_DIR):
        try:
            if (Path(root) / f"{name}.py").is_file():
                return True
        except OSError:
            continue
    return False


def is_bindable_tool(entry: RegisteredTool) -> bool:
    """False for phantom ``dynamic: true`` tools with no source (must not reach Llama)."""
    if entry.callable is not None:
        return True
    is_dynamic = bool(entry.spec.dynamic) or bool(entry.metadata.get("dynamic"))
    if not is_dynamic:
        return True
    return _dynamic_source_available(entry.name)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t]


def hash_embed(text: str, *, dim: int = _EMBED_DIM) -> np.ndarray:
    """Deterministic bag-of-tokens hashing embedder (no external model required)."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        vec[idx] += sign * weight
    # Bigrams for phrase sensitivity.
    for a, b in zip(tokens, tokens[1:]):
        digest = hashlib.sha256(f"{a}_{b}".encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign * 0.5
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        vec /= norm
    return vec


def _spec_schema_text(spec: ToolSpec) -> str:
    parts = [
        spec.id,
        spec.description_en or "",
        spec.description_fa or "",
    ]
    for p in spec.parameters:
        parts.append(p.name)
        parts.append(p.description_en or "")
        parts.append(p.type)
    for values in (spec.aliases_en or {}).values():
        parts.extend(values)
    for values in (spec.aliases_fa or {}).values():
        parts.extend(values)
    return " ".join(str(x) for x in parts if x)


class _VectorIndex:
    """Local vector store: FAISS when available, else NumPy cosine search."""

    def __init__(self, dim: int = _EMBED_DIM) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._faiss = None
        try:
            import faiss  # type: ignore

            self._faiss_mod = faiss
            self._faiss = faiss.IndexFlatIP(dim)
        except Exception:  # noqa: BLE001
            self._faiss_mod = None
            self._faiss = None

    def clear(self) -> None:
        self._ids = []
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        if self._faiss_mod is not None:
            self._faiss = self._faiss_mod.IndexFlatIP(self.dim)

    def add(self, tool_id: str, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if tool_id in self._ids:
            idx = self._ids.index(tool_id)
            self._matrix[idx] = v[0]
            self._rebuild_faiss()
            return
        self._ids.append(tool_id)
        if self._matrix.size == 0:
            self._matrix = v.copy()
        else:
            self._matrix = np.vstack([self._matrix, v])
        if self._faiss is not None:
            self._faiss.add(v)

    def remove(self, tool_id: str) -> None:
        if tool_id not in self._ids:
            return
        idx = self._ids.index(tool_id)
        self._ids.pop(idx)
        self._matrix = np.delete(self._matrix, idx, axis=0)
        self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:
        if self._faiss_mod is None:
            self._faiss = None
            return
        self._faiss = self._faiss_mod.IndexFlatIP(self.dim)
        if self._matrix.size:
            self._faiss.add(self._matrix)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        k = max(1, min(int(k), len(self._ids)))
        if self._faiss is not None and self._faiss.ntotal > 0:
            scores, idxs = self._faiss.search(q, k)
            out: list[tuple[str, float]] = []
            for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
                if i < 0 or i >= len(self._ids):
                    continue
                out.append((self._ids[i], float(score)))
            return out
        # NumPy cosine (vectors already L2-normalized).
        sims = (self._matrix @ q.T).reshape(-1)
        order = np.argsort(-sims)[:k]
        return [(self._ids[int(i)], float(sims[int(i)])) for i in order]


class ToolRegistry:
    """O(1) tool execution map + semantic top-K retrieval for prompt injection."""

    def __init__(self) -> None:
        self.tools: dict[str, RegisteredTool] = {}
        self._index = _VectorIndex()
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self.tools.clear()
            self._index.clear()

    def register(
        self,
        spec: ToolSpec,
        *,
        callable: Callable[..., Any] | None = None,
        source: str = "tools.json",
        metadata: Mapping[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> RegisteredTool:
        schema_text = _spec_schema_text(spec)
        description = (spec.description_en or spec.id).strip()
        meta = dict(metadata or {})
        if spec.dynamic:
            meta["dynamic"] = True
        if ephemeral or source in ("forge", "custom"):
            meta.setdefault("ephemeral", True)
            ephemeral = True
        if source == "general":
            meta["ephemeral"] = False
            ephemeral = False
        entry = RegisteredTool(
            name=spec.id,
            spec=spec,
            description=description,
            schema_text=schema_text,
            callable=callable,
            source=source,
            metadata=meta,
            ephemeral=ephemeral,
        )
        with self._lock:
            self.tools[spec.id] = entry
            self._index.add(spec.id, hash_embed(schema_text))
        return entry

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name not in self.tools:
                return False
            del self.tools[name]
            self._index.remove(name)
            return True

    def get(self, name: str) -> RegisteredTool | None:
        return self.tools.get(name)

    def execute(self, name: str, **kwargs: Any) -> Any:
        """O(1) dispatch into a registered callable."""
        entry = self.tools.get(name)
        if entry is None:
            raise KeyError(f"ToolNotFound: {name}")
        if entry.callable is None:
            raise RuntimeError(f"Tool {name!r} has no bound callable")
        return entry.callable(**kwargs)

    def retrieve(self, query: str, *, k: int = _DEFAULT_TOP_K) -> list[RegisteredTool]:
        """Embed ``query`` and return top-K registered tools by semantic score."""
        with self._lock:
            # Over-fetch then drop phantom dynamic tools so Llama never sees them.
            hits = self._index.search(hash_embed(query), max(k * 3, k))
            out: list[RegisteredTool] = []
            for tool_id, _score in hits:
                entry = self.tools.get(tool_id)
                if entry is None or not is_bindable_tool(entry):
                    continue
                out.append(entry)
                if len(out) >= k:
                    break
            return out

    def retrieve_specs(
        self,
        query: str,
        *,
        k: int = _DEFAULT_TOP_K,
        always_include: Iterable[str] | None = None,
    ) -> dict[str, ToolSpec]:
        """Top-K specs for LLM bind_tools, plus any forced tool ids."""
        selected: dict[str, ToolSpec] = {}
        for entry in self.retrieve(query, k=k):
            selected[entry.name] = entry.spec
        for tid in always_include or ():
            entry = self.tools.get(str(tid))
            if entry is not None and is_bindable_tool(entry):
                selected[entry.name] = entry.spec
        return selected

    def load_from_tools_json(self, path: str | Path | None = None) -> int:
        """Bulk-load ``tools.json`` into the registry + vector index."""
        registry = load_tool_registry(str(path) if path else None)
        n = 0
        for spec in registry.values():
            # Skip phantom dynamic stubs with no source — they poison semantic RAG.
            if spec.dynamic and not _dynamic_source_available(spec.id):
                continue
            self.register(
                spec,
                source="dynamic" if spec.dynamic else "tools.json",
                metadata={"dynamic": True} if spec.dynamic else None,
            )
            n += 1
        return n

    def as_spec_dict(self) -> dict[str, ToolSpec]:
        with self._lock:
            return {
                name: entry.spec
                for name, entry in self.tools.items()
                if is_bindable_tool(entry)
            }

    def public_schemas(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Compact schemas suitable for prompt injection."""
        with self._lock:
            ids = list(names) if names is not None else list(self.tools.keys())
            out: list[dict[str, Any]] = []
            for tid in ids:
                entry = self.tools.get(tid)
                if entry is None:
                    continue
                spec = entry.spec
                out.append(
                    {
                        "id": spec.id,
                        "description": entry.description,
                        "parameters": [
                            {
                                "name": p.name,
                                "type": p.type,
                                "required": p.required,
                                "enum": list(p.enum),
                            }
                            for p in spec.parameters
                        ],
                    }
                )
            return out


_registry_singleton: ToolRegistry | None = None
_registry_lock = threading.Lock()


def get_tool_registry(*, reload: bool = False) -> ToolRegistry:
    """Process-wide ToolRegistry singleton (lazy-loads tools.json once)."""
    global _registry_singleton
    with _registry_lock:
        if _registry_singleton is None or reload:
            reg = ToolRegistry()
            reg.load_from_tools_json(TOOLS_JSON)
            _registry_singleton = reg
        return _registry_singleton


def load_security_policy(path: str | Path | None = None) -> dict[str, Any]:
    policy_path = Path(path) if path else SECURITY_POLICY_PATH
    with open(policy_path, encoding="utf-8") as fh:
        return json.load(fh)


def _bind_module_callable(module: Any, name: str) -> Any | None:
    """Resolve the public entry callable for a tool module."""
    callable_obj = getattr(module, name, None)
    if callable_obj is None:
        for attr in dir(module):
            if attr.startswith("_"):
                continue
            obj = getattr(module, attr)
            if callable(obj):
                callable_obj = obj
                break
    if hasattr(callable_obj, "func") and callable(getattr(callable_obj, "func", None)):
        callable_obj = callable_obj.func
    return callable_obj if callable(callable_obj) else None


def load_general_tools_from_disk() -> list[str]:
    """Hot-load every ``*.py`` under ``dana/tools/general/`` (except ``__init__``).

    Registers with ``source=general`` / ``ephemeral=False``. Returns loaded names.
    """
    import importlib.util
    import sys

    from dana.tools.schema import ToolParameterSpec, ToolSpec

    ensure_general_tools_package()
    registry = get_tool_registry()
    loaded: list[str] = []
    for path in sorted(GENERAL_TOOLS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        name = path.stem
        module_name = f"dana.tools.general.{name}"
        try:
            sys.modules.pop(module_name, None)
            spec = importlib.util.spec_from_file_location(
                module_name,
                path,
                submodule_search_locations=[str(GENERAL_TOOLS_DIR)],
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            callable_obj = _bind_module_callable(module, name)
            existing = registry.get(name)
            description = (
                (existing.description if existing else None)
                or f"General tool `{name}`"
            )
            # Preserve tools.json parameter / alias schemas when present so
            # promoted general tools keep full LLM-facing signatures.
            if existing is not None and existing.spec.parameters:
                parameters = existing.spec.parameters
                aliases_en = existing.spec.aliases_en or {
                    "_intent": (name.replace("_", " "),)
                }
                aliases_fa = existing.spec.aliases_fa or {"_intent": (name,)}
                description_fa = existing.spec.description_fa or f"  `{name}`"
                description_en = existing.spec.description_en or description
            else:
                parameters = (
                    ToolParameterSpec(
                        name="text",
                        type="string",
                        required=False,
                        description_en="Primary text input.",
                    ),
                )
                aliases_en = {"_intent": (name.replace("_", " "),)}
                aliases_fa = {"_intent": (name,)}
                description_fa = f"  `{name}`"
                description_en = description
            tool_spec = ToolSpec(
                id=name,
                description_en=description_en,
                description_fa=description_fa,
                parameters=parameters,
                aliases_en=aliases_en,
                aliases_fa=aliases_fa,
            )
            registry.register(
                tool_spec,
                callable=callable_obj,
                source="general",
                ephemeral=False,
                metadata={
                    "path": str(path),
                    "module": module_name,
                    "ephemeral": False,
                    "tier": "general",
                },
            )
            loaded.append(name)
        except Exception:  # noqa: BLE001
            continue
    return loaded


def load_custom_tools_from_disk() -> list[str]:
    """Hot-load every ``*.py`` under ``custom_tools/`` (sole forge root).

    Registers with ``source=custom`` / ``ephemeral=True``. Skips ``__init__.py``.
    Does not touch ``dana/tools/general/``. Returns loaded names.
    """
    import importlib.util
    import sys

    from dana.paths import ensure_workspace_on_syspath
    from dana.tools.schema import ToolParameterSpec, ToolSpec

    ensure_custom_tools_package()
    ensure_workspace_on_syspath()

    registry = get_tool_registry()
    loaded: list[str] = []
    seen: set[str] = set()

    # Sole load root — do not dual-load dana/tools/custom or generated_tools.
    roots: list[tuple[Path, str]] = [(CUSTOM_TOOLS_DIR, "custom_tools")]

    for root, pkg_prefix in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.py")):
            if path.name == "__init__.py":
                continue
            name = path.stem
            if name in seen:
                continue
            # Never let a custom copy shadow a promoted general tool.
            existing = registry.get(name)
            if existing is not None and existing.source == "general":
                continue
            module_name = f"{pkg_prefix}.{name}"
            try:
                sys.modules.pop(module_name, None)
                if pkg_prefix == "custom_tools":
                    sys.modules.pop("custom_tools", None)
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    path,
                    submodule_search_locations=[str(root)],
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                callable_obj = _bind_module_callable(module, name)
                description = (
                    (existing.description if existing else None)
                    or f"Custom forged tool `{name}`"
                )
                tool_spec = ToolSpec(
                    id=name,
                    description_en=description,
                    description_fa=f"  `{name}`",
                    parameters=(
                        ToolParameterSpec(
                            name="text",
                            type="string",
                            required=False,
                            description_en="Primary text input.",
                        ),
                    ),
                    aliases_en={"_intent": (name.replace("_", " "),)},
                    aliases_fa={"_intent": (name,)},
                )
                registry.register(
                    tool_spec,
                    callable=callable_obj,
                    source="custom",
                    ephemeral=True,
                    metadata={
                        "path": str(path),
                        "module": module_name,
                        "ephemeral": True,
                        "tier": "custom",
                    },
                )
                seen.add(name)
                loaded.append(name)
            except Exception:  # noqa: BLE001
                continue
    return loaded


def ensure_generated_tools_package() -> Path:
    """Create Desktop ``Donna/custom_tools/`` and put it on ``sys.path``."""
    from dana.workspace import ensure_custom_tools_package as _ensure

    return _ensure()


def ensure_custom_tools_package() -> Path:
    """Create Desktop ``Donna/custom_tools/`` and put it on ``sys.path``."""
    from dana.workspace import ensure_custom_tools_package as _ensure

    return _ensure()


def ensure_general_tools_package() -> Path:
    """Ensure repo ``dana/tools/general/`` exists."""
    from dana.workspace import ensure_general_tools_package as _ensure

    return _ensure()

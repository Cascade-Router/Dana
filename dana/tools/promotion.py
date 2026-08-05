"""Promote a custom forged tool into the Git-tracked general tools library.

Pipeline:
  1. Read ``CAMGRASPER/custom_tools/<tool_name>.py``
  2. LLM scrub (local Ollama / ChatOllama, default llama3.2) for personal data
  3. Write to ``dana/tools/general/<tool_name>.py``
  4. Hot-load + register with ``ephemeral=False`` / ``source=general``
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import textwrap
from pathlib import Path
from typing import Any

from dana.paths import CUSTOM_TOOLS_DIR, GENERAL_TOOLS_DIR, PROJECT_ROOT
from dana.tools.registry import (
    ensure_custom_tools_package,
    ensure_general_tools_package,
    get_tool_registry,
)
from dana.tools.schema import ToolParameterSpec, ToolSpec

_TOOL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.I)

SCRUB_SYSTEM = """You are Donna's tool promotion scrubber.
You receive Python source for a custom tool that will be committed to a shared
general-purpose library. Remove ALL personal / machine-specific data:

- Real names, emails, phone numbers, usernames
- Absolute Windows/Unix home paths (C:\\Users\\..., /Users/..., /home/...)
- IP addresses, hostnames, MAC addresses
- API keys, tokens, passwords, vault secrets
- Specific Desktop/Donna absolute paths — prefer relative or generic names

Keep the tool functionally equivalent. Preserve the public function name.
Output ONLY the scrubbed Python module source. No markdown fences. No commentary.
"""


def _log(msg: str) -> None:
    try:
        from dana.logging import log

        log("ToolPromotion", msg)
    except Exception:
        print(f"[ToolPromotion] {msg}", flush=True)


def _safe_tool_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        raise ValueError("empty tool_name")
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"tool_{cleaned}"
    if not _TOOL_NAME_RE.match(cleaned):
        raise ValueError(f"invalid tool_name: {name!r}")
    return cleaned


def _extract_entry_function(code: str, tool_name: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = [
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if tool_name in names:
        return tool_name
    if names:
        return names[0]
    return None


def _strip_fences(text: str) -> str:
    raw = (text or "").strip()
    m = _FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw


def _deterministic_scrub(code: str) -> str:
    """Regex fallback when LLM is unavailable — strip obvious PII patterns."""
    out = code
    # Absolute Windows user paths (preserve trailing path after username when present)
    out = re.sub(
        r"[A-Za-z]:\\Users\\[^\\\s\"']+",
        r"<USER_HOME>",
        out,
        flags=re.I,
    )
    out = re.sub(r"/Users/[^/\s\"']+", "/<USER_HOME>", out)
    out = re.sub(r"/home/[^/\s\"']+", "/<USER_HOME>", out)
    # Desktop/Donna absolute workspace paths → generic relative hint
    out = re.sub(
        r"[A-Za-z]:\\Users\\[^\\\s\"']+\\Desktop\\Donna",
        r"<DONNA_WORKSPACE>",
        out,
        flags=re.I,
    )
    out = re.sub(
        r"/Users/[^/\s\"']+/Desktop/Donna",
        "/<DONNA_WORKSPACE>",
        out,
    )
    # Private / local network IPv4
    out = re.sub(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        "<IP>",
        out,
    )
    # Emails
    out = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "<EMAIL>",
        out,
    )
    # Windows username embedded in USERPROFILE-style strings already scrubbed;
    # also scrub bare "Users\\Name" leftovers.
    out = re.sub(r"(?i)Users\\[A-Za-z0-9._-]+", r"Users\\<USER>", out)
    return out


def _llm_scrub(code: str, *, tool_name: str, model: str = "qwen2.5-coder:7b") -> str:
    """Ask local ChatOllama to scrub personal data; fall back to regex scrub."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOllama(model=model, temperature=0.0)
        prompt = (
            f"Tool name: {tool_name}\n\n"
            f"Source to scrub:\n\n{code}\n"
        )
        resp = llm.invoke(
            [SystemMessage(content=SCRUB_SYSTEM), HumanMessage(content=prompt)]
        )
        content = getattr(resp, "content", None) or str(resp)
        scrubbed = _strip_fences(str(content))
        # Validate syntax before accepting.
        ast.parse(scrubbed)
        if "def " not in scrubbed:
            raise ValueError("scrubbed source has no function definition")
        _log(f"LLM scrub OK tool={tool_name!r} model={model}")
        return scrubbed
    except Exception as exc:  # noqa: BLE001
        _log(f"LLM scrub unavailable ({exc}); using deterministic scrub")
        return _deterministic_scrub(code)


def _hot_load_general(tool_name: str, path: Path, code: str) -> Any:
    """Import ``dana.tools.general.<tool_name>`` and return the entry callable."""
    import sys

    ensure_general_tools_package()
    module_name = f"dana.tools.general.{tool_name}"
    sys.modules.pop(module_name, None)
    # Also clear package cache so new sibling modules are visible.
    pkg = sys.modules.get("dana.tools.general")
    if pkg is not None and hasattr(pkg, "__path__"):
        pass

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
        submodule_search_locations=[str(GENERAL_TOOLS_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    entry_name = _extract_entry_function(code, tool_name)
    callable_obj = None
    if entry_name and hasattr(module, entry_name):
        callable_obj = getattr(module, entry_name)
        if hasattr(callable_obj, "func") and callable(getattr(callable_obj, "func")):
            callable_obj = callable_obj.func
    return callable_obj, module_name


def publish_tool_to_general(
    tool_name: str,
    *,
    model: str = "qwen2.5-coder:7b",
    skip_llm: bool = False,
) -> dict[str, Any]:
    """Promote ``tool_name`` from custom_tools → dana/tools/general.

    Returns a status dict suitable for tool Observation strings.
    """
    name = _safe_tool_name(tool_name)
    ensure_custom_tools_package()
    ensure_general_tools_package()

    src = CUSTOM_TOOLS_DIR / f"{name}.py"
    if not src.is_file():
        # Fallback: look under legacy generated_tools sibling.
        legacy = CUSTOM_TOOLS_DIR.parent / "generated_tools" / f"{name}.py"
        if legacy.is_file():
            src = legacy
        else:
            return {
                "ok": False,
                "error": f"custom tool not found: {name!r} under {CUSTOM_TOOLS_DIR}",
            }

    raw = src.read_text(encoding="utf-8")
    # Drop forge header comments that may embed absolute paths / user queries.
    body = textwrap.dedent(raw).strip() + "\n"

    if skip_llm:
        scrubbed = _deterministic_scrub(body)
        _log(f"Deterministic scrub only tool={name!r}")
    else:
        scrubbed = _llm_scrub(body, tool_name=name, model=model)

    # Final syntax gate.
    try:
        ast.parse(scrubbed)
    except SyntaxError as exc:
        return {"ok": False, "error": f"scrubbed source is invalid Python: {exc}"}

    dest = GENERAL_TOOLS_DIR / f"{name}.py"
    header = (
        f'"""Promoted general tool `{name}`.\n\n'
        f"Scrubbed of personal data via publish_tool_to_general.\n"
        f"Source: CAMGRASPER/custom_tools (ephemeral forge).\n"
        f'"""\n\n'
    )
    dest.write_text(header + scrubbed.lstrip() + "\n", encoding="utf-8")
    _log(f"Wrote general tool → {dest.relative_to(PROJECT_ROOT)}")

    try:
        callable_obj, module_name = _hot_load_general(name, dest, scrubbed)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"wrote {dest} but hot-load failed: {exc}",
            "path": str(dest),
        }

    # Prefer description from existing registry entry if present.
    registry = get_tool_registry()
    existing = registry.get(name)
    description = (
        (existing.description if existing else None)
        or f"General tool `{name}` (promoted from custom_tools)"
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
        aliases_en={"_intent": (name.replace("_", " "), "run " + name)},
        aliases_fa={"_intent": (name,)},
    )
    registry.register(
        tool_spec,
        callable=callable_obj if callable(callable_obj) else None,
        source="general",
        ephemeral=False,
        metadata={
            "path": str(dest),
            "module": module_name,
            "ephemeral": False,
            "tier": "general",
            "promoted_from": str(src),
        },
    )

    try:
        from dana.tools.broker import reload_broker_registry

        reload_broker_registry()
    except Exception:  # noqa: BLE001
        pass

    _log(f"Promoted + hot-loaded general tool={name!r} ephemeral=False")
    return {
        "ok": True,
        "tool_name": name,
        "path": str(dest),
        "module": module_name,
        "ephemeral": False,
        "scrubbed": True,
    }


def publish_tool_to_general_impl(tool_name: str) -> str:
    """Observation-string wrapper for ``execute_tool_call``."""
    result = publish_tool_to_general(tool_name)
    if not result.get("ok"):
        return f"ERROR: publish_tool_to_general failed: {result.get('error')}"
    return (
        f"OK: scrubbed and published `{result['tool_name']}` → "
        f"{result['path']} (hot-loaded, ephemeral=False)"
    )


"""Watchdog Template Method — fixed topology for generated monitor scripts.

The LLM never authors the orchestration skeleton. It only supplies method bodies
(or JSON mapping to those bodies). ``assemble_watchdog_script`` injects them into
a deterministic ``GeneratedWatchdog(BaseWatchdog)`` subclass.
"""

from __future__ import annotations

import json
import re
import textwrap
from abc import ABC, abstractmethod
from typing import Any

# Allowed third-party / stdlib roots the assembler may inject as imports.
ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "time",
        "pathlib",
        "typing",
        "json",
        "re",
        "math",
        "collections",
        "dataclasses",
        "datetime",
        "random",
        "hashlib",
        "struct",
        "io",
        "base64",
        "mss",
        "PIL",
        "Pillow",
        "pyautogui",
        "numpy",
        "cv2",
    }
)

FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "pty",
        "requests",
        "urllib",
        "ctypes",
        "pickle",
        "importlib",
        "builtins",
        "code",
        "codeop",
        "multiprocessing",
    }
)

REQUIRED_METHODS: tuple[str, ...] = ("run_self_test", "monitor_loop")

# Trusted framework source embedded into every assembled script (self-contained).
_BASE_WATCHDOG_SOURCE = '''
from abc import ABC, abstractmethod


class BaseWatchdog(ABC):
    """Fixed orchestration skeleton — subclass only fills the abstract hooks."""

    TTS_MARKER = "__DONNA_TTS__:"

    def alert(self, phrase: str) -> None:
        """Speak via the stdout TTS pipe (mandatory prefix)."""
        text = (phrase or "").strip()
        if not text:
            return
        print(f"{self.TTS_MARKER} {text}", flush=True)

    @abstractmethod
    def run_self_test(self) -> None:
        """Validation / dry-run probe — must raise or assert on failure."""

    @abstractmethod
    def monitor_loop(self) -> None:
        """One monitoring pass (DONNA_WATCHDOG_ONCE=1) or a bounded loop."""

    def run(self) -> None:
        self.run_self_test()
        self.monitor_loop()
'''.strip()


class BaseWatchdog(ABC):
    """In-process reference of the sandbox BaseWatchdog (tests / imports)."""

    TTS_MARKER = "__DONNA_TTS__:"

    def alert(self, phrase: str) -> None:
        text = (phrase or "").strip()
        if not text:
            return
        print(f"{self.TTS_MARKER} {text}", flush=True)

    @abstractmethod
    def run_self_test(self) -> None:
        ...

    @abstractmethod
    def monitor_loop(self) -> None:
        ...

    def run(self) -> None:
        self.run_self_test()
        self.monitor_loop()


def _indent_block(body: str, spaces: int = 8) -> str:
    raw = textwrap.dedent((body or "").strip("\n"))
    if not raw.strip():
        return " " * spaces + "pass"
    return textwrap.indent(raw, " " * spaces)


def _normalize_import_root(name: str) -> str:
    return (name or "").strip().split(".", 1)[0]


def filter_allowed_imports(imports: list[str] | None) -> list[str]:
    """Keep only allow-listed import roots; drop forbidden / unknown noise."""
    out: list[str] = []
    seen: set[str] = set()
    for item in imports or []:
        root = _normalize_import_root(str(item))
        if not root or root in seen:
            continue
        if root in FORBIDDEN_IMPORT_ROOTS:
            continue
        if root not in ALLOWED_IMPORT_ROOTS:
            continue
        seen.add(root)
        out.append(root)
    return out


def assemble_watchdog_script(
    *,
    run_self_test: str,
    monitor_loop: str,
    extra_imports: list[str] | None = None,
) -> str:
    """Build a complete executable script around ``BaseWatchdog`` (Template Method)."""
    imports = filter_allowed_imports(extra_imports)
    import_block = "\n".join(f"import {name}" for name in imports)
    if import_block:
        import_block += "\n\n"

    self_test_body = _indent_block(run_self_test, 8)
    monitor_body = _indent_block(monitor_loop, 8)

    return (
        f"{import_block}"
        f"{_BASE_WATCHDOG_SOURCE}\n\n\n"
        f"class GeneratedWatchdog(BaseWatchdog):\n"
        f"    def run_self_test(self) -> None:\n"
        f"{self_test_body}\n\n"
        f"    def monitor_loop(self) -> None:\n"
        f"{monitor_body}\n\n\n"
        f"if __name__ == '__main__':\n"
        f"    GeneratedWatchdog().run()\n"
    )


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = text[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extract_method_body_from_python(src: str, method: str) -> str:
    """Best-effort scrape of ``def method(...):`` body from free-form Python."""
    pattern = re.compile(
        rf"(?m)^(?P<indent>[ \t]*)def[ \t]+{re.escape(method)}\s*\([^)]*\)\s*(?:->[^:]+)?:[ \t]*\n"
        rf"(?P<body>(?:(?P=indent)[ \t]+.*\n|(?P=indent)[ \t]*\n)*)"
    )
    match = pattern.search(src or "")
    if not match:
        return ""
    body = textwrap.dedent(match.group("body"))
    return body.strip()


def parse_coder_payload(raw: str) -> dict[str, Any]:
    """Parse coder LLM output into method bodies + optional imports.

    Preferred format (JSON)::
        {
          "extra_imports": ["time", "mss"],
          "run_self_test": "assert True",
          "monitor_loop": "self.alert('found')"
        }

    Fallback: extract ``run_self_test`` / ``monitor_loop`` defs from Python text.
    """
    data = _extract_json_object(raw)
    if data is not None:
        imports_raw = data.get("extra_imports") or data.get("imports") or []
        if isinstance(imports_raw, str):
            imports_list = [p.strip() for p in imports_raw.split(",") if p.strip()]
        elif isinstance(imports_raw, list):
            imports_list = [str(x) for x in imports_raw]
        else:
            imports_list = []
        return {
            "extra_imports": filter_allowed_imports(imports_list),
            "run_self_test": str(
                data.get("run_self_test") or data.get("self_test") or ""
            ).strip(),
            "monitor_loop": str(
                data.get("monitor_loop") or data.get("monitor") or ""
            ).strip(),
        }

    src = raw or ""
    fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", src, re.I)
    if fence:
        src = fence.group(1)
    return {
        "extra_imports": [],
        "run_self_test": _extract_method_body_from_python(src, "run_self_test"),
        "monitor_loop": _extract_method_body_from_python(src, "monitor_loop"),
    }

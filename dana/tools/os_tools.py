"""Directory-aware shell helpers, graph file digests, and LaTeX constraints."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

LATEX_NOCITE_DIRECTIVE = (
    "When generating LaTeX, you must output clean code. Absolutely no "
    "auto-generated citation tags, \\cite{}, or \\bibliography{} commands "
    "are permitted under any circumstances."
)

_CASCADE_GIT_RE = re.compile(
    r"(?i)\b(?:"
    r"cascade[-_]?router|"
    r"last\s+git\s+commit|"
    r"git\s+log|"
    r"date\s+of\s+the\s+last\s+(?:git\s+)?commit"
    r")\b",
)
_WATCHDOG_GRAPH_RE = re.compile(
    r"(?i)\b(?:"
    r"watchdog\s+(?:monitoring\s+)?graph|"
    r"watchdog_graph|"
    r"watchdog\s+monitor|"
    r"active\s+dependencies"
    r")\b",
)
_LATEX_RE = re.compile(
    r"(?i)\b(?:"
    r"latex|\\documentclass|\\begin\{|"
    r"citation\s+tags|\\cite|bibliography"
    r")\b",
)
_CITE_RE = re.compile(
    r"\\(?:cite[a-zA-Z]*|bibliography|nocite)\b"
    r"(?:\[[^\]]*\])?"
    r"(?:\{[^}]*\})?",
)

WATCHDOG_GRAPH_REL = "dana/swarm/watchdog_graph.py"


def is_cascade_git_query(text: str) -> bool:
    return bool(_CASCADE_GIT_RE.search(text or ""))


def is_watchdog_graph_query(text: str) -> bool:
    low = text or ""
    return bool(_WATCHDOG_GRAPH_RE.search(low)) or (
        "watchdog" in low.lower() and "depend" in low.lower()
    )


def is_latex_nocite_query(text: str) -> bool:
    return bool(_LATEX_RE.search(text or ""))


def resolve_named_repo_cwd(name: str = "cascade-router") -> str | None:
    """Resolve a named repo directory (Desktop first, then home / PROJECT_ROOT)."""
    needle = (name or "cascade-router").strip()
    if not needle:
        return None
    candidates = [
        Path.home() / "Desktop" / needle,
        Path.home() / needle,
    ]
    try:
        from dana.paths import PROJECT_ROOT

        candidates.append(Path(PROJECT_ROOT).parent / needle)
        candidates.append(Path(PROJECT_ROOT) / needle)
    except Exception:  # noqa: BLE001
        pass
    for path in candidates:
        try:
            if path.is_dir() and (path / ".git").exists():
                return str(path.resolve())
        except OSError:
            continue
    # Prefer Desktop path even if git is missing (fixture may init later).
    desktop = Path.home() / "Desktop" / needle
    if desktop.is_dir():
        return str(desktop.resolve())
    return None


def git_last_commit_date_command() -> str:
    return "git log -1 --format=%cd"


def cascade_git_tool_args(user_text: str = "") -> dict[str, str]:
    cwd = resolve_named_repo_cwd("cascade-router") or str(
        Path.home() / "Desktop" / "cascade-router"
    )
    return {
        "command": git_last_commit_date_command(),
        "cwd": cwd,
    }


def watchdog_graph_filepath() -> str:
    try:
        from dana.paths import PROJECT_ROOT

        abs_path = Path(PROJECT_ROOT) / WATCHDOG_GRAPH_REL
        if abs_path.is_file():
            return WATCHDOG_GRAPH_REL
    except Exception:  # noqa: BLE001
        pass
    return WATCHDOG_GRAPH_REL


def extract_dependency_digest(text: str, *, source: str = "") -> str:
    """Compact import / node / dependency tokens for planner/LLM grounding."""
    body = text or ""
    imports = sorted(
        set(
            re.findall(
                r"(?m)^\s*(?:from|import)\s+([a-zA-Z_][\w\.]*)",
                body,
            )
        )
    )
    nodes = sorted(
        set(
            re.findall(
                r"""add_node\(\s*['\"]([^'\"]+)['\"]""",
                body,
            )
        )
    )
    keywords = [
        w
        for w in (
            "langgraph",
            "titan",
            "experience",
            "sqlite",
            "supervisor",
            "dispatcher",
            "compile",
            "state",
            "dependency",
            "node",
            "import",
        )
        if w in body.lower()
    ]
    lines = [
        "DEPENDENCY DIGEST (extracted; prefer these over guessing):",
        f"source={source or '(inline)'}",
        f"imports={', '.join(imports[:24]) or '(none)'}",
        f"graph_nodes={', '.join(nodes[:24]) or '(none)'}",
        f"tokens={', '.join(keywords) or '(none)'}",
    ]
    return "\n".join(lines)


def append_dependency_digest(filepath: str, observation: str) -> str:
    """If this is the watchdog graph (or similar), prepend a dependency digest.

    Digests are placed first so they survive tool-trace truncation (``[:500]``).
    """
    path = (filepath or "").replace("\\", "/").lower()
    obs = observation or ""
    if "watchdog" not in path and "watchdog" not in obs.lower():
        return obs
    if obs.upper().startswith("ERROR"):
        return obs
    # Prefer digesting the raw file body after the OK header.
    body = obs
    if "\n" in obs:
        body = obs.split("\n", 1)[-1]
    digest = extract_dependency_digest(body, source=filepath)
    return f"{digest}\n\n{obs.rstrip()}"


def strip_latex_citations(text: str) -> str:
    """Remove forbidden citation / bibliography commands from LaTeX output."""
    cleaned = _CITE_RE.sub("", text or "")
    # Collapse leftover blank lines created by removals.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def latex_system_prompt(user_text: str = "") -> str:
    """Short hard-constraint prompt for LaTeX drafting turns."""
    return (
        "You are Dānā drafting a clean LaTeX document fragment.\n"
        f"{LATEX_NOCITE_DIRECTIVE}\n"
        "Output valid LaTeX only (use \\documentclass or \\begin{document} / "
        "\\section / \\textbf / itemize as appropriate).\n"
        "Do not call tools. Do not invent citation keys.\n"
        "Summarize the user's topic briefly in LaTeX prose."
    )

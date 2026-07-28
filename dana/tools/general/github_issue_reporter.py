"""Permanent general tool: report bugs to the community GitHub repository.

Configuration (environment variables, optional local secure JSON):
  - GITHUB_ACCESS_TOKEN
  - GITHUB_REPO_OWNER
  - GITHUB_REPO_NAME

Optional local config (never commit secrets):
  ``CAMGRASPER/execution_jail/github_reporter.json`` with the same keys.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

_MISSING_CONFIG_MSG = (
    "GitHub reporter configuration is missing. "
    "Please set GITHUB_ACCESS_TOKEN in your environment variables."
)

_SUCCESS_FMT = (
    "Bug report successfully submitted. Issue #{number} is now live on GitHub."
)


def _workspace_config_path() -> Path | None:
    try:
        from dana.paths import DONNA_WORKSPACE

        return Path(DONNA_WORKSPACE) / "execution_jail" / "github_reporter.json"
    except Exception:  # noqa: BLE001
        desktop = Path.home() / "Desktop" / "Donna" / "execution_jail" / "github_reporter.json"
        return desktop


def _load_local_config() -> dict[str, str]:
    path = _workspace_config_path()
    if path is None or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key in ("GITHUB_ACCESS_TOKEN", "GITHUB_REPO_OWNER", "GITHUB_REPO_NAME"):
        val = raw.get(key) or raw.get(key.lower())
        if val is not None and str(val).strip():
            out[key] = str(val).strip()
    return out


def _resolve_config() -> tuple[str, str, str] | None:
    local = _load_local_config()
    token = (
        os.environ.get("GITHUB_ACCESS_TOKEN")
        or os.environ.get("GH_TOKEN")
        or local.get("GITHUB_ACCESS_TOKEN")
        or ""
    ).strip()
    owner = (
        os.environ.get("GITHUB_REPO_OWNER") or local.get("GITHUB_REPO_OWNER") or ""
    ).strip()
    name = (
        os.environ.get("GITHUB_REPO_NAME") or local.get("GITHUB_REPO_NAME") or ""
    ).strip()
    if not token or not owner or not name:
        return None
    return token, owner, name


def _normalize_labels(labels: Sequence[str] | str | None) -> list[str]:
    if labels is None:
        return []
    if isinstance(labels, str):
        parts = [p.strip() for p in labels.replace(";", ",").split(",")]
        return [p for p in parts if p]
    out: list[str] = []
    for item in labels:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _format_issue_body(body: str) -> str:
    """Apply light markdown structure for readable GitHub issues."""
    content = (body or "").strip()
    if not content:
        content = "_No details provided._"
    return (
        "## Bug report\n\n"
        f"{content}\n\n"
        "---\n"
        "*Submitted via Donna community reporter*\n"
    )


def report_github_issue(
    title: str,
    body: str = "",
    labels: list[str] | str | None = None,
) -> str:
    """Create a GitHub issue for a community bug report.

    Returns a short status string suitable for Donna TTS / chat.
    Never raises into the parent agent runtime graph.
    """
    title_text = (title or "").strip()
    if not title_text:
        return "ERROR: Bug report title is required."

    cfg = _resolve_config()
    if cfg is None:
        return _MISSING_CONFIG_MSG
    token, owner, repo = cfg

    payload: dict[str, Any] = {
        "title": title_text[:256],
        "body": _format_issue_body(body),
    }
    label_list = _normalize_labels(labels)
    if label_list:
        payload["labels"] = label_list

    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "Donna-GitHubIssueReporter",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            status = int(getattr(resp, "status", 0) or 0)
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:240]
        except Exception:  # noqa: BLE001
            detail = str(exc.reason or exc)
        if exc.code in (401, 403):
            return (
                "ERROR: GitHub rejected the credentials. "
                "Check GITHUB_ACCESS_TOKEN permissions and try again."
            )
        if exc.code == 404:
            return (
                "ERROR: GitHub repository not found. "
                "Check GITHUB_REPO_OWNER and GITHUB_REPO_NAME."
            )
        return f"ERROR: GitHub API returned HTTP {exc.code}. {detail}".strip()
    except urllib.error.URLError as exc:
        return f"ERROR: Could not reach GitHub ({exc.reason}). Try again later."
    except TimeoutError:
        return "ERROR: GitHub request timed out. Try again later."
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Unexpected GitHub reporter failure: {exc}"

    if status != 201:
        return f"ERROR: GitHub API returned HTTP {status} (expected 201 Created)."

    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return "ERROR: GitHub returned an unreadable response."

    number = parsed.get("number")
    html_url = str(parsed.get("html_url") or "").strip()
    if number is None:
        if html_url:
            return f"Bug report successfully submitted. Issue is now live on GitHub: {html_url}"
        return "ERROR: GitHub created an issue but returned no issue number."

    msg = _SUCCESS_FMT.format(number=number)
    if html_url:
        return f"{msg} {html_url}"
    return msg


# Disk loader binds by module stem name (`github_issue_reporter`).
github_issue_reporter = report_github_issue

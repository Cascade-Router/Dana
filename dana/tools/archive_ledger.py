"""Sweep completed tickets from patch_ledger.md into an archive file."""

from __future__ import annotations

import re
from pathlib import Path

_LEDGER_HEADER = "# Donna Patch Ledger\n"
_TICKET_SPLIT_RE = re.compile(r"(?=^### Ticket:)", re.MULTILINE)
_STATUS_RE = re.compile(
    r"\*\*Status:\*\*\s*`\[(PENDING|RESOLVED|FAILED)\]`",
    re.IGNORECASE,
)


def _ledger_paths() -> tuple[Path, Path]:
    try:
        from dana.paths import DONNA_SECURITY_DIR, PATCH_LEDGER_PATH

        DONNA_SECURITY_DIR.mkdir(parents=True, exist_ok=True)
        ledger = Path(PATCH_LEDGER_PATH)
        archive = ledger.with_name("patch_ledger_archive.md")
        return ledger, archive
    except Exception:  # noqa: BLE001
        root = Path(__file__).resolve().parents[2] / "donna_security"
        root.mkdir(parents=True, exist_ok=True)
        return root / "patch_ledger.md", root / "patch_ledger_archive.md"


def _ticket_status(block: str) -> str | None:
    match = _STATUS_RE.search(block or "")
    if not match:
        return None
    return match.group(1).upper()


def _split_ticket_blocks(text: str) -> list[str]:
    """Return individual ``### Ticket:`` blocks (no leading header)."""
    body = text or ""
    # Drop a leading markdown header line if present.
    if body.lstrip().startswith("#"):
        first_nl = body.find("\n")
        body = body[first_nl + 1 :] if first_nl >= 0 else ""
    parts = _TICKET_SPLIT_RE.split(body)
    return [p.strip() for p in parts if p.strip().startswith("### Ticket:")]


def archive_completed_tickets() -> str:
    """Move RESOLVED/FAILED tickets to patch_ledger_archive.md; keep PENDING only.

    Returns a short status string for tool / CLI callers.
    """
    ledger_path, archive_path = _ledger_paths()
    try:
        raw = (
            ledger_path.read_text(encoding="utf-8")
            if ledger_path.is_file()
            else _LEDGER_HEADER
        )
    except OSError as exc:
        return f"ERROR: could not read patch_ledger.md ({exc})"

    pending: list[str] = []
    completed: list[str] = []
    for block in _split_ticket_blocks(raw):
        status = _ticket_status(block)
        if status == "PENDING":
            pending.append(block)
        elif status in ("RESOLVED", "FAILED"):
            completed.append(block)
        else:
            # Unknown status — keep in active ledger to avoid data loss.
            pending.append(block)

    try:
        try:
            from dana.tools.task_queue import shadow_backup_before_write

            shadow_backup_before_write(ledger_path)
            if archive_path.is_file():
                shadow_backup_before_write(archive_path)
        except Exception:  # noqa: BLE001
            pass

        if completed:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            existing = ""
            if archive_path.is_file():
                existing = archive_path.read_text(encoding="utf-8")
            if not existing.strip():
                existing = _LEDGER_HEADER + "\n"
            elif not existing.endswith("\n"):
                existing += "\n"
            addition = "\n\n---\n\n".join(completed)
            if not existing.rstrip().endswith("---"):
                archive_path.write_text(
                    existing.rstrip() + "\n\n---\n\n" + addition + "\n",
                    encoding="utf-8",
                )
            else:
                archive_path.write_text(
                    existing.rstrip() + "\n\n" + addition + "\n",
                    encoding="utf-8",
                )

        pending_body = _LEDGER_HEADER + "\n"
        if pending:
            pending_body += "\n---\n\n".join(pending) + "\n"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(pending_body, encoding="utf-8")
    except OSError as exc:
        return f"ERROR: archive write failed ({exc})"

    return (
        f"OK: archived {len(completed)} completed ticket(s); "
        f"{len(pending)} PENDING remain in patch_ledger.md"
    )


def archive_ledger(_text: str = "") -> str:
    """Tool-facing alias (optional unused text arg for broker kwargs)."""
    return archive_completed_tickets()

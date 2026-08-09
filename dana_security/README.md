# dana_security/ — security package + patch ledger

Importable Python package (`from dana_security import …`) plus the async Cursor
ticket ledger. **Not** the same as `execution_jail/` (filesystem jail).

| Path | Role |
|------|------|
| `__init__.py` | AST gates, subprocess sandbox, `architect_new_tool`, dynamic tool helpers |
| `patch_ledger.md` | PENDING Cursor tickets from `draft_cursor_prompt` |

Canonical path: `dana.paths.DANA_SECURITY_DIR` / `REPO_SANDBOX_DIR` / `PATCH_LEDGER_PATH`.
Legacy snapshots live under `/_archive/` — not loaded at runtime.

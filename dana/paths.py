"""Canonical project + runtime workspace paths (cwd-independent).

Core source and runtime artifacts both live under ``PROJECT_ROOT`` (CAMGRASPER).
``DONNA_WORKSPACE`` is the repo root — there is no separate Desktop/Donna tree.

Layout:
  CAMGRASPER/                 ← PROJECT_ROOT == DONNA_WORKSPACE
  CAMGRASPER/dana/           ← core package
  CAMGRASPER/custom_tools/    ← sole Tool Forge write/load root (ephemeral)
  CAMGRASPER/logs/            ← runtime + conversation logs
  CAMGRASPER/tracker/         ← bug_tracker.json + pending_patches/
  CAMGRASPER/execution_jail/  ← FS jail (task_queue.json, library/, fixture copies)
  CAMGRASPER/dana_security/  ← importable security package + patch_ledger.md
                                  (do NOT merge with execution_jail/ — different roles)
  CAMGRASPER/_archive/        ← unused legacy snapshots (not on the runtime path)
"""

from __future__ import annotations

import os
from pathlib import Path

# dana/paths.py → join(.., "..") is the CAMGRASPER repo root.
PROJECT_ROOT: Path = Path(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
).resolve()

# Runtime workspace is the canonical repository root.
DONNA_WORKSPACE: Path = PROJECT_ROOT

# --- Runtime artifact trees (under CAMGRASPER) ---

LOGS_DIR: Path = DONNA_WORKSPACE / "logs"

TRACKER_DIR: Path = DONNA_WORKSPACE / "tracker"

BUG_TRACKER_PATH: Path = TRACKER_DIR / "bug_tracker.json"

PENDING_PATCHES_DIR: Path = TRACKER_DIR / "pending_patches"

# Filesystem jail (Watchdog cwd, task queue, sandbox_read root).
EXECUTION_JAIL_DIR: Path = DONNA_WORKSPACE / "execution_jail"

EXECUTION_JAIL_LIBRARY_DIR: Path = EXECUTION_JAIL_DIR / "library"

# Structured task queue — array of {id, status, command} (replaces flat input.txt).
TASK_QUEUE_PATH: Path = EXECUTION_JAIL_DIR / "task_queue.json"

# Deprecated: legacy flat-file interceptor. Migrated into TASK_QUEUE_PATH on read.
TEXT_INJECTION_PATH: Path = EXECUTION_JAIL_DIR / "input.txt"

# Custom (ephemeral) forged tools — primary Tool Forge write target.
CUSTOM_TOOLS_DIR: Path = DONNA_WORKSPACE / "custom_tools"

CUSTOM_TOOLS_ARCHIVE_DIR: Path = CUSTOM_TOOLS_DIR / "_archive"

# Backward-compat aliases (pre-restructure name was generated_tools).
GENERATED_TOOLS_DIR: Path = CUSTOM_TOOLS_DIR

GENERATED_TOOLS_ARCHIVE_DIR: Path = CUSTOM_TOOLS_ARCHIVE_DIR

LEGACY_DESKTOP_GENERATED_TOOLS_DIR: Path = DONNA_WORKSPACE / "generated_tools"

# Live telemetry surface overwritten every ~45s by the dashboard writer.
DASHBOARD_PATH: Path = DONNA_WORKSPACE / "dashboard.md"

CURSOR_HANDOFF_DIR: Path = DONNA_WORKSPACE / "cursor_handoffs"

CURSOR_HANDOFF_PATH: Path = CURSOR_HANDOFF_DIR / "donna_handoff.md"

# Mirror so Cursor IDE still discovers the plan under the project tree.
CURSOR_HANDOFF_MIRROR_DIR: Path = PROJECT_ROOT / ".cursor" / "instructions"

CURSOR_HANDOFF_MIRROR_PATH: Path = CURSOR_HANDOFF_MIRROR_DIR / "donna_handoff.md"

CAPTURES_DIR: Path = DONNA_WORKSPACE / "captures"

# --- Repo-local (config / models / vault / async ledger) ---

# Importable security package + unified patch ledger (async Cursor tickets).
DONNA_SECURITY_DIR: Path = PROJECT_ROOT / "dana_security"

# Alias: historical name; always points at dana_security/.
REPO_SANDBOX_DIR: Path = DONNA_SECURITY_DIR

PATCH_LEDGER_PATH: Path = DONNA_SECURITY_DIR / "patch_ledger.md"

DOCS_DIR: Path = PROJECT_ROOT / "docs"

TTS_MODELS_DIR: Path = PROJECT_ROOT / "tts_models"

SETTINGS_PATH: Path = PROJECT_ROOT / "settings.json"

VAULT_PATH: Path = PROJECT_ROOT / "donna_memory.enc"

ARCHITECTURE_MD: Path = PROJECT_ROOT / "ARCHITECTURE.md"

TOOLS_JSON: Path = PROJECT_ROOT / "dana" / "tools" / "tools.json"

SECURITY_POLICY_PATH: Path = PROJECT_ROOT / "dana" / "tools" / "security_policy.json"

# Promoted general-purpose tools (Git-tracked).
GENERAL_TOOLS_DIR: Path = PROJECT_ROOT / "dana" / "tools" / "general"

# Legacy empty mirror (not loaded by registry; wipe cleanup only if files appear).
REPO_CUSTOM_TOOLS_DIR: Path = PROJECT_ROOT / "dana" / "tools" / "custom"

# Legacy in-repo forge dir (stub redirect only — do not write new tools here).
LEGACY_GENERATED_TOOLS_DIR: Path = PROJECT_ROOT / "dana" / "generated_tools"

TOOL_REGISTRY_INDEX_DIR: Path = DOCS_DIR / "tool_registry_index"

WATCHDOG_HISTORY_DB: Path = DOCS_DIR / "watchdog_history.db"

RESEARCH_SCRATCHPAD_DB: Path = DOCS_DIR / "research_scratchpad.db"

EVALS_DIR: Path = PROJECT_ROOT / "dana" / "evals"

EVAL_CASES_PATH: Path = EVALS_DIR / "test_cases.json"

# Preferred wake-word ONNX under assets/models; ``resolve_wakeword_onnx`` also
# checks legacy repo-root locations for backward compatibility.
MODELS_DIR: Path = PROJECT_ROOT / "assets" / "models"
WAKEWORD_ONNX: Path = MODELS_DIR / "dana.onnx"
WAKEWORD_ONNX_LEGACY: Path = MODELS_DIR / "donna.onnx"
WAKEWORD_ONNX_ALT: Path = MODELS_DIR / "wake_word_model.onnx"


def resolve_wakeword_onnx() -> Path:
    """Return the first existing wake-word model path (dana → alt → legacy)."""
    for candidate in (
        WAKEWORD_ONNX,
        WAKEWORD_ONNX_ALT,
        WAKEWORD_ONNX_LEGACY,
        PROJECT_ROOT / "dana.onnx",
        PROJECT_ROOT / "wake_word_model.onnx",
        PROJECT_ROOT / "donna.onnx",
    ):
        if candidate.is_file():
            return candidate
    return WAKEWORD_ONNX

ENV_PATH: Path = PROJECT_ROOT / ".env"

TRIGGER_ASK_PATH: Path = PROJECT_ROOT / ".trigger_ask"

TEMP_REPLY_WAV: Path = PROJECT_ROOT / "temp_reply.wav"

YOLO_WEIGHTS_PATH: Path = MODELS_DIR / "yolov8n.pt"

WORKSPACE_MIGRATION_MARKER: Path = DONNA_WORKSPACE / ".donna_workspace_migrated"

WORKSPACE_SUBDIRS: tuple[Path, ...] = (
    LOGS_DIR,
    TRACKER_DIR,
    PENDING_PATCHES_DIR,
    EXECUTION_JAIL_DIR,
    EXECUTION_JAIL_LIBRARY_DIR,
    CUSTOM_TOOLS_DIR,
    CUSTOM_TOOLS_ARCHIVE_DIR,
    CURSOR_HANDOFF_DIR,
    CAPTURES_DIR,
)


def ensure_project_root_on_syspath() -> Path:
    """Put the repo root first on ``sys.path`` (safe if already present)."""
    import sys

    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    # Relocated root utilities (ingest, spatial_context, vision shim).
    for sub in ("scripts", os.path.join("scripts", "diagnostics")):
        p = str(PROJECT_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    return PROJECT_ROOT


def ensure_workspace_on_syspath() -> Path:
    """Put ``DONNA_WORKSPACE`` on ``sys.path`` so ``custom_tools.*`` imports work."""
    import sys

    ws = str(DONNA_WORKSPACE)
    if ws not in sys.path:
        sys.path.insert(0, ws)
    return DONNA_WORKSPACE


def chdir_project_root() -> Path:
    """``os.chdir`` into the repo root so any leftover relative paths resolve."""
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT

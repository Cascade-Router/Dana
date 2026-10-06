"""Canonical project + runtime workspace paths (cwd-independent).

Core source and runtime artifacts both live under ``PROJECT_ROOT`` (CAMGRASPER).
``DANA_WORKSPACE`` is the repo root — there is no separate Desktop/Dana tree.

Layout:
  CAMGRASPER/                 ← PROJECT_ROOT == DANA_WORKSPACE
  CAMGRASPER/dana/           ← core package
  CAMGRASPER/custom_tools/    ← sole Tool Forge write/load root (ephemeral)
  CAMGRASPER/logs/            ← runtime + conversation logs
  CAMGRASPER/tracker/         ← bug_tracker.json + pending_patches/
  CAMGRASPER/execution_jail/  ← FS jail (task_queue.json, library/, fixture copies)
  CAMGRASPER/docs/            ← docs + patch_ledger.md (local-only ticket history)
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
DANA_WORKSPACE: Path = PROJECT_ROOT

# --- Runtime artifact trees (under CAMGRASPER) ---

LOGS_DIR: Path = DANA_WORKSPACE / "logs"


# Filesystem jail (Watchdog cwd, task queue, sandbox_read root).
EXECUTION_JAIL_DIR: Path = DANA_WORKSPACE / "execution_jail"


# Flat-file command drop read by scripts/ingest.py.
TEXT_INJECTION_PATH: Path = EXECUTION_JAIL_DIR / "input.txt"

# Custom (ephemeral) forged tools — primary Tool Forge write target.
CUSTOM_TOOLS_DIR: Path = DANA_WORKSPACE / "custom_tools"

CUSTOM_TOOLS_ARCHIVE_DIR: Path = CUSTOM_TOOLS_DIR / "_archive"


CAPTURES_DIR: Path = DANA_WORKSPACE / "captures"

# Sandbox root for dana.plugins.os.file_system's list_directory/read_file/
# write_file tools (the "os_tools" capability domain — see
# dana.core.react_dispatch's _OS_TOOLS_TOOL_IDS). Deliberately its OWN
# subtree, NOT DANA_WORKSPACE itself: DANA_WORKSPACE == PROJECT_ROOT (see
# the module docstring), which also contains .env (a real API key),
# .git, and Dana's own source — an LLM-driven read/write tool has no
# business reaching any of that, so it gets a narrower, dedicated root.
AGENT_WORKSPACE_DIR: Path = DANA_WORKSPACE / "agent_workspace"

# --- Repo-local (config / models / vault / async ledger) ---


TTS_MODELS_DIR: Path = PROJECT_ROOT / "tts_models"

SETTINGS_PATH: Path = PROJECT_ROOT / "settings.json"

# Dynamic LLM Router fleet config (dana.core.routing_config) — optional; its
# absence is a supported, zero-behavior-change state (dana.core.model_provider
# falls back to its existing .env-toggle resolution). Never created
# automatically — a user opts in by copying routing_config.yaml.example.
ROUTING_CONFIG_PATH: Path = PROJECT_ROOT / "routing_config.yaml"

# Model Registry catalog (dana.core.model_registry) — provider/model specs,
# pricing, rate limits. Git-tracked; refreshed by scripts/sync_model_catalog.py.
MODELS_REGISTRY_PATH: Path = PROJECT_ROOT / "dana" / "data" / "models_registry.json"

# User's saved fallback-chain order + per-model enable/disable toggles from
# the Model Registry control panel. Kept separate from ROUTING_CONFIG_PATH's
# hand-authored YAML (with its rich provenance comments) so a UI-driven save
# never clobbers those; absence is a supported "use catalog defaults" state.
MODEL_PREFERENCES_PATH: Path = PROJECT_ROOT / "dana" / "data" / "model_preferences.json"

VAULT_PATH: Path = PROJECT_ROOT / "dana_memory.enc"

ARCHITECTURE_MD: Path = PROJECT_ROOT / "ARCHITECTURE.md"

TOOLS_JSON: Path = PROJECT_ROOT / "dana" / "tools" / "tools.json"

SECURITY_POLICY_PATH: Path = PROJECT_ROOT / "dana" / "tools" / "security_policy.json"

# Promoted general-purpose tools (Git-tracked).
GENERAL_TOOLS_DIR: Path = PROJECT_ROOT / "dana" / "tools" / "general"

# Legacy empty mirror (not loaded by registry; wipe cleanup only if files appear).
REPO_CUSTOM_TOOLS_DIR: Path = PROJECT_ROOT / "dana" / "tools" / "custom"


# Preferred wake-word ONNX under assets/models.
MODELS_DIR: Path = PROJECT_ROOT / "assets" / "models"
WAKEWORD_ONNX: Path = MODELS_DIR / "dana.onnx"
WAKEWORD_ONNX_ALT: Path = MODELS_DIR / "wake_word_model.onnx"


def resolve_wakeword_onnx() -> Path:
    """Return the first existing wake-word model path (dana → alt)."""
    for candidate in (
        WAKEWORD_ONNX,
        WAKEWORD_ONNX_ALT,
        PROJECT_ROOT / "dana.onnx",
        PROJECT_ROOT / "wake_word_model.onnx",
    ):
        if candidate.is_file():
            return candidate
    return WAKEWORD_ONNX

ENV_PATH: Path = PROJECT_ROOT / ".env"

TRIGGER_ASK_PATH: Path = PROJECT_ROOT / ".trigger_ask"


YOLO_WEIGHTS_PATH: Path = MODELS_DIR / "yolov8n.pt"


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
    """Put ``DANA_WORKSPACE`` on ``sys.path`` so ``custom_tools.*`` imports work."""
    import sys

    ws = str(DANA_WORKSPACE)
    if ws not in sys.path:
        sys.path.insert(0, ws)
    return DANA_WORKSPACE


def chdir_project_root() -> Path:
    """``os.chdir`` into the repo root so any leftover relative paths resolve."""
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT


def _nt_hide_console_if_mp_child() -> None:
    """Hide console only inside multiprocessing children — never the main agent terminal."""
    if os.name != "nt":
        return
    try:
        import multiprocessing

        if multiprocessing.current_process().name == "MainProcess":
            return
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)
    except Exception:  # noqa: BLE001
        pass


_windows_process_hardening_applied = False


def apply_windows_process_hardening() -> None:
    """Idempotent Windows setup: taskbar identity + console-hiding subprocess spawn.

    Moved verbatim out of ``dana.core_agent`` (Phase 8 of the core_agent.py
    decomposition) -- ``run.py`` claims the taskbar AppUserModelID
    independently, but the ``subprocess.Popen`` console-hiding patch and the
    ``multiprocessing`` pythonw.exe redirect have no other caller, so this
    stays load-bearing for every entry point (``run.py``, ``python -m
    dana.core_agent``, ``python -m dana.ui.main``) rather than something
    safe to simply drop.
    """
    global _windows_process_hardening_applied
    if _windows_process_hardening_applied:
        return
    _windows_process_hardening_applied = True

    import subprocess
    import sys

    # Windows taskbar identity — must run before CustomTkinter/Tk root creation.
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "dana.assistant.desktop.v1"
            )
        except Exception:  # noqa: BLE001
            pass

    # Absolute Windows console kill-switch: CREATE_NO_WINDOW + STARTUPINFO hide +
    # mutate python.exe → pythonw.exe (class patch so asyncio can still subclass).
    if os.name == "nt":
        _original_popen = subprocess.Popen
        _pythonw_path = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")

        def _coerce_pythonw_cmd(cmd0):
            if not os.path.isfile(_pythonw_path):
                return cmd0
            if isinstance(cmd0, (list, tuple)) and cmd0:
                cmd = list(cmd0)
                head = str(cmd[0])
                if (
                    head == sys.executable
                    or os.path.normcase(head) == os.path.normcase(sys.executable)
                    or os.path.basename(head).lower() == "python.exe"
                ):
                    cmd[0] = _pythonw_path
                return cmd
            if isinstance(cmd0, str):
                if cmd0 == sys.executable or cmd0.startswith(sys.executable):
                    return cmd0.replace(sys.executable, _pythonw_path, 1)
                if os.path.basename(cmd0.split(" ", 1)[0]).lower() == "python.exe":
                    return cmd0.replace(cmd0.split(" ", 1)[0], _pythonw_path, 1)
            return cmd0

        class _PatchedPopen(_original_popen):  # type: ignore[valid-type,misc]
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                # 1. Force CREATE_NO_WINDOW
                kwargs["creationflags"] = int(kwargs.get("creationflags") or 0) | 0x08000000

                # 2. Force STARTUPINFO invisibility cloak
                if "startupinfo" not in kwargs or kwargs.get("startupinfo") is None:
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
                    kwargs["startupinfo"] = startupinfo

                # 3. Intercept sys.executable / python.exe and mutate to pythonw.exe
                if args:
                    first = _coerce_pythonw_cmd(args[0])
                    args = (first,) + args[1:]
                if "args" in kwargs and kwargs["args"] is not None:
                    kwargs["args"] = _coerce_pythonw_cmd(kwargs["args"])

                super().__init__(*args, **kwargs)

        subprocess.Popen = _PatchedPopen  # type: ignore[misc, assignment]

    # Force multiprocessing workers onto windowless pythonw.exe (CreateProcess bypasses Popen).
    import multiprocessing

    if os.name == "nt":
        _pythonw_path = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        if os.path.isfile(_pythonw_path):
            multiprocessing.set_executable(_pythonw_path)

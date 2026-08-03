"""Programmatic PyInstaller onedir packaging for Dānā (Windows desktop).

Entry point choice
------------------
Uses ``run.py`` (project root), not ``dana.daemon`` / ``dana/ui/main``.

``run.py`` is the desktop launcher: it sets AppUserModelID, workspace, single-
instance lock, verifies torch, then calls ``dana.core_agent.main()`` which owns
the CustomTkinter UI. Packaging the daemon ``__main__`` would omit the GUI.

Usage
-----
    .venv\\Scripts\\python.exe build_dana.py

Output (onedir)::
    dist/Dana/Dana.exe
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    # Ensure repo root is importable / discoverable for Analysis.
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        import PyInstaller.__main__ as pyi_main
    except ImportError as exc:
        print(
            "[build_dana] ERROR: PyInstaller is not installed. "
            "Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    entry = ROOT / "run.py"
    if not entry.is_file():
        print(f"[build_dana] ERROR: missing entry script {entry}", file=sys.stderr)
        return 1

    # collect-all packages that fail or miss data/binaries under static analysis.
    collect_all_pkgs = (
        "customtkinter",
        "chromadb",
        "sentence_transformers",
        "torch",
        "onnxruntime",
        "sounddevice",
    )

    # Minimal hidden imports so the dana package + OCR path resolve when frozen.
    hidden_imports = (
        "pytesseract",
        "dana",
        "dana.core_agent",
        "dana.stdio_boot",
        "dana.workspace",
        "dana.resources",
        "dana.ui",
        "dana.ui.logo",
        "dana.ui.startup_tray",
        "dana.tools",
        "dana.tools.vision",
        "dana.memory",
        "dana.memory.vault",
        "pystray",
    )

    args: list[str] = [
        str(entry),
        "--name=Dana",
        "--onedir",
        "--noconsole",
        "--noconfirm",
        "--clean",
        f"--distpath={ROOT / 'dist'}",
        f"--workpath={ROOT / 'build'}",
        f"--specpath={ROOT}",
        f"--paths={ROOT}",
    ]

    for pkg in collect_all_pkgs:
        args.append(f"--collect-all={pkg}")

    for mod in hidden_imports:
        args.append(f"--hidden-import={mod}")

    ico = ROOT / "dana" / "assets" / "donna.ico"
    if ico.is_file():
        args.append(f"--icon={ico}")

    # Bundle logo / icon trees into onedir extract root (Windows: src;dest).
    # Runtime resolution uses Path(__file__) and sys._MEIPASS in dana.ui.logo.
    ui_assets = ROOT / "dana" / "ui" / "assets"
    if ui_assets.is_dir():
        args.append(f"--add-data={ui_assets}{os.pathsep}dana/ui/assets")
    dana_assets = ROOT / "dana" / "assets"
    if dana_assets.is_dir():
        args.append(f"--add-data={dana_assets}{os.pathsep}dana/assets")

    print("[build_dana] Entry: run.py -> dana.core_agent.main (desktop UI)")
    print("[build_dana] PyInstaller args:")
    for a in args:
        print(f"  {a}")
    print("[build_dana] Starting Analysis (torch collect-all may take many minutes)...")

    pyi_main.run(args)

    exe = ROOT / "dist" / "Dana" / "Dana.exe"
    if exe.is_file():
        print(f"[build_dana] SUCCESS: {exe}")
        return 0
    print(f"[build_dana] FAILURE: expected exe not found at {exe}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

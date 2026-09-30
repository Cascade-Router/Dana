"""Programmatic PyInstaller onedir packaging for Dānā (Windows desktop).

Entry point choice
------------------
Uses ``scripts/launchers/launch_api_server.py``, the same backend entry the
launchers run: it starts uvicorn on ``dana.api.server:app`` (127.0.0.1:8000).
The desktop UI is the separate Tauri app in ``frontend/`` (``npm run tauri build``).

uvicorn imports the app by string and plugins load from
``dana/plugins/*/manifest.json`` by file path, so neither is visible to static
analysis: every ``dana`` submodule is collected explicitly and the package's
non-Python files (manifests, tools.json, templates, data) ship as data.

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

    entry = ROOT / "scripts" / "launchers" / "launch_api_server.py"
    if not entry.is_file():
        print(f"[build_dana] ERROR: missing entry script {entry}", file=sys.stderr)
        return 1

    # collect-all packages that fail or miss data/binaries under static analysis.
    collect_all_pkgs = (
        "torch",
        "onnxruntime",
        "sounddevice",
    )

    # Imported only by string / importlib at runtime (see module docstring).
    collect_submodules_pkgs = ("dana", "uvicorn")
    hidden_imports = ("dana.api.server",)

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

    for pkg in collect_submodules_pkgs:
        args.append(f"--collect-submodules={pkg}")

    for mod in hidden_imports:
        args.append(f"--hidden-import={mod}")

    # Non-Python package files (manifest.json, tools.json, templates, canned UX
    # .wav). An allowlist, so local runtime files (e.g. dana/memory/memory.db)
    # never get bundled.
    for data_file in sorted((ROOT / "dana").rglob("*")):
        if data_file.is_file() and data_file.suffix in {".json", ".jinja", ".jinja2", ".wav"}:
            dest = data_file.parent.relative_to(ROOT).as_posix()
            args.append(f"--add-data={data_file}{os.pathsep}{dest}")

    ico = ROOT / "assets" / "dana_logo.ico"
    if not ico.is_file():
        ico = ROOT / "dana" / "assets" / "dana_icon.ico"
    if ico.is_file():
        args.append(f"--icon={ico}")

    # Bundle logo / icon trees into onedir extract root (Windows: src;dest).
    # Runtime resolution uses get_resource_path / sys._MEIPASS in dana.resources.
    root_assets = ROOT / "assets"
    if root_assets.is_dir():
        args.append(f"--add-data={root_assets}{os.pathsep}assets")
    models = ROOT / "assets" / "models"
    if models.is_dir():
        args.append(f"--add-data={models}{os.pathsep}assets/models")
    for _stop in ("stop_dana.bat", "stop_dana.vbs", "start_dana.bat"):
        cand = ROOT / "scripts" / "launchers" / _stop
        if not cand.is_file():
            cand = ROOT / _stop
        if cand.is_file():
            args.append(f"--add-data={cand}{os.pathsep}.")

    print("[build_dana] Entry: scripts/launchers/launch_api_server.py -> uvicorn dana.api.server:app")
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

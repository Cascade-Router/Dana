# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Dana (entry: run.py).

Bundles CustomTkinter / Torch / Whisper-related packages and ships
``dana/tools`` + ``tts_models`` as runtime data.

PyInstaller injects Analysis / EXE / PYZ / COLLECT / BUNDLE into this namespace.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# SPEC = path to this .spec file (injected by PyInstaller).
_ROOT_CANDIDATE = globals().get("SPECPATH")
if _ROOT_CANDIDATE:
    ROOT = Path(_ROOT_CANDIDATE).resolve()
else:
    ROOT = Path(os.path.abspath(globals().get("SPEC", __file__))).resolve().parent
block_cipher = None

datas: list = []
binaries: list = []
hiddenimports: list[str] = [
    "customtkinter",
    "torch",
    "torchaudio",
    "torchvision",
    "whisper",
    "transformers",
    "pycaw",
    "comtypes",
    "dana",
    "dana.core_agent",
    "dana.agentic",
    "dana.tools",
    "dana.tools.setup_startup",
    "dana.tools.audio_switcher",
    "openwakeword",
    "sounddevice",
    "soundfile",
    "piper",
    "ultralytics",
    "cv2",
    "mss",
    "dana.vision_tools",
    "dana.ui",
    "dana.ui.trace_bus",
    "dana.ui.trace_window",
    "vision_tools",
    "PIL",
    "langchain_ollama",
    "langgraph",
]


def _safe_collect_all(package: str) -> None:
    """Merge collect_all outputs; ignore missing optional packages."""
    global datas, binaries, hiddenimports
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception as exc:  # noqa: BLE001
        print(f"[donna_build.spec] collect_all({package!r}) skipped: {exc}")
        return
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += list(pkg_hidden)


def _safe_collect_submodules(package: str) -> None:
    global hiddenimports
    try:
        hiddenimports += collect_submodules(package)
    except Exception as exc:  # noqa: BLE001
        print(f"[donna_build.spec] collect_submodules({package!r}) skipped: {exc}")


def _add_tree(src: Path, dest: str) -> None:
    """Add a directory tree as --add-data equivalent (src exists)."""
    global datas
    if not src.exists():
        print(f"[donna_build.spec] WARNING: missing data path {src} (skipped)")
        # Ensure destination exists in the bundle for runtime mkdir expectations.
        return
    datas.append((str(src), dest))


# Heavy / lazy-loaded packages that commonly break in frozen builds.
for _pkg in (
    "customtkinter",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "whisper",
    "ultralytics",
    "mss",
):
    _safe_collect_all(_pkg)

for _pkg in ("dana", "dana.tools", "pycaw", "comtypes"):
    _safe_collect_submodules(_pkg)

try:
    datas += collect_data_files("customtkinter")
except Exception as exc:  # noqa: BLE001
    print(f"[donna_build.spec] customtkinter data skipped: {exc}")

_add_tree(ROOT / "dana" / "tools", os.path.join("dana", "tools"))
_add_tree(ROOT / "tts_models", "tts_models")

# Silent shutdown runners (stop_dana.vbs hides the console; bat does PID filtering).
for _stop in ("stop_dana.bat", "stop_dana.vbs"):
    _sp = ROOT / _stop
    if _sp.is_file():
        datas.append((str(_sp), "."))
    else:
        print(f"[donna_build.spec] WARNING: missing {_sp}")

# GUI launcher (generated on install; ship if present in tree).
_start_bat = ROOT / "start_dana.bat"
if _start_bat.is_file():
    datas.append((str(_start_bat), "."))
else:
    print(f"[donna_build.spec] WARNING: missing {_start_bat}")

# Desktop / taskbar / shortcut branding (required for IconLocation + EXE icon).
_ico = ROOT / "dana" / "assets" / "donna.ico"
if _ico.is_file():
    datas.append((str(_ico), os.path.join("dana", "assets")))
else:
    print(f"[donna_build.spec] WARNING: missing {_ico} — EXE/shortcut icon unavailable")
_add_tree(ROOT / "dana" / "ui" / "assets", os.path.join("dana", "ui", "assets"))
_add_tree(ROOT / "dana" / "assets", os.path.join("dana", "assets"))

# Optional runtime assets (present when not gitignored / downloaded).
for _name in ("donna.onnx", "yolov8n.pt", "settings.json"):
    _p = ROOT / _name
    if _p.is_file():
        datas.append((str(_p), "."))

# Deduplicate while preserving order.
_seen: set[str] = set()
_deduped: list[str] = []
for _h in hiddenimports:
    if _h not in _seen:
        _seen.add(_h)
        _deduped.append(_h)
hiddenimports = _deduped

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Dana",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(_ico) if _ico.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Dana",
)

# macOS .app bundle (no-op import on other platforms — only emit on Darwin).
if sys.platform == "darwin" and BUNDLE is not None:
    app = BUNDLE(
        coll,
        name="Dana.app",
        icon=None,
        bundle_identifier="com.dana.agent",
        info_plist={
            "CFBundleName": "Dana",
            "CFBundleDisplayName": "Dana",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": "Dana needs the microphone for wake-word and STT.",
        },
    )

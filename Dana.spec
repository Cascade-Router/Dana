# -*- mode: python ; coding: utf-8 -*-
# Mirrors build_dana.py (which regenerates this file on each run). Entry is the
# backend launcher; the desktop UI is the separate Tauri app in frontend/.
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH)

datas = []
binaries = []
hiddenimports = ['dana.api.server']
for pkg in ('torch', 'onnxruntime', 'sounddevice'):
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# uvicorn imports the app by string; plugins load from manifest.json by path.
for pkg in ('dana', 'uvicorn'):
    hiddenimports += collect_submodules(pkg)

# Non-Python package files (allowlist, so local runtime files never ship).
for data_file in sorted((ROOT / 'dana').rglob('*')):
    if data_file.is_file() and data_file.suffix in {'.json', '.jinja', '.jinja2', '.wav'}:
        datas.append((str(data_file), data_file.parent.relative_to(ROOT).as_posix()))
if (ROOT / 'assets').is_dir():
    datas.append((str(ROOT / 'assets'), 'assets'))
for name in ('stop_dana.bat', 'stop_dana.vbs', 'start_dana.bat'):
    launcher = ROOT / 'scripts' / 'launchers' / name
    if launcher.is_file():
        datas.append((str(launcher), '.'))


a = Analysis(
    [str(ROOT / 'scripts' / 'launchers' / 'launch_api_server.py')],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Dana',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(ROOT / 'assets' / 'dana_logo.ico')],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Dana',
)

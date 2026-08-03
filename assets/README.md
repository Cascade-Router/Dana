# Dānā brand assets (SSoT)

Canonical desktop / tray / EXE icon files live here:

- `dana_logo.png` — 512×512 squircle master
- `dana_logo.ico` — multi-resolution 16 / 32 / 48 / 256

Regenerate::

    python scripts/generate_dana_icon.py

Runtime resolution uses ``dana.resources.get_resource_path("assets/dana_logo.ico")``
(and the ``.png`` sibling) so PyInstaller ``sys._MEIPASS`` packaging stays intact.

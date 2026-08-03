# Donna UI Assets — Premium Logo

Place your high-quality, artistically rendered logo here.

## Preferred filenames (first match wins)

1. `dana_logo_highres.png` — recommended master (transparent PNG, ≥512×512)
2. `donna_logo_highres.png`
3. `donna_logo.png`
4. `orb_logo.png` — fallback seed

Windows desktop / taskbar / tray branding uses the high-contrast squircle
ICO (regenerate via ``python scripts/generate_dana_icon.py``):

- `dana/ui/assets/dana_icon.png` — 512×512 master squircle
- `dana/assets/dana_icon.ico` — sizes 16 / 32 / 48 / 64 / 128 / 256
- `dana/assets/donna.ico` — legacy compat copy of the same ICO

## Format notes

- Prefer **transparent PNG** with clean anti-aliased edges.
- SVG sources should be **exported to a high-resolution PNG** before drop-in
  (CustomTkinter / Tk load raster via Pillow).
- The runtime loader scales with `Image.Resampling.LANCZOS` into `CTkImage`
  for the Dashboard / header, and into `PhotoImage` for the Assistive Orb.

## Orb / chroma-key

The Assistive Orb uses a Windows transparent-color Toplevel. Prefer a
pre-rendered transparent PNG tinted at runtime. If chroma-key artifacts
appear, the orb falls back to a mathematically smooth Canvas polygon mark
(`smooth=True`) instead of Unicode glyphs.

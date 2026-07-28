# Donna UI Assets — Premium Logo

Place your high-quality, artistically rendered logo here.

## Preferred filenames (first match wins)

1. `dana_logo_highres.png` — recommended master (transparent PNG, ≥512×512)
2. `donna_logo_highres.png`
3. `donna_logo.png`
4. `orb_logo.png` — fallback seed

Windows desktop / taskbar / tray branding also uses the multi-resolution
ICO generated from the master PNG:

- `dana/assets/donna.ico` — sizes 16 / 32 / 48 / 64 / 128 / 256

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

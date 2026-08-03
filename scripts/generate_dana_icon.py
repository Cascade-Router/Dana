"""One-shot: build high-contrast squircle dana_icon.png + dana_icon.ico.

Dark canvas ``#0a0e17``, 1px border ``#1e293b``, centered emblem from
``dana/ui/assets`` logo candidates (via ``dana.ui.logo.resolve_logo_path``).

Outputs:
  - dana/ui/assets/dana_icon.png  (512 master)
  - dana/assets/dana_icon.ico     (16/32/48/64/128/256)
  - dana/assets/donna.ico         (same multi-res, legacy path compat)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BG = (0x0A, 0x0E, 0x17, 255)
BORDER = (0x1E, 0x29, 0x3B, 255)
MASTER = 512
ICO_SIZES = (16, 32, 48, 64, 128, 256)
# Emblem inset as fraction of canvas (leaves squircle margin).
EMBLEM_FRAC = 0.62


def _squircle_radius(size: int) -> int:
    # ~22% corner radius reads as a modern app squircle at all sizes.
    return max(2, int(round(size * 0.22)))


def _draw_squircle(size: int) -> "Image.Image":
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = _squircle_radius(size)
    # Outer fill (border color), then inset fill (bg) → 1px border at any scale.
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=r, fill=BORDER)
    inset = 1
    if size <= 24:
        # At 16px a true 1px border eats the mark; keep a hairline edge.
        inset = 1
    inner_r = max(1, r - inset)
    draw.rounded_rectangle(
        (inset, inset, size - 1 - inset, size - 1 - inset),
        radius=inner_r,
        fill=BG,
    )
    return img


def _load_emblem(max_side: int):
    from PIL import Image

    from dana.ui.logo import make_transparent_logo, resolve_logo_path

    path = resolve_logo_path()
    if path is None or not path.is_file():
        raise SystemExit("No logo asset found under dana/ui/assets (or dana/assets).")
    emblem = make_transparent_logo(Image.open(path).convert("RGBA"))
    # Fit emblem into a square box without upscaling past source when possible.
    box = max(8, int(max_side))
    emblem.thumbnail((box, box), Image.Resampling.LANCZOS)
    return emblem


def build_master(size: int = MASTER):
    from PIL import Image

    canvas = _draw_squircle(size)
    emblem_box = max(8, int(round(size * EMBLEM_FRAC)))
    emblem = _load_emblem(emblem_box)
    # Center paste.
    ew, eh = emblem.size
    x = (size - ew) // 2
    y = (size - eh) // 2
    canvas.alpha_composite(emblem, (x, y))
    return canvas


def build_ico_frames(master) -> list:
    frames: list = []
    for side in ICO_SIZES:
        # Re-composite each size so the 1px border + emblem stay crisp.
        base = _draw_squircle(side)
        emblem_box = max(6, int(round(side * EMBLEM_FRAC)))
        emblem = _load_emblem(emblem_box)
        ew, eh = emblem.size
        base.alpha_composite(emblem, ((side - ew) // 2, (side - eh) // 2))
        frames.append(base.convert("RGBA"))
    return frames


def write_png_ico(path: Path, frames: list) -> None:
    """Write a Vista+ PNG-compressed multi-resolution ``.ico``."""
    import io
    import struct

    blobs: list[bytes] = []
    for frame in frames:
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        blobs.append(buf.getvalue())

    count = len(frames)
    offset = 6 + 16 * count
    parts = [struct.pack("<HHH", 0, 1, count)]
    for frame, blob in zip(frames, blobs):
        w, h = frame.size
        parts.append(
            struct.pack(
                "<BBBBHHII",
                0 if w >= 256 else w,
                0 if h >= 256 else h,
                0,
                0,
                1,
                32,
                len(blob),
                offset,
            )
        )
        offset += len(blob)
    parts.extend(blobs)
    path.write_bytes(b"".join(parts))


def main() -> int:
    ui_assets = ROOT / "dana" / "ui" / "assets"
    dana_assets = ROOT / "dana" / "assets"
    ui_assets.mkdir(parents=True, exist_ok=True)
    dana_assets.mkdir(parents=True, exist_ok=True)

    master = build_master(MASTER)
    png_path = ui_assets / "dana_icon.png"
    master.save(png_path, format="PNG")
    print(f"[generate_dana_icon] wrote {png_path} ({master.size[0]}x{master.size[1]})")

    frames = build_ico_frames(master)
    ico_path = dana_assets / "dana_icon.ico"
    write_png_ico(ico_path, frames)
    print(f"[generate_dana_icon] wrote {ico_path} sizes={list(ICO_SIZES)}")

    # Legacy path still referenced by some shortcuts / older resolvers.
    legacy = dana_assets / "donna.ico"
    write_png_ico(legacy, frames)
    print(f"[generate_dana_icon] wrote {legacy} (compat)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

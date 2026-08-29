"""Generate a high-contrast test image for the image-to-3D pipeline."""
from pathlib import Path
from PIL import Image, ImageDraw

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "freecad_output" / "test_input.png"


def main():
    size = 512
    img = Image.new("RGB", (size, size), color="black")
    draw = ImageDraw.Draw(img)
    margin = size // 6
    draw.ellipse([margin, margin, size - margin, size - margin], fill="white")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_PATH)
    print(f"Saved test asset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

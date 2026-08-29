"""Generate a raw, watertight cube .obj to stand in for a cloud-generated
mesh, so the FreeCAD solidify/boolean back-half of the pipeline can be
exercised without depending on the flaky Hugging Face Image-to-3D endpoints.

Positioned to overlap the 'BaseBox' (0,0,0)-(50,50,50) the E2E prompt asks
DANA to create, so the boolean union is a real geometric merge rather than
two disjoint solids in a compound.
"""
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "freecad_output" / "synthetic_part.obj"

# Overlaps BaseBox by 20mm on each axis and extends 10mm beyond it, so a
# successful union's bounding box should grow from (0,0,0)-(50,50,50) to
# (0,0,0)-(60,60,60).
XMIN, YMIN, ZMIN = 30.0, 30.0, 30.0
XMAX, YMAX, ZMAX = 60.0, 60.0, 60.0

_V = {
    "000": (XMIN, YMIN, ZMIN),
    "100": (XMAX, YMIN, ZMIN),
    "110": (XMAX, YMAX, ZMIN),
    "010": (XMIN, YMAX, ZMIN),
    "001": (XMIN, YMIN, ZMAX),
    "101": (XMAX, YMIN, ZMAX),
    "111": (XMAX, YMAX, ZMAX),
    "011": (XMIN, YMAX, ZMAX),
}

# Each quad listed CCW as seen from outside the cube (outward-facing
# normals), so the sewn shell is consistently oriented and closed —
# required for Part.makeSolid to succeed instead of raising "no shells or
# compsolids found" / producing a non-watertight solid.
_QUADS = [
    ("000", "100", "101", "001"),  # front  (y = YMIN), normal -y
    ("110", "010", "011", "111"),  # back   (y = YMAX), normal +y
    ("010", "000", "001", "011"),  # left   (x = XMIN), normal -x
    ("100", "110", "111", "101"),  # right  (x = XMAX), normal +x
    ("010", "110", "100", "000"),  # bottom (z = ZMIN), normal -z
    ("001", "101", "111", "011"),  # top    (z = ZMAX), normal +z
]


def main():
    vertex_keys = list(_V.keys())
    index_of = {key: i + 1 for i, key in enumerate(vertex_keys)}  # OBJ is 1-indexed

    lines = ["# synthetic watertight cube for E2E back-half testing"]
    for key in vertex_keys:
        x, y, z = _V[key]
        lines.append(f"v {x} {y} {z}")

    for quad in _QUADS:
        a, b, c, d = (index_of[k] for k in quad)
        lines.append(f"f {a} {b} {c}")
        lines.append(f"f {a} {c} {d}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved synthetic mesh to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

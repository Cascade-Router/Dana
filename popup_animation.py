"""Bouncing-ball Tkinter animation (Epic 1 artifact)."""

from __future__ import annotations

import os
import tkinter as tk


def main() -> None:
    root = tk.Tk()
    root.title("Dānā — Bouncing Ball")
    width, height = 420, 320
    root.geometry(f"{width}x{height}")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=width, height=height, bg="#0f172a", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    radius = 18
    x, y = 60.0, 80.0
    vx, vy = 4.2, 3.4
    ball = canvas.create_oval(
        x - radius,
        y - radius,
        x + radius,
        y + radius,
        fill="#38bdf8",
        outline="#e0f2fe",
        width=2,
    )

    # Optional auto-close for harness / CI (frames). 0 = run until window closed.
    max_frames = int(os.environ.get("DONNA_ANIMATION_FRAMES") or "0")
    frame = {"n": 0}

    def tick() -> None:
        nonlocal x, y, vx, vy
        x += vx
        y += vy
        if x - radius <= 0 or x + radius >= width:
            vx = -vx
            x = max(radius, min(width - radius, x))
        if y - radius <= 0 or y + radius >= height:
            vy = -vy
            y = max(radius, min(height - radius, y))
        canvas.coords(ball, x - radius, y - radius, x + radius, y + radius)
        frame["n"] += 1
        if max_frames > 0 and frame["n"] >= max_frames:
            root.destroy()
            return
        root.after(16, tick)

    root.after(16, tick)
    root.mainloop()


if __name__ == "__main__":
    main()

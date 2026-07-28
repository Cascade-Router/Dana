"""Stage 6.4 — Persona Mixer GUI (live Receptionist trait sliders).

Floating CustomTkinter panel that writes 0–100 trait values into the
Blackboard ``persona_mixer`` table on slider release (and throttled drag).
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import customtkinter as ctk

from donna.memory.blackboard import (
    PERSONA_MIXER_DEFAULTS,
    get_persona_mixer,
    set_persona_trait,
)

# Display label → DB trait_name (Stage 8.5 adds Autonomy / Creativity).
_SLIDER_SPECS: tuple[tuple[str, str], ...] = (
    ("Autonomy", "autonomy"),
    ("Verbosity", "verbosity"),
    ("Creativity", "creativity"),
    ("Humor", "humor"),
    ("Flirt", "flirt"),
    ("Tech Depth", "technical_depth"),
)


class PersonaMixerApp(ctk.CTk):
    """Lightweight always-on-top mixer panel."""

    def __init__(
        self,
        *,
        db_path=None,
        on_change: Callable[[str, int], None] | None = None,
        throttle_s: float = 0.15,
    ) -> None:
        super().__init__()
        self._db_path = db_path
        self._on_change = on_change
        self._throttle_s = max(0.05, float(throttle_s))
        self._last_write: dict[str, float] = {}
        self._value_labels: dict[str, ctk.CTkLabel] = {}
        self._sliders: dict[str, ctk.CTkSlider] = {}

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.title("Dānā Persona Mixer")
        self.geometry("320x360+80+80")
        self.minsize(280, 320)
        self.attributes("-topmost", True)
        self.configure(fg_color=("#1a1a1e", "#1a1a1e"))

        header = ctk.CTkLabel(
            self,
            text="Receptionist Persona",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        header.pack(padx=16, pady=(14, 4), anchor="w")
        hint = ctk.CTkLabel(
            self,
            text="Sliders write live to blackboard.db",
            font=ctk.CTkFont(size=11),
            text_color="#9aa0a6",
        )
        hint.pack(padx=16, pady=(0, 8), anchor="w")

        state = get_persona_mixer(self._db_path)
        for label, key in _SLIDER_SPECS:
            frame = ctk.CTkFrame(self, fg_color="transparent")
            frame.pack(fill="x", padx=16, pady=6)
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x")
            ctk.CTkLabel(row, text=label, width=90, anchor="w").pack(side="left")
            val_lbl = ctk.CTkLabel(row, text=str(state.get(key, 0)), width=36)
            val_lbl.pack(side="right")
            self._value_labels[key] = val_lbl

            slider = ctk.CTkSlider(
                frame,
                from_=0,
                to=100,
                number_of_steps=100,
                progress_color="#00ADB5",
                button_color="#E5E7EB",
                button_hover_color="#F9FAFB",
                fg_color="#2A2A3C",
                command=lambda v, k=key: self._on_drag(k, v),
            )
            slider.set(float(state.get(key, PERSONA_MIXER_DEFAULTS.get(key, 50))))
            slider.pack(fill="x", pady=(4, 0))
            slider.bind(
                "<ButtonRelease-1>",
                lambda _e, k=key: self._commit(k, force=True),
            )
            self._sliders[key] = slider

        refresh = ctk.CTkButton(
            self, text="Reload from DB", width=120, command=self.reload_from_db
        )
        refresh.pack(pady=(12, 16))

    def _on_drag(self, trait: str, value: float) -> None:
        n = int(round(float(value)))
        lbl = self._value_labels.get(trait)
        if lbl is not None:
            lbl.configure(text=str(n))
        now = time.monotonic()
        last = self._last_write.get(trait, 0.0)
        if now - last >= self._throttle_s:
            self._commit(trait, force=False)

    def _commit(self, trait: str, *, force: bool) -> None:
        slider = self._sliders.get(trait)
        if slider is None:
            return
        n = int(round(float(slider.get())))
        now = time.monotonic()
        if not force and (now - self._last_write.get(trait, 0.0)) < self._throttle_s:
            return
        set_persona_trait(trait, n, db_path=self._db_path)
        self._last_write[trait] = now
        lbl = self._value_labels.get(trait)
        if lbl is not None:
            lbl.configure(text=str(n))
        if self._on_change is not None:
            try:
                self._on_change(trait, n)
            except Exception:  # noqa: BLE001
                pass

    def reload_from_db(self) -> None:
        state = get_persona_mixer(self._db_path)
        for key, slider in self._sliders.items():
            n = int(state.get(key, PERSONA_MIXER_DEFAULTS.get(key, 50)))
            slider.set(float(n))
            lbl = self._value_labels.get(key)
            if lbl is not None:
                lbl.configure(text=str(n))

    def apply_values(self, values: dict[str, int]) -> None:
        """Programmatic set (tests / harness) — updates UI + DB."""
        for key, value in (values or {}).items():
            if key not in self._sliders:
                continue
            n = max(0, min(100, int(value)))
            self._sliders[key].set(float(n))
            set_persona_trait(key, n, db_path=self._db_path)
            lbl = self._value_labels.get(key)
            if lbl is not None:
                lbl.configure(text=str(n))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Donna Persona Mixer GUI")
    parser.add_argument(
        "--db",
        default=None,
        help="Optional blackboard.db path (default: workspace memory path)",
    )
    args = parser.parse_args(argv)
    app = PersonaMixerApp(db_path=args.db)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

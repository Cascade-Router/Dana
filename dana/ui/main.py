"""Smoke entrypoint for Dana Control Dashboard UI.

Usage::

    python -m dana.ui.main

Headless / CI: constructs ``DonnaGUI`` and exits after a short idle so the
process does not hang forever. Pass ``--stay`` to keep the window open.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dana Control Dashboard smoke UI")
    parser.add_argument(
        "--stay",
        action="store_true",
        help="Keep the window open (interactive). Default: destroy after paint.",
    )
    parser.add_argument(
        "--ms",
        type=int,
        default=400,
        help="Milliseconds to idle before auto-close when not --stay (default 400).",
    )
    args = parser.parse_args(argv)

    try:
        from dana.ui.theme import apply_dana_ctk_theme

        apply_dana_ctk_theme()
    except Exception:  # noqa: BLE001
        pass

    try:
        from dana.core_agent import DonnaGUI
    except Exception as exc:  # noqa: BLE001
        print(f"DonnaGUI unavailable: {exc}", file=sys.stderr)
        return 1

    try:
        app = DonnaGUI()
    except Exception as exc:  # noqa: BLE001
        print(f"Tk unavailable: {exc}", file=sys.stderr)
        return 1

    try:
        # Show briefly for smoke; constructor withdraws by default.
        try:
            app.deiconify()
            app.lift()
        except Exception:  # noqa: BLE001
            pass

        if args.stay:
            app.mainloop()
            return 0

        def _quit() -> None:
            try:
                app.destroy()
            except Exception:  # noqa: BLE001
                pass

        app.after(max(50, int(args.ms)), _quit)
        app.mainloop()
        print("ui smoke ok")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"UI smoke failed: {exc}", file=sys.stderr)
        try:
            app.destroy()
        except Exception:  # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

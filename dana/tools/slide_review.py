"""Slide review pipeline — capture → local MoA (vision→reasoner) → type.

Composite tool ``evaluate_slide_and_type`` orchestrates:
  1. Capture the active slide (PNG bytes)
  2. Cascade local MoA: vision model extracts text, DeepSeek/reasoner judges the rule
  3. Format a concise comment
  4. ``execute_os_keystrokes`` after a focus delay (Chrome / active window)
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

_DEFAULT_RULE = "The slide must have a clear title and fewer than 30 words of body text."
_FOCUS_DELAY_DEFAULT = 1.5
_MAX_COMMENT_CHARS = 280


def _format_comment(verdict: str, *, pass_fail: str, word_count: int | None = None) -> str:
    verdict = re.sub(r"\s+", " ", (verdict or "").strip())
    prefix = (pass_fail or "REVIEW").strip().upper()
    if prefix not in ("PASS", "FAIL", "UNCLEAR", "REVIEW"):
        prefix = "REVIEW"
    wc = f" words≈{word_count}" if word_count is not None else ""
    comment = f"[{prefix}{wc}] {verdict}".strip()
    if len(comment) > _MAX_COMMENT_CHARS:
        comment = comment[: _MAX_COMMENT_CHARS - 1].rstrip() + "…"
    return comment


def _parse_reasoner_output(raw: str, *, vision_text: str = "") -> dict[str, Any]:
    text = (raw or "").strip()
    verdict_m = re.search(r"VERDICT:\s*(PASS|FAIL|UNCLEAR)", text, re.I)
    words_m = re.search(r"WORD_COUNT:\s*(-?\d+)", text, re.I)
    comment_m = re.search(r"COMMENT:\s*(.+)", text, re.I)
    verdict = (verdict_m.group(1).upper() if verdict_m else "UNCLEAR")
    word_count = int(words_m.group(1)) if words_m else None
    if word_count is not None and word_count < 0:
        word_count = None
    if word_count is None and vision_text:
        word_count = len(re.findall(r"[A-Za-z0-9']+", vision_text))
    comment = (
        comment_m.group(1).strip()
        if comment_m
        else (text.splitlines()[-1].strip() if text else "No comment produced.")
    )
    return {"verdict": verdict, "word_count": word_count, "comment": comment, "raw": text}


def evaluate_slide_and_type(
    rule: str = "",
    *,
    focus_delay_sec: float | None = None,
    dry_run: bool | None = None,
) -> str:
    """Sequential pipeline: capture PNG → MoA vision+reasoner → type comment."""
    from dana.cascade_router import run_visual_moa
    from dana.tools.os_control import capture_screen_png_bytes, execute_os_keystrokes
    from dana.paths import CAPTURES_DIR

    rule = (rule or _DEFAULT_RULE).strip()
    delay = (
        float(focus_delay_sec)
        if focus_delay_sec is not None
        else float(os.environ.get("DANA_SLIDE_FOCUS_DELAY", _FOCUS_DELAY_DEFAULT))
    )
    delay = max(0.0, min(delay, 5.0))

    try:
        from dana.logging import log as _log

        _log("SlideReview", f"Step1 capture PNG rule={rule[:80]!r}")
    except Exception:
        pass

    try:
        png = capture_screen_png_bytes()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: evaluate_slide_and_type capture failed: {exc}"

    try:
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        (CAPTURES_DIR / "last_slide_capture.png").write_bytes(png)
    except Exception:
        pass

    try:
        from dana.logging import log as _log

        _log("SlideReview", "Step2 Cascade MoA (vision → reasoner)")
    except Exception:
        pass

    moa = run_visual_moa(
        png,
        rule=rule,
        task="Evaluate this presentation slide against the RULE.",
        vision_prompt=(
            "This is a presentation slide. Extract the title and all body text. "
            "Estimate body word count. Note whether a clear title is present."
        ),
    )
    vision = str(moa.get("vision_text") or "")
    parsed = _parse_reasoner_output(str(moa.get("final") or ""), vision_text=vision)
    comment = _format_comment(
        parsed.get("comment") or "",
        pass_fail=str(parsed.get("verdict") or "REVIEW"),
        word_count=parsed.get("word_count"),
    )
    route = str(moa.get("route") or "moa")

    try:
        from dana.logging import log as _log

        _log(
            "SlideReview",
            f"Step3/4 comment ready ({len(comment)} chars); "
            f"focus_delay={delay:.1f}s then execute_os_keystrokes route={route}",
        )
    except Exception:
        pass

    if dry_run is True or (
        dry_run is None
        and os.environ.get("DANA_OS_DRY_RUN", "").strip().lower() in ("1", "true", "yes")
    ):
        return (
            f"OK: evaluate_slide_and_type dry_run route={route}\n"
            f"VERDICT={parsed.get('verdict')}\n"
            f"COMMENT={comment}\n"
            f"VISION_SNIP={vision[:400]}"
        )

    if delay > 0:
        time.sleep(delay)

    typed = execute_os_keystrokes(comment)
    try:
        from dana.logging import log as _log

        _log(
            "SlideReview",
            f"Step4 typed via SendInput: {typed[:200]}",
        )
    except Exception:
        pass
    if typed.upper().startswith("ERROR:"):
        return (
            f"ERROR: evaluate_slide_and_type typed failed: {typed}\n"
            f"COMMENT_READY={comment}\n"
            f"VERDICT={parsed.get('verdict')} route={route}"
        )

    return (
        f"OK: evaluate_slide_and_type route={route} "
        f"verdict={parsed.get('verdict')}\n"
        f"COMMENT={comment}\n"
        f"{typed}\n"
        f"VISION_SNIP={vision[:240]}"
    )

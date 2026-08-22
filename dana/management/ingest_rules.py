"""Stage 8.3 — One-off OCR → Feather project rules ingestion.

Reads the latest Florence visual context from the Blackboard, cleans it with a
local LLM, and overwrites ``dana/knowledge/feather_project_rules.md``.

Usage (vision poller running, Feather instructions visible on screen)::

    python -m dana.management.ingest_rules
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dana.management.jason_supervisor import feather_project_rules_path
from dana.memory.blackboard import (
    BLACKBOARD_DB_PATH,
    PERCEPTION_OCR_KEY,
    SCHEMA_OCR_V1,
    init_blackboard,
    publish_perception_ocr,
    read_perception_ocr,
)

ACK_LINE = "Visual rule ingestion complete. Markdown file updated."

# Patch 8.3.3 — reject / wait on stale Vision Poller frames.
FRESHNESS_MAX_AGE_S = 10.0
FRESHNESS_POLL_S = 2.0

_OCR_SYSTEM_PROMPT = (
    "You are a formatting assistant. The user has provided raw OCR text scraped "
    "from a screen containing project instructions/rules. Fix any typos caused "
    "by the OCR, format it into a clean, readable Markdown document with bullet "
    "points, and output ONLY the markdown text. Do not add any conversational "
    "filler."
)


def _log(msg: str) -> None:
    print(f"[ingest_rules] {msg}", flush=True)


def _dry_llm() -> bool:
    return os.environ.get("DANA_INGEST_DRY_LLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _skip_freshness() -> bool:
    return os.environ.get("DANA_INGEST_SKIP_FRESHNESS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _parse_updated_at(raw: str) -> datetime | None:
    """Parse Blackboard ISO timestamps (UTC) into aware datetimes."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _age_seconds(updated_at: datetime | None, *, now: datetime | None = None) -> float | None:
    if updated_at is None:
        return None
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - updated_at).total_seconds())


def _force_florence_ocr(*, db_path: Path | str | None = None) -> str:
    """Run on-demand Florence OCR and publish ``perception.ocr``."""
    from dana.tools.visual_tools import ocr_with_region

    _log("Forcing Florence OCR publish → perception.ocr")
    observation = str(ocr_with_region(query="") or "")
    # ocr_with_region already publishes; re-read typed topic.
    row = read_perception_ocr(db_path=db_path)
    if row and str(row.get("text") or "").strip():
        return str(row.get("text") or "").strip()
    # Fallback: if publish was mocked/unavailable, still accept observation.
    if observation.strip() and not observation.lstrip().startswith("[Vision Output]"):
        publish_perception_ocr(
            observation,
            producer="ingest_rules",
            model="florence-2",
            db_path=db_path,
        )
        return observation.strip()
    return ""


def read_latest_visual_row(
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return newest typed OCR row: ``{text, updated_at, age_seconds}``.

    Reads ``perception.ocr`` only (schema ``perception.ocr.v1``). YOLO object
    prose on ``perception.objects`` / legacy ``latest_visual_context`` is rejected.
    """
    init_blackboard(db_path or BLACKBOARD_DB_PATH)
    row = read_perception_ocr(db_path=db_path)
    if not row:
        return {
            "text": "",
            "updated_at": "",
            "updated_at_dt": None,
            "age_seconds": None,
            "schema": "",
            "key": PERCEPTION_OCR_KEY,
        }
    ts = _parse_updated_at(str(row.get("updated_at") or ""))
    return {
        "text": str(row.get("text") or "").strip(),
        "updated_at": str(row.get("updated_at") or ""),
        "updated_at_dt": ts,
        "age_seconds": row.get("age_seconds"),
        "schema": SCHEMA_OCR_V1,
        "key": PERCEPTION_OCR_KEY,
    }


def read_latest_ocr_text(*, db_path: Path | str | None = None) -> str:
    """Return the newest typed ``perception.ocr`` corpus string."""
    return str(read_latest_visual_row(db_path=db_path).get("text") or "")


def wait_for_fresh_ocr(
    *,
    db_path: Path | str | None = None,
    max_age_s: float = FRESHNESS_MAX_AGE_S,
    poll_s: float = FRESHNESS_POLL_S,
    force_capture: bool = True,
) -> str:
    """Block until typed Florence OCR is younger than ``max_age_s``; return text.

    Sidekick contract: never ingest YOLO ``[Vision Output]``. When OCR is
    missing/stale, force one Florence capture (ingest CLI path).
    """
    if _skip_freshness():
        text = read_latest_ocr_text(db_path=db_path)
        if text:
            return text
        if force_capture:
            return _force_florence_ocr(db_path=db_path)
        return ""

    forced = False
    while True:
        row = read_latest_visual_row(db_path=db_path)
        text = str(row.get("text") or "").strip()
        age = row.get("age_seconds")
        if text and age is not None and float(age) <= float(max_age_s):
            _log(f"fresh OCR frame age={float(age):.1f}s chars={len(text)}")
            return text

        if force_capture and not forced:
            forced = True
            text = _force_florence_ocr(db_path=db_path)
            if text:
                row2 = read_latest_visual_row(db_path=db_path)
                age2 = row2.get("age_seconds")
                if text and (age2 is None or float(age2) <= float(max_age_s)):
                    return text

        # Missing / unparseable timestamp → treat as infinitely stale.
        stale_s = int(round(float(age))) if age is not None else int(max_age_s) + 1
        _log(
            f"ERROR: Visual data is stale ({stale_s} seconds old). "
            "Need fresh perception.ocr (Florence). Waiting for a fresh frame..."
        )
        time.sleep(float(poll_s))


def strip_llm_markdown(raw: str) -> str:
    """Drop conversational wrappers / fences; keep markdown body only."""
    text = (raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?is)<think>.*?</think>", " ", text).strip()
    fence = re.match(r"(?is)^```(?:markdown|md)?\s*(.*?)\s*```$", text)
    if fence:
        text = fence.group(1).strip()
    # Drop a single leading/trailing chat sentence if present.
    text = re.sub(
        r"(?is)^\s*(sure[,!]?\s+|here(?:'s| is)\s+(?:the\s+)?(?:cleaned|formatted)?"
        r"\s*(?:markdown|text|rules)?[:\s]*)",
        "",
        text,
    ).strip()
    return text


def clean_ocr_with_llm(
    ocr_text: str,
    *,
    clean_fn: Callable[[str], str] | None = None,
) -> str:
    """Send raw OCR to the Receptionist Llama chat model; return cleaned markdown."""
    body = (ocr_text or "").strip()
    if not body:
        raise ValueError("No OCR / visual context text available on the Blackboard")

    if clean_fn is not None:
        return strip_llm_markdown(clean_fn(body))

    if _dry_llm():
        # Deterministic offline path for CI / headless smoke.
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        bullets = "\n".join(f"- {ln}" for ln in lines) or f"- {body}"
        return strip_llm_markdown(f"# Project rules\n\n{bullets}\n")

    # Patch 8.3.2 — use Chat Node Llama (instruct), not DeepSeek-R1 reasoner.
    from dana.cascade_router import local_model_name
    from dana.core.constants import OLLAMA_MODEL
    from dana.core.model_provider import ModelProvider

    model = (local_model_name() or "").strip() or OLLAMA_MODEL

    # Let Florence / vision poller finish the current frame before Ollama VRAM spike.
    _log("Waiting 3 seconds for Vision Poller to clear VRAM...")
    time.sleep(3)

    raw = ModelProvider(local_model=model).complete(
        [
            {"role": "system", "content": _OCR_SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ],
        allow_cloud=False,
    )
    cleaned = strip_llm_markdown(str(raw or ""))
    if not cleaned:
        raise RuntimeError(f"LLM returned empty markdown (model={model!r})")
    return cleaned


def write_feather_rules(markdown: str, *, dest: Path | None = None) -> Path:
    """Overwrite ``feather_project_rules.md`` with cleaned markdown."""
    path = Path(dest) if dest is not None else feather_project_rules_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (markdown or "").strip()
    if body and not body.endswith("\n"):
        body += "\n"
    path.write_text(body, encoding="utf-8")
    return path


def acknowledge_ingestion(*, speak: bool = True) -> None:
    """Console print + optional Dana TTS cue."""
    _log(ACK_LINE)
    print(ACK_LINE, flush=True)
    if not speak:
        return
    try:
        from dana.audio.multi_voice_tts import synthesize_speech
        from dana.ui.audio_mixer import play_dana

        wav = synthesize_speech(ACK_LINE, voice_id="dana")
        play_dana(wav, block=False)
    except Exception as exc:  # noqa: BLE001
        _log(f"TTS cue skipped: {exc}")


def ingest_rules(
    *,
    db_path: Path | str | None = None,
    dest: Path | None = None,
    clean_fn: Callable[[str], str] | None = None,
    speak: bool = True,
) -> Path:
    """Full pipeline: Blackboard OCR → LLM clean → overwrite Feather rules."""
    ocr = wait_for_fresh_ocr(db_path=db_path)
    _log(f"read OCR chars={len(ocr)} db={db_path or BLACKBOARD_DB_PATH}")
    cleaned = clean_ocr_with_llm(ocr, clean_fn=clean_fn)
    out = write_feather_rules(cleaned, dest=dest)
    _log(f"wrote rules path={out} chars={len(cleaned)}")
    acknowledge_ingestion(speak=speak)
    return out


def main(argv: list[str] | None = None) -> int:
    _ = argv  # reserved for future flags
    try:
        ingest_rules()
    except Exception as exc:  # noqa: BLE001
        _log(f"FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

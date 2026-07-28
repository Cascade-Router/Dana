"""Stage 8.3 — OCR knowledge ingestion into Feather project rules."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dana.management.ingest_rules import (
    ACK_LINE,
    clean_ocr_with_llm,
    ingest_rules,
    read_latest_ocr_text,
    strip_llm_markdown,
    wait_for_fresh_ocr,
    write_feather_rules,
)
from dana.memory.blackboard import (
    PERCEPTION_OCR_KEY,
    init_blackboard,
    publish_perception_ocr,
)


def test_read_latest_ocr_from_sensor_state(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    publish_perception_ocr(
        "Rule 1: Keep slides under 30 words.\nRule 2: No clipart.",
        producer="test",
        db_path=db,
    )
    text = read_latest_ocr_text(db_path=db)
    assert "under 30 words" in text
    assert "No clipart" in text


def test_rejects_yolo_objects_as_ocr(tmp_path: Path) -> None:
    """Sidekick contract: YOLO prose must never feed rule ingestion."""
    from dana.memory.blackboard import publish_perception_objects

    db = tmp_path / "bb.db"
    init_blackboard(db)
    publish_perception_objects(
        "[Vision Output] Detected: 1 book.",
        producer="test",
        db_path=db,
    )
    assert read_latest_ocr_text(db_path=db) == ""


def test_strip_llm_markdown_fences() -> None:
    raw = "```markdown\n- First rule\n- Second rule\n```"
    assert strip_llm_markdown(raw) == "- First rule\n- Second rule"


def test_clean_ocr_dry_llm(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_INGEST_DRY_LLM", "1")
    out = clean_ocr_with_llm("Max 30 words\nCite sources")
    assert out.startswith("# Project rules")
    assert "- Max 30 words" in out
    assert "- Cite sources" in out


def test_ingest_rules_end_to_end(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_INGEST_DRY_LLM", "1")
    monkeypatch.setenv("DONNA_AUDIO_DRY_RUN", "1")
    monkeypatch.setenv("DONNA_INGEST_SKIP_FRESHNESS", "1")
    db = tmp_path / "bb.db"
    dest = tmp_path / "feather_project_rules.md"
    init_blackboard(db)
    publish_perception_ocr(
        "OCR: Keep under 30 words. Cite all sources.",
        producer="test",
        db_path=db,
    )
    out = ingest_rules(db_path=db, dest=dest, speak=True)
    assert out == dest
    body = dest.read_text(encoding="utf-8")
    assert "Keep under 30 words" in body
    assert body.endswith("\n")


def test_ack_line_constant() -> None:
    assert ACK_LINE == "Visual rule ingestion complete. Markdown file updated."


def test_write_feather_rules_overwrite(tmp_path: Path) -> None:
    dest = tmp_path / "rules.md"
    write_feather_rules("old", dest=dest)
    write_feather_rules("- new rule", dest=dest)
    assert dest.read_text(encoding="utf-8") == "- new rule\n"


def test_clean_ocr_uses_local_llama_not_reasoner(monkeypatch) -> None:  # noqa: ANN001
    """Patch 8.3.2 — formatting must hit Chat Node Llama, with VRAM pause."""
    calls: dict[str, object] = {}

    def _fake_ask(messages, model=""):  # noqa: ANN001
        calls["model"] = model
        calls["messages"] = messages
        return "- Cleaned rule\n"

    def _fake_sleep(seconds: float) -> None:
        calls["slept"] = float(seconds)

    monkeypatch.delenv("DONNA_INGEST_DRY_LLM", raising=False)
    monkeypatch.setattr(
        "dana.cascade_router.local_model_name",
        lambda: "llama3.2:latest",
    )
    monkeypatch.setattr(
        "dana.core_agent.ask_ollama_messages",
        _fake_ask,
    )
    monkeypatch.setattr(
        "dana.management.ingest_rules.time.sleep",
        _fake_sleep,
    )

    out = clean_ocr_with_llm("Raw OCR rule text")
    assert "- Cleaned rule" in out
    assert calls.get("model") == "llama3.2:latest"
    assert calls.get("slept") == 3.0
    assert "deepseek" not in str(calls.get("model") or "").lower()


def _force_sensor_updated_at(db: Path, iso_ts: str) -> None:
    with sqlite3.connect(str(db), timeout=30.0) as conn:
        conn.execute(
            "UPDATE sensor_state SET updated_at = ? WHERE key = ?",
            (iso_ts, PERCEPTION_OCR_KEY),
        )
        conn.commit()


def test_wait_for_fresh_ocr_rejects_stale_then_accepts(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # noqa: ANN001
    """Patch 8.3.3 — stale frames poll every 2s until a fresh write arrives."""
    db = tmp_path / "bb.db"
    init_blackboard(db)
    publish_perception_ocr(
        "STALE OCR from yesterday",
        producer="test",
        db_path=db,
    )
    stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=45)).isoformat()
    _force_sensor_updated_at(db, stale_ts)

    sleeps: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleeps.append(float(seconds))
        # After first stale wait, publish a fresh frame.
        if len(sleeps) == 1:
            publish_perception_ocr(
                "FRESH Feather rules on screen",
                producer="test",
                db_path=db,
            )

    def _no_force(*, db_path=None):  # noqa: ANN001
        return ""

    monkeypatch.delenv("DONNA_INGEST_SKIP_FRESHNESS", raising=False)
    monkeypatch.setattr("dana.management.ingest_rules.time.sleep", _fake_sleep)
    # Avoid GPU Florence in unit test — exercise poll path only.
    monkeypatch.setattr(
        "dana.management.ingest_rules._force_florence_ocr",
        _no_force,
    )

    text = wait_for_fresh_ocr(db_path=db, force_capture=True)
    assert text == "FRESH Feather rules on screen"
    assert sleeps == [2.0]
    err = capsys.readouterr().out
    assert "ERROR: Visual data is stale" in err
    assert "Waiting for a fresh frame..." in err

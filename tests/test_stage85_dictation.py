"""Stage 8.5 — Dictation Loop + Behavior Mixer blackboard helpers."""

from __future__ import annotations

from pathlib import Path

from dana.cascade_router import decide_route
from dana.management.dictation import (
    handle_dictation,
    mentions_dictate,
    should_handle_dictation,
    strip_dictate_wrapper,
    toggle_dictation_mode,
)
from dana.memory.blackboard import (
    behavior_mixer_prompt_weights,
    init_blackboard,
    is_dictation_mode,
    list_dictation_sessions,
    set_persona_trait,
)


def test_mentions_and_strip_dictate() -> None:
    assert mentions_dictate("Donna, dictate click the next button")
    assert strip_dictate_wrapper("hey donna, dictate open the menu") == "open the menu"


def test_record_dictation_with_ocr_ref(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    db = tmp_path / "bb.db"
    init_blackboard(db)
    monkeypatch.setattr(
        "dana.management.dictation.read_perception_ocr_text",
        lambda **_k: "OCR: Enter Comments [10,20,100,40]",
    )
    monkeypatch.setattr(
        "dana.management.dictation._capture_visual_reference",
        lambda **_k: "OCR: Enter Comments [10,20,100,40]",
    )
    out = handle_dictation(
        "Donna, dictate type the evaluation into Enter Comments",
        db_path=db,
        force_ocr=False,
    )
    assert out["ok"] is True
    assert "type the evaluation" in out["command_text"]
    rows = list_dictation_sessions(db_path=db)
    assert len(rows) == 1
    assert "Enter Comments" in rows[0]["visual_state_reference"]
    assert rows[0]["status"] == "recorded"


def test_gui_dictation_latch(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    assert is_dictation_mode(db_path=db) is False
    assert toggle_dictation_mode(True, db_path=db) is True
    assert is_dictation_mode(db_path=db) is True
    assert should_handle_dictation("just click next", db_path=db) is True
    assert toggle_dictation_mode(False, db_path=db) is False
    assert should_handle_dictation("just click next", db_path=db) is False


def test_cascade_routes_dictate_keyword() -> None:
    d = decide_route("please dictate press the Target button")
    assert "dictation" in (d.reason or "").lower()


def test_behavior_mixer_weights(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    set_persona_trait("autonomy", 77, db_path=db)
    set_persona_trait("creativity", 66, db_path=db)
    block = behavior_mixer_prompt_weights(db_path=db)
    assert "Autonomy=77/100" in block
    assert "Creativity=66/100" in block

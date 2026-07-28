"""Sidekick reliability — typed perception topics + mode/lease helpers."""

from __future__ import annotations

from pathlib import Path

from dana.agentic import get_donna_mode, restore_voice_mode, set_donna_mode
from dana.memory.blackboard import (
    PERCEPTION_OBJECTS_KEY,
    PERCEPTION_OCR_KEY,
    SCHEMA_OBJECTS_V1,
    SCHEMA_OCR_V1,
    get_sensor_state,
    get_voice_session_mode,
    init_blackboard,
    publish_perception_objects,
    publish_perception_ocr,
    read_perception_objects,
    read_perception_ocr,
    read_perception_ocr_text,
    read_visual_state,
    release_actuator_lease,
    sidekick_health,
    try_acquire_actuator_lease,
)
from dana.tools.broker import IntentBroker


def test_objects_and_ocr_are_separate_topics(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    publish_perception_objects(
        "[Vision Output] Detected: 1 book.",
        producer="test",
        db_path=db,
    )
    publish_perception_ocr(
        "[Florence OCR] source=cold_screenshot regions=1\nRATIONALES [10, 20, 30, 40]",
        producer="test",
        db_path=db,
    )
    obj = read_perception_objects(db_path=db)
    ocr = read_perception_ocr(db_path=db)
    assert obj is not None
    assert ocr is not None
    assert obj["meta"]["schema"] == SCHEMA_OBJECTS_V1
    assert ocr["meta"]["schema"] == SCHEMA_OCR_V1
    assert "1 book" in read_visual_state(db_path=db)
    assert "RATIONALES" in read_perception_ocr_text(db_path=db)
    # Objects mirror still exists as legacy key, but OCR consumers ignore it.
    assert get_sensor_state(PERCEPTION_OBJECTS_KEY, db_path=db) is not None
    assert get_sensor_state(PERCEPTION_OCR_KEY, db_path=db) is not None


def test_ocr_reader_rejects_vision_output_prose(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    # Manually poison OCR key with YOLO prose + wrong kind — should reject.
    from dana.memory.blackboard import set_sensor_state

    set_sensor_state(
        PERCEPTION_OCR_KEY,
        "[Vision Output] Detected: 1 laptop.",
        meta={"schema": SCHEMA_OCR_V1, "kind": "ocr", "producer": "bad"},
        db_path=db,
    )
    assert read_perception_ocr(db_path=db) is None
    assert read_perception_ocr_text(db_path=db) == ""


def test_broker_routes_read_rules_to_florence() -> None:
    broker = IntentBroker()
    call = broker.parse_utterance("please read the rules on my screen")
    assert call is not None
    assert call.tool_id == "ocr_with_region"


def test_broker_routes_look_around_to_yolo() -> None:
    broker = IntentBroker()
    call = broker.parse_utterance("what do you see on my screen")
    assert call is not None
    assert call.tool_id == "analyze_visual_context"


def test_voice_mode_not_stolen_by_job_escalation(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    db = tmp_path / "bb.db"
    init_blackboard(db)
    monkeypatch.setattr(
        "dana.memory.blackboard.BLACKBOARD_DB_PATH",
        db,
    )
    set_donna_mode("chat", as_voice=True)
    assert get_voice_session_mode(db_path=db) == "chat"
    set_donna_mode("developer", as_voice=False)
    assert get_donna_mode() == "developer"
    assert get_voice_session_mode(db_path=db) == "chat"
    restored = restore_voice_mode()
    assert restored == "chat"
    assert get_donna_mode() == "chat"


def test_actuator_lease_is_exclusive(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    assert try_acquire_actuator_lease("owner-a", db_path=db) is True
    assert try_acquire_actuator_lease("owner-b", db_path=db) is False
    release_actuator_lease("owner-a", db_path=db)
    assert try_acquire_actuator_lease("owner-b", db_path=db) is True
    release_actuator_lease("owner-b", db_path=db)


def test_sidekick_health_degraded_without_heartbeats(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    # Point helpers at temp db via explicit path through sensor reads.
    from dana.memory.blackboard import publish_heartbeat, HEARTBEAT_VISION_KEY

    h = sidekick_health(db_path=db)
    assert h["degraded"] is True
    publish_heartbeat(HEARTBEAT_VISION_KEY, publisher="test", ok=True, db_path=db)
    h2 = sidekick_health(db_path=db)
    assert h2["vision_alive"] is True

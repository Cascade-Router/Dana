"""Stage 4.1 — vision sensor daemon + Chat blackboard visual hook."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from donna.memory.blackboard import (
    LATEST_VISUAL_CONTEXT_KEY,
    get_sensor_state,
    init_blackboard,
    read_visual_state,
    set_sensor_state,
)
from donna.middleware.vision_poller import publish_visual_context
from donna.tools.broker import merge_bound_tool_ids


def test_sensor_state_upsert_and_read_visual_state(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    assert read_visual_state(db_path=db) == ""

    set_sensor_state(
        LATEST_VISUAL_CONTEXT_KEY,
        "[Vision Output] Detected: 1 monitor.",
        meta={"publisher": "test"},
        db_path=db,
    )
    assert read_visual_state(db_path=db) == "[Vision Output] Detected: 1 monitor."
    row = get_sensor_state(LATEST_VISUAL_CONTEXT_KEY, db_path=db)
    assert row is not None
    assert row["meta"]["publisher"] == "test"

    set_sensor_state(
        LATEST_VISUAL_CONTEXT_KEY,
        "[Vision Output] Detected: 2 persons.",
        db_path=db,
    )
    assert read_visual_state(db_path=db) == "[Vision Output] Detected: 2 persons."


def test_publish_emits_sensor_vision_telemetry(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    from donna.memory.blackboard import publish_perception_objects

    db = tmp_path / "bb.db"
    out = tmp_path / "donna_telemetry.jsonl"
    monkeypatch.setattr("donna.telemetry.TELEMETRY_JSONL_PATH", out)
    monkeypatch.setattr(
        "donna.middleware.vision_poller.publish_perception_objects",
        lambda text, **kwargs: publish_perception_objects(
            text, db_path=db, **{k: v for k, v in kwargs.items() if k != "db_path"}
        ),
    )
    monkeypatch.setattr(
        "donna.middleware.vision_poller.publish_heartbeat",
        lambda *a, **k: None,
    )
    init_blackboard(db)
    publish_visual_context(
        "[Vision Output] Detected: 1 cup.",
        latency_ms=42.5,
    )
    assert read_visual_state(db_path=db) == "[Vision Output] Detected: 1 cup."
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    evt = json.loads(lines[0])
    assert evt["tag"] == "[SENSOR_VISION]"
    assert evt["latency_ms"] == 42.5


def test_chat_mode_strips_analyze_visual_context() -> None:
    known = (
        "analyze_visual_context",
        "draft_cursor_prompt",
        "web_search",
    )
    merged = merge_bound_tool_ids(
        user_text="please call analyze_visual_context on the screen",
        forced_tool_id="analyze_visual_context",
        mode="chat",
        known_ids=known,
    )
    assert "analyze_visual_context" not in merged


def test_concurrent_poller_write_and_chat_read(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for i in range(40):
                set_sensor_state(
                    LATEST_VISUAL_CONTEXT_KEY,
                    f"[Vision Output] Detected: frame {i}.",
                    db_path=db,
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(40):
                _ = read_visual_state(db_path=db)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert not errors, errors
    assert "Detected: frame" in read_visual_state(db_path=db)

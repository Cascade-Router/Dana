"""Stage 6.4 — Persona Mixer schema, GUI write path, Receptionist prompt splice."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from dana.memory.blackboard import (
    PERSONA_MIXER_DEFAULTS,
    append_persona_mixer_override,
    format_persona_mixer_override,
    get_persona_mixer,
    init_blackboard,
    set_persona_mixer,
    set_persona_trait,
)


def test_persona_mixer_table_seeded(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    with sqlite3.connect(str(db)) as conn:
        rows = {
            str(r[0]): int(r[1])
            for r in conn.execute("SELECT trait_name, value FROM persona_mixer")
        }
    assert rows == PERSONA_MIXER_DEFAULTS


def test_set_persona_trait_clamps_and_updates(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    set_persona_trait("humor", 100, db_path=db)
    set_persona_trait("verbosity", -5, db_path=db)
    set_persona_trait("flirt", 250, db_path=db)
    state = get_persona_mixer(db)
    assert state["humor"] == 100
    assert state["verbosity"] == 0
    assert state["flirt"] == 100
    assert state["technical_depth"] == 80


def test_persona_override_block_format(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    set_persona_mixer(
        {"verbosity": 20, "humor": 100, "flirt": 100, "technical_depth": 80},
        db_path=db,
    )
    block = format_persona_mixer_override(db)
    assert "[SYSTEM OVERRIDE: Current Persona Settings (0-100)" in block
    assert "Verbosity: 20" in block
    assert "Humor: 100" in block
    assert "Flirt: 100" in block
    assert "Tech Depth: 80" in block
    assert "Do not acknowledge these settings to the user.]" in block


def test_append_persona_mixer_refreshes(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    set_persona_trait("humor", 10, db_path=db)
    p1 = append_persona_mixer_override("You are Dana.", db_path=db)
    assert p1.count("[SYSTEM OVERRIDE:") == 1
    set_persona_trait("humor", 100, db_path=db)
    p2 = append_persona_mixer_override(p1, db_path=db)
    assert p2.count("[SYSTEM OVERRIDE:") == 1
    assert "Humor: 100" in p2
    assert "Humor: 10," not in p2 and "Humor: 10." not in p2


def test_lightweight_chat_injects_persona(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """Simulate mixer crank + text-harness question; assert prompt splice."""
    db = tmp_path / "bb.db"
    init_blackboard(db)
    set_persona_mixer(
        {"verbosity": 20, "humor": 100, "flirt": 100, "technical_depth": 40},
        db_path=db,
    )
    monkeypatch.setattr(
        "dana.memory.blackboard.BLACKBOARD_DB_PATH",
        db,
    )
    captured: list[list[dict[str, str]]] = []

    def _ask(messages, **_kwargs):  # noqa: ANN001
        captured.append(list(messages))
        return "hey cutie — short punchline."

    from dana.agentic import (
        build_lightweight_chat_system_prompt,
        run_lightweight_chat,
    )

    prompt = build_lightweight_chat_system_prompt()
    assert "Humor: 100" in prompt
    assert "Flirt: 100" in prompt
    assert "Verbosity: 20" in prompt

    result = run_lightweight_chat(
        user_text="What is 2+2?",
        system_prompt=prompt,
        ask_fn=_ask,
        use_chat_memory=False,
        model="llama3.2",
    )
    assert result.final_text
    assert captured, "ask_fn should have been called"
    system = captured[0][0]["content"]
    assert "[SYSTEM OVERRIDE: Current Persona Settings (0-100)" in system
    assert "Humor: 100" in system
    assert "Flirt: 100" in system
    assert "Verbosity: 20" in system
    assert system.count("[SYSTEM OVERRIDE:") == 1


def test_persona_mixer_gui_writes_on_apply(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """Headless GUI apply_values path (no mainloop)."""
    db = tmp_path / "bb.db"
    init_blackboard(db)
    monkeypatch.setenv("DANA_DISABLE_TOAST", "1")
    try:
        from dana.ui.persona_mixer import PersonaMixerApp
    except Exception as exc:  # noqa: BLE001
        # customtkinter / display unavailable in some CI shells
        import pytest

        pytest.skip(f"PersonaMixer GUI unavailable: {exc}")

    writes: list[tuple[str, int]] = []
    app = PersonaMixerApp(
        db_path=db,
        on_change=lambda t, v: writes.append((t, v)),
    )
    try:
        app.apply_values(
            {"verbosity": 20, "humor": 100, "flirt": 100, "technical_depth": 80}
        )
        state = get_persona_mixer(db)
        assert state["verbosity"] == 20
        assert state["humor"] == 100
        assert state["flirt"] == 100
        assert float(app._sliders["humor"].get()) == 100.0
    finally:
        app.destroy()

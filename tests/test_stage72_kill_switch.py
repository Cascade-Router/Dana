"""Stage 7.2 — Human panic button / kill switch."""

from __future__ import annotations

from pathlib import Path

from donna.memory.blackboard import (
    enqueue_action,
    get_action,
    init_blackboard,
)
from donna.middleware.kill_switch import (
    GLOBAL_HALT_EVENT,
    clear_global_halt,
    start_kill_switch_listener,
    trigger_halt,
)
from donna.operators.ghost_typist import GhostTypistOperator
from donna.operators.nav_and_click import NavigationOperator


def setup_function() -> None:  # noqa: D103
    clear_global_halt()


def teardown_function() -> None:  # noqa: D103
    clear_global_halt()


def test_trigger_halt_cancels_pending_and_running(tmp_path: Path) -> None:
    db = tmp_path / "bb.db"
    init_blackboard(db)
    a1 = enqueue_action("navigate_and_click", {"query": "X"}, db_path=db)
    a2 = enqueue_action("type_stealth_text", {"text": "hi"}, db_path=db)
    # Simulate in-flight claim.
    from donna.memory.blackboard import claim_next_pending

    claimed = claim_next_pending(db_path=db)
    assert claimed is not None

    out = trigger_halt(db_path=db, reason="test")
    assert out["ok"] is True
    assert GLOBAL_HALT_EVENT.is_set()
    assert out["cancelled"] >= 2

    r1 = get_action(a1, db_path=db)
    r2 = get_action(a2, db_path=db)
    assert r1 is not None and r1["status"] == "cancelled"
    assert r2 is not None and r2["status"] == "cancelled"


def test_ghost_typist_aborts_on_halt(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    monkeypatch.setenv("DONNA_GHOST_SKIP_HOTKEY", "1")
    clear_global_halt()
    typed: list[str] = []

    def _type(ch: str) -> bool:
        typed.append(ch)
        if len(typed) >= 3:
            GLOBAL_HALT_EVENT.set()
        return True

    op = GhostTypistOperator(
        type_char=_type,
        wait_hotkey_fn=lambda *_a, **_k: True,
        read_visual=lambda: "stable",
        chunk_size=20,
    )
    result = op.run("HELLO WORLD THIS IS LONG", wait_hotkey=False)
    assert result.get("halted") is True
    assert result.get("ok") is False
    assert len(typed) < len("HELLO WORLD THIS IS LONG")


def test_navigation_aborts_on_halt(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    clear_global_halt()
    moves = 0

    def _move(x: int, y: int) -> None:
        nonlocal moves
        moves += 1
        if moves >= 2:
            GLOBAL_HALT_EVENT.set()

    visual = "Target [400, 300, 560, 380]"
    op = NavigationOperator(
        read_visual=lambda: visual,
        move_cursor=_move,
        get_cursor=lambda: (280, 220),
        click=lambda: None,
        chunk_size=4,
    )
    result = op.navigate_and_click("Target", visual_context=visual)
    assert result.get("halted") is True
    assert result.get("ok") is False


def test_start_listener_idempotent(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_DISABLE_KILL_SWITCH", "1")
    # Force re-entry path for disabled flag.
    import donna.middleware.kill_switch as ks

    ks._LISTENER_STARTED = False
    assert start_kill_switch_listener() is False
    clear_global_halt()

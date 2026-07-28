"""Stage 6.3 — Navigation Operator Bezier SEA + actuator wiring."""

from __future__ import annotations

import json
import math
from pathlib import Path

from dana.memory.blackboard import (
    LATEST_VISUAL_CONTEXT_KEY,
    enqueue_action,
    init_blackboard,
    is_heavy_actuator_tool,
    set_sensor_state,
)
from dana.middleware.actuator_executor import process_action
from dana.operators.nav_and_click import (
    NavigationOperator,
    find_target_box,
    generate_bezier_path,
    navigate_and_click,
)


DUMMY_VISUAL = (
    "Florence-2 OCR: blue box labeled Target [400, 300, 560, 380]. "
    "Enter Comments [100, 500, 500, 560] empty input field."
)


def test_navigate_and_click_is_heavy() -> None:
    assert is_heavy_actuator_tool("navigate_and_click")


def test_find_target_box_by_label() -> None:
    box = find_target_box(DUMMY_VISUAL, "Target")
    assert box is not None
    assert box.x1 == 400 and box.y2 == 380
    comments = find_target_box(DUMMY_VISUAL, "Enter Comments")
    assert comments is not None
    assert "comment" in comments.label.lower()


def test_bezier_path_is_nonlinear_and_unique() -> None:
    start, end = (10, 10), (500, 400)
    p1 = generate_bezier_path(start, end, steps=24)
    p2 = generate_bezier_path(start, end, steps=24)
    assert len(p1) >= 8
    assert p1[-1][0] == end[0] or abs(p1[-1][0] - end[0]) <= 2
    # Paths should differ due to stochastic control points.
    assert p1 != p2
    # Midpoint should deviate from the straight line.
    mid = p1[len(p1) // 2]
    t = 0.5
    line_x = start[0] + (end[0] - start[0]) * t
    line_y = start[1] + (end[1] - start[1]) * t
    dist = math.hypot(mid[0] - line_x, mid[1] - line_y)
    assert dist > 1.0


def test_navigation_operator_clicks_dummy_target(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    clicks: list[tuple[int, int]] = []
    trail: list[tuple[int, int]] = []

    op = NavigationOperator(
        read_visual=lambda: DUMMY_VISUAL,
        chunk_size=6,
    )
    # Capture trail via wrapping move.
    orig_move = op._move

    def _move(x: int, y: int) -> None:
        trail.append((x, y))
        orig_move(x, y)

    op._move = _move  # type: ignore[method-assign]
    op.click = lambda: clicks.append(op._virt_cursor)  # type: ignore[assignment]

    result = op.navigate_and_click("Target", visual_context=DUMMY_VISUAL)
    assert result["ok"] is True
    assert clicks, "expected a click"
    cx, cy = clicks[0]
    box = find_target_box(DUMMY_VISUAL, "Target")
    assert box is not None and box.contains(cx, cy, pad=6)
    assert len(trail) >= 5
    # Movement should not be a single teleport.
    assert trail[0] != trail[-1]


def test_navigation_emits_operator_telemetry(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    out = tmp_path / "donna_telemetry.jsonl"
    monkeypatch.setattr("dana.telemetry.TELEMETRY_JSONL_PATH", out)
    result = navigate_and_click("Target", visual_context=DUMMY_VISUAL)
    assert result.startswith("OK: navigate_and_click")
    tags = [
        json.loads(line)["tag"]
        for line in out.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert "[OPERATOR_NAV_CLICK_COMPLETE]" in tags


def test_actuator_navigate_and_click(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    monkeypatch.setenv("DONNA_OS_DRY_RUN", "1")
    monkeypatch.setenv("DONNA_DISABLE_TOAST", "1")
    db = tmp_path / "bb.db"
    init_blackboard(db)
    set_sensor_state(LATEST_VISUAL_CONTEXT_KEY, DUMMY_VISUAL, db_path=db)
    monkeypatch.setattr(
        "dana.memory.blackboard.BLACKBOARD_DB_PATH",
        db,
    )
    import dana.memory.blackboard as bb

    monkeypatch.setattr(bb, "BLACKBOARD_DB_PATH", db)
    monkeypatch.setattr(
        "dana.middleware.actuator_executor.resolve_action",
        lambda action_id, status, result="", db_path=None, **kw: bb.resolve_action(
            action_id,
            status=status,
            result=result,
            error_context=kw.get("error_context", ""),
            db_path=db,
        ),
    )
    aid = enqueue_action(
        "navigate_and_click",
        {"query": "Target", "visual_context": DUMMY_VISUAL},
        session_id="nav-test",
        db_path=db,
    )
    claimed = bb.claim_next_pending(db_path=db)
    assert claimed is not None
    stats = process_action(claimed, db_path=db)
    assert stats["status"] == "completed"
    row = bb.get_action(aid, db_path=db)
    assert row is not None
    assert "OK: navigate_and_click" in (row.get("result") or "")


def test_live_dummy_target_window_harness(monkeypatch) -> None:  # noqa: ANN001
    """Optional visual demo — skipped unless DONNA_NAV_LIVE_DEMO=1."""
    import os

    if os.environ.get("DONNA_NAV_LIVE_DEMO", "").strip() not in {"1", "true", "yes"}:
        return
    # Live path: show a blue Target box, publish bbox, move for real.
    monkeypatch.delenv("DONNA_OS_DRY_RUN", raising=False)
    import tkinter as tk

    root = tk.Tk()
    root.title("Donna Nav Target")
    root.geometry("700x500+100+100")
    canvas = tk.Canvas(root, width=700, height=500, bg="#222")
    canvas.pack()
    # Box roughly at screen-relative coords after window maps.
    root.update_idletasks()
    root.update()
    x = root.winfo_rootx() + 250
    y = root.winfo_rooty() + 180
    w, h = 160, 80
    canvas.create_rectangle(250, 180, 410, 260, fill="#1e90ff", outline="white", width=2)
    canvas.create_text(330, 220, text="Target", fill="white", font=("Segoe UI", 16, "bold"))
    root.update()
    visual = f"Target [{x},{y},{x+w},{y+h}] blue box labeled Target"
    set_sensor_state(LATEST_VISUAL_CONTEXT_KEY, visual)
    root.after(400, lambda: None)
    root.update()
    result = navigate_and_click("Target", visual_context=visual)
    assert result.startswith("OK:")
    root.after(800, root.destroy)
    root.mainloop()

"""OSWorld benchmark harness: adversarial noise + adapter (offline, mocked, seeded)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from dana.agentic_react_graph import route_after_execution
from dana.graph.nodes.critic import (
    fail_closed_node,
    is_fatal_execution_error,
    python_repl_state_patch,
)
from dana.ui.watchdog import ShellWatchdog
from dana.vision.hybrid_grounding import HybridVisionGrounding
from dana.vision.uia_provider import Win32UIAProvider

_EVALS = Path(__file__).resolve().parent
if str(_EVALS) not in sys.path:
    sys.path.insert(0, str(_EVALS))

from noise_injector import (  # noqa: E402
    DesktopNoiseInjector,
    bbox_within_tolerance,
    translate_bboxes,
)
from osworld_adapter import OSWorldAdapter, load_osworld_fixture  # noqa: E402

_SUMMARY_PATH = _EVALS / "osworld_bench_summary.json"
_SEED = 42
_TOL_PX = 10.0


def _blank_rgb(w: int = 200, h: int = 120) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _paint_button(img: np.ndarray, bbox: list[float], rgb=(200, 80, 40)) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = img.shape[:2]
    img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = rgb


# ---------------------------------------------------------------------------
# 1) Florence-2 / UIA hybrid grounding under random visual shifts
# ---------------------------------------------------------------------------


def test_hybrid_grounding_survives_visual_shift() -> None:
    """Mock UIA + Florence; noise translates boxes; grounding within ±10px."""
    task = load_osworld_fixture()
    save = task["target_screen_state"]["elements"][0]
    clean_box = [float(v) for v in save["bbox"]]

    img = _blank_rgb()
    _paint_button(img, clean_box)

    injector = DesktopNoiseInjector(seed=_SEED, deterministic=True)
    noisy = injector.apply_visual_noise(img, bboxes=[clean_box], n_toasts=1)
    dx, dy = noisy.offset_xy
    assert abs(dx) >= 5 and abs(dx) <= 15
    assert abs(dy) >= 5 and abs(dy) <= 15
    assert noisy.overlays and noisy.overlays[0]["kind"] == "toast"

    expected_shifted = translate_bboxes([clean_box], (dx, dy))[0]
    assert noisy.adjusted_bboxes[0] == pytest.approx(expected_shifted)

    # UIA returns the *shifted* native bounds (desktop chrome moved with window).
    uia = Win32UIAProvider(
        control_tree=[
            {
                "name": "Save",
                "automation_id": "btnSave",
                "control_type": "Button",
                "bounds_norm": expected_shifted,  # pixel-space for this bench
            }
        ]
    )
    florence_calls: list[Any] = []

    def florence_spy(image: Any, label: str) -> Optional[list[float]]:
        florence_calls.append((image, label))
        # Coarse Florence on noisy frame returns shifted pixel box.
        return list(expected_shifted)

    grounder = HybridVisionGrounding(
        uia_provider=uia,
        florence_ground_fn=florence_spy,
        crop_zoom_fn=lambda image, box: (image, (0.0, 0.0, 1.0, 1.0)),
    )
    got = grounder.locate_ui_element(noisy.image, "Save")
    assert got is not None
    assert grounder.last_stage == "uia"
    assert florence_calls == []  # UIA hit short-circuits Florence
    assert bbox_within_tolerance(got, expected_shifted, tol_px=_TOL_PX)

    # Florence-only path (UIA miss) still recovers within tolerance after shift.
    grounder_f = HybridVisionGrounding(
        uia_provider=Win32UIAProvider(control_tree=[]),
        florence_ground_fn=florence_spy,
        crop_zoom_fn=lambda image, box: (image, (0.0, 0.0, 1.0, 1.0)),
    )
    got_f = grounder_f.locate_ui_element(noisy.image, "Save")
    assert got_f is not None
    assert grounder_f.last_stage in {"coarse", "zoom"}
    assert bbox_within_tolerance(got_f, expected_shifted, tol_px=_TOL_PX)

    # Adapter treats a click on the shifted center as a pass.
    adapter = OSWorldAdapter(click_tol_px=_TOL_PX, bbox_tol_px=_TOL_PX)
    cx = (expected_shifted[0] + expected_shifted[2]) / 2.0
    cy = (expected_shifted[1] + expected_shifted[3]) / 2.0
    step = adapter.evaluate_step(
        {"type": "click", "target": "Save", "x": cx, "y": cy},
        {"type": "click", "target": "Save", "bbox": expected_shifted},
    )
    assert step["passed"] is True
    assert step["score"] == 1.0


# ---------------------------------------------------------------------------
# 2) Self-Healing Critic / Shell Watchdog vs latency / background noise
# ---------------------------------------------------------------------------


def test_critic_and_watchdog_tolerate_latency_jitter() -> None:
    """Non-fatal delay / toast noise must not set fatal_block or fail-closed."""
    sleeps: list[float] = []
    injector = DesktopNoiseInjector(
        seed=_SEED,
        deterministic=True,
        sleep_fn=sleeps.append,
    )
    delay = injector.apply_latency_jitter()
    assert sleeps == [delay]
    assert 0.001 <= delay <= 0.005  # deterministic band

    # Successful REPL after jitter — no fatal / no fail-closed route.
    obs_ok = "exit_code=0\nstdout:\nok\n(latency_ms=%.1f)" % (delay * 1000.0)
    patch = python_repl_state_patch(code="print('ok')", observation=obs_ok)
    assert patch.get("fatal_block") is False
    assert patch.get("execution_error") is None
    assert route_after_execution({**patch, "halt": True}) == "__end__"

    # Background toast / log noise through ShellWatchdog — not a fatal OS block.
    events: list[tuple[str, str]] = []
    wd = ShellWatchdog(enabled=True, on_error=lambda t, s: events.append((t, s)), dedupe=True)
    toast_noise = (
        "[Toast] Update available\n"
        "INFO: background sync ok\n"
        f"latency_jitter_ms={delay * 1000.0:.1f}\n"
    )
    emitted = wd.feed_text(toast_noise)
    assert emitted == []
    assert events == []
    assert not is_fatal_execution_error(toast_noise)

    # Fixable code fault still routes to critic (not fail_closed) despite jitter note.
    obs_fix = (
        "exit_code=1\nstderr:\nZeroDivisionError: division by zero\n"
        f"# delayed {delay:.4f}s by DesktopNoiseInjector\n"
    )
    patch_fix = python_repl_state_patch(code="print(1/0)", observation=obs_fix)
    assert patch_fix.get("fatal_block") is False
    assert (
        route_after_execution(
            {**patch_fix, "retry_count": 0, "max_retries": 3}
        )
        == "critic"
    )
    closed = fail_closed_node(
        {
            "execution_error": None,
            "fatal_block": False,
            "critique_history": ["healed"],
            "retry_count": 0,
            "session_id": "osworld-latency",
        }
    )
    # Exhausted-heal fail_closed may halt, but latency alone must not invent fatal_block.
    assert closed.get("fatal_block") is not True
    assert "Fatal OS Block" not in str(closed.get("final_raw") or "")


# ---------------------------------------------------------------------------
# 3) Baseline precision + task completion across noisy runs
# ---------------------------------------------------------------------------


def test_osworld_noisy_runs_log_baseline_scores(tmp_path: Path) -> None:
    """Run fixture through noise + adapter; assert scores present; write summary JSON."""
    task = load_osworld_fixture()
    adapter = OSWorldAdapter(click_tol_px=_TOL_PX, bbox_tol_px=_TOL_PX)
    state = adapter.task_to_agent_state(task)
    assert state["session_id"].startswith("osworld-")
    assert state["active_intent"]
    assert state.get("fatal_block") is False
    assert state["env_context"]["benchmark"] == "osworld"
    assert state["env_context"]["task_id"] == task["id"]

    expected = list(task["expected_actions"])
    save_box = [float(v) for v in expected[0]["bbox"]]

    run_scores: list[dict[str, Any]] = []
    for run_i in range(5):
        inj = DesktopNoiseInjector(seed=_SEED + run_i, deterministic=True, sleep_fn=lambda _s: None)
        inj.apply_latency_jitter()
        noisy = inj.apply_visual_noise(_blank_rgb(), bboxes=[save_box], n_toasts=1)
        shifted = noisy.adjusted_bboxes[0]
        cx = (shifted[0] + shifted[2]) / 2.0
        cy = (shifted[1] + shifted[3]) / 2.0

        # Agent emits the shifted click + remaining expected steps (mock policy).
        actions = [
            {"type": "click", "target": "Save", "x": cx, "y": cy, "bbox": shifted},
            {"type": "type_text", "text": "hello.txt"},
            {"type": "tool", "name": "python_repl", "args": {"code": "print('saved')"}},
        ]
        # Evaluate against *shifted* first-step expectation (noise-aware oracle).
        exp_shifted = [
            {**expected[0], "bbox": shifted},
            expected[1],
            expected[2],
        ]
        result = adapter.evaluate_sequence(actions, exp_shifted)
        assert "precision" in result and "task_completion" in result
        assert "score" in result
        run_scores.append(
            {
                "run": run_i,
                "seed": _SEED + run_i,
                "offset_xy": list(noisy.offset_xy),
                "latency_s": inj.last_latency_s,
                "precision": result["precision"],
                "task_completion": result["task_completion"],
                "score": result["score"],
                "passed": result["passed"],
            }
        )
        assert result["passed"] is True
        assert result["precision"] == pytest.approx(1.0)
        assert result["task_completion"] == pytest.approx(1.0)

    summary = {
        "benchmark": "osworld_offline",
        "task_id": task["id"],
        "n_runs": len(run_scores),
        "mean_precision": sum(r["precision"] for r in run_scores) / len(run_scores),
        "mean_task_completion": sum(r["task_completion"] for r in run_scores)
        / len(run_scores),
        "mean_score": sum(r["score"] for r in run_scores) / len(run_scores),
        "runs": run_scores,
    }
    assert summary["mean_precision"] == pytest.approx(1.0)
    assert summary["mean_task_completion"] == pytest.approx(1.0)

    # Prefer repo-local summary when writable; else tmp (CI / sandbox).
    out = _SUMMARY_PATH
    try:
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except OSError:
        out = tmp_path / "osworld_bench_summary.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["n_runs"] == 5
    assert "mean_precision" in loaded and "mean_task_completion" in loaded
    print(
        "[osworld_bench] mean_precision=%.3f mean_task_completion=%.3f mean_score=%.3f"
        % (
            summary["mean_precision"],
            summary["mean_task_completion"],
            summary["mean_score"],
        )
    )


def test_noise_injector_seed_reproducible() -> None:
    a = DesktopNoiseInjector(seed=7, deterministic=True, sleep_fn=lambda _s: None)
    b = DesktopNoiseInjector(seed=7, deterministic=True, sleep_fn=lambda _s: None)
    img = _blank_rgb(64, 48)
    ra = a.apply_visual_noise(img, bboxes=[[10, 10, 20, 20]])
    rb = b.apply_visual_noise(img, bboxes=[[10, 10, 20, 20]])
    assert ra.offset_xy == rb.offset_xy
    assert ra.adjusted_bboxes == rb.adjusted_bboxes
    assert a.apply_latency_jitter() == b.apply_latency_jitter()

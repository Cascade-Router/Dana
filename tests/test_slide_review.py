"""Tests for Cascade local MoA + slide review routing."""

from __future__ import annotations

from dana.cascade_router import (
    decide_route,
    is_visual_task,
    reason_over_context,
    run_visual_moa,
)
from dana.tools.broker import _SLIDE_REVIEW_HINT_RE, get_broker, reload_broker_registry
from dana.tools.slide_review import _format_comment, _parse_reasoner_output, evaluate_slide_and_type


INJECT = (
    "Dana, I need you to evaluate the slide on my screen. "
    "Check if it follows the rule of having a clear title and less than 30 words. "
    "Then, type your evaluation summary directly into my active window."
)


def test_broker_routes_slide_review() -> None:
    reload_broker_registry()
    assert _SLIDE_REVIEW_HINT_RE.search(INJECT)
    call = get_broker().parse_utterance(INJECT)
    assert call is not None
    assert call.tool_id == "evaluate_slide_and_type", call
    print("[PASS] broker → evaluate_slide_and_type")


def test_decide_route_uses_moa_not_gpt() -> None:
    d = decide_route(INJECT, forced_tool="evaluate_slide_and_type")
    assert d.backend == "moa"
    assert "gpt" not in d.model.lower()
    assert is_visual_task(INJECT, forced_tool="evaluate_slide_and_type")
    print("[PASS] decide_route → moa (no gpt)")


def test_format_and_parse() -> None:
    c = _format_comment("Title clear; body is short.", pass_fail="PASS", word_count=12)
    assert c.startswith("[PASS")
    parsed = _parse_reasoner_output(
        "VERDICT: FAIL\nWORD_COUNT: 40\nCOMMENT: Too wordy and missing a clear title."
    )
    assert parsed["verdict"] == "FAIL"
    assert parsed["word_count"] == 40
    print("[PASS] format/parse")


def test_pipeline_dry_run_moa(monkeypatch) -> None:
    monkeypatch.setenv("DANA_OS_DRY_RUN", "1")

    monkeypatch.setattr(
        "dana.tools.os_control.capture_screen_png_bytes",
        lambda: b"\x89PNG\r\n\x1a\nfake",
    )
    monkeypatch.setattr(
        "dana.cascade_router.run_visual_moa",
        lambda *_a, **_k: {
            "vision_text": "Title: Roadmap. Body: Ship fast.",
            "final": "VERDICT: PASS\nWORD_COUNT: 4\nCOMMENT: Clear title; under 30 words.",
            "route": "moa/llava+deepseek-r1",
            "vision_model": "llava",
            "reasoner_model": "deepseek-r1",
        },
    )
    out = evaluate_slide_and_type(
        rule="clear title and less than 30 words",
        focus_delay_sec=0.0,
        dry_run=True,
    )
    assert out.startswith("OK: evaluate_slide_and_type")
    assert "PASS" in out
    assert "moa/" in out
    print("[PASS] dry-run MoA pipeline")


def test_reasoner_fallback_string(monkeypatch) -> None:
    calls = {"n": 0}

    def _chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("deepseek missing")
        return "VERDICT: PASS\nWORD_COUNT: 3\nCOMMENT: ok"

    monkeypatch.setattr("dana.cascade_router._ollama_chat", _chat)
    monkeypatch.setattr("dana.cascade_router.local_model_name", lambda: "llama3.2")
    out = reason_over_context("Title: Hi", rule="clear title", model="deepseek-r1")
    assert "PASS" in out or "VERDICT" in out
    assert calls["n"] >= 2
    print("[PASS] reasoner fallback")


if __name__ == "__main__":
    reload_broker_registry()
    test_broker_routes_slide_review()
    test_decide_route_uses_moa_not_gpt()
    test_format_and_parse()
    print("OK")

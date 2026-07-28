"""Smoke tests that parameterize the golden benchmark dataset."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_golden():
    path = Path(__file__).resolve().parent / "golden.py"
    spec = importlib.util.spec_from_file_location("evals_golden_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evals_golden_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_golden_dataset_loads_25_cases() -> None:
    cases = _load_golden().load_golden_dataset()
    assert len(cases) == 25
    by_cat = {c["category"] for c in cases}
    assert by_cat == {
        "routing_intent",
        "vision_grounding",
        "hitl_safety",
        "memory_recall",
    }
    counts = {c: 0 for c in by_cat}
    for case in cases:
        counts[case["category"]] += 1
    assert counts["routing_intent"] == 7
    assert counts["vision_grounding"] == 6
    assert counts["hitl_safety"] == 6
    assert counts["memory_recall"] == 6


def test_golden_case_schema(golden_case: dict[str, Any]) -> None:
    """Every golden case exposes the required evaluation fields."""
    assert golden_case["id"]
    assert golden_case["category"]
    assert isinstance(golden_case["user_input"], str) and golden_case["user_input"].strip()
    assert golden_case["expected_initial_node"] in {
        "chat",
        "planner",
        "agent",
        "tools",
        "ticket_validate",
    }
    assert isinstance(golden_case["requires_hitl"], bool)
    assert golden_case["ground_truth_output"].strip()


@pytest.mark.golden_category("hitl_safety")
def test_hitl_golden_cases_flag_requires_hitl(golden_case: dict[str, Any]) -> None:
    assert golden_case["category"] == "hitl_safety"
    assert golden_case["requires_hitl"] is True
    assert golden_case["expected_initial_node"] == "planner"


@pytest.mark.golden_category("vision_grounding")
def test_vision_golden_cases_target_planner(
    golden_case: dict[str, Any],
) -> None:
    assert golden_case["category"] == "vision_grounding"
    assert golden_case["expected_initial_node"] == "planner"
    assert golden_case["requires_hitl"] is False

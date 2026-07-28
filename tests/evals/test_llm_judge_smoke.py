"""Smoke: heuristic LLM-judge bench finishes offline without API keys."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_EVALS_DIR = Path(__file__).resolve().parent


def _load_judge_mod():
    path = _EVALS_DIR / "llm_judge.py"
    spec = importlib.util.spec_from_file_location("evals_llm_judge_smoke", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses require the module to exist in sys.modules during decoration.
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_heuristic_judge_smoke(tmp_path: Path) -> None:
    judge = _load_judge_mod()
    report = judge.run_benchmark(limit=3, judge="heuristic")
    assert report["case_count"] == 3
    assert "averages" in report
    assert report["averages"]["overall"] >= 1.0
    out = tmp_path / "smoke_eval_report.json"
    written = judge.write_report(report, out)
    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["case_count"] == 3
    assert isinstance(data.get("failures"), list)

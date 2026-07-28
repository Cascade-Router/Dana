"""Pytest fixtures for Dānā golden eval dataset parameterization."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_EVALS_DIR = Path(__file__).resolve().parent


def _load_golden_module():
    path = _EVALS_DIR / "golden.py"
    spec = importlib.util.spec_from_file_location("evals_golden", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load golden loader from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evals_golden"] = mod
    spec.loader.exec_module(mod)
    return mod


_golden = _load_golden_module()
load_golden_dataset = _golden.load_golden_dataset
CATEGORIES = _golden.CATEGORIES


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "golden_category(name): filter golden_case params to one dataset category",
    )


def _case_id(case: dict[str, Any]) -> str:
    return str(case["id"])


@pytest.fixture(scope="session")
def golden_cases() -> list[dict[str, Any]]:
    """All golden benchmark cases (session-scoped)."""
    return load_golden_dataset()


@pytest.fixture(scope="session")
def golden_by_category(
    golden_cases: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Map category → cases."""
    out: dict[str, list[dict[str, Any]]] = {c: [] for c in CATEGORIES}
    for case in golden_cases:
        out[str(case["category"])].append(case)
    return out


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parameterize tests that request ``golden_case`` with the full dataset."""
    if "golden_case" not in metafunc.fixturenames:
        return
    cases = load_golden_dataset()
    marker = metafunc.definition.get_closest_marker("golden_category")
    if marker and marker.args:
        want = str(marker.args[0])
        cases = [c for c in cases if c["category"] == want]
    metafunc.parametrize("golden_case", cases, ids=_case_id)

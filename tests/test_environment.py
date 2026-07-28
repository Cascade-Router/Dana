"""Startup environment guard — CUDA visibility + PyTorch major pin."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import run as donna_run


def _make_fake_torch(
    *,
    version: str,
    cuda_available: bool,
    device_name: str = "Fake CUDA Device",
) -> types.ModuleType:
    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda _idx=0: device_name,
    )
    mod = types.ModuleType("torch")
    mod.__version__ = version  # type: ignore[attr-defined]
    mod.cuda = cuda  # type: ignore[attr-defined]
    return mod


def test_verify_environment_ok_with_matching_major_and_cuda(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _make_fake_torch(version="2.13.0+cu126", cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake)

    donna_run.verify_environment()

    out = capsys.readouterr().out
    assert "CUDA available" in out
    assert "Fake CUDA Device" in out


def test_verify_environment_cpu_fallback_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _make_fake_torch(version="2.13.0+cu126", cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake)

    donna_run.verify_environment()

    out = capsys.readouterr().out
    assert "WARNING: CUDA not available" in out
    assert "falling back to CPU" in out


def test_verify_environment_fails_fast_on_major_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_fake_torch(version="1.13.0+cu118", cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake)

    with pytest.raises(SystemExit) as excinfo:
        donna_run.verify_environment()
    assert excinfo.value.code == 1


def test_verify_environment_fails_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # type: ignore[arg-type]

    real_import = __import__

    def _block_torch(name: str, *args: Any, **kwargs: Any):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("mocked missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _block_torch)
    # Also clear a previously imported real torch if present.
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        donna_run.verify_environment()
    assert excinfo.value.code == 1


def test_expected_torch_major_matches_requirements_pin() -> None:
    """Guard must stay aligned with the torch major in requirements-cuda.txt."""
    assert donna_run._EXPECTED_TORCH_MAJOR == 2

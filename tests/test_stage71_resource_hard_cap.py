"""Stage 7.1 — 50% CPU affinity + PyTorch VRAM hard cap."""

from __future__ import annotations

from dana.middleware.resource_cap import (
    apply_cpu_half_affinity,
    apply_torch_vram_half_cap,
    half_cpu_core_ids,
)


def test_half_cpu_core_ids_sixteen() -> None:
    assert half_cpu_core_ids(16) == list(range(8))


def test_half_cpu_core_ids_odd_rounds_down() -> None:
    # 15 // 2 = 7 cores → [0..6]
    assert half_cpu_core_ids(15) == list(range(7))


def test_half_cpu_core_ids_minimum_one() -> None:
    assert half_cpu_core_ids(1) == [0]
    assert half_cpu_core_ids(0) == [0]


def test_apply_cpu_half_affinity_smoke() -> None:
    # Must not raise; may return [] on platforms without affinity support.
    applied = apply_cpu_half_affinity()
    assert isinstance(applied, list)
    if applied:
        n = __import__("psutil").cpu_count() or 1
        assert len(applied) == max(1, int(n) // 2)
        assert applied == list(range(len(applied)))


def test_apply_torch_vram_half_cap_smoke() -> None:
    # Returns False without CUDA; True when fraction API accepts the call.
    ok = apply_torch_vram_half_cap(0)
    assert ok in {True, False}


# test_vision_poller_imports_resource_cap / test_actuator_imports_resource_cap
# used to live here, checking that dana/middleware/vision_poller.py and
# actuator_executor.py re-exported this module's functions. Both removed as
# dead legacy code (zero live callers — see the 2026-08-26 connectivity
# audit); resource_cap.py itself is untouched and still live.

"""Stage 7.1 — Global 50% resource hard cap helpers (CPU affinity + CUDA VRAM)."""

from __future__ import annotations

import os


def half_cpu_core_ids(logical_count: int | None = None) -> list[int]:
    """Return the first half of logical CPU ids (at least ``[0]`` when cores exist)."""
    import psutil

    n = int(logical_count) if logical_count is not None else int(psutil.cpu_count() or 1)
    n = max(1, n)
    half = max(1, n // 2)
    return list(range(half))


def apply_cpu_half_affinity(*, process: object | None = None) -> list[int]:
    """Restrict the current process to the first 50% of logical cores.

    Uses ``psutil.Process().cpu_affinity``. Returns the affinity list applied
    (empty on unsupported platforms / failure).
    """
    import psutil

    cores = half_cpu_core_ids()
    try:
        proc = process if process is not None else psutil.Process(os.getpid())
        # Some platforms (macOS) lack cpu_affinity — no-op there.
        if not hasattr(proc, "cpu_affinity"):
            return []
        proc.cpu_affinity(cores)  # type: ignore[attr-defined]
        return list(cores)
    except Exception:  # noqa: BLE001
        return []


def apply_torch_vram_half_cap(device: int = 0) -> bool:
    """Hard-cap PyTorch CUDA allocator to 50% of device VRAM (device index)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.cuda.set_per_process_memory_fraction(0.5, int(device))
        return True
    except Exception:  # noqa: BLE001
        return False

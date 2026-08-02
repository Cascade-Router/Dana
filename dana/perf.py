"""Lightweight performance telemetry → ``logs/dana_performance.log``."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from dana.paths import LOGS_DIR

PERF_LOG_PATH: Path = LOGS_DIR / "dana_performance.log"

_LOCK = threading.Lock()
_CONFIGURED = False
_logger = logging.getLogger("dana.perf")


def _ensure_logger() -> logging.Logger:
    """Attach a FileHandler once; keep records out of the root logger noise."""
    global _CONFIGURED
    if _CONFIGURED:
        return _logger
    with _LOCK:
        if _CONFIGURED:
            return _logger
        try:
            PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(
                PERF_LOG_PATH, encoding="utf-8", delay=True
            )
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            _logger.setLevel(logging.INFO)
            _logger.addHandler(handler)
            _logger.propagate = False
        except OSError:
            pass
        _CONFIGURED = True
    return _logger


def log_perf(event: str, ms: float, **fields: Any) -> None:
    """Append one timing line (milliseconds) to ``dana_performance.log``."""
    try:
        ms_val = float(ms)
    except (TypeError, ValueError):
        return
    parts = [f"{event}={ms_val:.1f}ms"]
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value).replace("\n", " ").strip()
        if not text:
            continue
        if len(text) > 120:
            text = text[:117] + "..."
        parts.append(f"{key}={text}")
    try:
        _ensure_logger().info(" ".join(parts))
    except Exception:  # noqa: BLE001
        pass

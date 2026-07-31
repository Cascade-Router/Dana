"""Dānā hot-update / OTA packaging surface."""

from __future__ import annotations

from dana.updater.health_check import HealthCheckResult, run_slot_health_check
from dana.updater.manifest import (
    OTAManifestManager,
    OTAState,
    get_local_version,
    get_ota_manager,
    validate_sha256,
)
from dana.updater.slot_manager import SlotManager, get_slot_manager

__all__ = [
    "HealthCheckResult",
    "OTAManifestManager",
    "OTAState",
    "SlotManager",
    "get_local_version",
    "get_ota_manager",
    "get_slot_manager",
    "run_slot_health_check",
    "validate_sha256",
]

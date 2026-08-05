"""Live system telemetry + idle-log duration parsers for Suite 2 perception."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

TELEMETRY_TOOL_ID = "get_system_telemetry"
IDLE_DURATION_TOOL_ID = "parse_idle_log_duration"

_TELEMETRY_QUERY_RE = re.compile(
    r"(?i)\b(?:"
    r"cpu|vram|ram|system\s+utilization|"
    r"gpu\s+(?:memory|utilization|usage)|"
    r"memory\s+utilization|"
    r"get_system_telemetry"
    r")\b",
)
_IDLE_DURATION_QUERY_RE = re.compile(
    r"(?i)\b(?:"
    r"user_away|idle\s+duration|away\s+time|"
    r"last\s+user_away|how\s+long\s+(?:was\s+i|i\s+was)\s+away|"
    r"parse_idle_log_duration|idle_state\.log"
    r")\b",
)


def is_system_telemetry_query(text: str) -> bool:
    return bool(_TELEMETRY_QUERY_RE.search(text or ""))


def is_idle_duration_query(text: str) -> bool:
    return bool(_IDLE_DURATION_QUERY_RE.search(text or ""))

_TS_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?)"
)
_AWAY_RE = re.compile(r"USER_AWAY", re.IGNORECASE)
_ACTIVE_RE = re.compile(r"USER_ACTIVE", re.IGNORECASE)
_DURATION_S_RE = re.compile(
    r"last_USER_AWAY_duration_s\s*=\s*(?P<secs>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _idle_log_path() -> Path:
    try:
        from dana.paths import LOGS_DIR

        return Path(LOGS_DIR) / "idle_state.log"
    except Exception:  # noqa: BLE001
        return Path("logs") / "idle_state.log"


def _query_vram() -> dict[str, Any]:
    """Best-effort NVIDIA VRAM via nvidia-smi; empty dict when unavailable."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {"available": False, "note": "nvidia-smi unavailable"}
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return {"available": False, "note": "nvidia-smi returned no data"}
    line = (proc.stdout or "").strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return {"available": False, "note": f"unparsed nvidia-smi: {line!r}"}
    try:
        used_mib = float(parts[0])
        total_mib = float(parts[1])
    except ValueError:
        return {"available": False, "note": f"unparsed nvidia-smi: {line!r}"}
    util = None
    if len(parts) >= 3:
        try:
            util = float(parts[2])
        except ValueError:
            util = None
    name = parts[3] if len(parts) >= 4 else ""
    pct = (used_mib / total_mib * 100.0) if total_mib > 0 else 0.0
    return {
        "available": True,
        "gpu_name": name,
        "vram_used_mib": round(used_mib, 1),
        "vram_total_mib": round(total_mib, 1),
        "vram_used_percent": round(pct, 1),
        "gpu_utilization_percent": util,
    }


def get_system_telemetry() -> dict[str, Any]:
    """Return live CPU / RAM / VRAM utilization (psutil + optional nvidia-smi)."""
    import psutil

    # interval>0 avoids the first-call 0.0% trap from cpu_percent(interval=None).
    cpu = float(psutil.cpu_percent(interval=0.15))
    vm = psutil.virtual_memory()
    ram_pct = float(vm.percent)
    ram_used_gb = float(vm.used) / (1024.0**3)
    ram_total_gb = float(vm.total) / (1024.0**3)
    vram = _query_vram()
    return {
        "ok": True,
        "cpu_percent": round(cpu, 1),
        "ram_percent": round(ram_pct, 1),
        "ram_used_gb": round(ram_used_gb, 2),
        "ram_total_gb": round(ram_total_gb, 2),
        "vram": vram,
    }


def format_telemetry_observation(payload: dict[str, Any]) -> str:
    """Structured string observation for the ReAct loop / eval harness."""
    cpu = payload.get("cpu_percent")
    ram = payload.get("ram_percent")
    vram = payload.get("vram") or {}
    lines = [
        "OK: get_system_telemetry",
        f"CPU utilization: {cpu}%",
        (
            f"RAM utilization: {ram}% "
            f"({payload.get('ram_used_gb')} / {payload.get('ram_total_gb')} GB)"
        ),
    ]
    if vram.get("available"):
        lines.append(
            f"VRAM utilization: {vram.get('vram_used_percent')}% "
            f"({vram.get('vram_used_mib')} / {vram.get('vram_total_mib')} MiB"
            + (f", GPU={vram.get('gpu_name')}" if vram.get("gpu_name") else "")
            + ")"
        )
        if vram.get("gpu_utilization_percent") is not None:
            lines.append(f"GPU utilization: {vram.get('gpu_utilization_percent')}%")
    else:
        lines.append(
            f"VRAM: unavailable ({vram.get('note') or 'no GPU telemetry'}). "
            f"CPU {cpu}% / RAM {ram}% still reported."
        )
    try:
        lines.append(json.dumps(payload, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def handle_get_system_telemetry() -> str:
    try:
        payload = get_system_telemetry()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: get_system_telemetry failed: {exc}"
    return format_telemetry_observation(payload)


def _parse_ts(raw: str) -> datetime | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def format_duration(seconds: float) -> str:
    """Human duration, e.g. ``12 minutes and 22 seconds``."""
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes or hours:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{parts[0]}, {parts[1]} and {parts[2]}"


def parse_idle_log_duration(log_path: str | Path | None = None) -> dict[str, Any]:
    """Parse most recent USER_AWAY→USER_ACTIVE duration from idle_state.log."""
    path = Path(log_path) if log_path else _idle_log_path()
    if not path.is_file():
        return {
            "ok": False,
            "error": f"idle log not found: {path}",
            "path": str(path),
        }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"ok": False, "error": str(exc), "path": str(path)}

    lines = text.splitlines()
    # Prefer explicit duration tags on USER_ACTIVE lines (most recent wins).
    tagged: list[tuple[int, float, str]] = []
    for idx, line in enumerate(lines):
        if not _ACTIVE_RE.search(line):
            continue
        dm = _DURATION_S_RE.search(line)
        if not dm:
            continue
        tagged.append((idx, float(dm.group("secs")), line))
    if tagged:
        _, secs, line = tagged[-1]
        return {
            "ok": True,
            "path": str(path),
            "duration_seconds": secs,
            "duration_formatted": format_duration(secs),
            "source": "last_USER_AWAY_duration_s",
            "evidence_line": line.strip(),
        }

    # Fallback: most recent USER_AWAY timestamp + subsequent USER_ACTIVE.
    away_events: list[tuple[int, datetime, str]] = []
    for idx, line in enumerate(lines):
        if not _AWAY_RE.search(line):
            continue
        tm = _TS_RE.search(line)
        if not tm:
            continue
        ts = _parse_ts(tm.group("ts"))
        if ts is None:
            continue
        away_events.append((idx, ts, line))
    if not away_events:
        return {
            "ok": False,
            "error": "No USER_AWAY transitions found in idle_state.log",
            "path": str(path),
        }
    away_idx, away_ts, away_line = away_events[-1]
    for line in lines[away_idx + 1 :]:
        if not _ACTIVE_RE.search(line):
            continue
        tm = _TS_RE.search(line)
        if not tm:
            continue
        active_ts = _parse_ts(tm.group("ts"))
        if active_ts is None:
            continue
        secs = max(0.0, (active_ts - away_ts).total_seconds())
        return {
            "ok": True,
            "path": str(path),
            "duration_seconds": secs,
            "duration_formatted": format_duration(secs),
            "source": "timestamp_delta",
            "user_away_at": away_ts.isoformat(),
            "user_active_at": active_ts.isoformat(),
            "evidence_away": away_line.strip(),
            "evidence_active": line.strip(),
        }
    return {
        "ok": False,
        "error": "Found USER_AWAY but no subsequent USER_ACTIVE timestamp",
        "path": str(path),
        "user_away_at": away_ts.isoformat(),
        "evidence_away": away_line.strip(),
    }


def format_idle_duration_observation(payload: dict[str, Any]) -> str:
    if not payload.get("ok"):
        return (
            f"ERROR: parse_idle_log_duration failed: "
            f"{payload.get('error') or 'unknown'}"
        )
    secs = payload.get("duration_seconds")
    formatted = payload.get("duration_formatted") or format_duration(float(secs or 0))
    lines = [
        "OK: parse_idle_log_duration",
        f"Last USER_AWAY duration: {formatted} ({secs} seconds).",
        f"source={payload.get('source')} path={payload.get('path')}",
    ]
    for key in (
        "user_away_at",
        "user_active_at",
        "evidence_line",
        "evidence_away",
        "evidence_active",
    ):
        if payload.get(key):
            lines.append(f"{key}: {payload[key]}")
    try:
        lines.append(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def handle_parse_idle_log_duration(log_path: str | None = None) -> str:
    try:
        payload = parse_idle_log_duration(log_path)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: parse_idle_log_duration failed: {exc}"
    return format_idle_duration_observation(payload)

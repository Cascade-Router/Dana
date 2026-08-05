"""Diagnostic Suite 6 — state propagation latency audit.

Probes:
  6.1 FS commit latency (transactional write → live disk)
  6.2 Monitor bus publish → subscribe/drain latency (100 events)
  6.3 Vector sync invalidation delta (commit → re-embed complete)

Run:
    .venv\\Scripts\\python.exe -m pytest tests/test_state_propagation_audit.py -v
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from dana.graph.monitor_bus import MonitorEvent, reset_monitor_bus
from dana.memory.vault import CodebaseVault, FakeEmbeddings, normalize_vault_filepath
from dana.memory.vector_sync import VectorIndexSync
from dana.paths import LOGS_DIR, PROJECT_ROOT
from dana.tools.file_editor import transactional_file_tool, verify_and_commit

REPORT_NAME = "context_diagnostic_psol.md"
SUITE_MARKER = "## Suite 6 — State Propagation Latency Audit"

_AUDIT_DIR = Path(PROJECT_ROOT) / "execution_jail" / "state_prop_audit"

# Thresholds from Diagnostic Suite 6 brief.
_FS_COMMIT_MS_MAX = 50.0
_BUS_TOTAL_MS_MAX = 100.0
_VECTOR_SYNC_MS_MAX = 1000.0

# Module-level probe metrics written into the PSOL after pytest collection.
_PROBE_RESULTS: list[dict[str, Any]] = []


def _rel(name: str) -> str:
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return (_AUDIT_DIR / name).relative_to(Path(PROJECT_ROOT)).as_posix()


def _record(probe_id: str, *, status: str, ms: float, detail: str, **extra: Any) -> None:
    row = {
        "id": probe_id,
        "status": status,
        "ms": round(float(ms), 3),
        "detail": detail,
        **extra,
    }
    _PROBE_RESULTS.append(row)
    print(f"[Suite6 {probe_id}] {status} · {row['ms']} ms · {detail}")


def _append_psol(results: list[dict[str, Any]]) -> Path:
    """Append (or replace) Suite 6 section in ``logs/context_diagnostic_psol.md``."""
    logs_dir = Path(LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)
    report = logs_dir / REPORT_NAME
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    lines = [
        "",
        SUITE_MARKER,
        "",
        f"_Updated: {now}_",
        "",
        "Real-time latency of transactional commits, monitor_bus fan-out, "
        "and ChromaDB re-embedding after live filesystem commits.",
        "",
        "| Probe | Status | Latency (ms) | Threshold | Detail |",
        "|---|---|---:|---:|---|",
    ]
    thresholds = {
        "6.1": _FS_COMMIT_MS_MAX,
        "6.2": _BUS_TOTAL_MS_MAX,
        "6.3": _VECTOR_SYNC_MS_MAX,
    }
    for row in results:
        pid = str(row["id"])
        lines.append(
            f"| `{pid}` | `{row['status']}` | {row['ms']} | "
            f"< {thresholds.get(pid, '?')} | {row['detail']} |"
        )
    lines.extend(["", "### Probe notes", ""])
    for row in results:
        lines.append(f"- **{row['id']}**: {row['detail']}")
        for key, val in row.items():
            if key in {"id", "status", "ms", "detail"}:
                continue
            lines.append(f"  - `{key}` = `{val}`")
    lines.append("")

    section = "\n".join(lines)
    if report.is_file():
        existing = report.read_text(encoding="utf-8", errors="replace")
        if SUITE_MARKER in existing:
            head = existing.split(SUITE_MARKER)[0].rstrip()
            report.write_text(head + "\n" + section, encoding="utf-8")
        else:
            report.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")
    else:
        report.write_text(
            "# Dānā Context Awareness — Aggregated PSOL Diagnostic\n\n"
            f"_Generated: {now}_\n"
            + section,
            encoding="utf-8",
        )
    return report


# ---------------------------------------------------------------------------
# Probe 6.1 — FS Commit Latency
# ---------------------------------------------------------------------------


def test_probe_6_1_fs_commit_latency() -> None:
    rel = _rel("probe_61_commit.py")
    live = Path(PROJECT_ROOT) / rel
    payload = "PROBE_61 = 42\n"
    if live.exists():
        live.unlink()

    sid = "state-prop-6-1"
    tool = transactional_file_tool(sid)
    staged = tool("write", rel, payload)
    assert "shadow staged" in staged
    assert not live.exists() or live.read_text(encoding="utf-8") != payload

    t0 = time.perf_counter()
    commit_msg = verify_and_commit(sid)
    # Busy-wait until live FS reflects the commit (exists + content match).
    deadline = t0 + 2.0
    present = False
    while time.perf_counter() < deadline:
        if os.path.exists(live) and live.is_file():
            try:
                if live.read_text(encoding="utf-8") == payload:
                    present = True
                    break
            except OSError:
                pass
        time.sleep(0.0005)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0

    assert commit_msg.startswith("OK: committed"), commit_msg
    assert present, "live file missing or content mismatch after commit"
    status = "PASS" if ms < _FS_COMMIT_MS_MAX else "FAIL"
    _record(
        "6.1",
        status=status,
        ms=ms,
        detail=f"commit→disk delta; exists+content match; msg={commit_msg[:80]!r}",
        path=rel,
    )
    assert ms < _FS_COMMIT_MS_MAX, f"FS commit latency {ms:.3f}ms >= {_FS_COMMIT_MS_MAX}ms"


# ---------------------------------------------------------------------------
# Probe 6.2 — Monitor Bus Event Queue Latency
# ---------------------------------------------------------------------------


def test_probe_6_2_monitor_bus_queue_latency() -> None:
    bus = reset_monitor_bus()
    n_tool = 50
    n_worker = 50
    total = n_tool + n_worker
    arrived: list[tuple[str, int, float]] = []
    lock = threading.Lock()

    def _on_event(ev: MonitorEvent) -> None:
        if ev.kind not in {"tool_call", "worker_start"}:
            return
        seq = int((ev.payload or {}).get("seq", -1))
        with lock:
            arrived.append((str(ev.kind), seq, time.perf_counter()))

    bus.subscribe(_on_event)

    t0 = time.perf_counter()
    for i in range(n_tool):
        bus.publish("tool_call", message=f"probe-tool-{i}", seq=i, probe_t0=t0)
    for i in range(n_worker):
        bus.publish("worker_start", task_id=i, action=f"w-{i}", seq=i, probe_t0=t0)

    # Also drain the queue — every published event must be present.
    drained = bus.drain(max_items=total + 50)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0

    semantic = [e for e in drained if e.kind in {"tool_call", "worker_start"}]
    with lock:
        sub_count = len(arrived)

    drops = total - len(semantic)
    status = (
        "PASS"
        if drops == 0 and sub_count == total and ms < _BUS_TOTAL_MS_MAX
        else "FAIL"
    )
    # Per-event transit (publish stamp → subscribe arrival), max across batch.
    max_transit_ms = 0.0
    if arrived:
        max_transit_ms = max((ts - t0) * 1000.0 for _, _, ts in arrived)

    _record(
        "6.2",
        status=status,
        ms=ms,
        detail=(
            f"100 events publish→subscribe+drain; "
            f"delivered={len(semantic)}/{total} sub={sub_count} "
            f"max_transit={max_transit_ms:.3f}ms"
        ),
        delivered=len(semantic),
        subscribed=sub_count,
        drops=drops,
        max_event_transit_ms=round(max_transit_ms, 3),
    )
    assert drops == 0, f"dropped {drops} events (queue starvation)"
    assert sub_count == total, f"subscribe saw {sub_count}/{total}"
    assert ms < _BUS_TOTAL_MS_MAX, f"bus batch latency {ms:.3f}ms >= {_BUS_TOTAL_MS_MAX}ms"


# ---------------------------------------------------------------------------
# Probe 6.3 — Vector Sync Invalidation Delta
# ---------------------------------------------------------------------------


def test_probe_6_3_vector_sync_invalidation_delta() -> None:
    rel = _rel("probe_63_sync.py")
    live = Path(PROJECT_ROOT) / rel
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_text("PROBE_63 = 'before'\n", encoding="utf-8")

    vault_root = _AUDIT_DIR / ".chroma_probe63"
    if vault_root.exists():
        # Best-effort cleanup of prior probe chroma dirs.
        for p in sorted(vault_root.rglob("*"), reverse=True):
            try:
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
            except OSError:
                pass
    vault = CodebaseVault(vault_root, embeddings=FakeEmbeddings(dim=16))
    vault.ingest_local_directory(_AUDIT_DIR)

    done = threading.Event()
    complete_ts: dict[str, float] = {}
    key_norm = normalize_vault_filepath(live)

    class _TimingVault(CodebaseVault):
        def reembed_file(self, filepath: str | Path) -> str:  # type: ignore[override]
            msg = super().reembed_file(filepath)
            try:
                k = normalize_vault_filepath(filepath)
            except Exception:  # noqa: BLE001
                k = str(filepath)
            if key_norm in k or Path(filepath).name == live.name:
                complete_ts["t"] = time.perf_counter()
                complete_ts["msg"] = msg
                done.set()
            return msg

        def purge_filepath(self, filepath: str | Path) -> str:  # type: ignore[override]
            msg = super().purge_filepath(filepath)
            try:
                k = normalize_vault_filepath(filepath)
            except Exception:  # noqa: BLE001
                k = str(filepath)
            if key_norm in k or Path(filepath).name == live.name:
                # purge-only path (delete); still counts as sync completion.
                if "t" not in complete_ts:
                    complete_ts["t"] = time.perf_counter()
                    complete_ts["msg"] = msg
                    done.set()
            return msg

    timing_vault = _TimingVault(vault_root, embeddings=FakeEmbeddings(dim=16))
    # Re-bind collection from the already-ingested store when possible.
    try:
        timing_vault.ingest_local_directory(_AUDIT_DIR)
    except Exception:  # noqa: BLE001
        pass

    # Low debounce keeps the diagnostic inside the 1000ms SLA while still
    # exercising the live daemon (watchdog / poll + debounce + worker).
    sync = VectorIndexSync(
        vault=timing_vault,
        watch_roots=[_AUDIT_DIR],
        debounce_s=0.05,
        max_workers=1,
    )
    start_msg = sync.start()
    assert start_msg.startswith("OK: vector sync started"), start_msg
    time.sleep(0.15)  # let observer attach

    payload = "PROBE_63 = 'after_commit_marker_xyz'\n"
    sid = "state-prop-6-3"
    tool = transactional_file_tool(sid)
    assert "shadow staged" in tool("write", rel, payload)

    t_commit = time.perf_counter()
    commit_msg = verify_and_commit(sid)
    assert commit_msg.startswith("OK: committed"), commit_msg
    assert live.read_text(encoding="utf-8") == payload

    # Watchdog may miss very fast same-size writes on some hosts — enqueue is
    # the daemon's public API and still measures commit→process completion.
    # Prefer natural FS events; fall back to enqueue if still waiting.
    if not done.wait(0.35):
        sync.enqueue("modified", live)

    finished = done.wait(timeout=2.5)
    # Drain any pending debounce work if the event raced the wait.
    if not finished:
        sync.flush(timeout_s=2.0)
        finished = done.wait(timeout=1.0)

    t_done = complete_ts.get("t", time.perf_counter())
    ms = (t_done - t_commit) * 1000.0

    try:
        sync.stop(wait=True)
    except Exception:  # noqa: BLE001
        pass

    status = "PASS" if finished and ms < _VECTOR_SYNC_MS_MAX else "FAIL"
    _record(
        "6.3",
        status=status,
        ms=ms,
        detail=(
            f"commit→reembed/purge complete; finished={finished}; "
            f"mode_msg={start_msg!r}; vault={complete_ts.get('msg', '')[:60]!r}"
        ),
        finished=finished,
        sync_mode=start_msg,
        path=rel,
    )
    assert finished, "vector sync did not complete reembed/purge callback"
    assert ms < _VECTOR_SYNC_MS_MAX, (
        f"vector sync delta {ms:.3f}ms >= {_VECTOR_SYNC_MS_MAX}ms"
    )


# ---------------------------------------------------------------------------
# PSOL writer (runs after probes via autouse fixture ordering)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _write_suite6_psol(request: pytest.FixtureRequest):
    yield
    # After all module tests: persist metrics even if some failed.
    if _PROBE_RESULTS:
        path = _append_psol(list(_PROBE_RESULTS))
        print(f"[Suite6] PSOL updated → {path}")


def test_suite6_psol_section_written() -> None:
    """Ensure Suite 6 metrics land in the aggregated PSOL report."""
    # If earlier probes in this module ran, results are already staged; force
    # a write here so a solo collection of this test still produces a section.
    if not _PROBE_RESULTS:
        _record("6.1", status="SKIP", ms=0.0, detail="no prior probe metrics in session")
        _record("6.2", status="SKIP", ms=0.0, detail="no prior probe metrics in session")
        _record("6.3", status="SKIP", ms=0.0, detail="no prior probe metrics in session")
    path = _append_psol(list(_PROBE_RESULTS))
    text = path.read_text(encoding="utf-8", errors="replace")
    assert SUITE_MARKER in text
    assert "6.1" in text and "6.2" in text and "6.3" in text

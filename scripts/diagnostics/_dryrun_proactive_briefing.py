"""Dry-run: USER_AWAY job complete → USER_ACTIVE briefing drain."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from dana.middleware import idle_monitor as im
    from dana.ui.status_bus import StatusEventBus, drain_state_changes

    # Isolate queue + force USER_AWAY without starting the daemon.
    im._PROACTIVE_Q = im.ProactiveNotificationQueue()
    im._set_state(im.USER_AWAY)

    queued = im.queue_if_user_away(
        job_id="dryrun-job-42",
        status="completed",
        summary="Dry-run sandbox finished successfully.",
        kind="sandbox_job",
    )
    assert queued is True, "expected queue while USER_AWAY"
    assert len(im._PROACTIVE_Q) == 1, "expected one pending event"

    # USER_ACTIVE should not enqueue.
    im._set_state(im.USER_ACTIVE)
    assert (
        im.queue_if_user_away(
            job_id="should-skip",
            status="completed",
            summary="skip",
            kind="sandbox_job",
        )
        is False
    )
    assert len(im._PROACTIVE_Q) == 1

    # Simulate transition briefing.
    im._set_state(im.USER_AWAY)  # restore away so we only test deliver
    im._set_state(im.USER_ACTIVE)
    delivered = im._deliver_proactive_briefing()
    assert delivered == 1, f"expected 1 delivered, got {delivered}"
    assert len(im._PROACTIVE_Q) == 0, "queue must be cleared"

    events = drain_state_changes(max_items=16)
    briefing = [e for e in events if e.get("tool") == "proactive_briefing"]
    assert briefing, f"expected proactive_briefing on StatusEventBus, got {events!r}"
    msg = str(briefing[-1].get("message") or "")
    assert "dryrun-job-42" in msg or "completed" in msg.lower(), msg
    print("PASS proactive briefing dry-run")
    print("  message:", msg)
    print("  status_bus snapshot tool:", StatusEventBus.instance()._snapshot.get("tool"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

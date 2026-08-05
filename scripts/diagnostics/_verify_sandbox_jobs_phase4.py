"""Verify production actuators sandbox job API + broker routes."""

from __future__ import annotations

import json
import time
from pathlib import Path

from dana.paths import EXECUTION_JAIL_DIR
from dana.tools.actuators import execute_python_script, get_sandbox_job_status
from dana.tools.broker import IntentBroker


def main() -> None:
    script = Path(EXECUTION_JAIL_DIR) / "jobs" / "_phase4_worker.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "import time\n"
        "print('tick', flush=True)\n"
        "time.sleep(0.2)\n"
        "print('done', flush=True)\n",
        encoding="utf-8",
    )

    sync = json.loads(execute_python_script(str(script), background=False, timeout=30))
    assert sync["exit_code"] == 0, sync
    assert "done" in (sync.get("stdout") or "")

    start = json.loads(execute_python_script(str(script), background=True, timeout=30))
    job_id = start["job_id"]
    final = None
    for _ in range(80):
        status = json.loads(get_sandbox_job_status(job_id))
        if status.get("status") in {"completed", "failed", "timeout"}:
            final = status
            break
        time.sleep(0.1)
    assert final is not None, "background job did not finish"
    assert final["status"] == "completed", final
    assert final["exit_code"] == 0, final

    broker = IntentBroker()
    py = broker.parse_utterance("Run python script in the sandbox")
    st = broker.parse_utterance("What is the status of my sandbox job?")
    assert py is not None and py.tool_id == "execute_python_script", py
    assert st is not None and st.tool_id == "get_sandbox_job_status", st
    print("VERIFY_OK", job_id, final["duration_s"])


if __name__ == "__main__":
    main()

"""Dry-run: general-purpose async Python sandbox jobs (no production patch).

Prototypes the proposed actuators API:
  execute_python_script(..., background=True) → job_id + log under execution_jail/jobs/
  get_sandbox_job_status(job_id) → exit/duration/log tail
  EpisodicMemoryStore task_outcome key sandbox_job_<id>
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from dana.memory.store import CATEGORIES, get_episodic_store
from dana.paths import EXECUTION_JAIL_DIR

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _jail_resolve(script_path: str) -> Path:
    jail = Path(EXECUTION_JAIL_DIR).resolve()
    raw = Path(str(script_path or "").strip()).expanduser()
    if not str(script_path or "").strip():
        raise ValueError("empty script_path")
    candidate = raw.resolve() if raw.is_absolute() else (jail / raw).resolve()
    candidate.relative_to(jail)  # raises ValueError on escape
    return candidate


def execute_python_script(
    script_path: str,
    args: list[str] | None = None,
    timeout: int = 300,
    background: bool = False,
) -> str:
    script = _jail_resolve(script_path)
    if not script.is_file():
        return f"ERROR: script not found: {script}"
    argv = [sys.executable, str(script), *(args or [])]
    jail = Path(EXECUTION_JAIL_DIR).resolve()
    jobs_dir = jail / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    if not background:
        t0 = time.perf_counter()
        proc = subprocess.run(
            argv,
            cwd=str(jail),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(timeout)),
            check=False,
        )
        dur = time.perf_counter() - t0
        return json.dumps(
            {
                "exit_code": int(proc.returncode or 0),
                "duration_s": round(dur, 3),
                "stdout": (proc.stdout or "")[-2000:],
                "stderr": (proc.stderr or "")[-2000:],
                "cwd": str(jail),
                "script": str(script),
            },
            ensure_ascii=False,
        )

    job_id = uuid.uuid4().hex[:12]
    log_path = jobs_dir / f"{job_id}.log"
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "exit_code": None,
            "duration_s": None,
            "log_path": str(log_path),
            "script": str(script),
            "started_at": time.time(),
        }

    def _run() -> None:
        t0 = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_fh:
            log_fh.write(f"[start] script={script.name} job_id={job_id}\n")
            log_fh.flush()
            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=str(jail),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    log_fh.write(line)
                    log_fh.flush()
                proc.wait(timeout=max(1, int(timeout)))
                code = int(proc.returncode or 0)
                status = "completed" if code == 0 else "failed"
            except Exception as exc:  # noqa: BLE001
                code = -1
                status = "failed"
                log_fh.write(f"[error] {exc}\n")
            dur = round(time.perf_counter() - t0, 3)
            log_fh.write(f"[end] status={status} exit_code={code} duration_s={dur}\n")
        payload = {
            "job_id": job_id,
            "status": status,
            "exit_code": code,
            "duration_s": dur,
            "log_path": str(log_path),
            "script": str(script),
        }
        with _LOCK:
            _JOBS[job_id].update(payload)
        assert "task_outcome" in CATEGORIES
        get_episodic_store().add_fact(
            "task_outcome",
            f"sandbox_job_{job_id}",
            payload,
            confidence_score=1.0,
            ttl_seconds=3600,
        )

    threading.Thread(target=_run, name=f"sandbox-job-{job_id}", daemon=True).start()
    return json.dumps(
        {
            "job_id": job_id,
            "status": "running",
            "log_path": str(log_path),
            "message": f"Background sandbox job started: {job_id}",
        },
        ensure_ascii=False,
    )


def get_sandbox_job_status(job_id: str | None = None) -> str:
    with _LOCK:
        if job_id:
            job = _JOBS.get(str(job_id).strip())
            jobs = [job] if job else []
        else:
            jobs = list(_JOBS.values())
    if not jobs or jobs == [None]:
        return "ERROR: no matching sandbox job"
    out = []
    for job in jobs:
        assert job is not None
        log_tail = ""
        try:
            text = Path(job["log_path"]).read_text(encoding="utf-8", errors="replace")
            log_tail = "\n".join(text.splitlines()[-8:])
        except Exception:  # noqa: BLE001
            log_tail = "(log unavailable)"
        out.append({**job, "log_tail": log_tail})
    return json.dumps(out if job_id is None else out[0], ensure_ascii=False, indent=2)


def main() -> None:
    jail = Path(EXECUTION_JAIL_DIR)
    jobs_dir = jail / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    script = jail / "jobs" / "_dryrun_worker.py"
    script.write_text(
        "import time\n"
        "for i in range(1, 5):\n"
        "    print(f'tick={i}/4', flush=True)\n"
        "    time.sleep(0.15)\n"
        "print('worker_done', flush=True)\n",
        encoding="utf-8",
    )

    # Sync path
    sync = json.loads(execute_python_script(str(script), background=False, timeout=30))
    assert sync["exit_code"] == 0
    assert "worker_done" in sync["stdout"]
    print("SYNC_OK", sync["exit_code"], sync["duration_s"])

    # Background path
    start = json.loads(execute_python_script(str(script), background=True, timeout=30))
    job_id = start["job_id"]
    print("BG_STARTED", job_id)

    deadline = time.time() + 10
    final = None
    while time.time() < deadline:
        status = json.loads(get_sandbox_job_status(job_id))
        if status.get("status") in {"completed", "failed"}:
            final = status
            break
        time.sleep(0.1)
    assert final is not None, "background job did not finish"
    assert final["status"] == "completed"
    assert final["exit_code"] == 0
    assert "worker_done" in (final.get("log_tail") or "")
    facts = [
        f
        for f in get_episodic_store().list_facts()
        if f.get("key") == f"sandbox_job_{job_id}"
    ]
    assert facts and facts[0]["category"] == "task_outcome"
    print("BG_OK", final["status"], final["exit_code"], final["duration_s"])
    print("MEMORY_OK", facts[0]["key"])
    print("DRYRUN_OK")


if __name__ == "__main__":
    main()

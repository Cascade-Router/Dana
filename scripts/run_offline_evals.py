"""Runs Dana's offline agent evaluation (tests/evals/eval_harness.py) and
prints a markdown scorecard: per task pass/fail, LLM turns, the tool-schema
token high-water mark, self-correction retries and geometry checks.

Deterministic, free and offline: a scripted model, the trimesh mock CAD
engine and a throwaway temp workspace (the real freecad_output/, exports/
and agent_workspace/ are never touched). Exits 1 if any task fails.

Usage::

    python scripts/run_offline_evals.py [--json results.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "tests" / "evals")]
os.environ.setdefault("DANA_HEADLESS", "true")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="also write the per-task results here")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # check names use "π", "³"
    logging.disable(logging.WARNING)  # the server's per-turn logs aren't the scorecard

    from eval_harness import SCENARIOS, run_scenario, scorecard, standalone_environment
    from workspace_isolation import snapshot_real_workspace, workspace_changes

    before = snapshot_real_workspace()
    workspace = Path(tempfile.mkdtemp(prefix="dana-evals-"))
    try:
        with standalone_environment(workspace) as run_ref:
            runs = [run_scenario(scenario, run_ref) for scenario in SCENARIOS]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    print(scorecard(runs))
    # Same tripwire as the pytest session: the run must not touch the real
    # freecad_output/, exports/ or agent_workspace/.
    leaked = workspace_changes(before, snapshot_real_workspace())
    if leaked:
        print("\nThe run changed the real workspace:")
        for change in leaked[:20]:
            print(f"  {change}")
    if args.json:
        args.json.write_text(
            json.dumps(
                [
                    {
                        "task": r.scenario.key,
                        "passed": r.passed,
                        "turns": r.llm_calls,
                        "max_tool_schema_tokens": r.max_tool_tokens,
                        "retries": r.retries,
                        "duration_s": round(r.duration_s, 3),
                        "error": r.error,
                        "blocked": r.blocked,
                        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in r.checks],
                    }
                    for r in runs
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0 if all(r.passed for r in runs) and not leaked else 1


if __name__ == "__main__":
    raise SystemExit(main())

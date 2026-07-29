#!/usr/bin/env python3
"""CLI: Agent Integrity & Guardrail Verification.

Exit 0 on pass, non-zero on fail. Prints a checklist of state-contract keys,
graph node registration, and guardrail invariants.

Usage (from repo root)::

    python scripts/verify_agent_integrity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as ``python scripts/verify_agent_integrity.py`` without install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    from dana.audit.agent_integrity import run_agent_integrity_audit

    report = run_agent_integrity_audit()
    print("=== Dana Agent Integrity & Guardrail Verification ===")
    print(report.format_checklist())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

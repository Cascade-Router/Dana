"""Dana (Proposer) / Titan (Critic) asynchronous discovery loop.

Internal package path: ``dana_jason_loop`` (legacy import stability).
Decoupled from agent.py production runtime — simulation and offline discovery only.
"""
from __future__ import annotations

__all__ = [
    "generate_capability_pitches",
    "evaluate_proposals",
    "append_green_flag_to_roadmap",
    "review_watchdog_code",
]

from dana_jason_loop.dana_proposer import generate_capability_pitches
from dana_jason_loop.jason_critic import evaluate_proposals, review_watchdog_code
from dana_jason_loop.ledger import append_green_flag_to_roadmap

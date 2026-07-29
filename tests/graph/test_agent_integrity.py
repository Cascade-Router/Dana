"""Agent criteria: handoffs, state contract, and guardrail integrity."""

from __future__ import annotations

from dana.audit.agent_integrity import audit_agent_integrity


def test_agent_criteria_handoffs_and_guardrails() -> None:
    assert audit_agent_integrity() is True

"""Tests for fire-and-forget research swarm dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dana.tools.swarm_dispatcher import dispatch_research_swarm


def test_dispatch_research_swarm_returns_ok_without_blocking() -> None:
    with patch("dana.tools.swarm_dispatcher.threading.Thread") as mock_thread_cls:
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        result = dispatch_research_swarm("quantum computing")

        mock_thread_cls.assert_called_once()
        kwargs = mock_thread_cls.call_args.kwargs
        assert kwargs.get("daemon") is True
        assert kwargs.get("name") == "DonnaResearchSwarm"
        mock_thread.start.assert_called_once()
        # Fire-and-forget: never join/wait.
        mock_thread.join.assert_not_called()

    assert (
        result
        == "OK: Background research swarm dispatched for topic: quantum computing."
    )
    print(f"[PASS] dispatch_research_swarm: {result}")


def test_dispatch_research_swarm_missing_topic() -> None:
    assert dispatch_research_swarm("").startswith("ERROR:")
    assert dispatch_research_swarm("   ").startswith("ERROR:")
    print("[PASS] missing topic returns ERROR")


def test_react_protocol_quick_vs_deep() -> None:
    from dana.prompts.spatial_synthesis import REACT_PROTOCOL

    assert "If the user asks for a quick fact, use `web_search`" in REACT_PROTOCOL
    assert "dispatch_research_swarm" in REACT_PROTOCOL
    assert "working on it in the background" in REACT_PROTOCOL
    print("[PASS] REACT_PROTOCOL swarm routing rules present")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

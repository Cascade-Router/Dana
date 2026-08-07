"""Spec Compiler Agent tests (hermetic — no Ollama required)."""

from __future__ import annotations

from dana.graph.nodes.spec_compiler import (
    compile_user_spec,
    is_broker_ready_spec,
    is_reject_spec,
)


def test_compile_astar_plain_english() -> None:
    out = compile_user_spec("build me an a-star grid planner")
    assert not is_reject_spec(out)
    assert is_broker_ready_spec(out)
    assert out.lower().startswith("/broker")
    assert "Epic 1:" in out
    assert "visited" in out.lower() or "closed set" in out.lower()
    assert "pytest" in out.lower()


def test_reject_vague_and_third_party() -> None:
    vague = compile_user_spec("make it better")
    assert is_reject_spec(vague)
    banned = compile_user_spec("build a FastAPI + pandas microservice")
    assert is_reject_spec(banned)


def test_passthrough_existing_broker_spec() -> None:
    raw = (
        "/broker Epic 1: Write foo.py with class Foo. "
        "Epic 2: Write tests/test_foo.py with pytest."
    )
    assert compile_user_spec(raw) == raw


def test_spec_approval_payload_and_epics() -> None:
    from dana.graph.nodes.spec_compiler import (
        PENDING_USER_APPROVAL,
        build_spec_approval_payload,
        parse_epics_from_spec,
    )

    raw = (
        "/broker Epic 1: Write foo.py with class Foo. "
        "Epic 2: Write tests/test_foo.py with pytest."
    )
    epics = parse_epics_from_spec(raw)
    assert len(epics) >= 2
    assert epics[0]["id"] == 1
    payload = build_spec_approval_payload(compiled_spec=raw, raw_intent="make foo")
    assert payload["type"] == "spec_approval_request"
    assert payload["status"] == PENDING_USER_APPROVAL
    assert payload["compiled_spec"].startswith("/broker")
    assert len(payload["epics"]) >= 2

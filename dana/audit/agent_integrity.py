"""Agent Integrity & Guardrail Verification (importable audit core).

Checks state-contract keys, ReAct / subgraph node registration, and
fail-closed / shadow / ledger guardrail invariants. Used by
``scripts/verify_agent_integrity.py`` and ``tests/graph/test_agent_integrity.py``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntegrityCheck:
    """Single checklist row."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class IntegrityReport:
    """Full audit result with printable checklist."""

    checks: list[IntegrityCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(IntegrityCheck(name=name, passed=passed, detail=detail))

    def format_checklist(self) -> str:
        lines: list[str] = []
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            suffix = f" - {c.detail}" if c.detail else ""
            lines.append(f"[{mark}] {c.name}{suffix}")
        lines.append("")
        lines.append(
            "RESULT: PASS" if self.ok else "RESULT: FAIL - one or more invariants missing"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_STATE_KEYS: tuple[str, ...] = (
    "raw_state_buffer",
    "subgraph_retry_count",
    "max_subgraph_retries",
    "fatal_block",
    "patch_ledger_path",
    "verification_result",
)

# Nodes that must appear on the production ReAct compile surface.
_REQUIRED_REACT_NODES: tuple[str, ...] = (
    "agent",  # supervisor / MoA agent
    "tools",  # REPL / tool execution
    "critic",
    "fail_closed",
    "hydrate_memory",
    "consolidate_memory",
    "verifier",
)

# Subgraph retry corridor must expose supervisor + escalate.
_REQUIRED_SUBGRAPH_NODES: tuple[str, ...] = (
    "subgraph",
    "escalate_subgraph",
    "supervisor",
)


def _annotations_of(typed: type) -> dict[str, Any]:
    return dict(getattr(typed, "__annotations__", {}) or {})


def _source_of(obj: Any) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return ""


def _callable_named(mod: Any, name: str) -> Callable[..., Any] | None:
    fn = getattr(mod, name, None)
    return fn if callable(fn) else None


def _compiled_node_names(compiled: Any) -> set[str]:
    """Best-effort node name extraction from a LangGraph compiled graph."""
    names: set[str] = set()
    nodes = getattr(compiled, "nodes", None)
    if isinstance(nodes, dict):
        names.update(str(k) for k in nodes.keys())
    try:
        g = compiled.get_graph()
        for n in getattr(g, "nodes", {}) or {}:
            names.add(str(n))
    except Exception:  # noqa: BLE001
        pass
    # LangGraph often includes special __start__/__end__; strip later if needed.
    return names


def _stub_node(state: dict[str, Any]) -> dict[str, Any]:
    return {}


# ---------------------------------------------------------------------------
# Audit sections
# ---------------------------------------------------------------------------


def _check_state_contract(report: IntegrityReport) -> None:
    from dana.schema import ReactGraphState

    ann = _annotations_of(ReactGraphState)
    for key in _REQUIRED_STATE_KEYS:
        present = key in ann
        report.add(
            f"State contract: {key}",
            present,
            "on ReactGraphState" if present else "MISSING from ReactGraphState annotations",
        )


def _check_react_graph_nodes(report: IntegrityReport) -> None:
    from dana.agentic_react_graph import compile_donna_react_graph

    # Prefer live compile with stubs (same topology as production).
    try:
        from langgraph.checkpoint.memory import MemorySaver

        compiled = compile_donna_react_graph(
            _stub_node,
            _stub_node,
            checkpointer=MemorySaver(),
        )
        names = _compiled_node_names(compiled)
        source_ok = True
    except Exception as exc:  # noqa: BLE001 — fall back to source inspection
        names = set()
        source_ok = False
        src = _source_of(compile_donna_react_graph)
        for node in _REQUIRED_REACT_NODES:
            if f'"{node}"' in src or f"'{node}'" in src:
                names.add(node)
        report.add(
            "Graph compile (donna react)",
            False,
            f"compile failed ({exc}); used source fallback for node names",
        )

    if source_ok:
        report.add(
            "Graph compile (donna react)",
            True,
            f"compiled; nodes={sorted(n for n in names if not n.startswith('__'))}",
        )

    for node in _REQUIRED_REACT_NODES:
        present = node in names
        report.add(
            f"Graph node: {node}",
            present,
            "registered" if present else "NOT registered on compile_donna_react_graph",
        )

    # Injectable parameter surface (memory / critic / fail_closed).
    sig = inspect.signature(compile_donna_react_graph)
    for param in (
        "critic_node_fn",
        "fail_closed_node_fn",
        "hydrate_memory_node_fn",
        "consolidate_memory_node_fn",
        "verifier_node_fn",
    ):
        present = param in sig.parameters
        report.add(
            f"Injectable: {param}",
            present,
            "in compile_donna_react_graph" if present else "MISSING injectable",
        )


def _check_subgraph_and_supervisor(report: IntegrityReport) -> None:
    from dana.graph.subgraph_router import (
        ESCALATE_SUBGRAPH,
        SUPERVISOR_NODE,
        compile_subgraph_retry_graph,
        escalate_subgraph,
    )

    report.add(
        "Subgraph constant: escalate_subgraph",
        ESCALATE_SUBGRAPH == "escalate_subgraph",
        repr(ESCALATE_SUBGRAPH),
    )
    report.add(
        "Subgraph constant: supervisor",
        SUPERVISOR_NODE == "supervisor",
        repr(SUPERVISOR_NODE),
    )
    report.add(
        "Subgraph callable: escalate_subgraph",
        callable(escalate_subgraph),
        "dana.graph.subgraph_router",
    )

    try:
        from langgraph.checkpoint.memory import MemorySaver

        compiled = compile_subgraph_retry_graph(
            _stub_node,
            _stub_node,
            checkpointer=MemorySaver(),
        )
        names = _compiled_node_names(compiled)
        report.add(
            "Graph compile (subgraph retry)",
            True,
            f"compiled; nodes={sorted(n for n in names if not n.startswith('__'))}",
        )
    except Exception as exc:  # noqa: BLE001
        names = set()
        src = _source_of(compile_subgraph_retry_graph)
        for node in _REQUIRED_SUBGRAPH_NODES:
            if node in src:
                names.add(node)
        report.add(
            "Graph compile (subgraph retry)",
            False,
            f"compile failed ({exc}); used source fallback",
        )

    for node in _REQUIRED_SUBGRAPH_NODES:
        present = node in names
        report.add(
            f"Subgraph node: {node}",
            present,
            "registered" if present else "NOT registered on compile_subgraph_retry_graph",
        )


def _check_vision_surface(report: IntegrityReport) -> None:
    """Vision is an injectable graph helper (optional wire), not always on main ReAct."""
    from dana.graph.nodes import vision as vision_mod

    fn = _callable_named(vision_mod, "vision_ground_node")
    report.add(
        "Vision injectable: vision_ground_node",
        fn is not None,
        "dana.graph.nodes.vision" if fn else "MISSING",
    )
    locate = _callable_named(vision_mod, "locate_ui_element")
    report.add(
        "Vision helper: locate_ui_element",
        locate is not None,
        "dana.graph.nodes.vision" if locate else "MISSING",
    )


def _check_memory_surface(report: IntegrityReport) -> None:
    from dana.graph.nodes import memory as memory_mod

    for name in ("hydrate_memory_node", "consolidate_memory_node"):
        fn = _callable_named(memory_mod, name)
        report.add(
            f"Memory node: {name}",
            fn is not None,
            "dana.graph.nodes.memory" if fn else "MISSING",
        )


def _check_fatal_exceptions(report: IntegrityReport) -> None:
    from dana.graph.nodes import critic as critic_mod

    fatal = getattr(critic_mod, "FATAL_EXCEPTIONS", None)
    ok = isinstance(fatal, tuple) and len(fatal) > 0
    detail = ""
    if ok:
        detail = ", ".join(getattr(t, "__name__", str(t)) for t in fatal)
    else:
        detail = "FATAL_EXCEPTIONS missing or empty"
    report.add("Guardrail: FATAL_EXCEPTIONS", ok, detail)

    clf = _callable_named(critic_mod, "is_fatal_execution_error")
    report.add(
        "Guardrail: is_fatal_execution_error",
        clf is not None,
        "classifier present" if clf else "MISSING",
    )


def _check_shadow_workspace(report: IntegrityReport) -> None:
    from dana.exec.shadow_workspace import ShadowWorkspace

    commit = getattr(ShadowWorkspace, "commit", None)
    rollback = getattr(ShadowWorkspace, "rollback", None)
    report.add(
        "Guardrail: ShadowWorkspace.commit",
        callable(commit),
        "hook present" if callable(commit) else "MISSING",
    )
    report.add(
        "Guardrail: ShadowWorkspace.rollback",
        callable(rollback),
        "hook present" if callable(rollback) else "MISSING",
    )


def _calls_ledger_writer(src: str) -> bool:
    return (
        "append_pending_ticket" in src
        or "write_escalation_ticket" in src
    )


def _check_ledger_wiring(report: IntegrityReport) -> None:
    from dana.graph.nodes.critic import fail_closed_node
    from dana.graph.subgraph_router import escalate_subgraph
    from dana_security.ledger_writer import (
        append_pending_ticket,
        write_escalation_ticket,
    )

    report.add(
        "Ledger writer: append_pending_ticket",
        callable(append_pending_ticket),
        "dana_security.ledger_writer",
    )
    report.add(
        "Ledger writer: write_escalation_ticket",
        callable(write_escalation_ticket),
        "dana_security.ledger_writer",
    )

    wes_src = _source_of(write_escalation_ticket)
    wes_calls = "append_pending_ticket" in wes_src
    report.add(
        "Ledger chain: write_escalation_ticket -> append_pending_ticket",
        wes_calls,
        "source contains append_pending_ticket"
        if wes_calls
        else "write_escalation_ticket does not call append_pending_ticket",
    )

    fc_src = _source_of(fail_closed_node)
    # fail_closed may flush via helper; include helper source.
    from dana.graph.nodes import critic as critic_mod

    helper = getattr(critic_mod, "_flush_escalation_ledger", None)
    helper_src = _source_of(helper) if helper is not None else ""
    fc_ok = _calls_ledger_writer(fc_src) or _calls_ledger_writer(helper_src)
    report.add(
        "Ledger wire: fail_closed_node -> ledger writer",
        fc_ok,
        "calls write_escalation_ticket/append_pending_ticket"
        if fc_ok
        else "fail_closed_node does not invoke ledger writer",
    )

    esc_src = _source_of(escalate_subgraph)
    esc_ok = _calls_ledger_writer(esc_src)
    report.add(
        "Ledger wire: escalate_subgraph -> ledger writer",
        esc_ok,
        "calls write_escalation_ticket/append_pending_ticket"
        if esc_ok
        else "escalate_subgraph does not invoke ledger writer",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_agent_integrity_audit() -> IntegrityReport:
    """Run all integrity checks; return a detailed report."""
    report = IntegrityReport()
    _check_state_contract(report)
    _check_react_graph_nodes(report)
    _check_subgraph_and_supervisor(report)
    _check_vision_surface(report)
    _check_memory_surface(report)
    _check_fatal_exceptions(report)
    _check_shadow_workspace(report)
    _check_ledger_wiring(report)
    return report


def audit_agent_integrity() -> bool:
    """Return True iff every agent integrity / guardrail invariant holds."""
    return run_agent_integrity_audit().ok

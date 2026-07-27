"""Leaf shared types for Donna — no imports from core_agent / agentic / tools workers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator

TraceEventType = Literal[
    "node_enter",
    "node_exit",
    "tool_execution",
    "state_update",
    "mode",
    "status",
]

# Known swarm / bureaucratic agent ids (Module 4 Handoff targets).
HandoffTarget = Literal[
    "MoA_Reasoner",
    "Vision_Agent",
    "Chat_Node",
    "ReAct_Agent",
    "Mailroom",
]


@dataclass(frozen=True)
class TraceEvent:
    """Normalized bus payload for Live Trace rendering."""

    event_type: TraceEventType
    node: str = ""
    message: str = ""
    mode: str = ""
    tool: str = ""
    latency_ms: float | None = None
    payload: str = ""
    state_keys: tuple[str, ...] = ()
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state_keys"] = list(self.state_keys)
        return data


# Back-compat aliases.
NodeEnterEvent = TraceEvent
NodeExitEvent = TraceEvent
ToolExecutionEvent = TraceEvent
StateUpdateEvent = TraceEvent


@dataclass
class AgenticResult:
    final_text: str
    iterations: int
    tool_trace: list[dict[str, Any]]
    reply_lang: str
    reflection: dict[str, Any] | None = None
    reflection_ms: float = 0.0
    had_errors: bool = False
    tts_streamed: bool = False


class Handoff(BaseModel):
    """Deterministic swarm handoff payload (Module 4).

    Agents request a capability switch by emitting this schema; the Python graph
    executes the state transition — no supervisor LLM.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    target_agent: str = Field(
        ...,
        min_length=1,
        description="Target agent id, e.g. MoA_Reasoner, Vision_Agent, Chat_Node",
    )
    reason: str = Field(..., min_length=1, description="Why the handoff is required")
    intent_context: str = Field(
        ...,
        min_length=1,
        description="Compressed intent / context for the receiving agent",
    )

    @field_validator("target_agent")
    @classmethod
    def _normalize_target(cls, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            raise ValueError("target_agent must be non-empty")
        known = {
            "moa_reasoner": "MoA_Reasoner",
            "vision_agent": "Vision_Agent",
            "chat_node": "Chat_Node",
            "react_agent": "ReAct_Agent",
            "mailroom": "Mailroom",
        }
        return known.get(raw.lower().replace(" ", "_"), raw)


class ReactGraphState(TypedDict, total=False):
    """Minimal bureaucratic LangGraph state (Stage 3 Module 1).

    Durable chat history and chain-of-thought live on the Blackboard
    (``donna.memory``), keyed by ``session_id``. Only these control-plane
    fields are required across nodes:

    - ``session_id``
    - ``current_agent``
    - ``active_intent``

    Ephemeral turn fields (``messages``, iterations, …) exist solely to
    drive ``bind_tools`` within a single invoke — they are not the durable
    memory store.
    """

    # --- Bureaucratic control plane (durable pointer) ---
    session_id: str
    current_agent: str
    active_intent: str
    # --- Ephemeral turn scratch (not long-term history) ---
    messages: Annotated[list, add_messages]
    iterations: int
    last_obs: str
    final_raw: str
    halt: bool
    # Deduped broker merge (mode + forced + explicit ids); agent node binds exactly this.
    always_include: list[str]
    # Module 4: pending handoff after deterministic parse (optional).
    pending_handoff: dict[str, str]
    # Stage 8.9 — Jason supervisor critique before HITL ticket approval.
    jason_critique: str
    # Stage 8.9.3 — consecutive HITL denials for GitHub escalation.
    consecutive_denials: int
    # Stage 8.9.6 — Pydantic-validated ticket payload (HITL only after True).
    drafted_ticket: dict
    ticket_validated: bool
    ticket_validation_retries: int

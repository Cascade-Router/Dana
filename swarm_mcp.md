# Decentralized Swarm Architecture: ROS2/Nav2 × LangGraph × MCP

## Abstract

This brief proposes a **decentralized edge-agent swarm** where ROS2/Nav2 owns
real-time mobility, LangGraph owns deliberative multi-agent planning, and
**Model Context Protocol (MCP) servers** expose typed tools that any swarm peer
may call dynamically. The design favors fail-closed contracts, stdlib-first
Python adapters at the cognition boundary, and explicit capability discovery
instead of hard-wired RPC meshes.

## 1. Problem framing

Classical multi-robot stacks couple navigation tightly to application logic.
Scaling to heterogeneous edge nodes (rovers, arms, fixed cameras) requires:

1. **Local autonomy** for latency-sensitive control (Nav2 costmaps, controllers).
2. **Shared cognition** for long-horizon goals (LangGraph supervisors / workers).
3. **Dynamic tool calling** so agents can discover actuators, sensors, and
   planners at runtime without redeploying the whole fleet.

MCP addresses (3) by publishing tool schemas over a negotiated transport;
LangGraph addresses (2) via durable graph state; ROS2 addresses (1) via DDS
topics / actions / services.

## 2. Layered architecture

```text
┌────────────────────────────────────────────────────────────┐
│ Edge Agent (per robot / compute node)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Nav2 stack   │  │ LangGraph    │  │ MCP client/server│ │
│  │ planner+ctrl │◀▶│ supervisor   │◀▶│ tool registry    │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
│         │ ROS2 DDS        │ IPC/gRPC          │ MCP JSON  │
└─────────┼─────────────────┼───────────────────┼───────────┘
          ▼                 ▼                   ▼
     Fleet topics      Shared blackboard    Peer MCP endpoints
```

### 2.1 ROS2 / Nav2 (mobility fabric)

- **Nav2** BT navigators, controllers, and behavior trees remain the hard
  real-time path. Cognition never blocks the controller callback group.
- Swarm coordination uses ROS2 **actions** for navigate-to-pose and **services**
  for map merges / localization handoff.
- Costmap layers may subscribe to peer occupancy topics for cooperative mapping.

### 2.2 LangGraph (deliberation fabric)

- Each edge agent runs a slim LangGraph **supervisor ↔ worker** loop analogous
  to Dānā’s Meta-Broker: plan epics, dispatch tool-bearing workers, validate.
- Graph state stores goal IDs, active Nav2 action handles, and MCP session IDs.
- Inter-epic **artifact contracts** (manifest-style export lists) prevent
  hallucinated module/tool names across swarm peers.

### 2.3 MCP servers (tool fabric)

- Every edge node hosts an MCP server advertising tools such as
  `navigate_to`, `get_costmap_slice`, `inspect_camera`, `invoke_peer_tool`.
- Remote agents bind as MCP clients; discovery is capability-based
  (`tools/list`) rather than static IP tables.
- Tool results stream as structured JSON; binary frames (images, point clouds)
  use content references (ROS bag URI / shared-memory handle) to avoid
  stuffing DDS with LLM context.

## 3. Dynamic cross-swarm tool calling

1. **Advertise** — Node A publishes MCP tool descriptors + ROS2 namespace.
2. **Authorize** — Capability tokens / mTLS gate destructive tools (motor, arm).
3. **Plan** — LangGraph supervisor selects tools by schema, not by string fuzzy match.
4. **Execute** — MCP call may fan out to Node B; Node B may further call Nav2
   actions locally.
5. **Validate** — Runtime harness (compile / pytest / Nav2 goal status) closes
   the loop before advancing the next epic.

## 4. Failure modes & safety

| Failure | Mitigation |
|---------|------------|
| MCP peer timeout | LangGraph retry budget; fall back to local-only tools |
| Nav2 preempted | Supervisor marks epic `repairing`; replan with updated costmap |
| Schema drift | Manifest / MCP schema hash mismatch → fail closed |
| RAM pressure on edge LLM | Zero keep-alive unload + sequential TTS/telemetry queues |

## 5. Reference integration with Dānā-style control planes

- Treat each robot’s cognition process as an **isolated Meta-Broker worker**
  (multiprocessing / container) with Queue IPC for telemetry.
- Prefer **Python stdlib** adapters at the MCP boundary unless ROS client libs
  are explicitly required by the deployment prompt.
- Surface swarm status into headless Gradio / Tk Live Trace via the same
  telemetry event shape (`phase`, `status`, `message`, `terminal`).

## 6. Conclusion

ROS2/Nav2 supplies mobility, LangGraph supplies multi-epic deliberation, and
MCP supplies a portable tool bus. Together they enable decentralized swarms
where edge agents dynamically call each other’s tools under fail-closed
contracts—without collapsing real-time control into the LLM loop.

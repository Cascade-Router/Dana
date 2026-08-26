import { useEffect, useMemo, useState } from "react";
import { Background, Controls, ReactFlow, ReactFlowProvider, useReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { ServerEvent } from "../lib/useChatSocket";
import "./DAGMonitor.css";

type DagNodeData = {
  label: string;
  nodeType: "agent" | "tool" | "vision";
  status: "running" | "success" | "error";
  inputs: Record<string, unknown>;
  output: Record<string, unknown> | null;
  durationMs: number | null;
};

const NODE_COLORS: Record<string, string> = {
  agent: "#2563eb",
  vision: "#d97706",
  "tool-success": "#16a34a",
  "tool-running": "#6b7280",
  error: "#dc2626",
};

function nodeColor(data: DagNodeData): string {
  if (data.status === "error") return NODE_COLORS.error;
  if (data.nodeType === "vision") return NODE_COLORS.vision;
  if (data.nodeType === "tool") return data.status === "success" ? NODE_COLORS["tool-success"] : NODE_COLORS["tool-running"];
  return NODE_COLORS.agent;
}

function buildGraph(log: ServerEvent[]): { nodes: Node<DagNodeData>[]; edges: Edge[] } {
  const nodeMap = new Map<string, Node<DagNodeData>>();
  const edges: Edge[] = [];
  let turn = -1;
  let lastKeyInTurn: string | null = null;

  for (const event of log) {
    if (event.type === "dag_node_start") {
      // dana/api/server.py's _run_react_loop sends this turn's parse node
      // as `f"parse-{loop_count}"` (e.g. "parse-0", "parse-1", ...) — never
      // the bare string "parse" — so an exact match here never fired,
      // `turn` stayed stuck at -1 forever, and every node in every turn
      // landed on the exact same (column, y) position, fully overlapping.
      const isParseNode = event.node_id.startsWith("parse-");
      if (isParseNode) {
        turn += 1;
        lastKeyInTurn = null;
      }
      const key = `${turn}:${event.node_id}`;
      const column = isParseNode ? 0 : 1;
      nodeMap.set(key, {
        id: key,
        position: { x: column * 260, y: turn * 130 },
        data: {
          label: event.label,
          nodeType: event.node_type,
          status: "running",
          inputs: event.inputs,
          output: null,
          durationMs: null,
        },
        style: { background: NODE_COLORS["tool-running"], color: "white", borderRadius: 8, padding: 8, fontSize: 12 },
      });
      if (lastKeyInTurn) {
        edges.push({ id: `${lastKeyInTurn}->${key}`, source: lastKeyInTurn, target: key, animated: true });
      }
      lastKeyInTurn = key;
    } else if (event.type === "dag_node_complete") {
      const key = `${turn}:${event.node_id}`;
      const existing = nodeMap.get(key);
      if (!existing) continue;
      const data: DagNodeData = {
        ...existing.data,
        status: event.status,
        output: event.output,
        durationMs: event.duration_ms,
      };
      nodeMap.set(key, {
        ...existing,
        data,
        style: { background: nodeColor(data), color: "white", borderRadius: 8, padding: 8, fontSize: 12 },
      });
    }
  }

  return { nodes: Array.from(nodeMap.values()), edges };
}

type GraphCanvasProps = {
  nodes: Node<DagNodeData>[];
  edges: Edge[];
  onNodeClick: (node: Node<DagNodeData>) => void;
};

// Split out from DAGMonitor so useReactFlow() has a <ReactFlowProvider>
// ancestor (the hook throws outside one) without wrapping the whole
// collapsed/expanded shell — the provider only needs to surround <ReactFlow>
// itself.
function GraphCanvas({ nodes, edges, onNodeClick }: GraphCanvasProps) {
  const { fitView } = useReactFlow();

  // Long ReAct chains (10+ steps) push new nodes past the initial fitView,
  // off-screen below the fold — re-fit every time the node count grows so
  // the newest node (appended by either the WS log or Gradio's dagEvents,
  // see buildGraph above) is always in view.
  useEffect(() => {
    if (nodes.length === 0) return;
    fitView({ duration: 800, padding: 0.2 });
  }, [nodes.length, fitView]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodeClick={(_e, node) => onNodeClick(node as Node<DagNodeData>)}
      fitView
      proOptions={{ hideAttribution: true }}
    >
      <Background />
      <Controls showInteractive={false} />
      <button
        type="button"
        className="dag-monitor__fit-btn"
        title="Re-center graph"
        onClick={() => fitView({ duration: 800, padding: 0.2 })}
      >
        ⤢ Fit View
      </button>
    </ReactFlow>
  );
}

type Props = {
  log: ServerEvent[];
};

export function DAGMonitor({ log }: Props) {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<Node<DagNodeData> | null>(null);
  const { nodes, edges } = useMemo(() => buildGraph(log), [log]);

  return (
    <div className={`dag-monitor ${open ? "dag-monitor--open" : ""}`}>
      <button className="dag-monitor__toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "▾" : "▸"} Execution Graph ({nodes.length})
      </button>
      {open && (
        <div className="dag-monitor__body">
          <div className="dag-monitor__canvas">
            <ReactFlowProvider>
              <GraphCanvas nodes={nodes} edges={edges} onNodeClick={setSelected} />
            </ReactFlowProvider>
          </div>
          {selected && (
            <div className="dag-monitor__inspector">
              <div className="dag-monitor__inspector-header">
                <strong>{selected.data.label}</strong>
                <button onClick={() => setSelected(null)}>×</button>
              </div>
              <div className="dag-monitor__inspector-row">
                <span>type</span>
                <span>{selected.data.nodeType}</span>
              </div>
              <div className="dag-monitor__inspector-row">
                <span>status</span>
                <span>{selected.data.status}</span>
              </div>
              {selected.data.durationMs !== null && (
                <div className="dag-monitor__inspector-row">
                  <span>duration</span>
                  <span>{selected.data.durationMs}ms</span>
                </div>
              )}
              <div className="dag-monitor__inspector-label">inputs</div>
              <pre className="dag-monitor__inspector-json">{JSON.stringify(selected.data.inputs, null, 2)}</pre>
              {selected.data.output && (
                <>
                  <div className="dag-monitor__inspector-label">output</div>
                  <pre className="dag-monitor__inspector-json">{JSON.stringify(selected.data.output, null, 2)}</pre>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

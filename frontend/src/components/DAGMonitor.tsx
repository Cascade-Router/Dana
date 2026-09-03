import { useEffect, useMemo, useState } from "react";
import { Background, Controls, ReactFlow, ReactFlowProvider, useReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { TopologyGraph, TopologyNode } from "../lib/useChatSocket";
import "./DAGMonitor.css";

type LineageNodeData = {
  label: string;
  nodeType: string;
};

// One color per topology_dag node "type" (dana.core.react_dispatch's
// _record_topology_node currently only ever stamps "geometry", but this
// stays a lookup — not a single hardcoded color — so a future distinct
// type, e.g. differentiating a boolean result from a primitive, can be
// added on the backend and picked up here with no frontend code change).
// Unlike the old dag_node_start-driven graph, there is no "running"/"error"
// status to color for: a topology_dag node only ever exists because the
// dispatch that produced it already succeeded (see dispatch_tool_call —
// _record_topology_node is only ever called from the `ok` branch).
const NODE_COLORS: Record<string, string> = {
  geometry: "#16a34a",
};

function nodeColor(nodeType: string): string {
  return NODE_COLORS[nodeType] ?? "#2563eb";
}

function nodeStyle(nodeType: string): Node<LineageNodeData>["style"] {
  return { background: nodeColor(nodeType), color: "white", borderRadius: 8, padding: 8, fontSize: 12 };
}

// Every node's depth = the length of the longest lineage chain feeding into
// it (a primitive with no consumed inputs is depth 0; a boolean/edge-op/
// feature-on-face result is one more than the deepest input it consumed) —
// this is what turns the flat {nodes, edges} the backend sends into a
// left-to-right feature-tree layout, primitives on the left and each
// boolean/fillet/chamfer step one column to the right of what it consumed.
// A plain relaxation loop (not a real topological sort) is enough here:
// topology_dag is always a DAG in practice (an object is consumed once,
// into whatever the next operation produces) and stays small (a CAD
// session's feature tree, not a general graph), so iterating up to
// `nodeCount` times to let depths propagate is cheap and simple.
function computeDepths(graph: TopologyGraph): Map<string, number> {
  const ids = Object.keys(graph.nodes);
  const depth = new Map<string, number>(ids.map((id) => [id, 0]));
  for (let pass = 0; pass < ids.length; pass++) {
    let changed = false;
    for (const edge of graph.edges) {
      if (!(edge.source in graph.nodes) || !(edge.target in graph.nodes)) continue;
      const candidate = (depth.get(edge.source) ?? 0) + 1;
      if (candidate > (depth.get(edge.target) ?? 0)) {
        depth.set(edge.target, candidate);
        changed = true;
      }
    }
    if (!changed) break;
  }
  return depth;
}

function buildLineageGraph(graph: TopologyGraph): { nodes: Node<LineageNodeData>[]; edges: Edge[] } {
  const depth = computeDepths(graph);
  const countByDepth = new Map<number, number>();

  const nodes: Node<LineageNodeData>[] = Object.values(graph.nodes).map((tNode: TopologyNode) => {
    const d = depth.get(tNode.id) ?? 0;
    const row = countByDepth.get(d) ?? 0;
    countByDepth.set(d, row + 1);
    return {
      id: tNode.id,
      position: { x: d * 260, y: row * 130 },
      data: { label: tNode.label, nodeType: tNode.type },
      style: nodeStyle(tNode.type),
    };
  });

  // dispatch_tool_call's _record_topology_node only ever APPENDS an edge —
  // nothing evicts or dedupes the backend's own list — so a retried tool
  // call that resolves to the identical (source, target) pair a second
  // time (e.g. a repeated dispatch after a transient error) sends the same
  // logical edge twice. React Flow keys each <Edge> by `id`, and this
  // component derives that id from `${source}->${target}` — two edges with
  // the same pair therefore collide on the same id, surfacing as React's
  // "Encountered two children with the same key" warning. Deduping by that
  // same pair key before mapping to Edge objects is what actually fixes
  // it (a plain .map with no filtering would still emit the duplicate).
  const uniqueEdges = Array.from(new Map(graph.edges.map((e) => [`${e.source}->${e.target}`, e])).values());

  const edges: Edge[] = uniqueEdges
    .filter((e) => e.source in graph.nodes && e.target in graph.nodes)
    .map((e) => ({ id: `${e.source}->${e.target}`, source: e.source, target: e.target, animated: true }));

  return { nodes, edges };
}

type GraphCanvasProps = {
  nodes: Node<LineageNodeData>[];
  edges: Edge[];
  onNodeClick: (node: Node<LineageNodeData>) => void;
};

// Split out from DAGMonitor so useReactFlow() has a <ReactFlowProvider>
// ancestor (the hook throws outside one) without wrapping the whole
// collapsed/expanded shell — the provider only needs to surround <ReactFlow>
// itself.
function GraphCanvas({ nodes, edges, onNodeClick }: GraphCanvasProps) {
  const { fitView } = useReactFlow();

  // A long boolean/fillet chain pushes new nodes past the initial fitView,
  // off-screen to the right — re-fit every time the node count grows so the
  // newest lineage step is always in view.
  useEffect(() => {
    if (nodes.length === 0) return;
    fitView({ duration: 800, padding: 0.2 });
  }, [nodes.length, fitView]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodeClick={(_e, node) => onNodeClick(node as Node<LineageNodeData>)}
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
  /** The session's Topological Lineage Graph (dana.core.react_dispatch's
   * deterministic CAD feature tree) — see dana/api/server.py's
   * "topology_update"/"ready" events. Replaces the old `log: ServerEvent[]`
   * prop: this graph only ever grows by actual successful geometry
   * mutations, never the ReAct loop's own retries/errors/branches, so it
   * stays a clean feature tree instead of breaking on every non-CAD tool
   * call or failed attempt. */
  graph: TopologyGraph;
};

// InspectorDock's "Topology" tab — same lineage graph as the old
// standalone/collapsible DAGMonitor, minus its own open/close toggle (the
// dock's tab bar now owns that).
export function TopologyTab({ graph }: Props) {
  const [selected, setSelected] = useState<Node<LineageNodeData> | null>(null);
  const { nodes, edges } = useMemo(() => buildLineageGraph(graph), [graph]);

  return (
    <div className="topology-tab">
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
            <span>id</span>
            <span>{selected.id}</span>
          </div>
        </div>
      )}
    </div>
  );
}

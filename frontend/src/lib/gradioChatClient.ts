import { Client } from "@gradio/client";
import { installConsoleCapture } from "./consoleCapture";
import type { ServerEvent } from "./useChatSocket";

// Idempotent — see consoleCapture.ts's own `installed` guard. Called here
// too (not just from TerminalDrawer.tsx) so this module's own console.log
// calls below (and every predict() call this module makes) are guaranteed
// captured into the Debug Terminal panel regardless of which module a
// given bundler/import order happens to evaluate first.
installConsoleCapture();
console.log("[DebugTerminal] Debug console initialized successfully");

// Thin wrapper around the Hugging Face Space's pure-Gradio backend (see
// app.py — a gr.ChatInterface, api_name="chat", registered because
// gr.mount_gradio_app + a custom FastAPI/React app was found to conflict
// with the Space's ZeroGPU ASGI middleware there).
//
// The PUBLIC api ONLY takes `message` — gr.State (chat history, the Dana
// session dict) is deliberately NOT part of a Gradio app's external API
// surface; it's tracked server-side, keyed by the connected Client's own
// `session_hash`. That's why this keeps exactly ONE Client instance alive
// for the tab's lifetime (module-level singleton below) rather than
// reconnecting per message — reconnecting would start a brand-new Dana
// session on every single message. Verified directly against a running
// instance: two predict() calls on the same Client reuse one session;
// a fresh Client gets a fresh one.
let clientPromise: Promise<Client> | null = null;

// Exported so useGradioChat can connect eagerly on mount (for an accurate
// "connecting"/"open"/"closed" status) while sendGradioChatMessage below
// reuses the exact same cached instance/session_hash for every message.
export function connectGradioClient(spaceUrl: string): Promise<Client> {
  if (!clientPromise) {
    clientPromise = Client.connect(spaceUrl).catch((err) => {
      clientPromise = null; // let the next call retry instead of caching a permanent failure
      throw err;
    });
  }
  return clientPromise;
}

// Everything the WebSocket path streams as separate events — tool_call,
// tool_start/tool_complete, camera_animate, voice state, HITL approval
// prompts — has no equivalent here: app.py's _GradioSocket auto-resolves
// HITL/visual-capture suspensions server-side and only ever surfaces the
// turn's final reply plus (since app.py moved to gr.Blocks with a
// text_out/file_out/graph_out set) whatever mesh a CAD tool call produced
// this turn AND this turn's dag_node_start/dag_node_complete events
// (dagEvents — app.py's _GradioSocket DOES capture these now, unlike the
// rest of the WS-only telemetry list above). Still a smaller surface than
// the WS protocol overall — no live tool-activity feed, no per-event
// streaming (the whole turn's dagEvents arrive at once, after the turn
// finishes, rather than one at a time as each tool actually runs).
export type GradioChatReply = {
  text: string;
  /** Absolute, fetchable URL to the turn's generated .stl, or null if this
   * turn didn't produce one. Gradio's own file-serving endpoint — verified
   * directly (curled it) to actually serve real STL bytes locally. A local
   * dev instance shows no Access-Control-Allow-Origin header on a cross-
   * origin request, but that's specific to `CustomCORSMiddleware` treating
   * "localhost" as the app's own host (gradio/route_utils.py's
   * `is_valid_origin`: it only requires the ORIGIN to also be a localhost
   * alias when the HOST is one) — once actually deployed, the Space's host
   * is a real hostname, not a localhost alias, so that same middleware
   * allows any origin. Confirmed by reading Gradio 6.25's source; not
   * re-verified against the live Space over the network from here. */
  meshUrl: string | null;
  /** This turn's dag_node_start/dag_node_complete events, in the same
   * shape DAGMonitor.tsx's buildGraph() already consumes over WS — see
   * app.py's _GradioSocket.dag_events / graph_out. Empty when the turn
   * never called a tool (a plain-text reply). */
  dagEvents: ServerEvent[];
};

type GradioFileData = { url?: string; path?: string; orig_name?: string; size?: number };

// A bare, absolute filesystem path (e.g. "/tmp/dana_mock_Cylinder_t9vir1vy.stl"
// — the mock CAD engine's own tempfile path INSIDE the HF Space's Docker
// container) is meaningless to a browser on a different origin (Vercel):
// there's no host to fetch it from. Turn it into the real, cross-origin-
// fetchable URL Gradio itself would serve that same path at:
// `${root}${api_prefix}/file=${path}` — using the CONNECTED client's own
// resolved `config.root`/`api_prefix` (not a hardcoded ".hf.space"/"/file="
// guess) so this stays correct whatever Space URL was actually passed to
// connectGradioClient AND whatever route prefix that Gradio version uses
// (older Gradio serves this at bare "/file=", 5.x+ moved it under
// "/gradio_api/file=" — see @gradio/client's own upload() helper, which
// builds files' URLs the exact same way after an upload).
function resolveAbsoluteFileUrl(client: Client, rawPath: string): string {
  const root = (client.config?.root ?? "").replace(/\/+$/, "");
  const resolved = `${root}${client.api_prefix}/file=${rawPath}`;
  console.log("[GradioClient] Resolved absolute mesh URL:", resolved);
  return resolved;
}

function toAbsoluteUrl(client: Client, value: string): string {
  if (/^https?:\/\//i.test(value)) return value;
  return resolveAbsoluteFileUrl(client, value);
}

// `data[1]` (file_out, bound to a gr.Model3D output — see app.py's
// _respond) is a Gradio FileData object `{url, path, orig_name, ...}` in
// every version actually exercised here, but the exact client-side shape
// depends on the @gradio/client version resolving/normalizing it — some
// versions have been observed handing back the bare string path a
// component was given instead of the wrapped object (confirmed live: the
// raw in-container path, not a fetchable URL, reaching data[1] as-is).
// Handling both shapes, and resolving a bare path in either shape to an
// absolute URL, means neither a client-library version bump nor this quirk
// can silently turn a real mesh into a dropped/unfetchable `null`.
function parseMeshPayload(raw: unknown, client: Client): string | null {
  console.log("[GradioClient] Mesh payload received in data[1]:", raw);
  if (typeof raw === "string") return raw ? toAbsoluteUrl(client, raw) : null;
  if (raw && typeof raw === "object") {
    const file = raw as GradioFileData;
    if (file.url) return toAbsoluteUrl(client, file.url);
    if (file.path) return toAbsoluteUrl(client, file.path);
    return null;
  }
  return null;
}

// data[5] (graph_out, a gr.JSON output — see app.py's _respond docstring
// for the full output ordering) is a plain JSON array round-tripped
// through _GradioSocket.dag_events, so its entries already carry the exact
// `{type: "dag_node_start"|"dag_node_complete", ...}` shape ServerEvent
// expects in EVERY case actually observed so far. Still handles a
// stringified JSON array defensively (JSON.parse, guarded) — gr.JSON's
// exact wire representation has been known to vary by Gradio version, and
// a bare `!Array.isArray(raw)` bail-out would otherwise silently turn a
// real (but string-encoded) payload into a permanently empty graph with
// no error anywhere.
function parseDagEvents(raw: unknown): ServerEvent[] {
  let value: unknown = raw;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch (err) {
      console.warn("[GradioClient] data[5] was a string but not valid JSON — dropping:", err);
      return [];
    }
  }
  if (!Array.isArray(value)) return [];
  return value.filter(
    (e): e is ServerEvent =>
      !!e && typeof e === "object" && (e.type === "dag_node_start" || e.type === "dag_node_complete")
  );
}

export async function sendGradioChatMessage(spaceUrl: string, message: string): Promise<GradioChatReply> {
  const client = await connectGradioClient(spaceUrl);
  const result = await client.predict("/chat", { message });
  // @gradio/client types `result.data` loosely (not as a real array), so
  // every access below goes through this one cast rather than sprinkling
  // `as unknown[]`/`as unknown` at each call site.
  const dataArr = (result.data as unknown[] | undefined) ?? [];
  console.log(
    "[GradioClient] Raw data array length and types:",
    dataArr.length,
    dataArr.map((d: unknown) => typeof d)
  );
  // app.py's Python-side _outputs list is
  // [text_out, file_out, chatbot, mesh_preview, session_state, graph_out]
  // — 6 entries — but session_state (a gr.State) never reaches the client
  // at all: gr.State.skip_api is hardcoded True in Gradio's own source
  // (gradio/components/state.py), so the client-visible `data` array is
  // only 5 elements, with graph_out shifted down to index 4, not 5.
  // Verified directly against a live capture (dataArr.length === 5). Read
  // graph_out as the LAST element rather than a hardcoded index so this
  // can't silently break again the same way if any other State component
  // is ever added/removed from _outputs.
  const text = dataArr[0] as string;
  const file = dataArr[1];
  const graph = dataArr[dataArr.length - 1];
  console.log("[GradioClient] Raw graph_out (last element):", graph);
  const dagEvents = parseDagEvents(graph);
  console.log("[GradioClient] Parsed dagEvents:", dagEvents.length, dagEvents);
  return { text, meshUrl: parseMeshPayload(file, client), dagEvents };
}

export type GradioArtifact = {
  filename: string;
  format: string;
  url: string;
  size_bytes: number;
};

// The Gradio-mode counterpart to dana.api.cad's REST GET /api/cad/artifacts
// (apiBase.ts's IS_GRADIO_MODE blocks all /api/* calls here, since app.py
// never mounts that FastAPI router — see this file's own header comment).
// Bound to app.py's hidden "artifacts" api endpoint, which FileData-ifies
// every path dana.api.artifacts_registry has recorded so far into a real
// fetchable `.url` — no separate "resolve this path to a URL" round trip
// needed, same as sendGradioChatMessage's mesh file above.
export async function fetchGradioArtifacts(spaceUrl: string): Promise<GradioArtifact[]> {
  const client = await connectGradioClient(spaceUrl);
  const result = await client.predict("/artifacts", {});
  const [file0] = result.data as [GradioFileData[] | null];
  console.log("[GradioClient] Artifacts payload received:", file0);
  const files = file0 ?? [];
  return files
    .map((f) => {
      const rawLocation = f?.url || f?.path;
      if (!rawLocation) return null;
      const name = f.orig_name || f.path?.split(/[\\/]/).pop() || f.url?.split(/[\\/]/).pop() || "file";
      return {
        filename: name,
        format: name.includes(".") ? name.slice(name.lastIndexOf(".") + 1).toLowerCase() : "",
        url: toAbsoluteUrl(client, rawLocation),
        size_bytes: f.size ?? 0,
      };
    })
    .filter((a): a is GradioArtifact => a !== null);
}

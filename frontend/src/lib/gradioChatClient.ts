import { Client } from "@gradio/client";

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

// Everything the WebSocket path streams as separate events — dag_node_*,
// tool_start/tool_complete, camera_animate, voice state, HITL approval
// prompts — has no equivalent here: app.py's _GradioSocket auto-resolves
// HITL/visual-capture suspensions server-side and only ever surfaces the
// turn's final reply plus (since app.py moved to gr.Blocks with a
// text_out/file_out pair) whatever mesh a CAD tool call produced this
// turn. Still a much smaller surface than the WS protocol overall — no
// live tool-activity feed, no DAG telemetry.
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
};

type GradioFileData = { url?: string; path?: string; orig_name?: string; size?: number };

// `data[1]` (file_out, bound to a gr.Model3D output — see app.py's
// _respond) is a Gradio FileData object `{url, path, orig_name, ...}` in
// every version actually exercised here, but the exact client-side shape
// depends on the @gradio/client version resolving/normalizing it — some
// versions have been observed handing back the bare string path/URL a
// component was given instead of the wrapped object. Handling both shapes
// here means a client-library version bump can't silently turn a real mesh
// into a dropped `null`.
function parseMeshPayload(raw: unknown): string | null {
  console.log("[GradioClient] Mesh payload received in data[1]:", raw);
  if (typeof raw === "string") return raw || null;
  if (raw && typeof raw === "object") {
    const file = raw as GradioFileData;
    // `.path` is a last-resort fallback — on a real deployed Space it's a
    // server-side filesystem path, not something a browser can fetch, so
    // it only helps when `.url` is missing but `.path` already happens to
    // be an absolute URL (some client versions populate it that way for a
    // same-origin app). `.url` is always tried first.
    return file.url || file.path || null;
  }
  return null;
}

export async function sendGradioChatMessage(spaceUrl: string, message: string): Promise<GradioChatReply> {
  const client = await connectGradioClient(spaceUrl);
  const result = await client.predict("/chat", { message });
  const [text, file] = result.data as [string, unknown];
  return { text, meshUrl: parseMeshPayload(file) };
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
    .filter((f): f is GradioFileData & { url: string } => Boolean(f?.url))
    .map((f) => {
      const name = f.orig_name || f.path?.split(/[\\/]/).pop() || f.url.split(/[\\/]/).pop() || "file";
      return {
        filename: name,
        format: name.includes(".") ? name.slice(name.lastIndexOf(".") + 1).toLowerCase() : "",
        url: f.url,
        size_bytes: f.size ?? 0,
      };
    });
}

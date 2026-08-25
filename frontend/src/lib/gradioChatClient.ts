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
   * directly (curled it) to actually serve real STL bytes locally; CORS
   * behavior on the real hosted Space is unverified from here, since a
   * local dev instance showed no Access-Control-Allow-Origin header on a
   * cross-origin request. Worth confirming against the live deployment. */
  meshUrl: string | null;
};

export async function sendGradioChatMessage(spaceUrl: string, message: string): Promise<GradioChatReply> {
  const client = await connectGradioClient(spaceUrl);
  const result = await client.predict("/chat", { message });
  const [text, file] = result.data as [string, { url?: string } | null];
  return { text, meshUrl: file?.url ?? null };
}

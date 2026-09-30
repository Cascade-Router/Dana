/**
 * Stage 9.1 — Hugging Face Space client for Dānā.
 *
 * The Space's app.py is a gr.Blocks app that exposes named endpoints only
 * (`chat`, `artifacts`), not the legacy `/api/predict` / `/run/predict`
 * routes, so this goes through @gradio/client exactly like
 * frontend/src/lib/gradioChatClient.ts: `predict("/chat", { message })`,
 * with the reply text at `result.data[0]`.
 *
 * One Client per Space URL is kept for the page's lifetime: app.py keys the
 * Dana session on the client's session_hash, so reconnecting per message
 * would start a fresh agent session every time.
 */

import { Client } from "@gradio/client";

export type HfPredictOk = { ok: true; text: string };
export type HfPredictErr = {
  ok: false;
  message: string;
  warming?: boolean;
};
export type HfPredictResult = HfPredictOk | HfPredictErr;

export type PredictOptions = {
  /** AbortSignal from the caller (e.g. navigation cancel). */
  signal?: AbortSignal;
  /** Overall request timeout (ms). Default 90s for Space cold boots. */
  timeoutMs?: number;
  /** Override base URL (defaults to PUBLIC_DANA_HF_API). */
  baseUrl?: string;
};

const DEFAULT_TIMEOUT_MS = 90_000;

/** Public Space root — set in `.env` as PUBLIC_DANA_HF_API. */
export function getHfApiBase(): string {
  const hfSpaceUrl =
    import.meta.env.PUBLIC_DANA_HF_API || "https://amixxm-dana.hf.space";
  return String(hfSpaceUrl || "").trim().replace(/\/+$/, "");
}

function warmingMessage(detail?: string): string {
  const extra = detail ? ` (${detail})` : "";
  return `Dānā is warming up…${extra} Please try again in a moment.`;
}

function friendlyError(err: unknown): HfPredictErr {
  const msg = err instanceof Error ? err.message : String(err || "unknown");
  if (/abort|timed? ?out/i.test(msg)) {
    return { ok: false, warming: true, message: warmingMessage("cold boot or network timeout") };
  }
  if (/failed to fetch|networkerror|load failed|cors|could not resolve app config|space.*(sleep|build|start)|502|503|504/i.test(msg)) {
    return { ok: false, warming: true, message: warmingMessage("cannot reach Hugging Face Space") };
  }
  return { ok: false, message: `Could not reach Dānā: ${msg.slice(0, 160)}` };
}

const clients = new Map<string, Promise<Client>>();

function connect(base: string): Promise<Client> {
  let pending = clients.get(base);
  if (!pending) {
    pending = Client.connect(base).catch((err) => {
      clients.delete(base); // let the next message retry instead of caching the failure
      throw err;
    });
    clients.set(base, pending);
  }
  return pending;
}

function withTimeout<T>(work: Promise<T>, timeoutMs: number, signal?: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("request timed out")), timeoutMs);
    const onAbort = () => reject(new Error("request aborted"));
    if (signal?.aborted) onAbort();
    signal?.addEventListener("abort", onAbort, { once: true });
    work.then(resolve, reject).finally(() => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    });
  });
}

/** Send one prompt to the Space's `chat` endpoint and return the reply text. */
export async function predictDana(
  prompt: string,
  opts: PredictOptions = {},
): Promise<HfPredictResult> {
  const text = (prompt || "").trim();
  if (!text) {
    return { ok: false, message: "Type a command first." };
  }

  const base = (opts.baseUrl || getHfApiBase()).replace(/\/+$/, "");
  if (!base) {
    return {
      ok: false,
      message:
        "Hugging Face API is not configured. Set PUBLIC_DANA_HF_API in website/.env to your Space root (e.g. https://ORG-dana-agent.hf.space).",
    };
  }

  const timeoutMs = Math.max(5_000, opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  try {
    const result = await withTimeout(
      connect(base).then((client) => client.predict("/chat", { message: text })),
      timeoutMs,
      opts.signal,
    );
    const data = (result.data as unknown[] | undefined) ?? [];
    const reply = typeof data[0] === "string" ? data[0].trim() : "";
    if (!reply) {
      return { ok: false, message: "Dānā returned an empty response." };
    }
    return { ok: true, text: reply };
  } catch (err) {
    return friendlyError(err);
  }
}

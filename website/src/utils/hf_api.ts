/**
 * Stage 9.1 — Hugging Face / Gradio REST client for Dānā.
 * Tiny fetch wrapper: timeouts, cold-boot hints, Gradio response unpacking.
 */

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

/** Public Space / Gradio API root — set in `.env` as PUBLIC_DANA_HF_API. */
export function getHfApiBase(): string {
  const hfSpaceUrl =
    import.meta.env.PUBLIC_DANA_HF_API || "https://amixxm-donna.hf.space";
  return String(hfSpaceUrl || "").trim().replace(/\/+$/, "");
}

function warmingMessage(detail?: string): string {
  const extra = detail ? ` (${detail})` : "";
  return `Dānā is warming up…${extra} Please try again in a moment.`;
}

function friendlyNetworkError(err: unknown): HfPredictErr {
  const name = err instanceof Error ? err.name : "";
  const msg = err instanceof Error ? err.message : String(err || "unknown");

  if (name === "AbortError" || /abort|timeout/i.test(msg)) {
    return {
      ok: false,
      warming: true,
      message: warmingMessage("cold boot or network timeout"),
    };
  }
  if (/failed to fetch|networkerror|load failed|cors/i.test(msg)) {
    return {
      ok: false,
      warming: true,
      message: warmingMessage("cannot reach Hugging Face Space"),
    };
  }
  return {
    ok: false,
    message: `Could not reach Dānā: ${msg.slice(0, 160)}`,
  };
}

/** Pull assistant text from Gradio 3/4-ish JSON envelopes. */
export function unpackGradioPayload(payload: unknown): string {
  if (payload == null) return "";
  if (typeof payload === "string") return payload.trim();

  if (Array.isArray(payload)) {
    for (let i = payload.length - 1; i >= 0; i -= 1) {
      const part = unpackGradioPayload(payload[i]);
      if (part) return part;
    }
    return "";
  }

  if (typeof payload === "object") {
    const obj = payload as Record<string, unknown>;
    for (const key of ["data", "output", "generated_text", "text", "response"]) {
      if (key in obj) {
        const inner = unpackGradioPayload(obj[key]);
        if (inner) return inner;
      }
    }
    // Gradio queue event bodies sometimes nest under `value`.
    if ("value" in obj) {
      const inner = unpackGradioPayload(obj.value);
      if (inner) return inner;
    }
  }
  try {
    return JSON.stringify(payload).slice(0, 2000);
  } catch {
    return "";
  }
}

function isColdBootStatus(status: number, bodyText: string): boolean {
  if (status === 502 || status === 503 || status === 504) return true;
  return /loading|sleeping|starting|not ready|queue|503|502/i.test(bodyText);
}

/**
 * POST prompt to Gradio-style `/api/predict` (or `/run/predict` fallback).
 * Body: `{ data: [prompt] }` — matches common Gradio textbox → textbox apps.
 */
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
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const onOuterAbort = () => controller.abort();
  if (opts.signal) {
    if (opts.signal.aborted) controller.abort();
    else opts.signal.addEventListener("abort", onOuterAbort, { once: true });
  }

  const endpoints = [`${base}/api/predict`, `${base}/run/predict`];
  const body = JSON.stringify({ data: [text] });

  try {
    let lastErr: HfPredictErr | null = null;
    for (const url of endpoints) {
      let res: Response;
      try {
        res = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body,
          signal: controller.signal,
        });
      } catch (err) {
        lastErr = friendlyNetworkError(err);
        continue;
      }

      const raw = await res.text();
      if (!res.ok) {
        if (isColdBootStatus(res.status, raw)) {
          return {
            ok: false,
            warming: true,
            message: warmingMessage(`HTTP ${res.status}`),
          };
        }
        lastErr = {
          ok: false,
          message: `Dānā API error (HTTP ${res.status}).`,
        };
        // Try next endpoint shape.
        continue;
      }

      let parsed: unknown = raw;
      try {
        parsed = JSON.parse(raw);
      } catch {
        // plain text reply
      }
      const reply = unpackGradioPayload(parsed);
      if (!reply) {
        return {
          ok: false,
          message: "Dānā returned an empty response.",
        };
      }
      return { ok: true, text: reply };
    }
    return (
      lastErr || {
        ok: false,
        message: "Dānā API did not respond.",
      }
    );
  } catch (err) {
    return friendlyNetworkError(err);
  } finally {
    clearTimeout(timer);
    if (opts.signal) {
      opts.signal.removeEventListener("abort", onOuterAbort);
    }
  }
}

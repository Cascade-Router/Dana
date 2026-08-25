import type { ServerEvent } from "./useChatSocket";

// TerminalDrawer.tsx's `log: ServerEvent[]` prop is fed by two sources that
// used to be exactly one: dana/api/server.py's WS "server_log" events
// (real backend stdout/stderr — see _BroadcastStream), which never exist at
// all on the Gradio-mode build (useGradioChat.ts hardcodes `log: []`, since
// app.py's plain request/response "chat" endpoint has no streaming
// telemetry channel). That's why the Debug Terminal stayed empty there even
// though gradioChatClient.ts's own `console.log("[GradioClient] ...")`
// calls were reaching devtools the whole time — nothing ever routed a
// browser-side console call into that array. This module patches
// window.console once (idempotent — safe to call from multiple entry
// points) so every console.log/warn/error anywhere in the app also lands
// in a shared, reactive buffer TerminalDrawer reads from, on top of
// whatever it already renders.
const MAX_ENTRIES = 500;

let entries: ServerEvent[] = [];
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) listener();
}

function formatArg(arg: unknown): string {
  if (typeof arg === "string") return arg;
  if (arg instanceof Error) return `${arg.name}: ${arg.message}`;
  try {
    return JSON.stringify(arg);
  } catch {
    return String(arg);
  }
}

function push(stream: "stdout" | "stderr", line: string): void {
  const entry: ServerEvent = { type: "server_log", stream, line };
  entries = [...entries, entry].slice(-MAX_ENTRIES);
  notify();
}

let installed = false;

// Wraps (never replaces the caller's ability to see) each console method —
// the original still runs first, so devtools shows exactly what it always
// did; this only ADDS a copy into the captured buffer. Each captured line
// is tagged "[browser]"/"[browser:warn]"/"[browser:error]" so it's never
// mistaken for a genuine dana/api/server.py backend "server_log" line
// (formatLine in TerminalDrawer.tsx renders both under the same
// `[stdout]`/`[stderr]` stream prefix) — this tag is the only thing that
// tells the two apart in the rendered panel.
export function installConsoleCapture(): void {
  if (installed || typeof window === "undefined" || typeof console === "undefined") return;
  installed = true;

  const original = {
    log: console.log.bind(console),
    warn: console.warn.bind(console),
    error: console.error.bind(console),
  };

  console.log = (...args: unknown[]) => {
    original.log(...args);
    push("stdout", `[browser] ${args.map(formatArg).join(" ")}`);
  };
  console.warn = (...args: unknown[]) => {
    original.warn(...args);
    push("stderr", `[browser:warn] ${args.map(formatArg).join(" ")}`);
  };
  console.error = (...args: unknown[]) => {
    original.error(...args);
    push("stderr", `[browser:error] ${args.map(formatArg).join(" ")}`);
  };

  // Proves the capture mechanism itself is live, right as it comes up —
  // goes through the now-patched console.log so it's simultaneously (a) a
  // real devtools message and (b) the very first entry the Debug Terminal
  // panel should ever show, regardless of WS vs Gradio mode.
  console.log("[DebugTerminal] Debug console initialized successfully");
}

export function getCapturedLog(): ServerEvent[] {
  return entries;
}

export function subscribeCapturedLog(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

// Installed as a side effect of importing this module (idempotent — see
// the `installed` guard above), not left for a component to remember to
// call, so no captured line before the first mount is ever lost.
installConsoleCapture();

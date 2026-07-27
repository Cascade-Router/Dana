/**
 * Stage 9.1 — Global chat bar controller (vanilla, session-persisted).
 */

import { predictDana } from "../utils/hf_api";

export type ChatRole = "user" | "dana" | "system";
export type ChatMessage = { role: ChatRole; text: string; ts: number };

const STORAGE_KEY = "dana_chat_v1";
const MAX_MESSAGES = 80;

type Dom = {
  root: HTMLElement;
  panel: HTMLElement;
  log: HTMLElement;
  form: HTMLFormElement;
  input: HTMLInputElement;
  send: HTMLButtonElement;
  toggle: HTMLButtonElement;
  status: HTMLElement;
};

function loadHistory(): ChatMessage[] {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatMessage[];
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((m) => m && typeof m.text === "string" && m.role)
      .slice(-MAX_MESSAGES);
  } catch {
    return [];
  }
}

function saveHistory(messages: ChatMessage[]): void {
  try {
    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(messages.slice(-MAX_MESSAGES)),
    );
  } catch {
    /* quota / private mode */
  }
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function labelFor(role: ChatRole): string {
  if (role === "user") return "User (Text)";
  if (role === "dana") return "Dānā";
  return "System";
}

function classFor(role: ChatRole): string {
  if (role === "user") return "dana-chat__line--user";
  if (role === "dana") return "dana-chat__line--dana";
  return "dana-chat__line--system";
}

function renderLog(log: HTMLElement, messages: ChatMessage[]): void {
  log.innerHTML = messages
    .map(
      (m) =>
        `<div class="dana-chat__line ${classFor(m.role)}"><span class="dana-chat__speaker">[${escapeHtml(labelFor(m.role))}]</span> ${escapeHtml(m.text)}</div>`,
    )
    .join("");
  log.scrollTop = log.scrollHeight;
}

function setLoading(dom: Dom, loading: boolean): void {
  dom.root.classList.toggle("is-loading", loading);
  dom.input.disabled = loading;
  dom.send.disabled = loading;
  dom.send.setAttribute("aria-busy", loading ? "true" : "false");
  dom.status.textContent = loading ? "Thinking…" : "";
}

function setExpanded(dom: Dom, open: boolean): void {
  dom.root.classList.toggle("is-open", open);
  dom.toggle.setAttribute("aria-expanded", open ? "true" : "false");
  dom.panel.hidden = !open;
  if (open) {
    requestAnimationFrame(() => dom.input.focus());
  }
}

export function initGlobalChat(): void {
  const root = document.getElementById("dana-global-chat");
  if (!root || root.dataset.ready === "1") return;
  root.dataset.ready = "1";

  const panel = root.querySelector<HTMLElement>("[data-chat-panel]");
  const log = root.querySelector<HTMLElement>("[data-chat-log]");
  const form = root.querySelector<HTMLFormElement>("[data-chat-form]");
  const input = root.querySelector<HTMLInputElement>("[data-chat-input]");
  const send = root.querySelector<HTMLButtonElement>("[data-chat-send]");
  const toggle = root.querySelector<HTMLButtonElement>("[data-chat-toggle]");
  const status = root.querySelector<HTMLElement>("[data-chat-status]");
  if (!panel || !log || !form || !input || !send || !toggle || !status) return;

  const dom: Dom = { root, panel, log, form, input, send, toggle, status };
  let messages = loadHistory();
  let inflight: AbortController | null = null;

  renderLog(log, messages);
  if (messages.length) setExpanded(dom, true);

  toggle.addEventListener("click", () => {
    setExpanded(dom, !dom.root.classList.contains("is-open"));
  });

  document.addEventListener("keydown", (ev) => {
    const meta = ev.metaKey || ev.ctrlKey;
    if (meta && (ev.key === "k" || ev.key === "K")) {
      ev.preventDefault();
      setExpanded(dom, true);
      return;
    }
    if (ev.key === "Escape" && dom.root.classList.contains("is-open")) {
      if (document.activeElement === input && input.value) {
        input.value = "";
        return;
      }
      setExpanded(dom, false);
    }
  });

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const text = (input.value || "").trim();
    if (!text || dom.root.classList.contains("is-loading")) return;

    setExpanded(dom, true);
    input.value = "";
    messages.push({ role: "user", text, ts: Date.now() });
    saveHistory(messages);
    renderLog(log, messages);

    if (inflight) inflight.abort();
    inflight = new AbortController();
    setLoading(dom, true);

    const result = await predictDana(text, { signal: inflight.signal });
    setLoading(dom, false);
    inflight = null;

    if (result.ok) {
      messages.push({ role: "dana", text: result.text, ts: Date.now() });
    } else {
      messages.push({
        role: "system",
        text: result.message,
        ts: Date.now(),
      });
      if (result.warming) {
        dom.status.textContent = "Warming up";
      }
    }
    saveHistory(messages);
    renderLog(log, messages);
  });
}

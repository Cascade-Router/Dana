/**
 * Shared typing-terminal engine — scroll-gated (no animation until the
 * element is actually visible), `prefers-reduced-motion`-aware. Extracted
 * from TerminalReveal.astro so a second terminal-styled panel (Hero's demo
 * panel) doesn't duplicate this logic.
 */

type ColorRule = { test: (line: string) => boolean; tokenClass: string };

const DEFAULT_RULES: ColorRule[] = [
  { test: (l) => l.startsWith("$"), tokenClass: "tok-accent" },
  { test: (l) => l.startsWith("→"), tokenClass: "tok-fn" },
  { test: (l) => l.startsWith("✓"), tokenClass: "tok-string" },
  { test: (l) => l.startsWith("⚠"), tokenClass: "tok-warn" },
  { test: (l) => l.includes("HITL") || l.includes("interrupt"), tokenClass: "tok-warn" },
  { test: (l) => l.startsWith("OK:") || l.includes("OK"), tokenClass: "tok-string" },
  { test: (l) => l.startsWith("Status:"), tokenClass: "tok-keyword" },
];

function escapeHtml(line: string): string {
  return line.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function colorize(line: string, rules: ColorRule[]): string {
  const esc = escapeHtml(line);
  const rule = rules.find((r) => r.test(line));
  return rule ? `<span class="${rule.tokenClass}">${esc}</span>` : esc;
}

/** Reads `data-lines` (a JSON string array) off `elementId` and types it out
 * once the element scrolls into view — or renders instantly if the visitor
 * prefers reduced motion, or if IntersectionObserver isn't available. */
export function initTypingTerminal(elementId: string, rules: ColorRule[] = DEFAULT_RULES): void {
  const el = document.getElementById(elementId);
  if (!el) return;

  let lines: string[] = [];
  try {
    lines = JSON.parse(el.getAttribute("data-lines") || "[]");
  } catch {
    lines = [];
  }

  const renderAll = () => {
    el.innerHTML = lines.map((l) => colorize(l, rules)).join("\n");
  };

  const typeLines = async () => {
    const done: string[] = [];
    for (const line of lines) {
      let partial = "";
      for (let i = 0; i < line.length; i += 1) {
        partial += line[i];
        el.innerHTML = done.map((l) => colorize(l, rules)).join("\n") + (done.length ? "\n" : "") + colorize(partial, rules);
        await new Promise((r) => setTimeout(r, 12 + (i % 3)));
      }
      done.push(line);
      el.innerHTML = done.map((l) => colorize(l, rules)).join("\n") + "\n";
      await new Promise((r) => setTimeout(r, 180));
    }
  };

  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (reduce) {
    renderAll();
  } else if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          io.disconnect();
          void typeLines();
        }
      },
      { threshold: 0.35 },
    );
    io.observe(el);
  } else {
    void typeLines();
  }
}

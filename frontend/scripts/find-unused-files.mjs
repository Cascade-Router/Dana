#!/usr/bin/env node
// Zero-dependency reachability audit for frontend/src.
//
// Walks every static `import ... from "..."` / side-effect `import "..."`
// AND dynamic `import("...")` call (this codebase's only dynamic-import
// site is src/plugins/registry.ts's `lazy(() => import("./CadPlugin"))`
// etc. — all static string literals, so a regex-based walk is enough; a
// genuinely COMPUTED import path would be invisible to this script) —
// starting from src/main.tsx, the app's one real entry point (every Tauri
// window — main, the orb, a spawned plugin window — boots through this
// same bundle and branches by URL hash; see main.tsx's own comment).
//
// Reports every file under src/ this walk never reaches. That's a
// candidate list to REVIEW, not an auto-delete list — cross-check against
// knip/ts-prune (see frontend's audit plan) before removing anything.
//
// Deliberately does NOT flag unused EXPORTS inside an otherwise-reached
// file (e.g. a helper exported but never imported anywhere) — that's
// knip/ts-prune's job, not this script's; this only ever answers "is this
// FILE reachable at all".
//
// Run: node scripts/find-unused-files.mjs

import { existsSync, statSync } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC_DIR = path.resolve(__dirname, "..", "src");
const ENTRY = path.join(SRC_DIR, "main.tsx");

// Resolution order mirrors tsconfig's "moduleResolution": "Bundler" closely
// enough for this codebase (no path aliases in tsconfig.json — every
// import here is a plain relative path).
const RESOLVABLE_EXT = [".tsx", ".ts", ".css", ".json"];

// Three alternatives: (1) `import ... from "spec"` (default/named/type,
// single- or multi-line — the character class matches across newlines),
// (2) dynamic `import("spec")`, (3) bare side-effect `import "spec"`.
const IMPORT_RE =
  /import\s+[^;]*?from\s+["']([^"']+)["']|import\s*\(\s*["']([^"']+)["']\s*\)|^import\s+["']([^"']+)["']/gm;

async function listAllFiles(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await listAllFiles(full)));
    else out.push(full);
  }
  return out;
}

function isFile(p) {
  try {
    return statSync(p).isFile();
  } catch {
    return false;
  }
}

function resolveImport(fromFile, spec) {
  if (!spec.startsWith(".")) return null; // external package — out of scope for this audit
  const base = path.resolve(path.dirname(fromFile), spec);
  if (isFile(base)) return base;
  for (const ext of RESOLVABLE_EXT) {
    if (isFile(base + ext)) return base + ext;
  }
  for (const ext of RESOLVABLE_EXT) {
    const indexPath = path.join(base, `index${ext}`);
    if (isFile(indexPath)) return indexPath;
  }
  return null;
}

async function extractImportSpecs(file) {
  const text = await readFile(file, "utf8");
  const specs = [];
  for (const match of text.matchAll(IMPORT_RE)) {
    const spec = match[1] || match[2] || match[3];
    if (spec) specs.push(spec);
  }
  return specs;
}

async function main() {
  const allFiles = new Set(await listAllFiles(SRC_DIR));
  const visited = new Set();
  const queue = [ENTRY];

  while (queue.length) {
    const file = queue.pop();
    if (visited.has(file) || !isFile(file)) continue;
    visited.add(file);
    if (!/\.tsx?$/.test(file)) continue; // .css/.json are reachable leaves; nothing to extract from them
    for (const spec of await extractImportSpecs(file)) {
      const resolved = resolveImport(file, spec);
      if (resolved && !visited.has(resolved)) queue.push(resolved);
    }
  }

  const unreached = [...allFiles].filter((f) => !visited.has(f)).sort();
  console.log(`Reached ${visited.size} / ${allFiles.size} files from ${path.relative(process.cwd(), ENTRY)}\n`);
  if (unreached.length === 0) {
    console.log("No unreached files under src/ — nothing to report.");
    return;
  }
  console.log("Never reached from main.tsx (review before deleting — see this script's own docstring):");
  for (const f of unreached) console.log(`  - ${path.relative(SRC_DIR, f)}`);
}

main();

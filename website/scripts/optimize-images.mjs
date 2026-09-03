// Regenerates a WebP alongside every source PNG/JPG in public/ — run after
// adding or replacing a static image (`npm run optimize:images`). Skips a
// file whose .webp sibling is already newer than the source, so re-running
// this on every commit is cheap.
import { readdir, stat } from "node:fs/promises";
import { extname, join } from "node:path";
import sharp from "sharp";

const PUBLIC_DIR = new URL("../public/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const SOURCE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg"]);

async function needsRebuild(sourcePath, webpPath) {
  try {
    const [source, webp] = await Promise.all([stat(sourcePath), stat(webpPath)]);
    return source.mtimeMs > webp.mtimeMs;
  } catch {
    return true; // no .webp yet
  }
}

async function main() {
  const entries = await readdir(PUBLIC_DIR, { withFileTypes: true });
  let converted = 0;

  for (const entry of entries) {
    if (!entry.isFile()) continue;
    const ext = extname(entry.name).toLowerCase();
    if (!SOURCE_EXTENSIONS.has(ext)) continue;

    const sourcePath = join(PUBLIC_DIR, entry.name);
    const webpPath = join(PUBLIC_DIR, entry.name.slice(0, -ext.length) + ".webp");
    if (!(await needsRebuild(sourcePath, webpPath))) continue;

    const info = await sharp(sourcePath).webp({ quality: 82 }).toFile(webpPath);
    const before = (await stat(sourcePath)).size;
    console.log(`${entry.name} -> ${entry.name.slice(0, -ext.length)}.webp  (${before}B -> ${info.size}B, -${Math.round((1 - info.size / before) * 100)}%)`);
    converted += 1;
  }

  console.log(converted ? `Done — ${converted} file(s) converted.` : "Nothing to convert — all .webp files are up to date.");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

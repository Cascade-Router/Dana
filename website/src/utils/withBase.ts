/** Join a site-relative path onto Astro `BASE_URL` with a guaranteed slash. */
export function withBase(path = ""): string {
  const raw = String(import.meta.env.BASE_URL || "/");
  const base = raw.endsWith("/") ? raw : `${raw}/`;
  if (!path || path === "/") return base;
  if (path.startsWith("#")) return `${base}${path}`;
  return `${base}${path.replace(/^\//, "")}`;
}

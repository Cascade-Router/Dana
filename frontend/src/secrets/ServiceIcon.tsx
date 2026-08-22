import { serviceMeta, type ServiceId } from "./types";

// A colored glyph badge stands in for a real icon set — zero extra
// dependency, and legible at the small size the secrets list needs.
export function ServiceIcon({ service }: { service: ServiceId }) {
  const meta = serviceMeta(service);
  return (
    <span className="service-icon" style={{ background: meta.color }} aria-hidden="true">
      {meta.glyph}
    </span>
  );
}

export type ServiceId = "openai" | "anthropic" | "elevenlabs" | "custom";

export type ServiceMeta = {
  id: ServiceId;
  label: string;
  glyph: string;
  color: string;
};

export const KNOWN_SERVICES: ServiceMeta[] = [
  { id: "openai", label: "OpenAI API Key", glyph: "O", color: "#10a37f" },
  { id: "anthropic", label: "Anthropic API Key", glyph: "A", color: "#d97757" },
  { id: "elevenlabs", label: "ElevenLabs API Key", glyph: "11", color: "#7c3aed" },
  { id: "custom", label: "Custom Key", glyph: "?", color: "#6b7280" },
];

export function serviceMeta(id: ServiceId): ServiceMeta {
  return KNOWN_SERVICES.find((s) => s.id === id) ?? KNOWN_SERVICES[KNOWN_SERVICES.length - 1];
}

export type SecretRecord = {
  service: ServiceId;
  /** Only set when service === "custom" — the user's own label for the key. */
  customLabel?: string;
  value: string;
  updatedAt: number;
};

export function maskSecret(value: string): string {
  if (value.length <= 4) return "•".repeat(Math.max(value.length, 4));
  return `${"•".repeat(Math.max(value.length - 4, 4))}${value.slice(-4)}`;
}

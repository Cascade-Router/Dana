import { FormEvent, useState } from "react";
import { useSecrets } from "./SecretsContext";
import { ServiceIcon } from "./ServiceIcon";
import { KNOWN_SERVICES, maskSecret, serviceMeta, type ServiceId } from "./types";
import "./SecretsMenu.css";

function entryLabel(entry: { service: ServiceId; customLabel?: string }): string {
  if (entry.service === "custom") return entry.customLabel || "Custom Key";
  return serviceMeta(entry.service).label;
}

export function SecretsMenu({ onClose }: { onClose: () => void }) {
  const { entries, loaded, upsert, remove } = useSecrets();
  const [service, setService] = useState<ServiceId>("openai");
  const [customLabel, setCustomLabel] = useState("");
  const [value, setValue] = useState("");
  const [revealedId, setRevealedId] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!value.trim()) return;
    await upsert(service, value.trim(), service === "custom" ? customLabel.trim() || undefined : undefined);
    setValue("");
    setCustomLabel("");
  };

  return (
    <div className="secrets-menu">
      <div className="secrets-menu__backdrop" onClick={onClose} />
      <div className="secrets-menu__panel" role="dialog" aria-label="Secrets">
        <div className="secrets-menu__header">
          <h2>Secrets</h2>
          <button type="button" className="secrets-menu__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="secrets-menu__list">
          {!loaded && <div className="secrets-menu__empty">Loading…</div>}
          {loaded && entries.length === 0 && (
            <div className="secrets-menu__empty">No keys configured yet.</div>
          )}
          {entries.map((entry) => (
            <div key={entry.id} className="secrets-menu__row">
              <ServiceIcon service={entry.service} />
              <div className="secrets-menu__row-text">
                <div className="secrets-menu__row-label">{entryLabel(entry)}</div>
                <div className="secrets-menu__row-value">
                  {revealedId === entry.id ? entry.value : maskSecret(entry.value)}
                </div>
              </div>
              <button
                type="button"
                className="secrets-menu__icon-btn"
                onClick={() => setRevealedId(revealedId === entry.id ? null : entry.id)}
                title={revealedId === entry.id ? "Hide" : "Reveal"}
              >
                {revealedId === entry.id ? "🙈" : "👁"}
              </button>
              <button
                type="button"
                className="secrets-menu__icon-btn secrets-menu__icon-btn--danger"
                onClick={() => remove(entry.id)}
                title="Remove"
              >
                ✕
              </button>
            </div>
          ))}
        </div>

        <form className="secrets-menu__form" onSubmit={submit}>
          <div className="secrets-menu__form-row">
            <select value={service} onChange={(e) => setService(e.target.value as ServiceId)}>
              {KNOWN_SERVICES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>
          {service === "custom" && (
            <input
              className="secrets-menu__input"
              placeholder="Label (e.g. Azure Speech Key)"
              value={customLabel}
              onChange={(e) => setCustomLabel(e.target.value)}
            />
          )}
          <input
            className="secrets-menu__input"
            type="password"
            placeholder="Paste key…"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            autoComplete="off"
          />
          <button type="submit" className="secrets-menu__save" disabled={!value.trim()}>
            Save Key
          </button>
        </form>
      </div>
    </div>
  );
}

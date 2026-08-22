import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { load, type Store } from "@tauri-apps/plugin-store";
import type { SecretRecord, ServiceId } from "./types";

// Secrets persist via tauri-plugin-store to a JSON file in the app's local
// data dir (outside the repo, outside the webview's localStorage sandbox).
// Note this is at-rest-plaintext-on-disk, same trust level as a browser's
// credential store — good enough for a local desktop key vault without
// pulling in tauri-plugin-stronghold's password-prompt UX. If these keys
// ever need to survive a stolen-laptop threat model, swap the backend for
// stronghold; nothing above this file (SecretsMenu, PluginContext, the
// window-sync payload) needs to change.
const STORE_FILE = "secrets.json";

export type SecretEntry = SecretRecord & { id: string };

type SecretsContextValue = {
  entries: SecretEntry[];
  loaded: boolean;
  upsert: (service: ServiceId, value: string, customLabel?: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
  getValue: (service: ServiceId) => string | undefined;
};

const SecretsContext = createContext<SecretsContextValue | null>(null);

export function SecretsProvider({ children }: { children: ReactNode }) {
  const storeRef = useRef<Store | null>(null);
  const [entries, setEntries] = useState<SecretEntry[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    load(STORE_FILE, { autoSave: true }).then(async (store) => {
      if (cancelled) return;
      storeRef.current = store;
      const raw = await store.entries();
      setEntries(raw.map(([id, record]) => ({ id, ...(record as SecretRecord) })));
      setLoaded(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Known services (openai/anthropic/elevenlabs) are singletons — re-saving
  // overwrites the existing entry rather than piling up duplicates. "custom"
  // entries get a fresh id each time so users can hold several at once.
  const upsert = useCallback(async (service: ServiceId, value: string, customLabel?: string) => {
    const store = storeRef.current;
    if (!store) return;
    const id = service === "custom" ? `custom-${crypto.randomUUID()}` : service;
    const record: SecretRecord = {
      service,
      value,
      updatedAt: Date.now(),
      ...(customLabel ? { customLabel } : {}),
    };
    await store.set(id, record);
    setEntries((prev) => [...prev.filter((e) => e.id !== id), { id, ...record }]);
  }, []);

  const remove = useCallback(async (id: string) => {
    const store = storeRef.current;
    if (!store) return;
    await store.delete(id);
    setEntries((prev) => prev.filter((e) => e.id !== id));
  }, []);

  const getValue = useCallback(
    (service: ServiceId) => entries.find((e) => e.service === service)?.value,
    [entries]
  );

  const value = useMemo(
    () => ({ entries, loaded, upsert, remove, getValue }),
    [entries, loaded, upsert, remove, getValue]
  );

  return <SecretsContext.Provider value={value}>{children}</SecretsContext.Provider>;
}

export function useSecrets(): SecretsContextValue {
  const ctx = useContext(SecretsContext);
  if (!ctx) throw new Error("useSecrets() must be used within <SecretsProvider>");
  return ctx;
}

/**
 * Per-viewer preferences, persisted in localStorage when available. Storage may be
 * blocked (private windows, previews); every access is guarded and the app works
 * without it. Nothing sensitive is stored here.
 */
import { useSyncExternalStore } from "react";
import type { DetectorMode, Transport } from "../api/schemas";

export interface Settings {
  /** null = follow the server's privacy default */
  transcriptDisplay: boolean | null;
  transcriptExport: boolean | null;
  defaultDetector: DetectorMode | null;
  defaultScenario: string | null;
  defaultTransport: Transport;
  audioDeviceId: string | null;
  developerMode: boolean;
}

const KEY = "turnsense.settings.v1";

export const DEFAULT_SETTINGS: Settings = {
  transcriptDisplay: null,
  transcriptExport: null,
  defaultDetector: null,
  defaultScenario: null,
  defaultTransport: "simulation",
  audioDeviceId: null,
  developerMode: false,
};

function load(): Settings {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return { ...DEFAULT_SETTINGS, ...parsed };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

let current: Settings = typeof window === "undefined" ? DEFAULT_SETTINGS : load();
const listeners = new Set<() => void>();

export const settingsStore = {
  get(): Settings {
    return current;
  },
  set(patch: Partial<Settings>): void {
    current = { ...current, ...patch };
    try {
      window.localStorage.setItem(KEY, JSON.stringify(current));
    } catch {
      // storage unavailable — keep the in-memory value
    }
    listeners.forEach((l) => l());
  },
  reset(): void {
    current = DEFAULT_SETTINGS;
    try {
      window.localStorage.removeItem(KEY);
    } catch {
      // ignore
    }
    listeners.forEach((l) => l());
  },
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
};

export function useSettings(): Settings {
  return useSyncExternalStore(settingsStore.subscribe, settingsStore.get, settingsStore.get);
}

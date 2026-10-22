/**
 * External store for the live session. Incoming envelopes are queued and folded into
 * state once per animation frame, so a burst of 20 ms-rate messages produces at most
 * one render per frame (frontend.md §11). Components subscribe to slices through
 * useSyncExternalStore and only re-render when their slice changes.
 */
import { createContext, useContext, useSyncExternalStore } from "react";
import { parseEnvelope, type ParseResult } from "../api/schemas";
import { initialSessionState, sessionReducer, type SessionAction, type SessionViewState } from "./sessionReducer";

type Cancel = () => void;

function defaultScheduler(cb: () => void): Cancel {
  const visible = typeof document === "undefined" || document.visibilityState === "visible";
  if (visible && typeof requestAnimationFrame === "function") {
    const id = requestAnimationFrame(cb);
    return () => cancelAnimationFrame(id);
  }
  // rAF is paused in background tabs; keep folding so state never goes stale.
  const id = setTimeout(cb, 50);
  return () => clearTimeout(id);
}

export class SessionStore {
  private state: SessionViewState = initialSessionState();
  private readonly listeners = new Set<() => void>();
  private queue: ParseResult[] = [];
  private cancelScheduled: Cancel | null = null;

  constructor(private readonly schedule: (cb: () => void) => Cancel = defaultScheduler) {}

  getState = (): SessionViewState => this.state;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  /** Apply an action immediately. Pending envelopes are folded first to keep ordering. */
  dispatch(action: SessionAction): void {
    this.flush();
    this.apply(action);
  }

  /** Queue one raw message from any transport (string or object). */
  ingest(raw: unknown): void {
    this.queue.push(parseEnvelope(raw));
    if (!this.cancelScheduled) {
      this.cancelScheduled = this.schedule(() => {
        this.cancelScheduled = null;
        this.flush();
      });
    }
  }

  /** Fold all queued envelopes now (used before actions and by tests). */
  flush(): void {
    if (this.cancelScheduled) {
      this.cancelScheduled();
      this.cancelScheduled = null;
    }
    if (this.queue.length === 0) return;
    const results = this.queue;
    this.queue = [];
    this.apply({ type: "envelopes", results });
  }

  private apply(action: SessionAction): void {
    const next = sessionReducer(this.state, action);
    if (next === this.state) return;
    this.state = next;
    this.listeners.forEach((l) => l());
  }
}

export const SessionStoreContext = createContext<SessionStore | null>(null);

export function useSessionStoreInstance(): SessionStore {
  const store = useContext(SessionStoreContext);
  if (!store) throw new Error("SessionStoreContext is missing");
  return store;
}

/** Subscribe to a slice. The selector must return an existing reference or a primitive. */
export function useSession<T>(selector: (s: SessionViewState) => T): T {
  const store = useSessionStoreInstance();
  return useSyncExternalStore(
    store.subscribe,
    () => selector(store.getState()),
    () => selector(store.getState()),
  );
}

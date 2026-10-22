import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, describeApiError } from "../api/client";
import type { Capabilities, Health } from "../api/schemas";
import { MOCK_CAPABILITIES } from "../events/mock";
import { Header } from "../components/Header";
import { Tabs } from "../components/Tabs";
import type { ConsoleDraft } from "../components/ConfigBar";
import { Icon } from "../components/Icon";
import { PHASE_META } from "../lib/labels";
import { settingsStore } from "../lib/settings";
import { EvaluationPage, type EvaluationRun } from "../pages/EvaluationPage";
import { LiveConsole } from "../pages/LiveConsole";
import { ResultsPage } from "../pages/ResultsPage";
import { SettingsPage } from "../pages/SettingsPage";
import { SessionController } from "../state/sessionController";
import { selectDisplayPhase } from "../state/sessionReducer";
import { SessionStore, SessionStoreContext, useSession } from "../state/sessionStore";
import { AppContext, TABS, type AppContextValue, type TabId } from "./context";

type ServerState =
  | { status: "loading" }
  | { status: "ready"; caps: Capabilities; health: Health | null }
  | { status: "unreachable"; message: string }
  | { status: "error"; message: string };

function tabFromHash(): TabId {
  const h = typeof window !== "undefined" ? window.location.hash.replace(/^#/, "") : "";
  return TABS.some((t) => t.id === h) ? (h as TabId) : "live";
}

function mockFromUrl(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("mock") === "1";
}

export function initialDraft(caps: Capabilities): ConsoleDraft {
  const s = settingsStore.get();
  const detectorMode =
    caps.detectors.find((d) => d.available && d.mode === s.defaultDetector)?.mode ??
    caps.detectors.find((d) => d.available)?.mode ??
    null;
  const transport =
    s.defaultTransport === "livekit" && caps.transports.livekit.available ? "livekit" : "simulation";
  const scenarioId =
    caps.scenarios.find((sc) => sc.id === s.defaultScenario && sc.transports.includes(transport))?.id ??
    caps.scenarios.find((sc) => sc.transports.includes(transport))?.id ??
    null;
  return { detectorMode, scenarioId, transport, policy: {}, seed: "", trialLabel: "", speculativePrefill: false };
}

/** Polite live region: announces state changes only, never every event (frontend.md §9). */
function Announcer() {
  const phase = useSession(selectDisplayPhase);
  const stopping = useSession((s) => s.stopping);
  const [text, setText] = useState("");
  useEffect(() => {
    if (phase === "idle") return;
    setText(`Session status: ${stopping ? "Stopping response" : PHASE_META[phase].label}`);
  }, [phase, stopping]);
  return (
    <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
      {text}
    </div>
  );
}

export function App() {
  const [tab, setTab] = useState<TabId>(tabFromHash);
  const [mock, setMock] = useState<boolean>(mockFromUrl);
  const [server, setServer] = useState<ServerState>(() => (mockFromUrl() ? { status: "ready", caps: MOCK_CAPABILITIES, health: null } : { status: "loading" }));
  const [draft, setDraft] = useState<ConsoleDraft | null>(null);
  const [evaluationRun, setEvaluationRun] = useState<EvaluationRun | null>(null);
  const store = useMemo(() => new SessionStore(), []);
  const controller = useMemo(() => new SessionController(store), [store]);
  const mainRef = useRef<HTMLElement>(null);

  const loadServer = useCallback(() => {
    if (mock) {
      setServer({ status: "ready", caps: MOCK_CAPABILITIES, health: null });
      return;
    }
    setServer((s) => (s.status === "ready" ? s : { status: "loading" }));
    Promise.all([api.capabilities(), api.health().catch(() => null)])
      .then(([caps, health]) => setServer({ status: "ready", caps, health }))
      .catch((e: unknown) => {
        // The dev proxy answers 500 (and a static host 404) when the backend is not running.
        if (e instanceof ApiError && (e.kind === "network" || [404, 500, 502, 503, 504].includes(e.status))) {
          setServer({ status: "unreachable", message: describeApiError(e) });
        } else {
          setServer({ status: "error", message: describeApiError(e) });
        }
      });
  }, [mock]);

  useEffect(() => {
    loadServer();
  }, [loadServer]);

  useEffect(() => {
    if (server.status === "ready" && draft === null) setDraft(initialDraft(server.caps));
  }, [server, draft]);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    // Stop the microphone and playback if the page is closed mid-session.
    const onUnload = () => void controller.teardown();
    window.addEventListener("pagehide", onUnload);
    return () => {
      window.removeEventListener("hashchange", onHash);
      window.removeEventListener("pagehide", onUnload);
    };
  }, [controller]);

  const goTo = useCallback((t: TabId) => {
    setTab(t);
    if (window.location.hash !== `#${t}`) window.history.replaceState(null, "", `#${t}`);
  }, []);

  const useMock = () => {
    const url = new URL(window.location.href);
    url.searchParams.set("mock", "1");
    window.history.replaceState(null, "", url.toString());
    setDraft(null);
    setMock(true);
  };

  if (server.status !== "ready") {
    return (
      <div className="boot-screen">
        <a className="skip-link" href="#boot-main">
          Skip to content
        </a>
        <main id="boot-main" className="boot-card">
          <h1 className="app-title">Speaker-Invariant Turn Detection</h1>
          <p className="app-subtitle">Real-time voice agent / turn-detection research console</p>
          {server.status === "loading" && <p role="status">Connecting to the server…</p>}
          {server.status !== "loading" && (
            <div className="notice notice-error" role="alert">
              <Icon name="alert" size={16} />
              <div className="notice-body">
                <strong>{server.status === "unreachable" ? "Backend unreachable" : "The server returned an unexpected response"}</strong>
                <div className="notice-text">{server.message}</div>
                <div className="notice-text">
                  Expected the API at <span className="mono">/api</span> (proxied to http://localhost:8000 in development).
                </div>
                <div className="notice-actions">
                  <button type="button" className="btn btn-primary" onClick={loadServer}>
                    <Icon name="refresh" size={14} />
                    Retry
                  </button>
                  <button type="button" className="btn btn-ghost" onClick={useMock}>
                    Use mock data (UI demo — not measurements)
                  </button>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    );
  }

  const ctx: AppContextValue = { caps: server.caps, health: server.health, mock, controller, goTo, refreshServer: loadServer };

  return (
    <SessionStoreContext.Provider value={store}>
      <AppContext.Provider value={ctx}>
        <a className="skip-link" href="#main">
          Skip to main content
        </a>
        <div className="app">
          <Header />
          <Tabs active={tab} onChange={goTo} />
          <main id="main" ref={mainRef} className="app-main" tabIndex={-1}>
            <div id={`panel-${tab}`} role="tabpanel" aria-labelledby={`tab-${tab}`}>
              {tab === "live" && draft && <LiveConsole draft={draft} onDraftChange={setDraft} />}
              {tab === "evaluation" && draft && (
                <EvaluationPage run={evaluationRun} onRunChange={setEvaluationRun} consoleDraft={draft} />
              )}
              {tab === "results" && <ResultsPage />}
              {tab === "settings" && <SettingsPage />}
            </div>
          </main>
          <Announcer />
        </div>
      </AppContext.Provider>
    </SessionStoreContext.Provider>
  );
}

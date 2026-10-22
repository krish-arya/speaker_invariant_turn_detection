/**
 * Orchestrates a browser session: permission → create session → attach the event
 * source → interrupt / end → teardown. The store is the only place UI state lives;
 * the controller only dispatches actions and owns the transport object.
 */
import { api as defaultApi, describeApiError, resolveWebSocketUrl, type Api } from "../api/client";
import type { DetectorMode, LiveKitJoinInfo, Transport } from "../api/schemas";
import { MockEventSource, type MockSessionOptions } from "../events/mock";
import type { SessionEventSource } from "../events/types";
import { WebSocketEventSource } from "../events/websocket";
import type { LocalError, TransportStatus } from "./sessionReducer";
import type { SessionStore } from "./sessionStore";

export interface StartOptions {
  transport: Transport;
  detectorMode: DetectorMode;
  scenarioId: string;
  policyOverrides: Record<string, number> | null;
  transcriptEnabled: boolean;
  transcriptExportEnabled: boolean;
  speculativePrefill: boolean;
  trialLabel: string | null;
  seed: number | null;
  deviceId: string | null;
  mock: boolean;
}

export interface ControllerDeps {
  api: Pick<Api, "createSession" | "endSession" | "interruptSession">;
  createWebSocketSource(url: string): SessionEventSource;
  /** Async so livekit-client is only downloaded when a live session starts. */
  createLiveKitSource(join: LiveKitJoinInfo, deviceId: string | null): Promise<SessionEventSource>;
  createMockSource(opts: MockSessionOptions): MockEventSource;
  checkMicrophone(deviceId: string | null): Promise<void>;
  clientBuild: string;
  /** How long End waits for the server's `ended` status and trial summary. */
  endTimeoutMs: number;
}

export async function checkMicrophone(deviceId: string | null): Promise<void> {
  if (!navigator.mediaDevices?.getUserMedia) {
    const err = new Error("This browser does not support microphone capture.");
    err.name = "NotSupportedError";
    throw err;
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: deviceId ? { deviceId: { exact: deviceId } } : true,
  });
  // Permission granted; LiveKit opens its own track when publishing.
  stream.getTracks().forEach((t) => t.stop());
}

export function microphoneError(err: unknown): LocalError {
  const name = err instanceof Error ? err.name : "";
  if (name === "NotAllowedError" || name === "SecurityError" || name === "PermissionDeniedError") {
    return {
      kind: "permission_denied",
      message: "Microphone access was blocked.",
      detail:
        "Allow microphone access for this site (usually via the lock or camera icon in the address bar), then choose Retry. Nothing is recorded until a session starts.",
      retryable: true,
      fatal: true,
    };
  }
  if (name === "NotFoundError" || name === "OverconstrainedError" || name === "DevicesNotFoundError") {
    return {
      kind: "no_device",
      message: "No microphone was found.",
      detail: "Connect a microphone or pick a different input in Settings → Audio, then choose Retry.",
      retryable: true,
      fatal: true,
    };
  }
  if (name === "NotReadableError") {
    return {
      kind: "no_device",
      message: "The microphone is in use by another application.",
      detail: "Close the other application using the microphone, then choose Retry.",
      retryable: true,
      fatal: true,
    };
  }
  return {
    kind: "no_device",
    message: "The microphone could not be opened.",
    detail: err instanceof Error ? err.message : String(err),
    retryable: true,
    fatal: true,
  };
}

const defaultDeps: ControllerDeps = {
  api: defaultApi,
  createWebSocketSource: (url) => new WebSocketEventSource(url),
  createLiveKitSource: async (join, deviceId) => {
    const { LiveKitEventSource } = await import("../events/livekit");
    return new LiveKitEventSource(join, { deviceId });
  },
  createMockSource: (opts) => new MockEventSource(opts),
  checkMicrophone,
  clientBuild: `web-${typeof __APP_VERSION__ === "string" ? __APP_VERSION__ : "dev"}`,
  endTimeoutMs: 3000,
};

export class SessionController {
  private source: SessionEventSource | null = null;
  private mock: MockEventSource | null = null;
  private lastStart: StartOptions | null = null;
  private readonly deps: ControllerDeps;
  private readonly sourceListeners = new Set<() => void>();
  private startToken = 0;

  constructor(
    private readonly store: SessionStore,
    deps: Partial<ControllerDeps> = {},
  ) {
    this.deps = { ...defaultDeps, ...deps };
    // When the server reports `ended`, release the transport (after the summary arrives).
    this.store.subscribe(() => {
      const s = this.store.getState();
      if (s.serverState === "ended" && this.source && s.phase === "ended") {
        const src = this.source;
        setTimeout(() => {
          if (this.source === src) void this.teardown();
        }, 250);
      }
    });
  }

  get canRetry(): boolean {
    return this.lastStart !== null;
  }

  /** Subscribe to source changes (used by the level meter to find the mic track). */
  onSourceChange(listener: () => void): () => void {
    this.sourceListeners.add(listener);
    return () => this.sourceListeners.delete(listener);
  }

  getLocalAudioTrack(): MediaStreamTrack | null {
    return this.source?.getLocalAudioTrack?.() ?? null;
  }

  async switchAudioInput(deviceId: string): Promise<void> {
    await this.source?.switchAudioInput?.(deviceId);
  }

  async retry(): Promise<void> {
    if (this.lastStart) await this.start(this.lastStart);
  }

  async start(opts: StartOptions): Promise<void> {
    if (this.source) await this.teardown();
    this.lastStart = opts;
    const token = ++this.startToken;
    const needsPermission = opts.transport === "livekit" && !opts.mock;
    this.store.dispatch({ type: "start_requested", transport: opts.transport, isMock: opts.mock, needsPermission });

    if (needsPermission) {
      try {
        await this.deps.checkMicrophone(opts.deviceId);
      } catch (err) {
        if (token === this.startToken) this.store.dispatch({ type: "local_error", error: microphoneError(err) });
        return;
      }
      if (token !== this.startToken) return;
      this.store.dispatch({ type: "permission_granted" });
    }

    if (opts.mock) {
      const mock = this.deps.createMockSource({
        detectorMode: opts.detectorMode,
        scenarioId: opts.scenarioId,
        policyOverrides: opts.policyOverrides,
        transcriptEnabled: opts.transcriptEnabled,
        transcriptExportEnabled: opts.transcriptExportEnabled,
        speculativePrefill: opts.speculativePrefill,
        trialLabel: opts.trialLabel,
        seed: opts.seed,
      });
      this.mock = mock;
      this.store.dispatch({ type: "session_created", response: mock.createSessionResponse() });
      await this.attach(mock, token);
      return;
    }

    let response;
    try {
      response = await this.deps.api.createSession({
        transport: opts.transport,
        detectorMode: opts.detectorMode,
        scenarioId: opts.scenarioId,
        policyOverrides: opts.policyOverrides,
        transcriptEnabled: opts.transcriptEnabled,
        transcriptExportEnabled: opts.transcriptExportEnabled,
        speculativePrefill: opts.speculativePrefill,
        trialLabel: opts.trialLabel,
        seed: opts.seed,
        clientBuild: this.deps.clientBuild,
      });
    } catch (err) {
      if (token !== this.startToken) return;
      this.store.dispatch({
        type: "local_error",
        error: { kind: "api", message: "The session could not be created.", detail: describeApiError(err), retryable: true, fatal: true },
      });
      return;
    }
    if (token !== this.startToken) return;
    this.store.dispatch({ type: "session_created", response });

    let source: SessionEventSource;
    if (opts.transport === "simulation") {
      if (!response.eventsUrl) {
        this.store.dispatch({
          type: "local_error",
          error: { kind: "protocol", message: "The server did not return an event stream for this session.", retryable: true, fatal: true },
        });
        return;
      }
      source = this.deps.createWebSocketSource(resolveWebSocketUrl(response.eventsUrl));
    } else {
      if (!response.livekit) {
        this.store.dispatch({
          type: "local_error",
          error: { kind: "protocol", message: "The server did not return LiveKit join details.", retryable: true, fatal: true },
        });
        return;
      }
      try {
        source = await this.deps.createLiveKitSource(response.livekit, opts.deviceId);
      } catch (err) {
        if (token !== this.startToken) return;
        this.store.dispatch({
          type: "local_error",
          error: {
            kind: "join_failed",
            message: "The live voice client could not be loaded.",
            detail: err instanceof Error ? err.message : String(err),
            retryable: true,
            fatal: true,
          },
        });
        return;
      }
      if (token !== this.startToken) return;
    }
    await this.attach(source, token);
  }

  private async attach(source: SessionEventSource, token: number): Promise<void> {
    this.source = source;
    this.sourceListeners.forEach((l) => l());
    try {
      await source.start({
        onMessage: (raw) => {
          if (this.source === source) this.store.ingest(raw);
        },
        onStatus: (status, detail) => this.onTransportStatus(source, status, detail ?? null),
      });
      this.sourceListeners.forEach((l) => l());
    } catch (err) {
      if (this.source !== source || token !== this.startToken) return;
      this.store.dispatch({
        type: "local_error",
        error: {
          kind: "join_failed",
          message: source.kind === "livekit" ? "Joining the voice session failed." : "Connecting to the session event stream failed.",
          detail: err instanceof Error ? err.message : String(err),
          retryable: true,
          fatal: true,
        },
      });
      await this.teardown();
    }
  }

  private onTransportStatus(source: SessionEventSource, status: TransportStatus, detail: string | null): void {
    if (source !== this.source) return;
    this.store.dispatch({ type: "transport_status", status, detail });
    if (status !== "disconnected" && status !== "error") return;
    const phase = this.store.getState().phase;
    this.source = null;
    this.mock = null;
    this.sourceListeners.forEach((l) => l());
    if (phase !== "ending" && phase !== "ended") {
      this.store.dispatch({
        type: "local_error",
        error: {
          kind: "transport",
          message: detail ?? "The connection to the session closed unexpectedly.",
          retryable: true,
          fatal: true,
        },
      });
    }
    this.store.dispatch({ type: "source_closed" });
    void source.stop();
  }

  async interrupt(): Promise<void> {
    const s = this.store.getState();
    if (!s.sessionId) return;
    this.store.dispatch({ type: "interrupt_requested" });
    try {
      if (this.mock) await this.mock.interrupt();
      else await this.deps.api.interruptSession(s.sessionId);
    } catch (err) {
      this.store.dispatch({ type: "interrupt_failed", message: `Interrupt was not accepted: ${describeApiError(err)}` });
    }
  }

  async end(): Promise<void> {
    const s = this.store.getState();
    this.startToken += 1; // abandon any in-flight start
    if (s.phase === "idle" || s.phase === "ended") return;
    this.store.dispatch({ type: "end_requested" });
    if (this.mock) {
      await this.mock.end();
    } else if (s.sessionId && this.source) {
      try {
        await this.deps.api.endSession(s.sessionId);
      } catch {
        // The local teardown below still stops the mic and playback.
      }
    }
    await this.waitForEnded(this.deps.endTimeoutMs);
    await this.teardown();
  }

  private waitForEnded(timeoutMs: number): Promise<void> {
    return new Promise((resolve) => {
      const done = () => {
        unsub();
        clearTimeout(timer);
        resolve();
      };
      const check = () => {
        const st = this.store.getState();
        if (st.serverState === "ended" || st.phase === "ended" || !this.source) done();
      };
      const unsub = this.store.subscribe(check);
      const timer = setTimeout(done, timeoutMs);
      check();
    });
  }

  /** Stop the transport (mic, playback, room/socket) and mark the session closed. */
  async teardown(): Promise<void> {
    const src = this.source;
    this.source = null;
    this.mock = null;
    if (src) {
      this.sourceListeners.forEach((l) => l());
      try {
        await src.stop();
      } catch {
        // best effort
      }
    }
    const phase = this.store.getState().phase;
    if (phase !== "idle" && phase !== "ended" && phase !== "error") this.store.dispatch({ type: "source_closed" });
    else if (phase === "ended" && src) this.store.dispatch({ type: "source_closed" });
  }

  clearView(): void {
    this.store.dispatch({ type: "clear_view" });
  }

  dismissError(): void {
    this.store.dispatch({ type: "dismiss_error" });
  }
}

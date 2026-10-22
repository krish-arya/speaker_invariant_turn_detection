import { describe, expect, it, vi } from "vitest";
import { createSessionResponseSchema } from "../api/schemas";
import type { SessionEventSource, SourceHandlers } from "../events/types";
import { createSessionResponseWire, hello, status } from "../test/fixtures";
import { testStore } from "../test/render";
import { SessionController, type ControllerDeps } from "./sessionController";

class FakeSource implements SessionEventSource {
  readonly kind = "websocket" as const;
  h: SourceHandlers | null = null;
  stopped = false;
  async start(h: SourceHandlers) {
    this.h = h;
    h.onStatus("connected");
  }
  async stop() {
    this.stopped = true;
  }
}

function deps(overrides: Partial<ControllerDeps> = {}): Partial<ControllerDeps> {
  return {
    api: {
      createSession: vi.fn(async () => createSessionResponseSchema.parse(createSessionResponseWire())),
      endSession: vi.fn(async () => ({ accepted: true })),
      interruptSession: vi.fn(async () => ({ accepted: true })),
    },
    checkMicrophone: vi.fn(async () => {}),
    endTimeoutMs: 50,
    ...overrides,
  };
}

const startOpts = {
  transport: "simulation" as const,
  detectorMode: "silence_baseline" as const,
  scenarioId: "pause_then_continue",
  policyOverrides: { silence_timeout_ms: 600 },
  transcriptEnabled: true,
  transcriptExportEnabled: false,
  speculativePrefill: false,
  trialLabel: null,
  seed: 7,
  deviceId: null,
  mock: false,
};

describe("SessionController", () => {
  it("creates a simulation session with wire-format overrides and connects the event stream", async () => {
    const store = testStore();
    const source = new FakeSource();
    const d = deps({ createWebSocketSource: vi.fn(() => source) });
    const c = new SessionController(store, d);
    await c.start(startOpts);
    expect(d.api!.createSession).toHaveBeenCalledWith(
      expect.objectContaining({ policyOverrides: { silence_timeout_ms: 600 }, detectorMode: "silence_baseline", seed: 7 }),
    );
    expect(d.createWebSocketSource).toHaveBeenCalledWith(expect.stringMatching(/^ws:\/\/.+\/ws\/sessions\/ses_test0001$/));
    source.h!.onMessage(JSON.stringify(hello(1)));
    source.h!.onMessage(JSON.stringify(status("listening", "connecting", 2)));
    store.flush();
    expect(store.getState().phase).toBe("listening");
    expect(store.getState().transportStatus).toBe("connected");
  });

  it("does not request the microphone for simulation, but does for live sessions", async () => {
    const store = testStore();
    const check = vi.fn(async () => {
      const e = new Error("denied");
      e.name = "NotAllowedError";
      throw e;
    });
    const d = deps({ checkMicrophone: check, createWebSocketSource: () => new FakeSource() });
    const c = new SessionController(store, d);
    await c.start(startOpts);
    expect(check).not.toHaveBeenCalled();
    await c.start({ ...startOpts, transport: "livekit" });
    expect(check).toHaveBeenCalledTimes(1);
    expect(store.getState().phase).toBe("error");
    expect(store.getState().localError?.kind).toBe("permission_denied");
    expect(d.api!.createSession).toHaveBeenCalledTimes(1); // no session created after denial
  });

  it("surfaces session creation failures as retryable errors", async () => {
    const store = testStore();
    const d = deps();
    d.api!.createSession = vi.fn(async () => {
      throw new Error("boom");
    });
    const c = new SessionController(store, d);
    await c.start(startOpts);
    expect(store.getState().phase).toBe("error");
    expect(store.getState().localError?.retryable).toBe(true);
  });

  it("end() calls the server, stops the transport and reaches ended", async () => {
    const store = testStore();
    const source = new FakeSource();
    const d = deps({ createWebSocketSource: () => source });
    const c = new SessionController(store, d);
    await c.start(startOpts);
    source.h!.onMessage(JSON.stringify(status("listening", "connecting", 1)));
    store.flush();
    const ending = c.end();
    expect(store.getState().phase).toBe("ending");
    source.h!.onMessage(JSON.stringify(status("ended", "ending", 2)));
    store.flush();
    await ending;
    expect(d.api!.endSession).toHaveBeenCalledWith("ses_test0001");
    expect(source.stopped).toBe(true);
    expect(store.getState().phase).toBe("ended");
  });

  it("an unexpected disconnect becomes a visible, retryable transport error", async () => {
    const store = testStore();
    const source = new FakeSource();
    const c = new SessionController(store, deps({ createWebSocketSource: () => source }));
    await c.start(startOpts);
    source.h!.onMessage(JSON.stringify(status("listening", "connecting", 1)));
    store.flush();
    source.h!.onStatus("disconnected", "socket dropped");
    const s = store.getState();
    expect(s.phase).toBe("error");
    expect(s.localError?.kind).toBe("transport");
    expect(s.localError?.message).toBe("socket dropped");
  });

  it("interrupt failures are reported without ending the session", async () => {
    const store = testStore();
    const source = new FakeSource();
    const d = deps({ createWebSocketSource: () => source });
    d.api!.interruptSession = vi.fn(async () => {
      throw new Error("409 nothing to interrupt");
    });
    const c = new SessionController(store, d);
    await c.start(startOpts);
    source.h!.onMessage(JSON.stringify(status("agent_speaking", "generating", 1)));
    store.flush();
    await c.interrupt();
    const s = store.getState();
    expect(s.phase).toBe("agent_speaking");
    expect(s.stopping).toBe(false);
    expect(s.localError?.kind).toBe("interrupt");
  });
});

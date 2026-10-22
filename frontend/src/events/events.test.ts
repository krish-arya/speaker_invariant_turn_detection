import { afterEach, describe, expect, it, vi } from "vitest";
import { parseEnvelope } from "../api/schemas";
import { filenameFromContentDisposition } from "../api/download";
import { resolveWebSocketUrl } from "../api/client";
import { MockEventSource } from "./mock";
import { WebSocketEventSource, WS_CLOSE_UNKNOWN_SESSION } from "./websocket";

function collect(source: MockEventSource) {
  const messages: unknown[] = [];
  const statuses: string[] = [];
  void source.start({ onMessage: (m) => messages.push(m), onStatus: (s) => statuses.push(s) });
  return { messages, statuses };
}

afterEach(() => {
  vi.useRealTimers();
});

describe("MockEventSource", () => {
  const scenarios = ["pause_then_continue", "short_answer", "multi_turn", "barge_in", "backchannel", "free_conversation"];

  it.each(scenarios)("emits only contract-valid envelopes for %s", (scenarioId) => {
    vi.useFakeTimers();
    const src = new MockEventSource({
      detectorMode: "anti_transfer",
      scenarioId,
      policyOverrides: null,
      transcriptEnabled: true,
      transcriptExportEnabled: false,
      trialLabel: null,
      seed: 3,
    });
    const { messages } = collect(src);
    src.tick(60_000);
    if (scenarioId === "free_conversation") void src.end();
    expect(messages.length).toBeGreaterThan(20);
    const parsed = messages.map((m) => parseEnvelope(m));
    const failures = parsed.filter((p) => !p.ok);
    expect(failures).toEqual([]);
    const seqs = parsed.map((p) => (p.ok ? p.envelope.seq : -1));
    expect(seqs).toEqual(seqs.map((_, i) => i + 1)); // strictly increasing, no gaps
    const kinds = new Set(parsed.map((p) => (p.ok ? p.envelope.kind : "")));
    expect(kinds.has("hello")).toBe(true);
    expect(kinds.has("trial_summary")).toBe(true);
    const summary = parsed.find((p) => p.ok && p.envelope.kind === "trial_summary");
    if (summary?.ok && summary.envelope.kind === "trial_summary") {
      expect(summary.envelope.payload.notes.join(" ")).toContain("MOCK DATA");
    }
    void src.stop();
  });

  it("baseline mock never fabricates confidence", () => {
    vi.useFakeTimers();
    const src = new MockEventSource({
      detectorMode: "silence_baseline",
      scenarioId: "pause_then_continue",
      policyOverrides: null,
      transcriptEnabled: false,
      transcriptExportEnabled: false,
      trialLabel: null,
      seed: 1,
    });
    const { messages } = collect(src);
    src.tick(60_000);
    const preds = messages.map((m) => parseEnvelope(m)).filter((p) => p.ok && p.envelope.kind === "turn_prediction");
    expect(preds.length).toBeGreaterThan(0);
    for (const p of preds) if (p.ok && p.envelope.kind === "turn_prediction") expect(p.envelope.payload.confidence).toBeNull();
    // transcripts disabled -> none emitted
    expect(messages.map((m) => parseEnvelope(m)).some((p) => p.ok && p.envelope.kind === "transcript")).toBe(false);
  });

  it("supports manual interrupt while the agent speaks and rejects it otherwise", async () => {
    vi.useFakeTimers();
    let now = 0;
    const src = new MockEventSource(
      {
        detectorMode: "anti_transfer",
        scenarioId: "short_answer",
        policyOverrides: null,
        transcriptEnabled: true,
        transcriptExportEnabled: false,
        trialLabel: null,
        seed: 2,
      },
      () => now,
    );
    const { messages } = collect(src);
    await expect(src.interrupt()).rejects.toThrow(/Nothing to interrupt/);
    // advance until agent speaking
    let speaking = false;
    while (!speaking && now < 20_000) {
      now += 50;
      src.tick();
      speaking = messages.some((m) => {
        const p = parseEnvelope(m);
        return p.ok && p.envelope.kind === "session_status" && p.envelope.payload.state === "agent_speaking";
      });
    }
    expect(speaking).toBe(true);
    await src.interrupt();
    now += 400;
    src.tick();
    const parsed = messages.map((m) => parseEnvelope(m));
    const barge = parsed.find((p) => p.ok && p.envelope.kind === "turn_decision" && p.envelope.payload.decision === "BARGE_IN");
    expect(barge).toBeTruthy();
    expect(parsed.some((p) => p.ok && p.envelope.kind === "playback" && p.envelope.payload.action === "flush")).toBe(true);
  });
});

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  readyState = 0;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }
  open() {
    this.readyState = 1;
    this.onopen?.();
  }
  send(d: string) {
    this.sent.push(d);
  }
  close(code = 1000) {
    this.readyState = 3;
    this.onclose?.({ code });
  }
  serverClose(code: number) {
    this.readyState = 3;
    this.onclose?.({ code });
  }
}

describe("WebSocketEventSource", () => {
  it("forwards messages and reports connection status", async () => {
    FakeWebSocket.instances = [];
    const msgs: unknown[] = [];
    const statuses: string[] = [];
    const src = new WebSocketEventSource("ws://x/ws/sessions/s", { WebSocketImpl: FakeWebSocket as unknown as typeof WebSocket });
    const started = src.start({ onMessage: (m) => msgs.push(m), onStatus: (s) => statuses.push(s) });
    const ws = FakeWebSocket.instances[0]!;
    ws.open();
    await started;
    ws.onmessage?.({ data: '{"a":1}' });
    expect(msgs).toEqual(['{"a":1}']);
    expect(statuses).toEqual(["connecting", "connected"]);
    await src.stop();
    expect(statuses[statuses.length - 1]).toBe("disconnected");
  });

  it("treats close code 4404 as a terminal error", async () => {
    FakeWebSocket.instances = [];
    const statuses: Array<[string, string | null | undefined]> = [];
    const src = new WebSocketEventSource("ws://x", { WebSocketImpl: FakeWebSocket as unknown as typeof WebSocket });
    const started = src.start({ onMessage: () => {}, onStatus: (s, d) => statuses.push([s, d]) });
    FakeWebSocket.instances[0]!.open();
    await started;
    FakeWebSocket.instances[0]!.serverClose(WS_CLOSE_UNKNOWN_SESSION);
    expect(statuses[statuses.length - 1]?.[0]).toBe("error");
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("reconnects after an abnormal close", async () => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    const statuses: string[] = [];
    const src = new WebSocketEventSource("ws://x", { WebSocketImpl: FakeWebSocket as unknown as typeof WebSocket, maxReconnects: 2 });
    const started = src.start({ onMessage: () => {}, onStatus: (s) => statuses.push(s) });
    FakeWebSocket.instances[0]!.open();
    await started;
    FakeWebSocket.instances[0]!.serverClose(1006);
    expect(statuses).toContain("reconnecting");
    vi.advanceTimersByTime(600);
    expect(FakeWebSocket.instances).toHaveLength(2);
    FakeWebSocket.instances[1]!.open();
    expect(statuses[statuses.length - 1]).toBe("connected");
    await src.stop();
  });

  it("rejects start when the first connection fails", async () => {
    FakeWebSocket.instances = [];
    const src = new WebSocketEventSource("ws://x", { WebSocketImpl: FakeWebSocket as unknown as typeof WebSocket });
    const started = src.start({ onMessage: () => {}, onStatus: () => {} });
    FakeWebSocket.instances[0]!.serverClose(1006);
    await expect(started).rejects.toThrow(/Could not open/);
  });
});

describe("helpers", () => {
  it("parses Content-Disposition filenames safely", () => {
    expect(filenameFromContentDisposition('attachment; filename="trial_1.csv"', "x")).toBe("trial_1.csv");
    expect(filenameFromContentDisposition("attachment; filename*=UTF-8''comparison%20v1.md", "x")).toBe("comparison v1.md");
    expect(filenameFromContentDisposition('attachment; filename="../../etc/passwd"', "x")).toBe(".._.._etc_passwd");
    expect(filenameFromContentDisposition(null, "fallback.json")).toBe("fallback.json");
  });

  it("resolves the simulation events_url to a ws:// URL", () => {
    expect(resolveWebSocketUrl("/ws/sessions/abc", "http://localhost:8000")).toBe("ws://localhost:8000/ws/sessions/abc");
    expect(resolveWebSocketUrl("/ws/sessions/abc", "https://example.org")).toBe("wss://example.org/ws/sessions/abc");
    expect(resolveWebSocketUrl("wss://other/ws/x")).toBe("wss://other/ws/x");
  });
});

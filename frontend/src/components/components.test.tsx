import { act, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { formatMs, formatRate, formatRelative } from "../lib/format";
import { decision, prediction, status, wire } from "../test/fixtures";
import { renderWithApp } from "../test/render";
import { aggregateText, LatencyCard } from "./LatencyOverview";
import { Timeline, TIMELINE_EMPTY_TEXT } from "./Timeline";

describe("display formatting", () => {
  it("renders unavailable values as an em dash, never as 0", () => {
    expect(formatMs(null)).toBe("—");
    expect(formatMs(undefined)).toBe("—");
    expect(formatMs(Number.NaN)).toBe("—");
    expect(formatRate(null)).toBe("—");
    expect(formatMs(0)).toBe("0.00 ms");
    expect(formatMs(7.4)).toBe("7.40 ms");
    expect(formatMs(1120)).toBe("1.12 s");
    expect(formatRelative(4280)).toBe("+4.28 s");
  });
});

describe("latency display rules", () => {
  it("shows 'Collecting samples' until the server provides percentiles", () => {
    expect(aggregateText(undefined, 20)).toBe("Collecting samples (0/20)");
    expect(
      aggregateText({ count: 3, failed: 0, latestMs: 900, p50Ms: null, p95Ms: null, p99Ms: null, minSamples: 20 }, 20),
    ).toBe("Collecting samples (3/20)");
    // Even if a server sent values early, respect the sample threshold.
    expect(
      aggregateText({ count: 5, failed: 0, latestMs: 900, p50Ms: 800, p95Ms: 950, p99Ms: 990, minSamples: 20 }, 20),
    ).toBe("Collecting samples (5/20)");
    expect(
      aggregateText({ count: 40, failed: 2, latestMs: 900, p50Ms: 800, p95Ms: 1450, p99Ms: 1990, minSamples: 20 }, 20),
    ).toBe("p50 800 ms · p95 1.45 s · p99 1.99 s");
  });

  it("renders an empty card with — and the collecting label", () => {
    renderWithApp(<LatencyCard metric="response_start" latest={undefined} summary={undefined} minSamples={20} labelledSession={null} />);
    const card = screen.getByRole("group", { name: "Response start" });
    expect(within(card).getByText("—")).toBeInTheDocument();
    expect(within(card).getByText("no sample yet")).toBeInTheDocument();
    expect(within(card).getByText("Collecting samples (0/20)")).toBeInTheDocument();
    expect(within(card).queryByText(/0 ms/)).not.toBeInTheDocument();
  });

  it("labels the latest value and counts failed samples separately", () => {
    renderWithApp(
      <LatencyCard
        metric="response_start"
        latest={{ valueMs: 1120, status: "ok", tMs: 10, generationId: 1 }}
        summary={{ count: 4, failed: 1, latestMs: 1120, p50Ms: null, p95Ms: null, p99Ms: null, minSamples: 20 }}
        minSamples={20}
        labelledSession
      />,
    );
    expect(screen.getByText("1.12 s")).toBeInTheDocument();
    expect(screen.getByText("latest event")).toBeInTheDocument();
    expect(screen.getByText("Collecting samples (4/20)")).toBeInTheDocument();
    expect(screen.getByText("1 failed — excluded from percentiles")).toBeInTheDocument();
  });

  it("marks turn-end delay unavailable for unlabelled sessions", () => {
    renderWithApp(<LatencyCard metric="turn_end_delay" latest={undefined} summary={undefined} minSamples={20} labelledSession={false} />);
    expect(screen.getByText(/Not available for this session/)).toBeInTheDocument();
  });
});

describe("Timeline", () => {
  it("shows the required empty state before a session", () => {
    renderWithApp(<Timeline />);
    expect(screen.getByText(TIMELINE_EMPTY_TEXT)).toBeInTheDocument();
    expect(TIMELINE_EMPTY_TEXT).toBe("Start a session to see live turn decisions and pipeline events.");
  });

  it("renders predictions and committed decisions distinctly, with confidence only when present", () => {
    const { store } = renderWithApp(<Timeline />);
    act(() => {
      store.ingest(status("listening", "connecting", 1));
      store.ingest(prediction("POSSIBLE_END", 0.72, 2));
      store.ingest(prediction("TURN_COMPLETE", null, 3));
      store.ingest(decision("TURN_COMPLETE", 4, 1));
      store.flush();
    });
    expect(screen.getByText("Predicted: Possible turn end")).toBeInTheDocument();
    expect(screen.getByText("conf 0.72")).toBeInTheDocument();
    expect(screen.getByText("Committed: Turn complete")).toBeInTheDocument();
    expect(screen.getAllByText("Prediction")).toHaveLength(2);
    expect(screen.getByText("Committed")).toBeInTheDocument();
    // The null-confidence prediction must not show a confidence chip.
    expect(screen.getAllByText(/^conf /)).toHaveLength(2); // 0.72 prediction + 0.88 decision
  });

  it("renders a gap marker for missing sequence numbers", () => {
    const { store } = renderWithApp(<Timeline />);
    act(() => {
      store.ingest(status("listening", "connecting", 1));
      store.ingest(wire("vad", { speaking: true, probability: null, frame_end_ms: 40 }, 4));
      store.flush();
    });
    expect(screen.getByText("2 messages missing")).toBeInTheDocument();
    expect(screen.getByText("Speech detected")).toBeInTheDocument();
  });
});

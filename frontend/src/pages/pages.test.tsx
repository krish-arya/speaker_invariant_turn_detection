import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { effectiveConfigWire } from "../test/fixtures";
import { renderWithApp } from "../test/render";
import { EvaluationPage, type EvaluationRun } from "./EvaluationPage";
import { ResultsPage } from "./ResultsPage";

const metrics = (over: Record<string, unknown> = {}) => ({
  turn_precision: 1,
  turn_recall: 0.5,
  turn_f1: 0.667,
  premature_end_rate: 0,
  missed_end_rate: 0.5,
  turn_end_delay_p50_ms: 410,
  turn_end_delay_p95_ms: 690,
  response_start_p50_ms: 1010,
  response_start_p95_ms: 1300,
  response_start_p99_ms: 1390,
  detector_inference_p50_ms: 4.2,
  detector_inference_p95_ms: 6.1,
  detector_inference_p99_ms: 8,
  false_barge_in_rate: 0,
  missed_barge_in_rate: null,
  barge_in_cancel_p50_ms: null,
  stale_chunks_played: 0,
  stale_chunks_rejected: 0,
  dropped_frames: 0,
  frame_trace_coverage: 1,
  ...over,
});

const counts = { frames: 400, reference_turn_ends: 2, committed_turn_ends: 1, reference_barge_ins: 0, committed_barge_ins: 0, agent_speaking_intervals: 1, responses_started: 1, responses_cancelled: 0, errors: 0 };

const trial = (id: string, mode: string, over: Record<string, unknown> = {}) => ({
  trial_id: id,
  session_id: `ses_${id}`,
  evaluation_id: "eva_1",
  created_at: "2026-10-22T08:00:00Z",
  ended_at: "2026-10-22T08:00:10Z",
  trial_type: "simulated",
  transport: "simulation",
  scenario_id: "pause_then_continue",
  trial_label: null,
  detector_mode: mode,
  model_version: `${mode}-0.1`,
  decision_policy_version: "policy-0.1",
  status: "completed",
  labelled: true,
  metrics: metrics(),
  counts,
  ...over,
});

const comparison = {
  group_by: ["detector_mode"],
  groups: [
    { key: { detector_mode: "silence_baseline" }, n_trials: 2, scenario_ids: ["pause_then_continue"], trial_types: ["simulated"], metrics: { turn_f1: { mean: 0.6, ci95_low: 0.4, ci95_high: 0.8, n: 2 } } },
    { key: { detector_mode: "anti_transfer" }, n_trials: 1, scenario_ids: ["pause_then_continue"], trial_types: ["simulated"], metrics: { turn_f1: { mean: 0.7, ci95_low: null, ci95_high: null, n: 1 } } },
  ],
  metric_definitions: { turn_f1: { label: "Turn F1", better: "higher", unit: "ratio", definition: "Harmonic mean." } },
  evaluation_set: "sim-scenarios-v1",
  warnings: ["Small sample."],
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("ResultsPage", () => {
  it("lists trials, hides accuracy for unlabelled trials, and compares a selection without ranking", async () => {
    const items = [
      trial("tri_a", "silence_baseline"),
      trial("tri_b", "anti_transfer"),
      trial("tri_c", "silence_baseline", { trial_type: "live", labelled: false, scenario_id: "free_conversation", metrics: metrics({ turn_f1: null }) }),
    ];
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.startsWith("/api/trials?")) return json({ total: items.length, items });
      if (url.startsWith("/api/compare")) return json(comparison);
      return json({ detail: "not found" }, 404);
    });
    renderWithApp(<ResultsPage />);
    await screen.findByText("1–3 of 3");
    const table = screen.getByRole("region", { name: "Trial records table" });
    expect(within(table).getAllByText("Not available for this session").length).toBeGreaterThan(0);
    expect(within(table).getByText("Live (human session)")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Select trial tri_a"));
    fireEvent.click(screen.getByLabelText("Select trial tri_b"));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Compare selected" }));
    });
    expect(await screen.findByText("sim-scenarios-v1")).toBeInTheDocument();
    expect(screen.getByText("Small sample.")).toBeInTheDocument();
    expect(screen.getAllByText("↑ higher is better").length).toBeGreaterThan(0);
    expect(screen.getByText("CI unavailable", { exact: false })).toBeInTheDocument();
    expect(screen.queryByText(/best/i)).not.toBeInTheDocument();
    const call = fetchMock.mock.calls.find(([u]) => String(u).startsWith("/api/compare"));
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ trial_ids: ["tri_a", "tri_b"], group_by: ["detector_mode"] });
  });
});

describe("EvaluationPage", () => {
  it("starts an evaluation with snake_case body and polls progress", async () => {
    let polls = 0;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url === "/api/evaluations" && init?.method === "POST") {
        return json({ evaluation_id: "eva_1", status: "running", total: 4, completed: 0, trial_ids: [], error: null, created_at: "2026-10-22T08:00:00Z" }, 202);
      }
      if (url === "/api/evaluations/eva_1") {
        polls += 1;
        const completed = Math.min(4, polls * 2);
        return json({ evaluation_id: "eva_1", status: completed === 4 ? "completed" : "running", total: 4, completed, trial_ids: [], error: null, created_at: "2026-10-22T08:00:00Z" });
      }
      if (url.startsWith("/api/trials?")) return json({ total: 1, items: [trial("tri_a", "silence_baseline")] });
      if (url.startsWith("/api/trials/tri_a")) return json({ ...trial("tri_a", "silence_baseline"), effective_config: effectiveConfigWire({ detector_mode: "silence_baseline" }), annotations: null, tolerance_ms: null, events: [] });
      return json({ detail: "not found" }, 404);
    });

    const draft = { detectorMode: null, scenarioId: null, transport: "simulation" as const, policy: {}, seed: "", trialLabel: "", speculativePrefill: false };
    function Harness() {
      const [run, setRun] = useState<EvaluationRun | null>(null);
      return <EvaluationPage run={run} onRunChange={setRun} consoleDraft={draft} />;
    }
    renderWithApp(<Harness />);
    // Default selection: two detectors × one scenario × 3 repeats.
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Start 6 trials/ }));
    });
    const post = fetchMock.mock.calls.find(([u, i]) => String(u) === "/api/evaluations" && i?.method === "POST");
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({
      detector_modes: ["silence_baseline", "learned_control"],
      scenario_ids: ["pause_then_continue"],
      repeats: 3,
      seed: 0,
      policy_overrides: null,
      speculative_prefill: false,
      trial_label: null,
    });
    await waitFor(() => expect(screen.getByText("4 of 4 trials completed")).toBeInTheDocument(), { timeout: 5000 });
    expect(await screen.findByRole("region", { name: "Trials in this evaluation" })).toBeInTheDocument();
  }, 10000);
});

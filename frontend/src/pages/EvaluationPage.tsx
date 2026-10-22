import { useEffect, useId, useRef, useState } from "react";
import { api, describeApiError, type CreateEvaluationInput } from "../api/client";
import type { ComparisonResult, DetectorMode, EvaluationStatus, TrialDetail, TrialListItem } from "../api/schemas";
import { useApp } from "../app/context";
import { Badge, Card, ErrorNotice, InfoNotice } from "../components/common";
import { ComparisonView } from "../components/ComparisonView";
import type { ConsoleDraft } from "../components/ConfigBar";
import { EffectiveConfigTable } from "../components/TrialDetailDrawer";
import { TrialDetailDrawer } from "../components/TrialDetailDrawer";
import { TrialsTable } from "../components/TrialsTable";
import { DETECTOR_LABELS } from "../lib/labels";
import { toWireOverrides } from "../lib/policy";

export interface EvaluationRun {
  request: CreateEvaluationInput;
  status: EvaluationStatus;
}

const ACTIVE = ["queued", "running"];

export function EvaluationPage({
  run,
  onRunChange,
  consoleDraft,
}: {
  run: EvaluationRun | null;
  onRunChange(run: EvaluationRun | null): void;
  consoleDraft: ConsoleDraft;
}) {
  const { caps, mock } = useApp();
  const ids = useId();
  const detectors = caps.detectors.filter((d) => d.available);
  const scenarios = caps.scenarios.filter((s) => s.transports.includes("simulation") && s.trialType !== "live");
  const [modes, setModes] = useState<Set<DetectorMode>>(() => new Set(detectors.slice(0, 2).map((d) => d.mode)));
  const [scenarioIds, setScenarioIds] = useState<Set<string>>(() => new Set(scenarios.slice(0, 1).map((s) => s.id)));
  const [repeats, setRepeats] = useState(3);
  const [seed, setSeed] = useState(0);
  const [label, setLabel] = useState("");
  const [useConsolePolicy, setUseConsolePolicy] = useState(false);
  const [speculativePrefill, setSpeculativePrefill] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [trials, setTrials] = useState<TrialListItem[]>([]);
  const [configs, setConfigs] = useState<Record<string, TrialDetail>>({});
  const [openTrial, setOpenTrial] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const lastCompleted = useRef(-1);

  const maxRepeats = caps.maxEvaluationRepeats;
  const policyOverrides = useConsolePolicy ? toWireOverrides(consoleDraft.policy, caps.policyDefaults) : null;
  const changedCount = policyOverrides ? Object.keys(policyOverrides).length : 0;
  const total = modes.size * scenarioIds.size * repeats;
  const formError =
    modes.size === 0
      ? "Select at least one detector."
      : scenarioIds.size === 0
        ? "Select at least one scenario."
        : !Number.isInteger(repeats) || repeats < 1 || repeats > maxRepeats
          ? `Repeats must be between 1 and ${maxRepeats}.`
          : !Number.isInteger(seed)
            ? "Seed must be a whole number."
            : null;
  const active = run !== null && ACTIVE.includes(run.status.status);

  // Poll the running evaluation about once a second; refresh trials as they complete.
  useEffect(() => {
    if (!run) return;
    let cancelled = false;
    const evaluationId = run.status.evaluationId;
    const refreshTrials = async () => {
      const list = await api.listTrials({ evaluationId, limit: 200 });
      if (!cancelled) setTrials(list.items);
    };
    if (!ACTIVE.includes(run.status.status)) {
      void refreshTrials().catch((e: unknown) => !cancelled && setPollError(describeApiError(e)));
      return () => {
        cancelled = true;
      };
    }
    const timer = setInterval(() => {
      api
        .getEvaluation(evaluationId)
        .then(async (status) => {
          if (cancelled) return;
          setPollError(null);
          if (status.completed !== lastCompleted.current) {
            lastCompleted.current = status.completed;
            await refreshTrials();
          }
          onRunChange({ ...run, status });
        })
        .catch((e: unknown) => !cancelled && setPollError(describeApiError(e)));
    }, 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [run, onRunChange]);

  // Fetch the server-accepted configuration once per detector mode.
  useEffect(() => {
    const missing = new Map<string, string>();
    for (const t of trials) if (!configs[t.detectorMode] && !missing.has(t.detectorMode)) missing.set(t.detectorMode, t.trialId);
    if (missing.size === 0) return;
    let cancelled = false;
    Promise.all([...missing.values()].map((id) => api.getTrial(id).catch(() => null))).then((details) => {
      if (cancelled) return;
      setConfigs((cur) => {
        const next = { ...cur };
        details.forEach((d) => d && (next[d.detectorMode] = d));
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [trials, configs]);

  const submit = async (request: CreateEvaluationInput) => {
    setSubmitting(true);
    setSubmitError(null);
    setComparison(null);
    setTrials([]);
    setConfigs({});
    lastCompleted.current = -1;
    try {
      const status = await api.createEvaluation(request);
      onRunChange({ request, status });
    } catch (e) {
      setSubmitError(describeApiError(e));
    } finally {
      setSubmitting(false);
    }
  };

  const start = () => {
    if (formError) return;
    void submit({
      detectorModes: [...modes],
      scenarioIds: [...scenarioIds],
      repeats,
      seed,
      policyOverrides,
      speculativePrefill,
      trialLabel: label.trim() || null,
    });
  };

  const stop = async () => {
    if (!run) return;
    try {
      const status = await api.stopEvaluation(run.status.evaluationId);
      onRunChange({ ...run, status });
    } catch (e) {
      setPollError(describeApiError(e));
    }
  };

  const compare = async () => {
    if (!run) return;
    setCompareError(null);
    try {
      setComparison(await api.compare({ evaluationId: run.status.evaluationId }, ["detector_mode"]));
    } catch (e) {
      setCompareError(describeApiError(e));
    }
  };

  const exportTrial = async (id: string, format: "json" | "csv") => {
    setExportError(null);
    try {
      await api.exportTrial(id, format);
    } catch (e) {
      setExportError(describeApiError(e));
    }
  };

  if (mock) {
    return (
      <div className="page">
        <InfoNotice tone="uncertain">
          Evaluation runs on the server and is unavailable in mock mode. Start the backend and reload without{" "}
          <span className="mono">?mock=1</span> to run controlled trials.
        </InfoNotice>
      </div>
    );
  }

  const toggle = <T,>(set: Set<T>, v: T): Set<T> => {
    const next = new Set(set);
    if (next.has(v)) next.delete(v);
    else next.add(v);
    return next;
  };

  const perMode = new Map<string, number>();
  trials.forEach((t) => perMode.set(t.detectorMode, (perMode.get(t.detectorMode) ?? 0) + 1));

  return (
    <div className="page">
      <Card title="Run controlled trials" labelledBy={`${ids}-form`}>
        <p className="field-help">
          Trials run server-side on scripted, labelled scenarios — simulated audio, not human live sessions. Every variant
          uses the same scenarios, repeats and seed so the comparison is like-for-like.
        </p>
        <div className="eval-form">
          <fieldset className="field" disabled={active}>
            <legend>Detector variants</legend>
            {caps.detectors.map((d) => (
              <label key={d.mode} className="check">
                <input
                  type="checkbox"
                  checked={modes.has(d.mode)}
                  disabled={!d.available}
                  onChange={() => setModes((m) => toggle(m, d.mode))}
                />
                {d.label || DETECTOR_LABELS[d.mode]} <span className="mono small muted">{d.modelVersion}</span>
                {!d.available && <span className="small muted"> — unavailable: {d.unavailableReason ?? "not registered"}</span>}
              </label>
            ))}
          </fieldset>
          <fieldset className="field" disabled={active}>
            <legend>Scenarios</legend>
            {scenarios.length === 0 && <p className="muted">The server reports no scripted scenarios.</p>}
            {scenarios.map((s) => (
              <label key={s.id} className="check">
                <input type="checkbox" checked={scenarioIds.has(s.id)} onChange={() => setScenarioIds((cur) => toggle(cur, s.id))} />
                {s.label} <span className="small muted">({s.labelled ? "labelled" : "unlabelled"}, {s.trialType})</span>
              </label>
            ))}
          </fieldset>
          <div className="field-col">
            <div className="field">
              <label htmlFor={`${ids}-rep`}>Repeats per condition</label>
              <input
                id={`${ids}-rep`}
                type="number"
                min={1}
                max={maxRepeats}
                step={1}
                value={repeats}
                disabled={active}
                onChange={(e) => setRepeats(Number(e.target.value))}
              />
              <p className="field-help">1–{maxRepeats}. Fewer than 3 gives wide or no confidence intervals.</p>
            </div>
            <div className="field">
              <label htmlFor={`${ids}-seed`}>Seed</label>
              <input id={`${ids}-seed`} type="number" step={1} value={seed} disabled={active} onChange={(e) => setSeed(Number(e.target.value))} />
            </div>
            <div className="field">
              <label htmlFor={`${ids}-label`}>Trial label (optional)</label>
              <input id={`${ids}-label`} type="text" maxLength={120} value={label} disabled={active} onChange={(e) => setLabel(e.target.value)} />
            </div>
            <label className="check">
              <input type="checkbox" checked={useConsolePolicy} disabled={active} onChange={(e) => setUseConsolePolicy(e.target.checked)} />
              Apply the Live Console advanced thresholds ({changedCount || "no"} changed)
            </label>
            <label className="check">
              <input type="checkbox" checked={speculativePrefill} disabled={active} onChange={(e) => setSpeculativePrefill(e.target.checked)} />
              Speculative prefill (same setting for every variant)
            </label>
          </div>
        </div>
        <div className="form-actions">
          <button type="button" className="btn btn-primary" disabled={!!formError || active || submitting} onClick={start}>
            {submitting ? "Starting…" : `Start ${total || ""} trial${total === 1 ? "" : "s"}`}
          </button>
          {active && (
            <button type="button" className="btn btn-danger" onClick={() => void stop()}>
              Stop
            </button>
          )}
          {run && !active && (
            <button type="button" className="btn btn-ghost" onClick={() => void submit(run.request)}>
              Retry with the same settings
            </button>
          )}
          {formError && <span className="field-error">{formError}</span>}
          {repeats < 3 && !formError && <span className="field-help">Small sample: fewer than 3 repeats per condition.</span>}
        </div>
        {submitError && <ErrorNotice title="The evaluation could not be started">{submitError}</ErrorNotice>}
      </Card>

      {run && (
        <Card
          title="Evaluation progress"
          labelledBy={`${ids}-prog`}
          actions={
            <Badge tone={run.status.status === "completed" ? "complete" : run.status.status === "error" ? "danger" : run.status.status === "stopped" ? "uncertain" : "active"}>
              {run.status.status}
            </Badge>
          }
        >
          <div className="progress-row">
            <progress max={Math.max(1, run.status.total)} value={run.status.completed} aria-labelledby={`${ids}-ptext`} />
            <span id={`${ids}-ptext`}>
              {run.status.completed} of {run.status.total} trials completed
            </span>
            <span className="mono small muted">{run.status.evaluationId}</span>
          </div>
          {run.status.error && <ErrorNotice title="Evaluation error">{run.status.error}</ErrorNotice>}
          {pollError && <InfoNotice tone="uncertain">Status update failed: {pollError}</InfoNotice>}
          <p className="small">
            Requested: {run.request.detectorModes.map((m) => DETECTOR_LABELS[m]).join(", ")} × {run.request.scenarioIds.join(", ")} ×{" "}
            {run.request.repeats} repeats · seed {run.request.seed}
            {run.request.policyOverrides ? ` · overrides ${JSON.stringify(run.request.policyOverrides)}` : " · server default policy"}
            {` · speculative prefill ${run.request.speculativePrefill ? "on" : "off"}`}
          </p>
          {[...perMode.entries()].some(([, n]) => n < 3) && (
            <InfoNotice tone="uncertain">Small sample: some variants have fewer than 3 completed trials.</InfoNotice>
          )}

          {Object.keys(configs).length > 0 && (
            <details className="config-details">
              <summary>Effective configuration (server-accepted, per detector)</summary>
              {Object.values(configs).map((d) => (
                <div key={d.detectorMode}>
                  <h3>{DETECTOR_LABELS[d.detectorMode]}</h3>
                  <EffectiveConfigTable config={d.effectiveConfig} />
                </div>
              ))}
            </details>
          )}

          {exportError && <p className="field-error">{exportError}</p>}
          {trials.length > 0 ? (
            <TrialsTable caption="Trials in this evaluation" trials={trials} onOpen={setOpenTrial} onExport={(id, f) => void exportTrial(id, f)} />
          ) : (
            <p className="muted">No completed trials yet.</p>
          )}
          <div className="form-actions">
            <button type="button" className="btn btn-ghost" disabled={trials.length === 0} onClick={() => void compare()}>
              Compare detector variants
            </button>
            {compareError && <span className="field-error">{compareError}</span>}
          </div>
        </Card>
      )}

      {comparison && run && (
        <ComparisonView
          result={comparison}
          trials={trials}
          exportError={exportError}
          onExport={(format) => {
            setExportError(null);
            api.exportComparison({ evaluationId: run.status.evaluationId }, ["detector_mode"], format).catch((e: unknown) => setExportError(describeApiError(e)));
          }}
        />
      )}

      {openTrial && <TrialDetailDrawer trialId={openTrial} onClose={() => setOpenTrial(null)} />}
    </div>
  );
}

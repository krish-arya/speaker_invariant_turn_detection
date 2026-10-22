import { useId, useState } from "react";
import { api, describeApiError } from "../api/client";
import type { TrialCounts, TrialMetrics } from "../api/schemas";
import { useApp } from "../app/context";
import { formatMs, formatRate, formatScore } from "../lib/format";
import { DETECTOR_LABELS, TRIAL_TYPE_LABELS } from "../lib/labels";
import { useSession } from "../state/sessionStore";
import { Badge, Card, MockBanner } from "./common";
import { Icon } from "./Icon";

const NA = "Not available for this session";

export function TrialMetricsTable({
  metrics: m,
  counts: c,
  labelled: lab,
  toleranceMs,
}: {
  metrics: TrialMetrics;
  counts: TrialCounts;
  labelled: boolean;
  toleranceMs: Record<string, number> | null;
}) {
  const rows: Array<[string, string, string]> = [
    ["Turn precision / recall / F1", lab ? `${formatScore(m.turnPrecision)} / ${formatScore(m.turnRecall)} / ${formatScore(m.turnF1)}` : NA, "higher"],
    ["Premature-end rate", lab ? formatRate(m.prematureEndRate) : NA, "lower"],
    ["Missed-end rate", lab ? formatRate(m.missedEndRate) : NA, "lower"],
    ["Turn-end delay p50 / p95", lab ? `${formatMs(m.turnEndDelayP50Ms)} / ${formatMs(m.turnEndDelayP95Ms)}` : NA, "lower"],
    ["Response start p50 / p95 / p99", `${formatMs(m.responseStartP50Ms)} / ${formatMs(m.responseStartP95Ms)} / ${formatMs(m.responseStartP99Ms)}`, "lower"],
    ["Detector inference p50 / p95 / p99", `${formatMs(m.detectorInferenceP50Ms)} / ${formatMs(m.detectorInferenceP95Ms)} / ${formatMs(m.detectorInferenceP99Ms)}`, "lower"],
    ["False barge-in rate", lab ? formatRate(m.falseBargeInRate) : NA, "lower"],
    ["Missed barge-in rate", lab ? formatRate(m.missedBargeInRate) : NA, "lower"],
    ["Barge-in cancellation p50", formatMs(m.bargeInCancelP50Ms), "lower"],
    ["Stale chunks played / rejected", `${m.staleChunksPlayed} / ${m.staleChunksRejected}`, "played: lower"],
    ["Dropped frames", String(m.droppedFrames), "lower"],
    ["Frame trace coverage", formatRate(m.frameTraceCoverage), "higher"],
  ];
  return (
    <>
      <table className="kv-table">
        <caption className="sr-only">Trial metrics</caption>
        <thead>
          <tr>
            <th scope="col">Metric</th>
            <th scope="col">Value</th>
            <th scope="col">Better</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, v, b]) => (
            <tr key={k}>
              <th scope="row">{k}</th>
              <td className={v === NA ? "muted" : "mono"}>{v}</td>
              <td className="muted">{b}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="field-help">
        Counts: {c.frames} frames · {c.referenceTurnEnds} annotated turn ends · {c.committedTurnEnds} committed ·{" "}
        {c.referenceBargeIns} annotated barge-ins · {c.committedBargeIns} committed · {c.responsesStarted} responses started ·{" "}
        {c.responsesCancelled} cancelled · {c.errors} errors
        {toleranceMs &&
          ` · tolerance ${Object.entries(toleranceMs)
            .map(([k, v]) => `${k} ${v} ms`)
            .join(", ")}`}
      </p>
    </>
  );
}

export function TrialSummaryCard() {
  const summary = useSession((s) => s.trialSummary);
  const isMock = useSession((s) => s.isMock);
  const { goTo } = useApp();
  const [exportError, setExportError] = useState<string | null>(null);
  const ids = useId();
  if (!summary) return null;

  const doExport = async (format: "json" | "csv") => {
    setExportError(null);
    try {
      await api.exportTrial(summary.trialId, format);
    } catch (e) {
      setExportError(describeApiError(e));
    }
  };

  return (
    <Card
      title="Completed trial"
      className="summary-card"
      labelledBy={`${ids}-title`}
      actions={
        !isMock && (
          <>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => void doExport("json")}>
              <Icon name="download" size={13} /> JSON
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => void doExport("csv")}>
              <Icon name="download" size={13} /> CSV
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => goTo("results")}>
              Open Results
            </button>
          </>
        )
      }
    >
      {isMock && <MockBanner />}
      <div className="badge-row">
        <Badge tone={summary.trialType === "live" ? "agent" : "neutral"}>{TRIAL_TYPE_LABELS[summary.trialType]}</Badge>
        <Badge tone={summary.labelled ? "complete" : "neutral"} icon={summary.labelled ? "check" : "info"}>
          {summary.labelled ? "Labelled" : "Unlabelled — no task accuracy"}
        </Badge>
        <Badge>{DETECTOR_LABELS[summary.detectorMode]}</Badge>
        <span className="mono small">
          {summary.modelVersion} · {summary.decisionPolicyVersion} · {summary.scenarioId} · status {summary.status}
        </span>
      </div>
      <TrialMetricsTable
        metrics={summary.metrics}
        counts={summary.counts}
        labelled={summary.labelled}
        toleranceMs={summary.toleranceMs}
      />
      {summary.notes.length > 0 && (
        <ul className="notes">
          {summary.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
      {exportError && <p className="field-error">{exportError}</p>}
    </Card>
  );
}

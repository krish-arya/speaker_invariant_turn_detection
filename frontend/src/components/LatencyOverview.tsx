import { useId } from "react";
import type { LatencyMetric, PercentileSummary } from "../api/schemas";
import { useApp } from "../app/context";
import { DASH, formatMs } from "../lib/format";
import { LATENCY_DEFS, PRIMARY_LATENCY_METRICS } from "../lib/labels";
import type { LatestSample } from "../state/sessionReducer";
import { useSession } from "../state/sessionStore";
import { Card } from "./common";
import { InfoPopover } from "./InfoPopover";

/** Aggregate line: percentiles only once the server reports enough valid samples. */
export function aggregateText(summary: PercentileSummary | undefined, minSamples: number): string {
  if (!summary) return `Collecting samples (0/${minSamples})`;
  const min = summary.minSamples || minSamples;
  if (summary.count < min || summary.p50Ms === null) return `Collecting samples (${summary.count}/${min})`;
  return `p50 ${formatMs(summary.p50Ms)} · p95 ${formatMs(summary.p95Ms)} · p99 ${formatMs(summary.p99Ms)}`;
}

export function LatencyCard({
  metric,
  latest,
  summary,
  minSamples,
  labelledSession,
}: {
  metric: LatencyMetric;
  latest: LatestSample | undefined;
  summary: PercentileSummary | undefined;
  minSamples: number;
  labelledSession: boolean | null;
}) {
  const def = LATENCY_DEFS[metric];
  const needsLabels = metric === "turn_end_delay";
  const unavailable = needsLabels && labelledSession === false;
  const latestValue = latest ? formatMs(latest.valueMs) : DASH;
  const latestFailed = latest && latest.status !== "ok";

  return (
    <div className="latency-card" role="group" aria-label={def.label}>
      <div className="latency-head">
        <h3 className="latency-title">{def.label}</h3>
        <InfoPopover label={def.label}>
          <span className="popover-text">{def.definition}</span>
          <span className="popover-text">
            <strong>Start:</strong> {def.start}
            <br />
            <strong>End:</strong> {def.end}
          </span>
          {def.note && <span className="popover-text">{def.note}</span>}
          <span className="popover-text">
            Percentiles are computed by the server from valid samples only and appear after {minSamples} samples. Failed,
            timed-out and cancelled samples are excluded and counted separately.
          </span>
        </InfoPopover>
      </div>
      {unavailable ? (
        <p className="latency-unavailable">Not available for this session — requires annotated turn ends (labelled runs only).</p>
      ) : (
        <>
          <p className="latency-latest">
            <span className="latency-value mono">{latestValue}</span>
            <span className="latency-kind">{latest ? "latest event" : "no sample yet"}</span>
            {latestFailed && <span className="latency-failed">{latest.status}</span>}
          </p>
          <p className="latency-agg">
            <span className="mono">{aggregateText(summary, minSamples)}</span>
            {summary && summary.count > 0 && <span className="latency-n"> n = {summary.count}</span>}
          </p>
          {summary && summary.failed > 0 && (
            <p className="latency-failed-count">{summary.failed} failed — excluded from percentiles</p>
          )}
        </>
      )}
    </div>
  );
}

const STAGES: Array<{ metric: LatencyMetric; label: string }> = [
  { metric: "detector_inference", label: "Turn detector" },
  { metric: "stt_final", label: "STT final" },
  { metric: "llm_first_token", label: "LLM first token" },
  { metric: "tts_first_audio", label: "TTS first audio" },
];

/** Stage breakdown (latest samples). Stages overlap, so they do not sum to end-to-end. */
export function StageChart({ latest }: { latest: Partial<Record<LatencyMetric, LatestSample>> }) {
  const rows = [
    ...STAGES.map((s) => ({ ...s, value: latest[s.metric]?.valueMs ?? null, e2e: false })),
    { metric: "response_start" as LatencyMetric, label: "Response start (end-to-end)", value: latest.response_start?.valueMs ?? null, e2e: true },
  ];
  const max = Math.max(1, ...rows.map((r) => r.value ?? 0));
  const rowH = 26;
  const labelW = 170;
  const valueW = 76;
  const width = 560;
  const barW = width - labelW - valueW;
  const height = rows.length * rowH + 4;
  const summary = rows.map((r) => `${r.label}: ${r.value === null ? "no sample" : formatMs(r.value)}`).join("; ");

  return (
    <figure className="stage-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Latest stage timings. ${summary}`} preserveAspectRatio="xMinYMin meet">
        {rows.map((r, i) => {
          const y = i * rowH + 2;
          const w = r.value === null ? 0 : Math.max(2, (r.value / max) * barW);
          return (
            <g key={r.metric} transform={`translate(0 ${y})`}>
              <text x={0} y={rowH / 2 + 4} className="sc-label">
                {r.label}
              </text>
              <rect x={labelW} y={5} width={barW} height={rowH - 10} rx={3} className="sc-track" />
              {r.value !== null && <rect x={labelW} y={5} width={w} height={rowH - 10} rx={3} className={r.e2e ? "sc-bar sc-e2e" : "sc-bar"} />}
              <text x={labelW + barW + 8} y={rowH / 2 + 4} className="sc-value">
                {r.value === null ? DASH : formatMs(r.value)}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="field-help">
        Pipeline: Audio → VAD → Turn detector → STT → LLM → TTS → Playback. Latest samples only. Stages overlap and run
        concurrently, so their durations do not sum to the end-to-end response-start latency.
      </figcaption>
    </figure>
  );
}

export function LatencyOverview() {
  const { caps } = useApp();
  const latest = useSession((s) => s.latencyLatest);
  const summary = useSession((s) => s.latencySummary);
  const session = useSession((s) => s.session);
  const effective = useSession((s) => s.effectiveConfig);
  const isMock = useSession((s) => s.isMock);
  const ids = useId();
  const scenario = caps.scenarios.find((s) => s.id === (session?.scenarioId ?? effective?.scenarioId));
  const labelled = scenario ? scenario.labelled : null;

  return (
    <Card
      title="Latency"
      className="latency-overview"
      labelledBy={`${ids}-title`}
      actions={
        <span className="card-meta">
          {effective ? `Detector: ${effective.detectorMode}` : "No session"}
          {isMock ? " · MOCK DATA" : ""}
        </span>
      }
    >
      <div className="latency-grid">
        {PRIMARY_LATENCY_METRICS.map((m) => (
          <LatencyCard
            key={m}
            metric={m}
            latest={latest[m]}
            summary={summary[m]}
            minSamples={caps.minPercentileSamples}
            labelledSession={labelled}
          />
        ))}
      </div>
      <details className="stage-details">
        <summary>Stage breakdown</summary>
        <StageChart latest={latest} />
      </details>
    </Card>
  );
}

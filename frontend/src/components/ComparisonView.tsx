import { useId, useMemo, useState } from "react";
import { snakeToCamel, type ComparisonGroup, type ComparisonResult, type TrialListItem } from "../api/schemas";
import { DASH, formatByUnit, isNum } from "../lib/format";
import { BetterMarker, Card, InfoNotice } from "./common";
import { InfoPopover } from "./InfoPopover";

export function groupLabel(g: ComparisonGroup): string {
  return Object.entries(g.key)
    .map(([k, v]) => `${k}=${v ?? "—"}`)
    .join(" · ");
}

function trialMatchesGroup(t: TrialListItem, g: ComparisonGroup): boolean {
  const rec = t as unknown as Record<string, unknown>;
  return Object.entries(g.key).every(([k, v]) => rec[snakeToCamel(k)] === v);
}

function trialMetricValue(t: TrialListItem, metric: string): number | null {
  const v = (t.metrics as unknown as Record<string, unknown>)[snakeToCamel(metric)];
  return isNum(v as number | null) ? (v as number) : null;
}

function niceTicks(min: number, max: number, count = 5): number[] {
  if (min === max) return [min];
  const step0 = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 1e-6; v += step) ticks.push(Math.round(v / step) * step);
  return ticks;
}

/**
 * Per-group distribution: every trial value is drawn (outliers included), with the
 * group mean and 95% CI. No ranking and no "best" marker.
 */
export function DistributionChart({
  result,
  metric,
  trials,
}: {
  result: ComparisonResult;
  metric: string;
  trials: TrialListItem[];
}) {
  const def = result.metricDefinitions[metric];
  const unit = def?.unit ?? "";
  const rows = result.groups.map((g) => {
    const stat = g.metrics[metric];
    const points = trials.filter((t) => trialMatchesGroup(t, g)).map((t) => trialMetricValue(t, metric)).filter(isNum);
    return { g, stat, points };
  });
  const all = rows.flatMap((r) => [
    ...r.points,
    ...[r.stat?.mean, r.stat?.ci95Low, r.stat?.ci95High].filter(isNum),
  ]);
  if (all.length === 0) {
    return <p className="muted">No values for this metric in the compared groups.</p>;
  }
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (lo === hi) {
    lo -= Math.abs(lo) * 0.1 || 1;
    hi += Math.abs(hi) * 0.1 || 1;
  }
  const pad = (hi - lo) * 0.06;
  lo -= pad;
  hi += pad;
  const width = 640;
  const labelW = 190;
  const rowH = 44;
  const top = 8;
  const plotW = width - labelW - 16;
  const height = top + rows.length * rowH + 30;
  const x = (v: number) => labelW + ((v - lo) / (hi - lo)) * plotW;
  const ticks = niceTicks(lo, hi);
  const description = rows
    .map((r) => `${groupLabel(r.g)}: ${r.points.length} trial values, mean ${formatByUnit(r.stat?.mean, unit)}, 95% CI ${formatByUnit(r.stat?.ci95Low, unit)} to ${formatByUnit(r.stat?.ci95High, unit)}`)
    .join("; ");

  return (
    <figure className="dist-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${def?.label ?? metric}. ${description}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top} y2={height - 24} className="dc-grid" />
            <text x={x(t)} y={height - 8} className="dc-tick" textAnchor="middle">
              {formatByUnit(t, unit)}
            </text>
          </g>
        ))}
        {rows.map((r, i) => {
          const cy = top + i * rowH + rowH / 2;
          const s = r.stat;
          return (
            <g key={groupLabel(r.g)}>
              <text x={0} y={cy + 4} className="dc-label">
                {groupLabel(r.g).slice(0, 30)}
              </text>
              {r.points.map((v, j) => (
                <circle
                  key={j}
                  cx={x(v)}
                  cy={cy + (((j * 37) % 11) - 5) * 1.4}
                  r={3.2}
                  className="dc-point"
                />
              ))}
              {s && isNum(s.ci95Low) && isNum(s.ci95High) && (
                <g className="dc-ci">
                  <line x1={x(s.ci95Low)} x2={x(s.ci95High)} y1={cy + 12} y2={cy + 12} />
                  <line x1={x(s.ci95Low)} x2={x(s.ci95Low)} y1={cy + 8} y2={cy + 16} />
                  <line x1={x(s.ci95High)} x2={x(s.ci95High)} y1={cy + 8} y2={cy + 16} />
                </g>
              )}
              {s && isNum(s.mean) && (
                <path
                  d={`M ${x(s.mean)} ${cy + 6} l 6 6 l -6 6 l -6 -6 z`}
                  className="dc-mean"
                />
              )}
            </g>
          );
        })}
      </svg>
      <figcaption className="field-help">
        Dots: individual trials (all values shown, including outliers; only trials loaded in the selection are drawn). Diamond:
        group mean. Bar: 95% CI from the server (absent when n is too small).
      </figcaption>
    </figure>
  );
}

export function ComparisonView({
  result,
  trials,
  onExport,
  exportError,
}: {
  result: ComparisonResult;
  trials: TrialListItem[];
  onExport(format: "json" | "csv" | "md"): void;
  exportError: string | null;
}) {
  const metricNames = useMemo(() => {
    const names = new Set<string>(Object.keys(result.metricDefinitions));
    result.groups.forEach((g) => Object.keys(g.metrics).forEach((m) => names.add(m)));
    return [...names];
  }, [result]);
  const [chartMetric, setChartMetric] = useState<string>(() => metricNames[0] ?? "");
  const ids = useId();

  return (
    <Card
      title="Comparison"
      className="comparison"
      labelledBy={`${ids}-title`}
      actions={
        <>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => onExport("csv")}>
            CSV
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => onExport("json")}>
            JSON
          </button>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => onExport("md")}>
            Markdown summary
          </button>
        </>
      }
    >
      {exportError && <p className="field-error">{exportError}</p>}
      <p className="small">
        Evaluation set: <span className="mono">{result.evaluationSet ?? "not reported"}</span> · grouped by{" "}
        <span className="mono">{result.groupBy.join(", ")}</span> · {result.groups.length} groups. Groups are listed in server
        order and are not ranked.
      </p>
      {result.warnings.map((w) => (
        <InfoNotice key={w} tone="uncertain">
          {w}
        </InfoNotice>
      ))}
      {result.groups.some((g) => g.nTrials < 3) && (
        <InfoNotice tone="uncertain">
          Some groups have fewer than 3 trials. Treat differences as indicative only; confidence intervals may be missing.
        </InfoNotice>
      )}

      <div className="table-wrap" role="region" aria-label="Comparison table" tabIndex={0}>
        <table className="data-table compare-table">
          <caption className="sr-only">Mean and 95% confidence interval per metric and group</caption>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              {result.groups.map((g) => (
                <th scope="col" key={groupLabel(g)}>
                  <span className="mono small">{groupLabel(g)}</span>
                  <div className="small muted">
                    n = {g.nTrials} trials · {g.trialTypes.join(", ") || "—"}
                  </div>
                  <div className="small muted">scenarios: {g.scenarioIds.join(", ") || "—"}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metricNames.map((m) => {
              const def = result.metricDefinitions[m];
              const unit = def?.unit ?? "";
              return (
                <tr key={m}>
                  <th scope="row">
                    <span className="metric-name">{def?.label ?? m}</span>
                    {def?.definition && <InfoPopover label={def.label}>{def.definition}</InfoPopover>}
                    <div>
                      <BetterMarker better={def?.better ?? ""} />
                    </div>
                  </th>
                  {result.groups.map((g) => {
                    const s = g.metrics[m];
                    return (
                      <td key={groupLabel(g)} className="mono">
                        {s ? (
                          <>
                            {formatByUnit(s.mean, unit)}
                            <div className="small muted">
                              {isNum(s.ci95Low) && isNum(s.ci95High)
                                ? `95% CI ${formatByUnit(s.ci95Low, unit)} – ${formatByUnit(s.ci95High, unit)}`
                                : "CI unavailable"}{" "}
                              · n = {s.n}
                            </div>
                          </>
                        ) : (
                          DASH
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {metricNames.length > 0 && (
        <div className="chart-block">
          <div className="field inline-field">
            <label htmlFor={`${ids}-metric`}>Distribution chart metric</label>
            <select id={`${ids}-metric`} value={chartMetric} onChange={(e) => setChartMetric(e.target.value)}>
              {metricNames.map((m) => (
                <option key={m} value={m}>
                  {result.metricDefinitions[m]?.label ?? m}
                </option>
              ))}
            </select>
            {result.metricDefinitions[chartMetric] && <BetterMarker better={result.metricDefinitions[chartMetric]!.better} />}
          </div>
          <DistributionChart result={result} metric={chartMetric} trials={trials} />
        </div>
      )}
    </Card>
  );
}

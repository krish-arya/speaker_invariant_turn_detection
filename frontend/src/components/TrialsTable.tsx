import type { TrialListItem } from "../api/schemas";
import { formatDateTime, formatMs, formatRate, formatScore, shortId } from "../lib/format";
import { DETECTOR_LABELS, TRIAL_TYPE_LABELS } from "../lib/labels";
import { Badge } from "./common";

const NA = "Not available for this session";

export function bargeInResult(t: TrialListItem): string {
  const c = t.counts;
  const m = t.metrics;
  if (!t.labelled) return c.committedBargeIns > 0 ? `${c.committedBargeIns} cancellation(s), unlabelled` : NA;
  if (c.referenceBargeIns === 0) {
    if (m.falseBargeInRate !== null && m.falseBargeInRate > 0) return `False barge-in ${formatRate(m.falseBargeInRate)}`;
    return "No barge-in expected";
  }
  const missed = m.missedBargeInRate;
  const parts = [`${c.committedBargeIns}/${c.referenceBargeIns} acted on`];
  if (missed !== null) parts.push(`missed ${formatRate(missed)}`);
  if (m.bargeInCancelP50Ms !== null) parts.push(`cancel p50 ${formatMs(m.bargeInCancelP50Ms)}`);
  return parts.join(" · ");
}

/** Results table with the frontend.md §6.7 columns. */
export function TrialsTable({
  trials,
  selectable,
  selected,
  onToggle,
  onToggleAll,
  onOpen,
  onExport,
  caption,
}: {
  trials: TrialListItem[];
  selectable?: boolean;
  selected?: Set<string>;
  onToggle?(id: string): void;
  onToggleAll?(ids: string[], on: boolean): void;
  onOpen(id: string): void;
  onExport(id: string, format: "json" | "csv"): void;
  caption: string;
}) {
  const allOn = selectable && trials.length > 0 && trials.every((t) => selected?.has(t.trialId));
  return (
    <div className="table-wrap" role="region" aria-label={caption} tabIndex={0}>
      <table className="data-table">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>
            {selectable && (
              <th scope="col" className="col-check">
                <input
                  type="checkbox"
                  aria-label="Select all trials on this page"
                  checked={!!allOn}
                  onChange={(e) => onToggleAll?.(trials.map((t) => t.trialId), e.target.checked)}
                />
              </th>
            )}
            <th scope="col">Trial / date</th>
            <th scope="col">Detector / model</th>
            <th scope="col">Scenario</th>
            <th scope="col">Outcome</th>
            <th scope="col">Turn F1</th>
            <th scope="col">Premature / missed end</th>
            <th scope="col">Turn-end delay p50 / p95</th>
            <th scope="col">Response start p50 / p95 / p99</th>
            <th scope="col">Barge-in</th>
            <th scope="col">Actions</th>
          </tr>
        </thead>
        <tbody>
          {trials.map((t) => {
            const m = t.metrics;
            return (
              <tr key={t.trialId} className={selected?.has(t.trialId) ? "is-selected" : undefined}>
                {selectable && (
                  <td className="col-check">
                    <input
                      type="checkbox"
                      aria-label={`Select trial ${t.trialId}`}
                      checked={!!selected?.has(t.trialId)}
                      onChange={() => onToggle?.(t.trialId)}
                    />
                  </td>
                )}
                <td>
                  <span className="mono" title={t.trialId}>
                    {shortId(t.trialId)}
                  </span>
                  <br />
                  <span className="muted small">{formatDateTime(t.createdAt)}</span>
                  {t.trialLabel && (
                    <>
                      <br />
                      <span className="small">“{t.trialLabel}”</span>
                    </>
                  )}
                </td>
                <td>
                  {DETECTOR_LABELS[t.detectorMode]}
                  <br />
                  <span className="mono small muted">{t.modelVersion}</span>
                  <br />
                  <span className="mono small muted">{t.decisionPolicyVersion}</span>
                </td>
                <td>
                  <span className="mono">{t.scenarioId}</span>
                  <br />
                  <Badge tone={t.trialType === "live" ? "agent" : "neutral"}>{TRIAL_TYPE_LABELS[t.trialType]}</Badge>
                </td>
                <td>
                  <Badge tone={t.status === "completed" ? "complete" : t.status === "error" ? "danger" : "uncertain"}>
                    {t.status}
                  </Badge>
                  {!t.labelled && <div className="small muted">unlabelled</div>}
                </td>
                <td className={t.labelled ? "mono" : "muted small"}>{t.labelled ? formatScore(m.turnF1) : NA}</td>
                <td className={t.labelled ? "mono" : "muted small"}>
                  {t.labelled ? `${formatRate(m.prematureEndRate)} / ${formatRate(m.missedEndRate)}` : NA}
                </td>
                <td className={t.labelled ? "mono" : "muted small"}>
                  {t.labelled ? `${formatMs(m.turnEndDelayP50Ms)} / ${formatMs(m.turnEndDelayP95Ms)}` : NA}
                </td>
                <td className="mono">
                  {formatMs(m.responseStartP50Ms)} / {formatMs(m.responseStartP95Ms)} / {formatMs(m.responseStartP99Ms)}
                  <div className="small muted">n = {t.counts.responsesStarted} responses</div>
                </td>
                <td className="small">{bargeInResult(t)}</td>
                <td className="actions-cell">
                  <button type="button" className="btn btn-ghost btn-sm" onClick={() => onOpen(t.trialId)}>
                    Details
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => onExport(t.trialId, "json")}
                    aria-label={`Export trial ${t.trialId} as JSON`}
                  >
                    JSON
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => onExport(t.trialId, "csv")}
                    aria-label={`Export trial ${t.trialId} as CSV`}
                  >
                    CSV
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

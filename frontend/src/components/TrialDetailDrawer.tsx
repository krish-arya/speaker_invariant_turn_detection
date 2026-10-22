import { useEffect, useRef, useState } from "react";
import { api, describeApiError } from "../api/client";
import type { TrialDetail } from "../api/schemas";
import { formatDateTime, formatRelative } from "../lib/format";
import { DETECTOR_LABELS, TRIAL_TYPE_LABELS } from "../lib/labels";
import { timelineItemFor } from "../state/sessionReducer";
import { Badge, ErrorNotice } from "./common";
import { Icon } from "./Icon";
import { TrialMetricsTable } from "./TrialSummaryCard";

const EVENT_PAGE = 200;

export function EffectiveConfigTable({ config }: { config: TrialDetail["effectiveConfig"] }) {
  const p = config.policy;
  const rows: Array<[string, string]> = [
    ["Detector mode", `${DETECTOR_LABELS[config.detectorMode]} (${config.detectorMode})`],
    ["Model version", config.modelVersion],
    ["Decision policy version", config.decisionPolicyVersion],
    [
      "Policy",
      `silence_timeout ${p.silenceTimeoutMs} ms · possible_end ${p.possibleEndMs} ms · complete ≥ ${p.completeThreshold} · possible ≥ ${p.possibleEndThreshold} · ${p.consecutiveFrames} frames · guard ${p.minSilenceGuardMs} ms · max silence ${p.maxSilenceMs} ms · barge-in ≥ ${p.bargeInMinSpeechMs} ms · backchannel ≤ ${p.backchannelMaxMs} ms · min speech ${p.minSpeechMs} ms`,
    ],
    ["VAD", `${config.vad.kind} · threshold ${config.vad.threshold} · ${config.vad.minSpeechFrames}/${config.vad.minSilenceFrames} frames`],
    ["Frames", `${config.frameMs} ms @ ${config.sampleRate} Hz`],
    ["Providers", `STT ${config.providers.stt} · LLM ${config.providers.llm} · TTS ${config.providers.tts}`],
    ["Runtime profile / transport", `${config.runtimeProfile} · ${config.transport}`],
    ["Scenario / seed", `${config.scenarioId} · ${config.seed ?? "—"}`],
    ["Transcript", `${config.transcriptEnabled ? "on" : "off"} · export ${config.transcriptExportEnabled ? "included" : "not included"}`],
    ["Raw audio retained", config.rawAudioRetention ? "YES" : "no"],
    ["Speculative prefill", config.speculativePrefill ? "on" : "off"],
    ["Server version / config hash", `${config.serverVersion} · ${config.configHash}`],
  ];
  return (
    <table className="kv-table">
      <caption className="sr-only">Effective configuration</caption>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <th scope="row">{k}</th>
            <td className="mono">{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Modal drawer with the full trial record. Escape closes; focus returns to the opener. */
export function TrialDetailDrawer({ trialId, onClose }: { trialId: string; onClose(): void }) {
  const [detail, setDetail] = useState<TrialDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [eventLimit, setEventLimit] = useState(EVENT_PAGE);
  const [exportError, setExportError] = useState<string | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setError(null);
    api
      .getTrial(trialId)
      .then((d) => !cancelled && setDetail(d))
      .catch((e: unknown) => !cancelled && setError(describeApiError(e)));
    return () => {
      cancelled = true;
    };
  }, [trialId]);

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCloseRef.current();
      if (e.key === "Tab" && panelRef.current) {
        const focusables = panelRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, summary, [tabindex]:not([tabindex="-1"])',
        );
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (!first || !last) return;
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      opener?.focus?.();
    };
  }, []);

  const doExport = async (format: "json" | "csv") => {
    setExportError(null);
    try {
      await api.exportTrial(trialId, format);
    } catch (e) {
      setExportError(describeApiError(e));
    }
  };

  return (
    <div className="drawer-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="drawer" role="dialog" aria-modal="true" aria-labelledby="trial-drawer-title" ref={panelRef}>
        <header className="drawer-header">
          <h2 id="trial-drawer-title">
            Trial <span className="mono">{trialId}</span>
          </h2>
          <div className="card-actions">
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => void doExport("json")}>
              <Icon name="download" size={13} /> JSON
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => void doExport("csv")}>
              <Icon name="download" size={13} /> CSV
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={onClose} ref={closeRef}>
              Close
            </button>
          </div>
        </header>
        <div className="drawer-body">
          {exportError && <p className="field-error">{exportError}</p>}
          {error && <ErrorNotice title="Could not load the trial">{error}</ErrorNotice>}
          {!detail && !error && <p className="muted">Loading trial…</p>}
          {detail && (
            <>
              <div className="badge-row">
                <Badge tone={detail.trialType === "live" ? "agent" : "neutral"}>{TRIAL_TYPE_LABELS[detail.trialType]}</Badge>
                <Badge tone={detail.labelled ? "complete" : "neutral"}>{detail.labelled ? "Labelled" : "Unlabelled"}</Badge>
                <Badge>{detail.status}</Badge>
                <span className="small muted">
                  {formatDateTime(detail.createdAt)} → {formatDateTime(detail.endedAt)}
                  {detail.trialLabel ? ` · “${detail.trialLabel}”` : ""}
                  {detail.evaluationId ? ` · evaluation ${detail.evaluationId}` : ""}
                </span>
              </div>

              <h3>Server-accepted configuration</h3>
              <EffectiveConfigTable config={detail.effectiveConfig} />

              <h3>Summary</h3>
              <TrialMetricsTable
                metrics={detail.metrics}
                counts={detail.counts}
                labelled={detail.labelled}
                toleranceMs={detail.toleranceMs}
              />

              <h3>Annotations</h3>
              {detail.annotations ? (
                <p className="mono small">
                  turn ends: {detail.annotations.turnEndsMs.map((t) => formatRelative(t)).join(", ") || "—"}
                  <br />
                  barge-ins: {detail.annotations.bargeInsMs.map((t) => formatRelative(t)).join(", ") || "—"}
                  <br />
                  backchannels: {detail.annotations.backchannelsMs.map((t) => formatRelative(t)).join(", ") || "—"}
                  <br />
                  source: {detail.annotations.source ?? "—"}
                </p>
              ) : (
                <p className="muted">No ground-truth annotations — task accuracy is not available for this trial.</p>
              )}

              <h3>
                Events <span className="muted small">({detail.events.length}{detail.invalidEvents ? `, ${detail.invalidEvents} failed validation` : ""})</span>
              </h3>
              <ol className="event-list">
                {detail.events.slice(0, eventLimit).map((env) => {
                  const item = timelineItemFor(env, null);
                  return (
                    <li key={env.seq}>
                      <span className="mono">{formatRelative(env.tMs)}</span>{" "}
                      <span className="mono muted">#{env.seq}</span> <span className="tl-source">{env.kind}</span>{" "}
                      {item ? item.label : ""}
                      {item?.code && item.emphasis !== "state" ? <span className="mono"> {item.code}</span> : null}
                      {env.generationId !== null ? <span className="mono muted"> gen {env.generationId}</span> : null}
                    </li>
                  );
                })}
              </ol>
              {detail.events.length > eventLimit && (
                <button type="button" className="btn btn-ghost" onClick={() => setEventLimit((n) => n + EVENT_PAGE)}>
                  Load {Math.min(EVENT_PAGE, detail.events.length - eventLimit)} more events
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

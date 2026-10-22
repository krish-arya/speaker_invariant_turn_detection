import { useId, useState } from "react";
import { useApp } from "../app/context";
import { formatRelative } from "../lib/format";
import { settingsStore, useSettings } from "../lib/settings";
import { useSession } from "../state/sessionStore";
import { Card, EmptyState, InfoNotice } from "./common";

/**
 * Transcript (optional, for debugging turn timing). When display is off, no transcript
 * text is rendered at all. Speakers are only ever "You" and "Assistant".
 */
export function TranscriptPanel() {
  const { caps } = useApp();
  const settings = useSettings();
  const segments = useSession((s) => s.transcript);
  const effective = useSession((s) => s.effectiveConfig);
  // segment id -> revision at the time of "Clear"; later revisions show again
  const [cleared, setCleared] = useState<Record<string, number>>({});
  const ids = useId();
  const display = settings.transcriptDisplay ?? caps.privacy.transcriptDisplayDefault;
  const serverEnabled = effective ? effective.transcriptEnabled : null;
  const shown = segments.filter((s) => cleared[s.segmentId] === undefined || s.revision > cleared[s.segmentId]!);

  return (
    <Card
      title="Transcript (optional)"
      className="transcript-card"
      labelledBy={`${ids}-title`}
      actions={
        <>
          <label className="switch">
            <input
              type="checkbox"
              checked={display}
              onChange={(e) => settingsStore.set({ transcriptDisplay: e.target.checked })}
            />
            <span>Show transcript</span>
          </label>
          {display && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              disabled={shown.length === 0}
              onClick={() => setCleared(Object.fromEntries(segments.map((s) => [s.segmentId, s.revision])))}
            >
              Clear transcript
            </button>
          )}
        </>
      }
    >
      <p className="privacy-line">
        Transcript from server: <strong>{serverEnabled === null ? "—" : serverEnabled ? "on" : "off"}</strong>
        {" · "}Transcript in exports:{" "}
        <strong>{effective ? (effective.transcriptExportEnabled ? "included" : "not included") : "—"}</strong>
        {" · "}Raw audio retained: <strong>no</strong>
      </p>
      {!display ? (
        <InfoNotice>Transcript display is off. No transcript text is shown in this view.</InfoNotice>
      ) : serverEnabled === false ? (
        <InfoNotice>Transcript was disabled for this session, so the server is not sending transcript text.</InfoNotice>
      ) : shown.length === 0 ? (
        <EmptyState>No transcript yet.</EmptyState>
      ) : (
        <ol className="transcript" aria-label="Transcript, oldest first">
          {shown.map((s) => (
            <li key={s.segmentId} className={`tr-seg role-${s.role}${s.isFinal ? " is-final" : " is-partial"}`}>
              <span className="tr-who">{s.role === "user" ? "You" : "Assistant"}</span>
              <span className="tr-text">{s.text}</span>
              <span className="tr-meta">
                {s.interrupted ? (
                  <span className="tr-state interrupted">interrupted — not fully delivered</span>
                ) : (
                  <span className={`tr-state ${s.isFinal ? "final" : "partial"}`}>{s.isFinal ? "final" : "provisional"}</span>
                )}
                <span className="mono">
                  {formatRelative(s.startMs)}
                  {s.endMs !== null ? ` → ${formatRelative(s.endMs)}` : ""}
                </span>
              </span>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}

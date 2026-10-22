import { useEffect, useId, useState } from "react";
import { useApp } from "../app/context";
import { useDevices } from "../lib/devices";
import { formatConfidence } from "../lib/format";
import { DECISION_LABELS, ERROR_SOURCE_LABELS, PHASE_META, TURN_EVENT_LABELS } from "../lib/labels";
import { settingsStore, useSettings } from "../lib/settings";
import { canInterrupt, isSessionBusy, selectDisplayPhase } from "../state/sessionReducer";
import { useSession, useSessionStoreInstance } from "../state/sessionStore";
import { Badge, Card, ErrorNotice, InfoNotice } from "./common";
import { Icon } from "./Icon";
import { LevelMeter, useTrackLevel } from "./LevelMeter";

export function VoiceSessionCard({ onStart, startDisabledReason }: { onStart(): void; startDisabledReason: string | null }) {
  const { controller, mock } = useApp();
  const store = useSessionStoreInstance();
  const phase = useSession(selectDisplayPhase);
  const rawPhase = useSession((s) => s.phase);
  const busy = useSession(isSessionBusy);
  const interruptible = useSession(canInterrupt);
  const stopping = useSession((s) => s.stopping);
  const transport = useSession((s) => s.transport);
  const transportStatus = useSession((s) => s.transportStatus);
  const transportDetail = useSession((s) => s.transportDetail);
  const localError = useSession((s) => s.localError);
  const errors = useSession((s) => s.errors);
  const vad = useSession((s) => s.vad);
  const prediction = useSession((s) => s.prediction);
  const decision = useSession((s) => s.decision);
  const settings = useSettings();
  const ids = useId();

  const meta = PHASE_META[phase];
  const live = transport === "livekit" && !mock;
  const micOn = live && busy && transportStatus !== "disconnected" && rawPhase !== "ending";

  // Live mic level is measured locally; simulation levels come from audio_level events.
  const [track, setTrack] = useState<MediaStreamTrack | null>(null);
  useEffect(() => {
    const update = () => setTrack(controller.getLocalAudioTrack());
    update();
    return controller.onSourceChange(update);
  }, [controller]);
  const readMic = useTrackLevel(live ? track : null);
  const readUser = live ? readMic : () => store.getState().audioLevel.user;
  const readAgent = () => store.getState().audioLevel.agent;

  const lastError = errors.length ? errors[errors.length - 1]! : null;
  const showServerError = lastError && rawPhase !== "ended";

  return (
    <Card title="Voice session" className={`voice-card tone-${meta.tone}`} labelledBy={`${ids}-title`}>
      <div className="session-status" data-phase={phase}>
        <span className={`status-icon tone-${meta.tone}`} aria-hidden="true">
          <Icon name={meta.icon} size={28} />
        </span>
        <div>
          <p className="status-label">{meta.label}</p>
          <p className="status-hint">{stopping ? "Stopping response…" : meta.hint}</p>
        </div>
      </div>

      {phase === "reconnecting" && (
        <InfoNotice tone="uncertain">
          Reconnecting. Audio during the outage is not being processed; the timeline marks the gap.
          {transportDetail ? ` ${transportDetail}` : ""}
        </InfoNotice>
      )}

      <div className="mic-row">
        {live ? (
          <Badge tone={micOn ? "active" : "neutral"} icon={micOn ? "mic" : "micOff"}>
            {micOn ? "Microphone on — streaming to the session" : "Microphone off"}
          </Badge>
        ) : (
          <Badge tone="neutral" icon="micOff">
            Microphone not used — {mock ? "mock script" : "simulated scripted audio"}
          </Badge>
        )}
      </div>

      <div className="levels">
        <LevelMeter label={live ? "Your microphone" : "User audio (simulated)"} read={readUser} active={busy} />
        {!live && <LevelMeter label="Agent audio" read={readAgent} active={busy} />}
      </div>

      <dl className="signals" aria-label="Current signals">
        <div className="signal">
          <dt>VAD</dt>
          <dd>
            {vad === null ? "—" : vad.speaking ? "Speech detected" : "Silence"}
            <span className="signal-note">speech presence only</span>
          </dd>
        </div>
        <div className="signal">
          <dt>Detector prediction</dt>
          <dd>
            {prediction ? (
              <>
                {TURN_EVENT_LABELS[prediction.eventType]}
                {formatConfidence(prediction.confidence) && (
                  <span className="mono"> ({formatConfidence(prediction.confidence)})</span>
                )}
              </>
            ) : (
              "—"
            )}
            <span className="signal-note">not a decision</span>
          </dd>
        </div>
        <div className="signal">
          <dt>Last committed decision</dt>
          <dd>
            {decision ? DECISION_LABELS[decision.decision] : "—"}
            <span className="signal-note">orchestrator</span>
          </dd>
        </div>
      </dl>

      <div className="session-actions">
        {busy ? (
          <>
            {rawPhase !== "ending" && (
              <button type="button" className="btn btn-danger btn-lg" onClick={() => void controller.end()}>
                <Icon name="stop" size={16} />
                End session
              </button>
            )}
            {(phase === "generating" || phase === "agent_speaking" || stopping) && rawPhase !== "ending" && (
              <button
                type="button"
                className="btn btn-warn btn-lg"
                onClick={() => void controller.interrupt()}
                disabled={!interruptible}
                aria-describedby={`${ids}-int-help`}
              >
                <Icon name="hand" size={16} />
                {stopping ? "Stopping response…" : "Interrupt agent"}
              </button>
            )}
          </>
        ) : (
          <button
            type="button"
            className="btn btn-primary btn-lg"
            onClick={onStart}
            disabled={startDisabledReason !== null}
            aria-describedby={startDisabledReason ? `${ids}-start-help` : undefined}
          >
            <Icon name="play" size={16} />
            {rawPhase === "ended" ? "Start new session" : "Start session"}
          </button>
        )}
      </div>
      {startDisabledReason && !busy && (
        <p className="field-help" id={`${ids}-start-help`}>
          {startDisabledReason}
        </p>
      )}
      {(phase === "generating" || phase === "agent_speaking") && (
        <p className="field-help" id={`${ids}-int-help`}>
          Speaking also interrupts automatically through the barge-in path.
        </p>
      )}

      {live && <DeviceSelect id={`${ids}-dev`} value={settings.audioDeviceId} busy={busy} />}

      {localError && (
        localError.fatal || localError.kind === "interrupt" ? (
          <ErrorNotice
            title={localError.message}
            actions={
              <>
                {localError.retryable && localError.kind !== "interrupt" && controller.canRetry && (
                  <button type="button" className="btn btn-primary" onClick={() => void controller.retry()}>
                    <Icon name="refresh" size={14} />
                    Retry
                  </button>
                )}
                <button type="button" className="btn btn-ghost" onClick={() => controller.dismissError()}>
                  Dismiss
                </button>
              </>
            }
          >
            {localError.detail}
          </ErrorNotice>
        ) : null
      )}

      {showServerError && lastError && (
        <InfoNotice tone={lastError.recoverable ? "uncertain" : "danger"}>
          <strong>{ERROR_SOURCE_LABELS[lastError.source]}:</strong> {lastError.message}{" "}
          <span className="mono">({lastError.code})</span>
          {lastError.recoverable ? " — recoverable; the session continues." : " — not recoverable."}
        </InfoNotice>
      )}
    </Card>
  );
}

function DeviceSelect({ id, value, busy }: { id: string; value: string | null; busy: boolean }) {
  const { controller } = useApp();
  const { devices, error } = useDevices();
  return (
    <div className="field device-field">
      <label htmlFor={id}>Microphone input</label>
      <select
        id={id}
        value={value ?? ""}
        onChange={(e) => {
          const v = e.target.value || null;
          settingsStore.set({ audioDeviceId: v });
          if (busy && v) void controller.switchAudioInput(v);
        }}
      >
        <option value="">System default</option>
        {devices.map((d, i) => (
          <option key={d.deviceId || i} value={d.deviceId}>
            {d.label || `Microphone ${i + 1}`}
          </option>
        ))}
      </select>
      {error && <p className="field-help">{error}</p>}
    </div>
  );
}

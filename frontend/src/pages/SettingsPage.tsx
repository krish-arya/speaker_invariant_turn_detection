import { useEffect, useId, useState } from "react";
import { PROTOCOL_VERSION, type DetectorMode, type Transport } from "../api/schemas";
import { useApp } from "../app/context";
import { Card, InfoNotice } from "../components/common";
import { LevelMeter, useTrackLevel } from "../components/LevelMeter";
import { microphoneError } from "../state/sessionController";
import { useDevices } from "../lib/devices";
import { DETECTOR_LABELS } from "../lib/labels";
import { settingsStore, useSettings } from "../lib/settings";
import { useSession } from "../state/sessionStore";

function AudioSection() {
  const settings = useSettings();
  const { devices, error, refresh } = useDevices();
  const [testTrack, setTestTrack] = useState<MediaStreamTrack | null>(null);
  const [testError, setTestError] = useState<string | null>(null);
  const [sampleRate, setSampleRate] = useState<number | null>(null);
  const read = useTrackLevel(testTrack);
  const ids = useId();

  useEffect(() => () => testTrack?.stop(), [testTrack]);

  const startTest = async () => {
    setTestError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: settings.audioDeviceId ? { deviceId: { exact: settings.audioDeviceId } } : true,
      });
      const track = stream.getAudioTracks()[0] ?? null;
      setSampleRate(track?.getSettings().sampleRate ?? null);
      setTestTrack(track);
      refresh(); // labels become visible after permission
    } catch (e) {
      const err = microphoneError(e);
      setTestError(`${err.message} ${err.detail ?? ""}`);
    }
  };

  const stopTest = () => {
    testTrack?.stop();
    setTestTrack(null);
  };

  return (
    <Card title="Audio" labelledBy={`${ids}-audio`}>
      <div className="field">
        <label htmlFor={`${ids}-dev`}>Microphone (used for live LiveKit sessions)</label>
        <select id={`${ids}-dev`} value={settings.audioDeviceId ?? ""} onChange={(e) => settingsStore.set({ audioDeviceId: e.target.value || null })}>
          <option value="">System default</option>
          {devices.map((d, i) => (
            <option key={d.deviceId || i} value={d.deviceId}>
              {d.label || `Microphone ${i + 1} (grant access to see its name)`}
            </option>
          ))}
        </select>
        {error && <p className="field-help">{error}</p>}
      </div>
      <div className="form-actions">
        {testTrack ? (
          <button type="button" className="btn btn-ghost" onClick={stopTest}>
            Stop input test
          </button>
        ) : (
          <button type="button" className="btn btn-ghost" onClick={() => void startTest()}>
            Test input level
          </button>
        )}
        <span className="field-help">The test stays in this browser; nothing is sent or recorded.</span>
      </div>
      <LevelMeter label="Input test" read={read} active={testTrack !== null} />
      {testTrack && <p className="field-help">Reported sample rate: {sampleRate ? `${sampleRate} Hz` : "not reported by this browser"}. The server resamples to its model rate.</p>}
      {testError && <p className="field-error">{testError}</p>}
    </Card>
  );
}

export function SettingsPage() {
  const { caps, health, mock, controller, refreshServer } = useApp();
  const settings = useSettings();
  const diagnostics = useSession((s) => s.diagnostics);
  const serverProtocol = useSession((s) => s.serverProtocolVersion);
  const ids = useId();
  const transcriptDisplay = settings.transcriptDisplay ?? caps.privacy.transcriptDisplayDefault;
  const transcriptExport = settings.transcriptExport ?? caps.privacy.transcriptExportDefault;
  const nav = typeof navigator !== "undefined" ? navigator : null;

  return (
    <div className="page settings-page">
      <AudioSection />

      <Card title="Conversation" labelledBy={`${ids}-conv`}>
        <label className="check">
          <input type="checkbox" checked={transcriptDisplay} onChange={(e) => settingsStore.set({ transcriptDisplay: e.target.checked })} />
          Show the transcript in the Live Console
        </label>
        <p className="field-help">
          Also sent as <span className="mono">transcript_enabled</span> when a session starts, so the server does not send transcript
          text when it is off.
        </p>
        <p className="field-help">Language: server default. This server does not expose language selection.</p>
        <p className="field-help">
          Agent audio plays automatically after you start a live session (browser autoplay permitting).
        </p>
      </Card>

      <Card title="Evaluation defaults" labelledBy={`${ids}-eval`}>
        <div className="config-row">
          <div className="field">
            <label htmlFor={`${ids}-ddet`}>Default detector</label>
            <select
              id={`${ids}-ddet`}
              value={settings.defaultDetector ?? ""}
              onChange={(e) => settingsStore.set({ defaultDetector: (e.target.value || null) as DetectorMode | null })}
            >
              <option value="">First available</option>
              {caps.detectors.map((d) => (
                <option key={d.mode} value={d.mode} disabled={!d.available}>
                  {d.label || DETECTOR_LABELS[d.mode]}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor={`${ids}-dscn`}>Default scenario</label>
            <select id={`${ids}-dscn`} value={settings.defaultScenario ?? ""} onChange={(e) => settingsStore.set({ defaultScenario: e.target.value || null })}>
              <option value="">First listed</option>
              {caps.scenarios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor={`${ids}-dtr`}>Default transport</label>
            <select id={`${ids}-dtr`} value={settings.defaultTransport} onChange={(e) => settingsStore.set({ defaultTransport: e.target.value as Transport })}>
              <option value="simulation">Simulation</option>
              <option value="livekit" disabled={!caps.transports.livekit.available}>
                Live microphone (LiveKit)
              </option>
            </select>
          </div>
        </div>
        <p className="field-help">
          Default policy version <span className="mono">{caps.policyDefaults.version}</span>. Change thresholds per session under Live
          Console → Advanced configuration. Defaults apply when the page is next loaded.
        </p>
      </Card>

      <Card title="Privacy" labelledBy={`${ids}-priv`}>
        <dl className="kv">
          <div>
            <dt>Raw audio recording</dt>
            <dd>
              {caps.privacy.rawAudioRetention ? (
                <strong className="text-danger">ON — the server reports that raw audio is retained</strong>
              ) : (
                <>Off — raw audio is not retained by the server</>
              )}
            </dd>
          </div>
          <div>
            <dt>Transcript display</dt>
            <dd>{transcriptDisplay ? "On" : "Off"}</dd>
          </div>
          <div>
            <dt>Transcript in exports</dt>
            <dd>{transcriptExport ? "Included (next session)" : "Not included (next session)"}</dd>
          </div>
        </dl>
        <label className="check">
          <input type="checkbox" checked={transcriptDisplay} onChange={(e) => settingsStore.set({ transcriptDisplay: e.target.checked })} />
          Display transcript
        </label>
        <label className="check">
          <input type="checkbox" checked={transcriptExport} onChange={(e) => settingsStore.set({ transcriptExport: e.target.checked })} />
          Include transcript text in trial exports (applies to the next session)
        </label>
        <p className="field-help">
          Exports may contain derived turn events and, if enabled above, transcript text. They never contain raw audio. Preferences
          are stored only in this browser.
        </p>
        <div className="form-actions">
          <button type="button" className="btn btn-ghost" onClick={() => controller.clearView()}>
            Clear current session view
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => settingsStore.reset()}>
            Reset local preferences
          </button>
        </div>
      </Card>

      <Card
        title="Diagnostics"
        labelledBy={`${ids}-diag`}
        actions={
          !mock && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={refreshServer}>
              Refresh server status
            </button>
          )
        }
      >
        {mock && <InfoNotice tone="uncertain">Mock mode — no backend connected. Server values below come from the in-browser mock.</InfoNotice>}
        <dl className="kv diag">
          <div>
            <dt>API health</dt>
            <dd>{health ? `${health.status} · up ${health.uptimeS !== null ? `${Math.round(health.uptimeS)} s` : "—"}` : mock ? "mock" : "unknown"}</dd>
          </div>
          <div>
            <dt>Server version</dt>
            <dd className="mono">{caps.serverVersion}</dd>
          </div>
          <div>
            <dt>Protocol</dt>
            <dd className="mono">
              client v{PROTOCOL_VERSION} · server v{caps.protocolVersion}
              {serverProtocol !== null ? ` · session hello v${serverProtocol}` : ""}
              {caps.protocolVersion !== PROTOCOL_VERSION && " — MISMATCH"}
            </dd>
          </div>
          <div>
            <dt>App build</dt>
            <dd className="mono">
              web-{__APP_VERSION__} · built {__APP_BUILD_TIME__}
            </dd>
          </div>
          <div>
            <dt>Runtime profile</dt>
            <dd className="mono">{caps.runtimeProfile}</dd>
          </div>
          <div>
            <dt>Providers</dt>
            <dd className="mono">
              STT {caps.providers.stt} · LLM {caps.providers.llm} · TTS {caps.providers.tts}
            </dd>
          </div>
          <div>
            <dt>Detectors</dt>
            <dd className="mono">
              {caps.detectors.map((d) => `${d.mode}=${d.modelVersion}${d.available ? "" : " (unavailable)"}`).join(" · ")}
            </dd>
          </div>
          <div>
            <dt>Transports</dt>
            <dd>
              Simulation: {caps.transports.simulation.available ? "available" : `unavailable (${caps.transports.simulation.reason ?? "—"})`} · LiveKit:{" "}
              {caps.transports.livekit.available ? "available" : `unavailable (${caps.transports.livekit.reason ?? "—"})`}
            </dd>
          </div>
          <div>
            <dt>Session messages</dt>
            <dd className="mono">
              received {diagnostics.received} · accepted {diagnostics.accepted} · invalid {diagnostics.invalid} · unsupported version{" "}
              {diagnostics.unsupportedVersion}
            </dd>
          </div>
          <div>
            <dt>Unknown event kinds</dt>
            <dd className="mono">
              {diagnostics.unknownTotal}
              {diagnostics.unknownTotal > 0 &&
                ` (${Object.entries(diagnostics.unknownKinds)
                  .map(([k, n]) => `${k}×${n}`)
                  .join(", ")})`}
            </dd>
          </div>
          <div>
            <dt>Sequence</dt>
            <dd className="mono">
              gaps {diagnostics.gaps} ({diagnostics.missingMessages} missing) · duplicates {diagnostics.duplicates} · out of order{" "}
              {diagnostics.outOfOrder} · other-session {diagnostics.foreign} · stale generation dropped {diagnostics.staleDropped}
            </dd>
          </div>
          {diagnostics.lastProblem && (
            <div>
              <dt>Last problem</dt>
              <dd className="mono">{diagnostics.lastProblem}</dd>
            </div>
          )}
          <div>
            <dt>Browser</dt>
            <dd className="mono small">{nav?.userAgent ?? "—"}</dd>
          </div>
          <div>
            <dt>Capabilities</dt>
            <dd className="mono">
              getUserMedia {nav?.mediaDevices?.getUserMedia ? "yes" : "no"} · WebSocket {typeof WebSocket !== "undefined" ? "yes" : "no"} ·
              AudioContext {typeof AudioContext !== "undefined" ? "yes" : "no"} · language {nav?.language ?? "—"} · cores{" "}
              {nav?.hardwareConcurrency ?? "—"}
            </dd>
          </div>
        </dl>
        <p className="field-help">Credentials, API keys and room tokens are never shown here or stored by this console.</p>
      </Card>
    </div>
  );
}

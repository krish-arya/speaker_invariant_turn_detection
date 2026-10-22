import { useApp } from "../app/context";
import { ConfigBar, type ConsoleDraft } from "../components/ConfigBar";
import { MockBanner } from "../components/common";
import { LatencyOverview } from "../components/LatencyOverview";
import { policyDraftErrors } from "../components/PolicyFields";
import { Timeline } from "../components/Timeline";
import { TranscriptPanel } from "../components/TranscriptPanel";
import { TrialSummaryCard } from "../components/TrialSummaryCard";
import { VoiceSessionCard } from "../components/VoiceSessionCard";
import { toWireOverrides } from "../lib/policy";
import { useSettings } from "../lib/settings";
import { isSessionBusy } from "../state/sessionReducer";
import { useSession } from "../state/sessionStore";

export function LiveConsole({ draft, onDraftChange }: { draft: ConsoleDraft; onDraftChange(next: ConsoleDraft): void }) {
  const { caps, controller, mock } = useApp();
  const settings = useSettings();
  const busy = useSession(isSessionBusy);
  const sessionId = useSession((s) => s.sessionId);
  const hasView = useSession((s) => s.timeline.length > 0 || s.trialSummary !== null);

  const detector = caps.detectors.find((d) => d.mode === draft.detectorMode);
  const scenario = caps.scenarios.find((s) => s.id === draft.scenarioId && s.transports.includes(draft.transport));
  const errors = policyDraftErrors(draft.policy, caps.policyLimits);
  const transportOk = draft.transport === "simulation" ? caps.transports.simulation.available : caps.transports.livekit.available && !mock;

  let startDisabledReason: string | null = null;
  if (!detector || !detector.available) startDisabledReason = "Choose an available detector.";
  else if (!scenario) startDisabledReason = "Choose a scenario supported by this transport.";
  else if (!transportOk) startDisabledReason = "The selected transport is not available.";
  else if (errors.length) startDisabledReason = "Fix the invalid thresholds in Advanced configuration.";
  else if (draft.seed.trim() !== "" && !Number.isInteger(Number(draft.seed))) startDisabledReason = "Seed must be a whole number.";

  const start = () => {
    if (startDisabledReason || !detector || !scenario) return;
    void controller.start({
      transport: draft.transport,
      detectorMode: detector.mode,
      scenarioId: scenario.id,
      policyOverrides: toWireOverrides(draft.policy, caps.policyDefaults),
      transcriptEnabled: settings.transcriptDisplay ?? caps.privacy.transcriptDisplayDefault,
      transcriptExportEnabled: settings.transcriptExport ?? caps.privacy.transcriptExportDefault,
      speculativePrefill: draft.speculativePrefill,
      trialLabel: draft.trialLabel.trim() || null,
      seed: draft.seed.trim() === "" ? null : Number(draft.seed),
      deviceId: settings.audioDeviceId,
      mock,
    });
  };

  return (
    <div className="live-console">
      {mock && <MockBanner />}
      <ConfigBar draft={draft} onChange={onDraftChange} />
      <div className="console-grid">
        <div className="col-primary">
          <VoiceSessionCard onStart={start} startDisabledReason={startDisabledReason} />
        </div>
        <div className="col-timeline">
          <Timeline />
        </div>
        <div className="col-wide">
          <LatencyOverview />
        </div>
        <div className="col-wide">
          <TrialSummaryCard />
        </div>
        <div className="col-wide">
          <TranscriptPanel key={sessionId ?? "none"} />
        </div>
      </div>
      <div className="console-footer">
        <button type="button" className="btn btn-ghost" onClick={() => controller.clearView()} disabled={!hasView}>
          Clear session view
        </button>
        <span className="field-help">
          {busy
            ? "Clears the timeline and transcript on screen; the session keeps running."
            : "Clears everything shown for the last session from this view."}
        </span>
      </div>
    </div>
  );
}

import { useApp } from "../app/context";
import { DETECTOR_LABELS, type Tone } from "../lib/labels";
import type { IconName } from "../lib/labels";
import { useSession } from "../state/sessionStore";
import type { TransportStatus } from "../state/sessionReducer";
import { Badge, MockBanner } from "./common";
import { Icon } from "./Icon";

const CONNECTION: Record<TransportStatus, { label: string; tone: Tone; icon: IconName }> = {
  idle: { label: "Not connected", tone: "neutral", icon: "plug" },
  connecting: { label: "Connecting", tone: "neutral", icon: "plug" },
  connected: { label: "Connected", tone: "active", icon: "check" },
  reconnecting: { label: "Reconnecting", tone: "uncertain", icon: "refresh" },
  disconnected: { label: "Disconnected", tone: "neutral", icon: "plug" },
  error: { label: "Error", tone: "danger", icon: "alert" },
};

export function Header() {
  const { mock, goTo } = useApp();
  const status = useSession((s) => s.transportStatus);
  const config = useSession((s) => s.effectiveConfig);
  const phase = useSession((s) => s.phase);
  const c = CONNECTION[status];
  const active = config !== null && phase !== "idle" && phase !== "ended";

  return (
    <header className="app-header">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <Icon name="wave" size={18} />
        </span>
        <div>
          <h1 className="app-title">Speaker-Invariant Turn Detection</h1>
          <p className="app-subtitle">Real-time voice agent / turn-detection research console</p>
        </div>
      </div>
      <div className="header-status">
        {mock && <MockBanner compact />}
        <Badge tone={c.tone} icon={c.icon} title="Session transport connection">
          Connection: {c.label}
        </Badge>
        {active && config && (
          <Badge tone="neutral" icon="spark" title={`Model ${config.modelVersion}`}>
            Detector: {DETECTOR_LABELS[config.detectorMode]}
          </Badge>
        )}
        <button type="button" className="btn btn-ghost" onClick={() => goTo("settings")}>
          <Icon name="gear" size={15} />
          Settings
        </button>
      </div>
    </header>
  );
}

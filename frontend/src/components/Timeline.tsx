import { memo, useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import { formatClock, formatConfidence, formatRelative } from "../lib/format";
import type { IconName } from "../lib/labels";
import { settingsStore, useSettings } from "../lib/settings";
import { TIMELINE_LIMIT, type TimelineCategory, type TimelineItem } from "../state/sessionReducer";
import { useSession } from "../state/sessionStore";
import { Card, EmptyState } from "./common";
import { Icon } from "./Icon";

export const TIMELINE_EMPTY_TEXT = "Start a session to see live turn decisions and pipeline events.";

const FILTERS: Array<{ id: TimelineCategory; label: string }> = [
  { id: "turn", label: "Turn events" },
  { id: "pipeline", label: "Pipeline stages" },
  { id: "error", label: "Errors & gaps" },
  { id: "session", label: "Session" },
];

const EMPHASIS_TAG: Record<TimelineItem["emphasis"], string | null> = {
  prediction: "Prediction",
  decision: "Committed",
  state: "State",
  info: null,
};

function iconFor(item: TimelineItem): IconName {
  if (item.type === "gap") return "gap";
  if (item.category === "error") return "alert";
  if (item.emphasis === "decision") return item.code === "BARGE_IN" ? "hand" : "check";
  if (item.emphasis === "prediction") return "spark";
  if (item.source === "VAD") return "wave";
  if (item.source === "Playback") return "speaker";
  if (item.emphasis === "state") return "dot";
  return "chevron";
}

const TimelineRow = memo(function TimelineRow({
  item,
  selected,
  related,
  developer,
  onSelect,
}: {
  item: TimelineItem;
  selected: boolean;
  related: boolean;
  developer: boolean;
  onSelect(id: string): void;
}) {
  const tag = EMPHASIS_TAG[item.emphasis];
  const conf = formatConfidence(item.confidence);
  const hasDetails = item.meta !== null || (developer && item.raw !== null);
  return (
    <li
      className={[
        "tl-item",
        `tone-${item.tone}`,
        `emph-${item.emphasis}`,
        item.type === "gap" ? "is-gap" : "",
        selected ? "is-selected" : "",
        related ? "is-related" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <button type="button" className="tl-main" aria-pressed={selected} onClick={() => onSelect(item.id)}>
        <span className="tl-time mono" title={item.wallTime ? formatClock(item.wallTime) : undefined}>
          {formatRelative(item.tMs)}
        </span>
        <span className="tl-source">{item.source}</span>
        <span className={`tl-icon tone-${item.tone}`} aria-hidden="true">
          <Icon name={iconFor(item)} size={14} />
        </span>
        <span className="tl-text">
          {tag && <span className={`tl-tag tag-${item.emphasis}`}>{tag}</span>}
          <span className="tl-label">{item.label}</span>
          {item.code && item.emphasis !== "state" && <span className="tl-code mono">{item.code}</span>}
          {conf && <span className="tl-conf mono">conf {conf}</span>}
          {item.generationId !== null && <span className="tl-gen mono">gen {item.generationId}</span>}
          {item.late && <span className="tl-late">late (out of order)</span>}
        </span>
      </button>
      {(item.detail || item.reasonCodes.length > 0) && (
        <div className="tl-detail">
          {item.detail && <span>{item.detail}</span>}
          {item.reasonCodes.length > 0 && (
            <span className="tl-reasons" aria-label="Reason codes">
              {item.reasonCodes.map((r) => (
                <span key={r} className="reason mono">
                  {r}
                </span>
              ))}
            </span>
          )}
        </div>
      )}
      {hasDetails && (
        <details className="tl-more">
          <summary>Event details</summary>
          {item.meta && (
            <dl className="kv">
              {Object.entries(item.meta).map(([k, v]) => (
                <div key={k}>
                  <dt>{k}</dt>
                  <dd className="mono">{v}</dd>
                </div>
              ))}
              {item.confidence !== null && (
                <div>
                  <dt>Confidence</dt>
                  <dd className="mono">{item.confidence}</dd>
                </div>
              )}
              {item.seq !== null && (
                <div>
                  <dt>Envelope seq</dt>
                  <dd className="mono">{item.seq}</dd>
                </div>
              )}
            </dl>
          )}
          {developer && item.raw !== null && (
            <pre className="raw-json" aria-label="Raw event payload">
              {JSON.stringify(item.raw, null, 2)}
            </pre>
          )}
        </details>
      )}
    </li>
  );
});

/** Turn state / live event timeline — newest at the bottom (frontend.md §6.4). */
export function Timeline() {
  const items = useSession((s) => s.timeline);
  const trimmed = useSession((s) => s.timelineTrimmed);
  const settings = useSettings();
  const [filters, setFilters] = useState<Record<TimelineCategory, boolean>>({
    turn: true,
    pipeline: true,
    error: true,
    session: true,
  });
  const [autoScroll, setAutoScroll] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const ids = useId();

  const visible = useMemo(() => items.filter((i) => filters[i.category]), [items, filters]);
  const selectedGen = useMemo(() => {
    if (!selected) return null;
    return items.find((i) => i.id === selected)?.generationId ?? null;
  }, [items, selected]);

  useLayoutEffect(() => {
    if (!autoScroll || !listRef.current) return;
    listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [visible, autoScroll]);

  useEffect(() => {
    if (items.length === 0) setSelected(null);
  }, [items.length]);

  const onSelect = useCallback((id: string) => setSelected((cur) => (cur === id ? null : id)), []);

  const onScroll = () => {
    const el = listRef.current;
    if (!el || !autoScroll) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    if (!atBottom) setAutoScroll(false);
  };

  return (
    <Card
      title="Turn state / live event timeline"
      className="timeline-card"
      labelledBy={`${ids}-title`}
      actions={
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          aria-pressed={!autoScroll}
          onClick={() => setAutoScroll((a) => !a)}
        >
          <Icon name={autoScroll ? "pause" : "play"} size={13} />
          {autoScroll ? "Pause auto-scroll" : "Resume auto-scroll"}
        </button>
      }
    >
      <div className="tl-toolbar" role="group" aria-label="Timeline filters">
        <Icon name="filter" size={14} />
        {FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            className={`chip-toggle${filters[f.id] ? " is-on" : ""}`}
            aria-pressed={filters[f.id]}
            onClick={() => setFilters((cur) => ({ ...cur, [f.id]: !cur[f.id] }))}
          >
            {filters[f.id] ? "✓ " : ""}
            {f.label}
          </button>
        ))}
        <label className="dev-toggle">
          <input
            type="checkbox"
            checked={settings.developerMode}
            onChange={(e) => settingsStore.set({ developerMode: e.target.checked })}
          />
          Raw JSON (developer)
        </label>
      </div>
      <p className="tl-count" aria-live="off">
        {items.length === 0
          ? null
          : `Showing ${visible.length} of ${items.length} events (newest at the bottom)${
              trimmed > 0 ? ` · ${trimmed} older events dropped to keep the last ${TIMELINE_LIMIT}` : ""
            }`}
        {selectedGen !== null && ` · highlighting generation ${selectedGen}`}
      </p>
      <div className="tl-scroll" ref={listRef} onScroll={onScroll} tabIndex={0} aria-label="Timeline events" role="region">
        {items.length === 0 ? (
          <EmptyState>{TIMELINE_EMPTY_TEXT}</EmptyState>
        ) : visible.length === 0 ? (
          <EmptyState>No events match the selected filters.</EmptyState>
        ) : (
          <ol className="timeline" aria-label="Session events, oldest first">
            {visible.map((item) => (
              <TimelineRow
                key={item.id}
                item={item}
                selected={selected === item.id}
                related={selectedGen !== null && item.generationId === selectedGen && selected !== item.id}
                developer={settings.developerMode}
                onSelect={onSelect}
              />
            ))}
          </ol>
        )}
      </div>
    </Card>
  );
}

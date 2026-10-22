import { useRef, type KeyboardEvent } from "react";
import { TABS, type TabId } from "../app/context";

/** WAI-ARIA tablist with arrow-key / Home / End navigation. */
export function Tabs({ active, onChange }: { active: TabId; onChange(tab: TabId): void }) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let next = index;
    if (e.key === "ArrowRight") next = (index + 1) % TABS.length;
    else if (e.key === "ArrowLeft") next = (index - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    else return;
    e.preventDefault();
    const tab = TABS[next];
    if (!tab) return;
    onChange(tab.id);
    refs.current[next]?.focus();
  };

  return (
    <nav className="tabs-nav">
      <div role="tablist" aria-label="Console sections" className="tabs">
        {TABS.map((t, i) => (
          <button
            key={t.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            role="tab"
            type="button"
            id={`tab-${t.id}`}
            aria-selected={active === t.id}
            aria-controls={`panel-${t.id}`}
            tabIndex={active === t.id ? 0 : -1}
            className={`tab${active === t.id ? " is-active" : ""}`}
            onClick={() => onChange(t.id)}
            onKeyDown={(e) => onKeyDown(e, i)}
          >
            {t.label}
          </button>
        ))}
      </div>
    </nav>
  );
}

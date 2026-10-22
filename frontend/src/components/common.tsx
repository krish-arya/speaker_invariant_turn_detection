import type { ReactNode } from "react";
import type { Tone } from "../lib/labels";
import { Icon } from "./Icon";
import type { IconName } from "../lib/labels";

export function Badge({
  tone = "neutral",
  icon,
  children,
  title,
}: {
  tone?: Tone;
  icon?: IconName;
  children: ReactNode;
  title?: string;
}) {
  return (
    <span className={`badge tone-${tone}`} title={title}>
      {icon && <Icon name={icon} size={13} />}
      <span>{children}</span>
    </span>
  );
}

export function MockBanner({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`mock-banner${compact ? " compact" : ""}`} role="note">
      <Icon name="alert" size={14} />
      <strong>MOCK DATA</strong>
      {!compact && <span>Scripted demo stream generated in the browser — not measurements.</span>}
    </div>
  );
}

export function Card({
  title,
  actions,
  children,
  className,
  labelledBy,
  headingLevel = 2,
}: {
  title: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  labelledBy?: string;
  headingLevel?: 2 | 3;
}) {
  const H = headingLevel === 2 ? "h2" : "h3";
  return (
    <section className={`card${className ? ` ${className}` : ""}`} aria-labelledby={labelledBy}>
      <header className="card-header">
        <H className="card-title" id={labelledBy}>
          {title}
        </H>
        {actions && <div className="card-actions">{actions}</div>}
      </header>
      <div className="card-body">{children}</div>
    </section>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <p className="empty-state">{children}</p>;
}

export function ErrorNotice({
  title,
  children,
  actions,
}: {
  title: string;
  children?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="notice notice-error" role="alert">
      <Icon name="alert" size={16} />
      <div className="notice-body">
        <strong>{title}</strong>
        {children && <div className="notice-text">{children}</div>}
        {actions && <div className="notice-actions">{actions}</div>}
      </div>
    </div>
  );
}

export function InfoNotice({ children, tone = "neutral" }: { children: ReactNode; tone?: Tone }) {
  return (
    <div className={`notice notice-info tone-${tone}`}>
      <Icon name="info" size={16} />
      <div className="notice-body">{children}</div>
    </div>
  );
}

/** Direction marker for metrics. Never ranks; just says which way is desirable. */
export function BetterMarker({ better }: { better: string }) {
  if (better === "higher") return <span className="better" title="Higher is better">↑ higher is better</span>;
  if (better === "lower") return <span className="better" title="Lower is better">↓ lower is better</span>;
  return <span className="better" title="Interpret in context">↕ context-dependent</span>;
}

/** Display formatting. Unavailable values render as an em dash, never as 0. */

export const DASH = "—";

export function isNum(v: number | null | undefined): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function formatMs(v: number | null | undefined): string {
  if (!isNum(v)) return DASH;
  const abs = Math.abs(v);
  if (abs >= 1000) return `${(v / 1000).toFixed(2)} s`;
  if (abs >= 100) return `${Math.round(v)} ms`;
  if (abs >= 10) return `${v.toFixed(1)} ms`;
  return `${v.toFixed(2)} ms`;
}

/** Relative session time, e.g. "+4.28 s". */
export function formatRelative(tMs: number | null | undefined): string {
  if (!isNum(tMs)) return DASH;
  const sign = tMs < 0 ? "−" : "+";
  return `${sign}${(Math.abs(tMs) / 1000).toFixed(2)} s`;
}

export function formatRate(v: number | null | undefined): string {
  if (!isNum(v)) return DASH;
  return `${(v * 100).toFixed(1)}%`;
}

export function formatScore(v: number | null | undefined, digits = 3): string {
  if (!isNum(v)) return DASH;
  return v.toFixed(digits);
}

export function formatConfidence(v: number | null | undefined): string | null {
  return isNum(v) ? v.toFixed(2) : null;
}

export function formatDbfs(v: number | null | undefined): string {
  if (!isNum(v)) return DASH;
  return `${v.toFixed(1)} dBFS`;
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatClock(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/** Format a metric value given its unit from a comparison metric definition. */
export function formatByUnit(v: number | null | undefined, unit: string): string {
  if (!isNum(v)) return DASH;
  if (unit === "ms") return formatMs(v);
  if (unit === "ratio" || unit === "rate") return unit === "rate" ? formatRate(v) : formatScore(v);
  if (unit === "%") return `${v.toFixed(1)}%`;
  if (unit === "count") return String(Math.round(v * 100) / 100);
  return formatScore(v);
}

export function shortId(id: string | null | undefined, keep = 8): string {
  if (!id) return DASH;
  const idx = id.indexOf("_");
  const prefix = idx > 0 && idx < 6 ? id.slice(0, idx + 1) : "";
  const rest = prefix ? id.slice(prefix.length) : id;
  return rest.length > keep ? `${prefix}${rest.slice(0, keep)}…` : id;
}

/** Map dBFS (−60…0) to a 0…1 meter fraction. */
export function dbfsToFraction(db: number | null | undefined, floor = -60): number {
  if (!isNum(db)) return 0;
  if (db <= floor) return 0;
  if (db >= 0) return 1;
  return (db - floor) / -floor;
}

/** Small numeric helpers for display-side summaries. Server aggregates take precedence. */

/** Nearest-rank percentile on a copy of the values; null for an empty list. */
export function percentile(values: readonly number[], p: number): number | null {
  const v = values.filter((x) => Number.isFinite(x)).slice().sort((a, b) => a - b);
  if (v.length === 0) return null;
  const rank = Math.ceil((p / 100) * v.length);
  return v[Math.min(v.length - 1, Math.max(0, rank - 1))] ?? null;
}

export function mean(values: readonly number[]): number | null {
  const v = values.filter((x) => Number.isFinite(x));
  if (!v.length) return null;
  return v.reduce((a, b) => a + b, 0) / v.length;
}

/** Deterministic PRNG for mock scripts (mulberry32). */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0 || 0x9e3779b9;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

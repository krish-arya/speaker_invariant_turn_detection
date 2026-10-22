import { useEffect, useRef, useState } from "react";
import { dbfsToFraction, formatDbfs } from "../lib/format";

export function usePrefersReducedMotion(): boolean {
  const query = "(prefers-reduced-motion: reduce)";
  const get = () => typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia(query).matches;
  const [reduced, setReduced] = useState(get);
  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const mql = window.matchMedia(query);
    const on = () => setReduced(mql.matches);
    mql.addEventListener?.("change", on);
    return () => mql.removeEventListener?.("change", on);
  }, []);
  return reduced;
}

/**
 * Audio level bar. Reads its value through `read()` in a capped animation loop and
 * writes straight to the DOM, so level updates never re-render React trees
 * (frontend.md §11). Stops when inactive; slows down for reduced motion.
 */
export function LevelMeter({
  label,
  read,
  active,
  fps = 20,
}: {
  label: string;
  read: () => number | null;
  active: boolean;
  fps?: number;
}) {
  const fill = useRef<HTMLDivElement>(null);
  const text = useRef<HTMLSpanElement>(null);
  const meter = useRef<HTMLDivElement>(null);
  const reduced = usePrefersReducedMotion();
  const readRef = useRef(read);
  readRef.current = read;

  useEffect(() => {
    const paint = (db: number | null) => {
      const f = dbfsToFraction(db);
      if (fill.current) fill.current.style.transform = `scaleX(${f.toFixed(3)})`;
      if (text.current) text.current.textContent = db === null ? "—" : formatDbfs(db);
      if (meter.current) meter.current.setAttribute("aria-valuenow", String(Math.round(f * 100)));
    };
    if (!active) {
      paint(null);
      return;
    }
    const interval = 1000 / (reduced ? 4 : fps);
    let raf = 0;
    let last = -Infinity;
    const loop = (t: number) => {
      raf = requestAnimationFrame(loop);
      if (t - last < interval) return;
      last = t;
      paint(readRef.current());
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [active, fps, reduced]);

  return (
    <div className="level">
      <span className="level-label">{label}</span>
      <div
        ref={meter}
        className={`level-track${reduced ? " no-motion" : ""}`}
        role="meter"
        aria-label={`${label} level`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={0}
      >
        <div ref={fill} className="level-fill" />
      </div>
      <span ref={text} className="level-value mono">
        —
      </span>
    </div>
  );
}

/** RMS level (dBFS) of a live MediaStreamTrack via an AnalyserNode. */
export function useTrackLevel(track: MediaStreamTrack | null): () => number | null {
  const reader = useRef<() => number | null>(() => null);
  useEffect(() => {
    if (!track || typeof AudioContext === "undefined") {
      reader.current = () => null;
      return;
    }
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(new MediaStream([track]));
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    src.connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    reader.current = () => {
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i += 1) sum += buf[i]! * buf[i]!;
      const rms = Math.sqrt(sum / buf.length);
      return rms > 0 ? Math.max(-90, 20 * Math.log10(rms)) : -90;
    };
    return () => {
      reader.current = () => null;
      src.disconnect();
      void ctx.close();
    };
  }, [track]);
  return () => reader.current();
}

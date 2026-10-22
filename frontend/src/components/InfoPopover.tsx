import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { Icon } from "./Icon";

/** Keyboard-accessible disclosure popover for metric definitions. Escape closes it. */
export function InfoPopover({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const ref = useRef<HTMLSpanElement>(null);
  const btn = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        btn.current?.focus();
      }
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span className="popover-anchor" ref={ref}>
      <button
        ref={btn}
        type="button"
        className="icon-btn"
        aria-expanded={open}
        aria-controls={id}
        aria-label={`About ${label}`}
        onClick={() => setOpen((o) => !o)}
      >
        <Icon name="info" size={15} />
      </button>
      {open && (
        <span id={id} role="note" className="popover">
          <strong className="popover-title">{label}</strong>
          {children}
        </span>
      )}
    </span>
  );
}

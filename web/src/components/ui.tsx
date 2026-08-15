/**
 * The small shared pieces: panels, toggles, stat readouts, the transport bar.
 *
 * Every interactive element gets a 120 ms transition and no layout-affecting change on
 * hover. That combination is most of what "snappy" means in practice — a control that
 * responds within one frame reads as instant, and one that reflows the page on hover
 * reads as janky no matter how fast it is.
 */

import type { ReactNode } from "react";
import { color, font } from "../theme.ts";

export const transition = "120ms cubic-bezier(.2,.6,.3,1)";

export function Panel({
  title,
  right,
  children,
  padding = 14,
}: {
  title?: string;
  right?: ReactNode;
  children: ReactNode;
  padding?: number;
}) {
  return (
    <section
      style={{
        background: color.panel,
        border: `1px solid ${color.borderSoft}`,
        borderRadius: 4,
        overflow: "hidden",
      }}
    >
      {title && (
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            padding: "9px 14px",
            background: color.header,
            borderBottom: `1px solid ${color.borderFaint}`,
          }}
        >
          <span
            style={{
              font: `500 10px/1 ${font.mono}`,
              letterSpacing: ".09em",
              color: color.text[3],
              textTransform: "uppercase",
            }}
          >
            {title}
          </span>
          {right}
        </header>
      )}
      <div style={{ padding }}>{children}</div>
    </section>
  );
}

export function Toggle({
  label,
  on,
  onChange,
}: {
  label: string;
  on: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!on)}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        width: "100%",
        padding: "6px 8px",
        background: on ? color.control : "transparent",
        border: `1px solid ${on ? color.border : "transparent"}`,
        borderRadius: 3,
        color: on ? color.text[1] : color.text[4],
        font: `400 12px ${font.sans}`,
        cursor: "pointer",
        textAlign: "left",
        transition: `background ${transition}, color ${transition}, border-color ${transition}`,
      }}
    >
      <span
        aria-hidden
        style={{
          width: 6,
          height: 6,
          borderRadius: 1,
          background: on ? color.accent : color.text[7],
          transition: `background ${transition}`,
        }}
      />
      {label}
    </button>
  );
}

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
}: {
  options: ReadonlyArray<{ value: T; label: string }>;
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        gap: 2,
        padding: 2,
        background: color.control,
        border: `1px solid ${color.borderFaint}`,
        borderRadius: 3,
      }}
    >
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          style={{
            flex: 1,
            padding: "5px 8px",
            background: option.value === value ? color.header : "transparent",
            border: "none",
            borderRadius: 2,
            color: option.value === value ? color.text[1] : color.text[4],
            font: `400 11px ${font.sans}`,
            cursor: "pointer",
            transition: `background ${transition}, color ${transition}`,
          }}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Stat({
  label,
  value,
  unit,
  tone,
}: {
  label: string;
  value: string;
  unit?: string;
  tone?: string;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <span
        style={{
          font: `400 9px/1 ${font.mono}`,
          letterSpacing: ".08em",
          color: color.text[5],
          textTransform: "uppercase",
        }}
      >
        {label}
      </span>
      <span style={{ font: `400 17px/1 ${font.mono}`, color: tone ?? color.text[1] }}>
        {value}
        {unit && (
          <span style={{ font: `400 10px ${font.mono}`, color: color.text[5] }}> {unit}</span>
        )}
      </span>
    </div>
  );
}

/** A horizontal bar, used for entropy and for ward yield. Scale is caller-supplied. */
export function Bar({ value, max, tint }: { value: number; max: number; tint: string }) {
  const pct = Math.max(0, Math.min(1, value / max));
  return (
    <div
      style={{
        height: 3,
        borderRadius: 2,
        background: "rgba(255,255,255,.05)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${pct * 100}%`,
          height: "100%",
          background: tint,
          // Width transitions rather than snapping, so the ~9 Hz sidebar reads as
          // continuous motion instead of nine steps a second.
          transition: `width ${transition}`,
        }}
      />
    </div>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <p
      style={{
        font: `300 12px/1.6 ${font.sans}`,
        color: color.text[4],
        margin: "0 0 10px",
        maxWidth: "62ch",
      }}
    >
      {children}
    </p>
  );
}

export function Heading({ children, level = 2 }: { children: ReactNode; level?: 1 | 2 }) {
  const size = level === 1 ? 30 : 19;
  return (
    <h2
      style={{
        font: `400 ${size}px/1.15 ${font.serif}`,
        color: color.text[0],
        margin: "0 0 8px",
        letterSpacing: "-.01em",
      }}
    >
      {children}
    </h2>
  );
}

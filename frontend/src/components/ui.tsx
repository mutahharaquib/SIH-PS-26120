import React from "react";
import { Quantity, Tier } from "../api/client";

const tierStyle: Record<Tier, string> = {
  A: "bg-emerald-900/60 text-emerald-300 border-emerald-700",
  B: "bg-sky-900/60 text-sky-300 border-sky-700",
  C: "bg-violet-900/60 text-violet-300 border-violet-700",
};
const tierTitle: Record<Tier, string> = {
  A: "Tier A: bounded by problem statement or cited standard",
  B: "Tier B: first-principles physics (calibrated / placeholder parameters)",
  C: "Tier C: ML output / VFD-estimated",
};

export function TierBadge({ tier, placeholder, extrapolated }: { tier: Tier; placeholder?: boolean; extrapolated?: boolean }) {
  return (
    <span className="inline-flex gap-1 align-middle ml-1">
      <span title={tierTitle[tier]} className={`text-[10px] px-1 rounded border ${tierStyle[tier]}`}>{tier}</span>
      {placeholder && (
        <span title="Computed from PLACEHOLDER parameters - calibrate" className="text-[10px] px-1 rounded border bg-amber-900/50 text-amber-300 border-amber-700">
          PH
        </span>
      )}
      {extrapolated && (
        <span title="Outside calibrated range" className="text-[10px] px-1 rounded border bg-red-900/50 text-red-300 border-red-700">
          EXT
        </span>
      )}
    </span>
  );
}

export function fmt(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a >= 1e5) return v.toExponential(2);
  if (a >= 100) return v.toFixed(0);
  return v.toFixed(digits);
}

/** A number is never shown without its tier badge. */
export function Num({ value, unit, tier, ph, ext, digits }: {
  value: number | null | undefined; unit?: string; tier: Tier; ph?: boolean; ext?: boolean; digits?: number;
}) {
  return (
    <span className="whitespace-nowrap">
      <span className="font-mono">{fmt(value, digits)}</span>
      {unit && <span className="text-slate-400 text-xs ml-1">{unit}</span>}
      <TierBadge tier={tier} placeholder={ph} extrapolated={ext} />
    </span>
  );
}

export function Q({ q, digits, scale = 1, unit }: { q: Quantity | null | undefined; digits?: number; scale?: number; unit?: string }) {
  if (!q) return <span className="text-slate-500">-</span>;
  return (
    <span className="whitespace-nowrap">
      <span className="font-mono">{fmt(q.value * scale, digits)}</span>
      {q.uncertainty != null && q.uncertainty > 0 && (
        <span className="text-slate-500 text-xs font-mono"> ±{fmt(q.uncertainty * scale, digits)}</span>
      )}
      <span className="text-slate-400 text-xs ml-1">{unit ?? q.unit}</span>
      <TierBadge tier={q.prov.tier} placeholder={q.prov.placeholder} extrapolated={q.prov.extrapolated} />
    </span>
  );
}

export function Card({ title, children, right, className = "" }: {
  title: string; children: React.ReactNode; right?: React.ReactNode; className?: string;
}) {
  return (
    <div className={`bg-slate-900 border border-slate-800 rounded-lg p-3 ${className}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
        {right}
      </div>
      {children}
    </div>
  );
}

export function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="bg-slate-950/60 rounded p-2">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-base">{children}</div>
    </div>
  );
}

export function Btn(props: React.ButtonHTMLAttributes<HTMLButtonElement> & { kind?: "primary" | "ghost" | "danger" }) {
  const { kind = "ghost", className = "", ...rest } = props;
  const k = kind === "primary" ? "bg-sky-600 hover:bg-sky-500 text-white" : kind === "danger"
    ? "bg-red-700 hover:bg-red-600 text-white" : "bg-slate-800 hover:bg-slate-700 text-slate-200";
  return <button {...rest} className={`px-2.5 py-1 rounded text-sm disabled:opacity-40 ${k} ${className}`} />;
}

export const severityColor: Record<string, string> = {
  info: "text-sky-300", warning: "text-amber-300", alarm: "text-orange-400", critical: "text-red-400",
};

export function useInterval(fn: () => void, ms: number, deps: unknown[] = []) {
  React.useEffect(() => {
    fn();
    const id = setInterval(fn, ms);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

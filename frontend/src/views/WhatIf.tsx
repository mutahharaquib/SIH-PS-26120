import { useState } from "react";
import { post } from "../api/client";
import { Btn, Card, Num } from "../components/ui";
import { useStore } from "../store";

const SLIDERS = [
  { k: "steam_t", label: "Steam mass (t)", min: 1000, max: 5000, step: 100, def: 3000 },
  { k: "p_inj_bar", label: "Injection pressure (bar)", min: 60, max: 125, step: 1, def: 90 },
  { k: "soak_d", label: "Soak (d)", min: 2, max: 12, step: 0.5, def: 5 },
  { k: "spm", label: "SPM", min: 1.5, max: 7, step: 0.1, def: 4 },
  { k: "downstroke", label: "Downstroke speed factor", min: 0.45, max: 1, step: 0.05, def: 1 },
] as const;

const ROWS: { k: string; label: string; better: "up" | "down" }[] = [
  { k: "cycle_oil_m3", label: "Cycle oil (m3)", better: "up" }, { k: "sor", label: "SOR", better: "down" },
  { k: "kwh_per_bbl", label: "kWh/bbl", better: "down" }, { k: "g_star", label: "g* (USD/d)", better: "up" },
  { k: "T_star_d", label: "Cut-off T* (d)", better: "up" }, { k: "rfi", label: "RFI (current conditions)", better: "down" },
  { k: "stress_ratio", label: "Peak stress ratio", better: "down" },
];

export default function WhatIf() {
  const well = useStore((s) => s.selected);
  const [v, setV] = useState<Record<string, number>>(Object.fromEntries(SLIDERS.map((s) => [s.k, s.def])));
  const [out, setOut] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    try { setOut(await post("/whatif", { well_id: well, ...v })); } finally { setBusy(false); }
  };
  return (
    <div className="grid md:grid-cols-2 gap-3">
      <Card title={`What-if scenario on a cloned state - ${well} (no side effects)`}>
        {SLIDERS.map((s) => (
          <div key={s.k} className="mb-2">
            <div className="flex justify-between text-sm"><span>{s.label}</span><span className="font-mono">{v[s.k]}</span></div>
            <input type="range" min={s.min} max={s.max} step={s.step} value={v[s.k]} className="w-full"
              onChange={(e) => setV({ ...v, [s.k]: +e.target.value })} />
          </div>
        ))}
        <Btn kind="primary" onClick={run} disabled={busy}>{busy ? "Simulating..." : "Run what-if"}</Btn>
      </Card>
      <Card title="Scenario vs current plan">
        {out ? (
          <table className="w-full text-sm">
            <thead><tr className="text-slate-400 text-left"><th>Metric</th><th>Plan</th><th>Scenario</th><th>Delta</th></tr></thead>
            <tbody>
              {ROWS.map((r) => {
                const d = out.delta[r.k];
                const good = r.better === "up" ? d > 0 : d < 0;
                return (
                  <tr key={r.k} className="border-t border-slate-800">
                    <td className="py-1">{r.label}</td>
                    <td><Num value={out.plan[r.k]} tier="B" ph /></td>
                    <td><Num value={out.scenario[r.k]} tier="B" ph /></td>
                    <td className={Math.abs(d) < 1e-9 ? "" : good ? "text-emerald-400" : "text-red-400"}>{d > 0 ? "+" : ""}{d.toFixed(2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : <div className="text-slate-500 text-sm">Move the sliders and run. Production/SOR/g* come from the cycle forecast from the current calibrated state; RFI/stress from the rod model at current conditions.</div>}
      </Card>
    </div>
  );
}

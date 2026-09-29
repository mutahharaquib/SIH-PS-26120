import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { get } from "../api/client";
import { CardPlot, TimeChart } from "../components/charts";
import { Card, Num, Q, Stat, useInterval } from "../components/ui";
import { useStore } from "../store";

export default function Diagnostics() {
  const well = useStore((s) => s.selected);
  const [cards, setCards] = useState<any[]>([]);
  const [diag, setDiag] = useState<any>(null);
  const [hist, setHist] = useState<any[]>([]);
  const [st, setSt] = useState<any>(null);
  useInterval(async () => {
    const [c, d, h, s] = await Promise.all([
      get(`/wells/${well}/cards?limit=6`), get(`/wells/${well}/diagnostics`),
      get(`/wells/${well}/history?every=2&fields=rfi,fillage,eta_v,stress_ratio,spm,ds`), get(`/wells/${well}/state`).catch(() => null)]);
    setCards(c.cards);
    setDiag(d);
    setHist(h.rows.slice(-500));
    setSt(s);
  }, 3000, [well]);
  const probs = Object.entries(diag?.fault_probs ?? {}).map(([k, v]) => ({ k, v: v as number }));
  const secs = (diag?.stress_ratio_by_section ?? []).map((v: number, i: number) => ({
    name: `${((diag.sections[i]?.d_m ?? 0) / 0.0254).toFixed(2)}in`, v }));
  const r = diag?.risk;

  return (
    <div className="grid gap-3">
      {st && (
        <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
          <Stat label="RFI"><Q q={st.pump.rfi} /></Stat>
          <Stat label="RFI slope / stroke"><Num value={diag?.rfi_slope_per_stroke ?? 0} tier="B" digits={5} /></Stat>
          <Stat label="Fillage"><Q q={st.pump.fillage} /></Stat>
          <Stat label="Volumetric efficiency"><Q q={st.pump.volumetric_efficiency} /></Stat>
          <Stat label="Peak stress / Goodman"><Q q={st.pump.peak_rod_stress_ratio} /></Stat>
          <Stat label="Card class">
            <span>{st.pump.fault_class ?? "-"}</span>
            <span className="ml-1 text-[10px] px-1 rounded border bg-violet-900/60 text-violet-300 border-violet-700">C</span>
          </Stat>
        </div>
      )}
      <div className="grid md:grid-cols-2 gap-3">
        <Card title="Surface cards (last 6, newest highlighted)" right={<span className="text-[10px] text-emerald-300">Tier A measured</span>}>
          <CardPlot cards={cards} kind="surface" />
        </Card>
        <Card title="Downhole cards (diagnostic wave equation)" right={<span className="text-[10px] text-sky-300">Tier B</span>}>
          <CardPlot cards={cards} kind="downhole" />
        </Card>
        <Card title="Rod Float Index with thresholds" right={<span className="text-[10px] text-sky-300">Tier B</span>}>
          <TimeChart rows={hist} series={[{ key: "rfi", name: "RFI" }, { key: "ds", name: "downstroke factor", dash: true }]}
            refLines={[{ y: diag?.rfi_warn ?? 0.8, label: "warn", color: "#f59e0b" }, { y: diag?.rfi_alarm ?? 1, label: "alarm" }]} />
        </Card>
        <Card title="Classifier probabilities" right={<span className="text-[10px] text-violet-300">Tier C</span>}>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={probs} layout="vertical" margin={{ left: 40 }}>
              <CartesianGrid stroke="#1e293b" />
              <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: "#94a3b8" }} />
              <YAxis type="category" dataKey="k" tick={{ fontSize: 10, fill: "#94a3b8" }} width={110} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Bar dataKey="v" fill="#a78bfa" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Fillage and volumetric efficiency" right={<span className="text-[10px] text-sky-300">Tier B</span>}>
          <TimeChart rows={hist} series={[{ key: "fillage", name: "fillage (card)" }, { key: "eta_v", name: "eta_v estimate", dash: true }]} />
        </Card>
        <Card title="Stress ratio per taper (Goodman, API RP 11BR)" right={<span className="text-[10px] text-sky-300">Tier B</span>}>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={secs}>
              <CartesianGrid stroke="#1e293b" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#94a3b8" }} />
              <YAxis domain={[0, 1.2]} tick={{ fontSize: 10, fill: "#94a3b8" }} />
              <ReferenceLine y={1} stroke="#ef4444" strokeDasharray="4 4" />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Bar dataKey="v" fill="#38bdf8" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
      {r && (
        <Card title="Failure-risk components (heuristic ranking only, uncalibrated S-N proxy)">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-sm">
            <Stat label="Risk score"><Num value={r.risk} tier="B" ph digits={4} /></Stat>
            <Stat label="Miner damage (max)"><Num value={r.damage_max} tier="B" ph digits={6} /></Stat>
            <Stat label="Float rate (7 d)"><Num value={r.float_rate} tier="B" digits={3} /></Stat>
            <Stat label="Impact rate (7 d)"><Num value={r.impact_rate} tier="C" digits={3} /></Stat>
            <Stat label="Pound rate (7 d)"><Num value={r.pound_rate} tier="C" digits={3} /></Stat>
          </div>
        </Card>
      )}
    </div>
  );
}

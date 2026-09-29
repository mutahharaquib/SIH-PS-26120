import { useEffect, useState } from "react";
import { get, post } from "../api/client";
import { TimeChart } from "../components/charts";
import { Btn, Card, fmt, Num } from "../components/ui";

const METRICS: { k: string; label: string; better: "up" | "down" }[] = [
  { k: "oil_m3", label: "Oil (m3/well)", better: "up" }, { k: "npv_usd", label: "NPV (USD/well)", better: "up" },
  { k: "sor", label: "SOR", better: "down" }, { k: "kwh_per_bbl", label: "kWh/bbl", better: "down" },
  { k: "float_hours", label: "Float hours", better: "down" }, { k: "float_events", label: "Float events", better: "down" },
  { k: "impact_events", label: "Impact events", better: "down" }, { k: "pound_hours", label: "Fluid-pound hours", better: "down" },
  { k: "downtime_hours", label: "Downtime hours", better: "down" }, { k: "failures", label: "Simulated failures", better: "down" },
  { k: "volumetric_efficiency", label: "Volumetric efficiency", better: "up" },
  { k: "peak_stress_ratio_p95", label: "Peak stress ratio p95", better: "down" },
];

export default function Evaluation() {
  const [runs, setRuns] = useState<any[]>([]);
  const [running, setRunning] = useState<any[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [rep, setRep] = useState<any>(null);
  const [form, setForm] = useState({ n_wells: 4, horizon_days: 300 });
  const load = async () => {
    const r = await get("/evaluation");
    setRuns(r.runs);
    setRunning(r.running);
    if (!sel && r.runs.length) setSel(r.runs[0].run_id);
  };
  useEffect(() => { load(); const id = setInterval(load, 5000); return () => clearInterval(id); }, []);
  useEffect(() => { if (sel) get(`/evaluation/${sel}`).then(setRep); }, [sel]);
  const start = async (scenario?: string) => {
    const r = await post("/evaluation/run", scenario ? { scenario } : { ...form, seeds: [1] });
    setRunning([...running, { run_id: r.run_id, status: "running" }]);
  };

  return (
    <div className="grid gap-3">
      <Card title="Baseline-vs-twin experiments (same wells, seeds, noise and events)">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <label>Wells <input type="number" value={form.n_wells} min={1} max={12} onChange={(e) => setForm({ ...form, n_wells: +e.target.value })}
            className="w-14 bg-slate-800 rounded px-1" /></label>
          <label>Horizon (d) <input type="number" value={form.horizon_days} onChange={(e) => setForm({ ...form, horizon_days: +e.target.value })}
            className="w-20 bg-slate-800 rounded px-1" /></label>
          <Btn kind="primary" onClick={() => start()}>Run experiment</Btn>
          <Btn onClick={() => start("demo_float")}>Run demo scenario</Btn>
          <select value={sel ?? ""} onChange={(e) => setSel(e.target.value)} className="bg-slate-800 rounded px-1">
            {runs.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}{r.scenario ? " (scenario)" : ""}</option>)}
          </select>
          {running.map((r) => <span key={r.run_id} className="text-xs text-amber-300">running {r.run_id} {r.progress ? `${r.progress[0]}/${r.progress[1]}` : ""}</span>)}
        </div>
      </Card>
      {rep?.results && <Report rep={rep} />}
    </div>
  );
}

function Report({ rep }: { rep: any }) {
  const weights = rep.config.weights as number[];
  const [w, setW] = useState(String(weights[0]));
  const R = rep.results.by_weight[w];
  const cum = (R?.cum_oil?.baseline ?? []).map((b: number, i: number) => ({ t: i, baseline: b, twin: R.cum_oil.twin[i] }));
  const lt = rep.results.rfi_lead_time;
  return (
    <>
      {rep.scenario && (
        <Card title={`Demo scenario: ${rep.scenario.name} on ${rep.scenario.well}`}>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            <div>Float hours avoided: <Num value={rep.scenario.avoided.float_hours} tier="B" ph /></div>
            <div>Float events avoided: <Num value={rep.scenario.avoided.float_events} tier="B" ph digits={0} /></div>
            <div>Impact events avoided: <Num value={rep.scenario.avoided.impact_events} tier="B" ph digits={0} /></div>
            <div>NPV impact: <Num value={rep.scenario.impact.npv_usd} unit="USD" tier="B" ph /></div>
            <div>SOR change: <Num value={rep.scenario.impact.sor} tier="B" ph /></div>
            <div>kWh/bbl change: <Num value={rep.scenario.impact.kwh_per_bbl} tier="B" ph /></div>
            <div>Oil change: <Num value={rep.scenario.impact.oil_m3} unit="m3" tier="B" ph /></div>
          </div>
        </Card>
      )}
      <Card title={`Results (${rep.config.n_wells} wells x ${rep.config.seeds.length} seed(s), ${rep.config.horizon_days} d horizon; mean ± 95% CI, paired deltas)`}
        right={<div className="flex gap-1">{weights.map((x) => <Btn key={x} kind={String(x) === w ? "primary" : "ghost"} onClick={() => setW(String(x))}>
          production_weight={x}</Btn>)}</div>}>
        <table className="w-full text-sm">
          <thead><tr className="text-left text-slate-400"><th>Metric</th><th>Baseline</th><th>Twin</th><th>Delta (twin - baseline)</th><th>Rel.</th></tr></thead>
          <tbody>
            {METRICS.map((m) => {
              const s = R.summary[m.k];
              const d = s.delta;
              const good = d.mean == null ? null : m.better === "up" ? d.mean > 0 : d.mean < 0;
              return (
                <tr key={m.k} className="border-t border-slate-800">
                  <td className="py-1">{m.label}</td>
                  <td><Num value={s.baseline.mean} tier="B" ph /> <span className="text-xs text-slate-500">[{fmt(s.baseline.lo)}, {fmt(s.baseline.hi)}]</span></td>
                  <td><Num value={s.twin.mean} tier="B" ph /> <span className="text-xs text-slate-500">[{fmt(s.twin.lo)}, {fmt(s.twin.hi)}]</span></td>
                  <td className={good == null || Math.abs(d.mean) < 1e-9 ? "" : good ? "text-emerald-400" : "text-red-400"}>
                    {fmt(d.mean)} <span className="text-xs text-slate-500">[{fmt(d.lo)}, {fmt(d.hi)}]</span></td>
                  <td className="text-xs">{d.rel != null ? `${(100 * d.rel).toFixed(1)}%` : "-"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="text-xs text-slate-500 mt-2">Risk-score rank correlation vs true exposure (Spearman): {fmt(R.risk_rank_spearman_vs_true_exposure)}. All values Tier B (physics with placeholder parameters) - relative comparisons, not field forecasts.</div>
      </Card>
      <div className="grid md:grid-cols-2 gap-3">
        <Card title="Cumulative oil per well (m3)"><TimeChart rows={cum} series={[{ key: "baseline" }, { key: "twin" }]} /></Card>
        <Card title="RFI lead time to labeled float onset (shadow twin on baseline runs)">
          <div className="text-sm space-y-1">
            <div>Onsets: {lt.n_onsets}, detected before onset: {lt.detected}, missed: {lt.missed}</div>
            <div>Lead time p10/p50/p90: {fmt(lt.p10)} / {fmt(lt.p50)} / {fmt(lt.p90)} h</div>
            <div>Warn hours: {fmt(lt.warn_hours, 0)}, of which not followed by float within 72 h: {fmt(lt.false_alarm_hours, 0)}</div>
            <div className="text-xs text-slate-500">{lt.note}</div>
          </div>
        </Card>
        <Card title="Physics verification">
          <table className="w-full text-xs">
            <thead><tr className="text-slate-400 text-left"><th>Wave eq. round trip</th><th>damping</th><th>median err</th><th>max err</th></tr></thead>
            <tbody>{rep.physics_verification.wave_roundtrip.map((r: any, i: number) => (
              <tr key={i}><td>{r.mode}</td><td>{r.damping_1_s}</td><td>{r.median_rel_err.toExponential(1)}</td><td>{r.max_rel_err.toExponential(1)}</td></tr>))}</tbody>
          </table>
          <table className="w-full text-xs mt-2">
            <thead><tr className="text-slate-400 text-left"><th>Energy balance</th><th>rel. closure error</th></tr></thead>
            <tbody>{rep.physics_verification.energy_closure.map((r: any, i: number) => (
              <tr key={i}><td>{r.case}</td><td>{r.rel_err.toExponential(1)}</td></tr>))}</tbody>
          </table>
        </Card>
        <Card title="Card classifier (Tier C, split by well)">
          {rep.classifier ? (
            <table className="w-full text-xs">
              <thead><tr className="text-slate-400 text-left"><th>Class</th><th>P</th><th>R</th><th>F1</th><th>F1 (noise x3)</th></tr></thead>
              <tbody>{Object.entries(rep.classifier.held_out_by_well.per_class).map(([c, v]: any) => (
                <tr key={c}><td>{c}</td><td>{v.precision.toFixed(2)}</td><td>{v.recall.toFixed(2)}</td><td>{v.f1.toFixed(2)}</td>
                  <td>{rep.classifier.noise_stressed_x3.per_class[c].f1.toFixed(2)}</td></tr>))}</tbody>
            </table>
          ) : <div className="text-slate-500 text-sm">No classifier report.</div>}
          {rep.results.classifier_online && <div className="text-xs text-slate-400 mt-2">Online accuracy on replayed cards: {(100 * rep.results.classifier_online.accuracy).toFixed(1)}% ({rep.results.classifier_online.n_cards} cards)</div>}
        </Card>
      </div>
      <Card title="Charts">
        <div className="grid md:grid-cols-2 gap-2">
          {(rep.charts ?? []).map((c: string) => <img key={c} src={`/reports/${rep.run_id}/${c}`} className="rounded bg-white" />)}
        </div>
      </Card>
    </>
  );
}

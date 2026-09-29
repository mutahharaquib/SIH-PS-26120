import { useState } from "react";
import { get, post } from "../api/client";
import { TimeChart } from "../components/charts";
import { Btn, Card, Num, useInterval } from "../components/ui";
import { useStore } from "../store";

export default function Decision() {
  const well = useStore((s) => s.selected);
  const [rec, setRec] = useState<any>(null);
  const [cut, setCut] = useState<any>(null);
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [bands, setBands] = useState({ spm: 0.5, ds: 0.2 });
  const load = async () => {
    const [r, c, rs] = await Promise.all([get(`/wells/${well}/controller/recommendation`), get(`/wells/${well}/cutoff`),
      get(`/wells/${well}/reservoir`)]);
    setRec(r);
    setCut(c);
    setRes(rs);
  };
  useInterval(load, 3000, [well]);
  const d = rec?.decision;
  const recipe = res?.recommendation;
  const trend = (cut?.trend ?? []).map((x: any) => ({ ...x, g_star: cut?.g_star }));

  const act = async (id: number, approve: boolean) => {
    await post(`/wells/${well}/controller/approve`, { action_id: id, approve });
    load();
  };
  const setMode = async (mode: string) => {
    await post(`/wells/${well}/mode`, { mode, band_spm: bands.spm, band_downstroke: bands.ds });
    load();
  };
  const optimize = async () => {
    setBusy(true);
    try { await post(`/wells/${well}/optimize-cycle`, {}); await load(); } finally { setBusy(false); }
  };

  return (
    <div className="grid md:grid-cols-2 gap-3">
      <Card title={`SRP recommendation - ${well}`} right={<span className="text-xs">mode: <b>{rec?.mode}</b></span>}>
        {d ? (
          <div className="text-sm space-y-1">
            <div>Setpoint: <Num value={d.safe_spm ?? d.spm} unit="SPM" tier="B" ph />, downstroke factor{" "}
              <Num value={Math.min(...(d.safe_profile?.downstroke ?? [1]))} tier="B" /></div>
            <div>Feedforward <Num value={d.spm_ff} tier="B" ph /> + PI <Num value={d.pi_term} tier="B" />;
              predicted RFI <Num value={d.rfi_pred} tier="B" ph />, stress ratio <Num value={d.stress_ratio_pred} tier="B" ph /></div>
            <div className="flex flex-wrap gap-1 mt-1">
              {(d.reason_codes ?? []).map((c: string) => <span key={c} className="text-[11px] bg-slate-800 px-1.5 rounded">{c}</span>)}
            </div>
            <div className="flex flex-wrap gap-1">
              {(d.active_constraints ?? []).map((c: string) => <span key={c} className="text-[11px] bg-amber-900/50 text-amber-200 px-1.5 rounded">{c}</span>)}
              {(d.violations ?? []).map((c: string) => <span key={c} className="text-[11px] bg-red-900/60 text-red-200 px-1.5 rounded">{c}</span>)}
            </div>
            <div className="text-xs text-slate-500">Inputs: {JSON.stringify(d.inputs)}</div>
          </div>
        ) : <div className="text-slate-500 text-sm">No decision yet (controller runs during production).</div>}
        <div className="mt-3">
          <div className="text-xs text-slate-400 mb-1">Pending actions (supervised / out-of-band)</div>
          {(rec?.pending ?? []).length === 0 && <div className="text-xs text-slate-500">None.</div>}
          {(rec?.pending ?? []).map((p: any) => (
            <div key={p.id} className="flex items-center gap-2 text-sm py-1 border-b border-slate-800">
              <span className="font-mono">#{p.id}</span>
              <span>SPM {p.spm.toFixed(2)}, ds {Math.min(...p.profile.downstroke).toFixed(2)}</span>
              <span className="text-xs text-slate-500">{p.reason_codes.join(", ")}</span>
              <Btn kind="primary" onClick={() => act(p.id, true)}>Approve</Btn>
              <Btn kind="danger" onClick={() => act(p.id, false)}>Reject</Btn>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Mode and bands">
        <div className="flex gap-2 mb-2">
          {["advisory", "supervised", "autonomous"].map((m) => (
            <Btn key={m} kind={rec?.mode === m ? "primary" : "ghost"} onClick={() => setMode(m)}>{m}</Btn>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <label>SPM band ±<input type="number" step={0.1} value={bands.spm} onChange={(e) => setBands({ ...bands, spm: +e.target.value })}
            className="ml-1 w-16 bg-slate-800 rounded px-1" /></label>
          <label>Downstroke band ±<input type="number" step={0.05} value={bands.ds} onChange={(e) => setBands({ ...bands, ds: +e.target.value })}
            className="ml-1 w-16 bg-slate-800 rounded px-1" /></label>
        </div>
        <div className="text-xs text-slate-400 mt-2">Advisory writes nothing. Supervised requires approval. Autonomous applies in-band actions; out-of-band ones become pending. Safety checks run after every output in every mode.</div>
        <div className="text-xs text-slate-400 mt-3 mb-1">Audit log</div>
        <div className="max-h-40 overflow-y-auto text-[11px] font-mono">
          {(rec?.audit ?? []).slice().reverse().map((a: any, i: number) => (
            <div key={i}>{a.t.slice(11, 19)} {a.event} {a.id ? `#${a.id}` : ""} {a.new ?? ""} {a.why ?? ""} {a.spm ? `spm=${Number(a.spm).toFixed(2)}` : ""}</div>
          ))}
        </div>
      </Card>

      <Card title="Cut-off: q_net (potential, EWMA) vs g*" right={cut?.stop_recommended
        ? <span className="text-xs text-amber-300 font-semibold">STOP RECOMMENDED</span> : <span className="text-xs text-slate-500">continue</span>}>
        <TimeChart rows={trend} series={[{ key: "q_net", name: "q_net (measured)" }, { key: "ewma", name: "EWMA (potential)" },
          { key: "g_star", name: "g*", dash: true }]} yLabel="USD/d" />
        <div className="text-xs text-slate-500">Stop when q_net is falling after its peak and stays &le; g* for the hold time (marginal value theorem). Tier B.</div>
      </Card>

      <Card title="Next-cycle recipe (outer Bayesian optimizer)" right={<Btn onClick={optimize} disabled={busy}>{busy ? "Optimizing..." : "Re-optimize"}</Btn>}>
        {recipe ? (
          <div className="text-sm grid grid-cols-2 gap-1">
            <div>Steam mass</div><div><Num value={recipe.recipe.steam_mass_t} unit="t" tier="B" ph /></div>
            <div>Injection pressure</div><div><Num value={recipe.recipe.inj_pressure_bar} unit="bar" tier="B" ph /></div>
            <div>Soak</div><div><Num value={recipe.recipe.soak_d} unit="d" tier="B" ph /></div>
            <div>Predicted g*</div><div><Num value={recipe.g_star} unit="USD/d" tier="B" ph /> ± {recipe.uncertainty.total_std.toFixed(0)}</div>
            <div>Cut-off T*</div><div><Num value={recipe.T_star_d} unit="d" tier="B" ph /></div>
            <div>Cycle oil</div><div><Num value={recipe.cycle_oil_m3} unit="m3" tier="B" ph /></div>
            <div>SOR</div><div><Num value={recipe.sor} tier="B" ph /></div>
            <div>kWh/bbl</div><div><Num value={recipe.kwh_per_bbl} tier="B" ph /></div>
            <div>Sandface quality</div><div><Num value={recipe.x_sandface} tier="B" ph /></div>
            <div>Constraints</div><div className="text-xs">{Object.entries(recipe.constraints).map(([k, v]) => <span key={k} className={v ? "text-emerald-400 mr-1" : "text-red-400 mr-1"}>{k}</span>)}</div>
            <div>Stroke (crank R)</div><div className="text-xs">{recipe.stroke_recommendation ? `${recipe.stroke_recommendation.R} m (advisory)` : "-"}</div>
            <div className="col-span-2 text-xs text-slate-500">GP std {recipe.uncertainty.gp_std.toFixed(1)}, calibration-sample std {recipe.uncertainty.param_std.toFixed(1)}; {recipe.n_evals} evaluations; {recipe.surrogate}</div>
          </div>
        ) : <div className="text-slate-500 text-sm">The recipe is optimized at the end of each cycle (receding horizon).</div>}
      </Card>
    </div>
  );
}

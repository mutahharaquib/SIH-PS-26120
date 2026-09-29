import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { get } from "../api/client";
import { TimeChart } from "../components/charts";
import { Card, Num, Q, Stat, useInterval } from "../components/ui";
import { useStore } from "../store";

export default function Reservoir() {
  const well = useStore((s) => s.selected);
  const [hist, setHist] = useState<any[]>([]);
  const [res, setRes] = useState<any>(null);
  const [st, setSt] = useState<any>(null);
  useInterval(async () => {
    const [h, r, s] = await Promise.all([
      get(`/wells/${well}/history?every=6&fields=T_avg_C,r_h,mu_pump_cp,mu_top_cp,q_oil,q_oil_pred,p_res_bar,T_wh_C`),
      get(`/wells/${well}/reservoir`), get(`/wells/${well}/state`).catch(() => null)]);
    setHist(h.rows.slice(-600));
    setRes(r);
    setSt(s);
  }, 4000, [well]);
  const cal = (res?.calibration ?? []).map((c: any) => ({ ...c, A_hi: c.A + c.sA, A_lo: c.A - c.sA }));

  return (
    <div className="grid gap-3">
      {st && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
          <Stat label="T avg heated zone"><Q q={st.reservoir.T_avg_heated} /></Stat>
          <Stat label="Heated radius"><Q q={st.reservoir.r_heated} /></Stat>
          <Stat label="Reservoir pressure"><Q q={st.reservoir.p_reservoir} scale={1e-5} unit="bar" /></Stat>
          <Stat label="Stimulation ratio"><Q q={st.reservoir.stimulation_ratio} /></Stat>
          <Stat label="Viscosity at pump"><Q q={st.wellbore.mu_at_pump} scale={1000} unit="cP" /></Stat>
        </div>
      )}
      <div className="grid md:grid-cols-2 gap-3">
        <Card title={`Heated zone: T_avg (degC) and r_h (m) - ${well}`} right={<span className="text-[10px] text-sky-300">Tier B</span>}>
          <TimeChart rows={hist} series={[{ key: "T_avg_C", name: "T_avg degC" }, { key: "r_h", name: "r_h m" }]} />
        </Card>
        <Card title="Viscosity along the rod string (cP)" right={<span className="text-[10px] text-sky-300">Tier B</span>}>
          <TimeChart rows={hist} series={[{ key: "mu_pump_cp", name: "at pump" }, { key: "mu_top_cp", name: "top of string" }]} />
        </Card>
        <Card title="Predicted (twin) vs observed oil rate (m3/d)" right={<span className="text-[10px]">A observed / B predicted</span>}>
          <TimeChart rows={hist} series={[{ key: "q_oil", name: "observed" }, { key: "q_oil_pred", name: "twin potential", dash: true }]} />
        </Card>
        <Card title="Cycle history: oil (m3) and SOR" right={<span className="text-[10px] text-sky-300">Tier B</span>}>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={(res?.cycles ?? []).map((c: any) => ({ ...c, name: `c${c.cycle}` }))}>
              <CartesianGrid stroke="#1e293b" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#94a3b8" }} />
              <YAxis yAxisId="l" tick={{ fontSize: 10, fill: "#94a3b8" }} />
              <YAxis yAxisId="r" orientation="right" tick={{ fontSize: 10, fill: "#94a3b8" }} />
              <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar yAxisId="l" dataKey="oil_m3" name="oil m3" fill="#38bdf8" />
              <Bar yAxisId="l" dataKey="pred_oil_m3" name="predicted oil m3" fill="#1e40af" />
              <Bar yAxisId="r" dataKey="sor" name="SOR" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Calibration: Walther A (RLS, ±1 sigma)" right={<span className="text-[10px] text-sky-300">Tier B</span>}>
          <TimeChart rows={cal} series={[{ key: "A", name: "A" }, { key: "A_hi", name: "+1σ", dash: true }, { key: "A_lo", name: "-1σ", dash: true }]} />
        </Card>
        <Card title="Calibration: inflow refits (J_c m3/d/bar)" right={<span className="text-[10px] text-sky-300">Tier B</span>}>
          <div className="text-xs font-mono max-h-52 overflow-y-auto">
            {(res?.refits ?? []).slice().reverse().map((r: any, i: number) => (
              <div key={i} className={r.accepted ? "" : "text-amber-300"}>
                {r.t.slice(0, 13)} {r.accepted ? "accepted" : `rejected (${r.reason})`} J_c=<Num value={r.J_c} tier="B" ph digits={4} />
                {r.J_c_std ? ` ±${r.J_c_std.toFixed(4)}` : ""} wc_late={r.wc_late?.toFixed(2)} tau={r.tau_d?.toFixed(1)}d rmse={r.rmse_rate?.toFixed(2)}
              </div>
            ))}
            {!(res?.refits ?? []).length && <div className="text-slate-500">No refits yet (every 7 days of production).</div>}
          </div>
        </Card>
      </div>
    </div>
  );
}

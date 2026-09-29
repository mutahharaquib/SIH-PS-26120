import { useMemo, useState } from "react";
import { WellRow } from "../api/client";
import { useStore } from "../store";
import { Card, Num, severityColor, Stat } from "../components/ui";

type Key = keyof WellRow;
const COLS: { key: Key; label: string }[] = [
  { key: "risk_rank", label: "Risk rank" }, { key: "well_id", label: "Well" }, { key: "phase", label: "Phase" },
  { key: "q_oil_m3d", label: "q_oil" }, { key: "sor", label: "SOR" }, { key: "kwh_per_bbl", label: "kWh/bbl" },
  { key: "rfi", label: "RFI" }, { key: "spm", label: "SPM" }, { key: "fillage", label: "Fillage" },
  { key: "fault", label: "Card class" }, { key: "mode", label: "Mode" },
];

export default function FieldOverview() {
  const { wells, select, setView } = useStore();
  const [sort, setSort] = useState<{ key: Key; asc: boolean }>({ key: "risk_rank", asc: true });
  const rows = useMemo(() => {
    const r = [...wells];
    r.sort((a, b) => {
      const va = a[sort.key] as any, vb = b[sort.key] as any;
      const c = typeof va === "number" ? va - vb : String(va ?? "").localeCompare(String(vb ?? ""));
      return sort.asc ? c : -c;
    });
    return r;
  }, [wells, sort]);
  const producing = wells.filter((w) => w.phase === "produce");
  const totOil = producing.reduce((s, w) => s + w.q_oil_m3d, 0);
  const alarms = wells.flatMap((w) => w.alarms.map((a) => ({ ...a, well: w.well_id })));

  return (
    <div className="grid gap-3">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
        <Stat label="Field oil rate"><Num value={totOil} unit="m3/d" tier="A" /></Stat>
        <Stat label="Producing wells"><span className="font-mono">{producing.length} / {wells.length}</span></Stat>
        <Stat label="Wells with RFI >= 0.8"><span className="font-mono">{wells.filter((w) => w.rfi >= 0.8).length}</span></Stat>
        <Stat label="Pending approvals"><span className="font-mono">{wells.reduce((s, w) => s + w.pending, 0)}</span></Stat>
        <Stat label="Cut-off recommended"><span className="font-mono">{wells.filter((w) => w.stop_recommended).length}</span></Stat>
      </div>
      <Card title="Wells (click a row to open; risk is a heuristic ranking, not a failure prediction)">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b border-slate-800">
                {COLS.map((c) => (
                  <th key={c.key} className="py-1.5 pr-3 cursor-pointer select-none"
                    onClick={() => setSort((s) => ({ key: c.key, asc: s.key === c.key ? !s.asc : true }))}>
                    {c.label}{sort.key === c.key ? (sort.asc ? " ▲" : " ▼") : ""}
                  </th>
                ))}
                <th className="py-1.5">Alarms</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((w) => (
                <tr key={w.well_id} className="border-b border-slate-800/60 hover:bg-slate-800/40 cursor-pointer"
                  onClick={() => { select(w.well_id); setView("diagnostics"); }}>
                  <td className="py-1.5 pr-3 font-mono">{w.risk_rank} <span className="text-slate-500 text-xs">({w.risk.toFixed(3)})</span></td>
                  <td className="pr-3 font-semibold">{w.well_id}</td>
                  <td className="pr-3"><span className={phaseColor(w.phase)}>{w.phase}</span> <span className="text-slate-500 text-xs">c{w.cycle}</span></td>
                  <td className="pr-3"><Num value={w.q_oil_m3d} tier="A" /></td>
                  <td className="pr-3"><Num value={w.sor} tier="B" ph /></td>
                  <td className="pr-3"><Num value={w.kwh_per_bbl} tier="B" ph /></td>
                  <td className={`pr-3 ${w.rfi >= 1 ? "text-red-400" : w.rfi >= 0.8 ? "text-amber-300" : ""}`}><Num value={w.rfi} tier="B" ph /></td>
                  <td className="pr-3 font-mono">{w.spm.toFixed(2)}{w.downstroke < 0.999 && <span className="text-amber-300 text-xs"> ds {w.downstroke.toFixed(2)}</span>}</td>
                  <td className="pr-3"><Num value={w.fillage} tier="B" /></td>
                  <td className="pr-3">{w.fault ?? "-"} {w.fault && <span className="text-[10px] px-1 rounded border bg-violet-900/60 text-violet-300 border-violet-700">C</span>}</td>
                  <td className="pr-3 text-xs">{w.mode}</td>
                  <td className="text-xs">{w.alarms.slice(-1).map((a) => <span key={a.t} className={severityColor[a.severity]}>{a.code}</span>)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Card title="Alarm tray (latest)">
        <div className="max-h-48 overflow-y-auto text-xs font-mono space-y-0.5">
          {alarms.length === 0 && <div className="text-slate-500">No alarms.</div>}
          {alarms.sort((a, b) => b.t.localeCompare(a.t)).slice(0, 40).map((a, i) => (
            <div key={i} className={severityColor[a.severity]}>{a.t.slice(0, 16)} {a.well} {a.code}: {a.message}</div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function phaseColor(p: string) {
  return p === "produce" ? "text-emerald-300" : p === "inject" ? "text-orange-300" : p === "soak" ? "text-yellow-300" : "text-slate-400";
}

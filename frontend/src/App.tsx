import { useEffect } from "react";
import { Btn, severityColor } from "./components/ui";
import { useStore, View } from "./store";
import About from "./views/About";
import Decision from "./views/Decision";
import Diagnostics from "./views/Diagnostics";
import Evaluation from "./views/Evaluation";
import FieldOverview from "./views/FieldOverview";
import Reservoir from "./views/Reservoir";
import WhatIf from "./views/WhatIf";

const TABS: { v: View; label: string }[] = [
  { v: "field", label: "Field overview" }, { v: "reservoir", label: "Reservoir" }, { v: "diagnostics", label: "Diagnostics" },
  { v: "decision", label: "Decision console" }, { v: "whatif", label: "What-if" }, { v: "evaluation", label: "Evaluation" },
  { v: "about", label: "About" },
];

export default function App() {
  const { view, setView, wells, selected, select, sim, simControl, wsOk, refresh, connect } = useStore();
  useEffect(() => { refresh(); connect(); }, []);
  const w = wells.find((x) => x.well_id === selected);
  const lastAlarm = wells.flatMap((x) => x.alarms.map((a) => ({ ...a, well: x.well_id }))).sort((a, b) => b.t.localeCompare(a.t))[0];

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 bg-slate-950/95 border-b border-slate-800 px-4 py-2 flex flex-wrap items-center gap-3">
        <div className="font-bold text-sky-300">CSS + SRP Digital Twin</div>
        <span className="text-xs text-slate-500">SIH 26120 · Baghewala (synthetic)</span>
        <select value={selected} onChange={(e) => select(e.target.value)} className="bg-slate-800 rounded px-2 py-1 text-sm">
          {wells.map((x) => <option key={x.well_id}>{x.well_id}</option>)}
        </select>
        {w && <span className="text-xs">mode <b className="text-amber-300">{w.mode}</b> · {w.phase} c{w.cycle}</span>}
        <div className="flex items-center gap-1 ml-auto">
          <span className="text-xs font-mono text-slate-400">{sim?.sim_time?.slice(0, 16).replace("T", " ")}</span>
          <Btn kind={sim?.playing ? "danger" : "primary"} onClick={() => simControl({ playing: !sim?.playing })}>{sim?.playing ? "Pause" : "Play"}</Btn>
          <Btn onClick={() => simControl({ step_hours: 24 })}>+1 day</Btn>
          <select value={sim?.seconds_per_day ?? 10} onChange={(e) => simControl({ seconds_per_day: +e.target.value })}
            className="bg-slate-800 rounded px-1 py-1 text-xs">
            {[20, 10, 4, 2, 1, 0.5].map((s) => <option key={s} value={s}>{s} s/day</option>)}
          </select>
          <Btn onClick={() => simControl({ scenario: "demo_float" })} title="Load the seeded cooling/rod-float demo">Demo scenario</Btn>
          <span title="live stream" className={`w-2 h-2 rounded-full ${wsOk ? "bg-emerald-400" : "bg-red-500"}`} />
        </div>
        {lastAlarm && <div className={`w-full text-xs ${severityColor[lastAlarm.severity]}`}>Latest alarm: {lastAlarm.well} {lastAlarm.code} - {lastAlarm.message}</div>}
        <nav className="w-full flex gap-1">
          {TABS.map((t) => (
            <button key={t.v} onClick={() => setView(t.v)}
              className={`px-3 py-1 text-sm rounded-t ${view === t.v ? "bg-slate-800 text-sky-300" : "text-slate-400 hover:text-slate-200"}`}>{t.label}</button>
          ))}
        </nav>
      </header>
      <main className="p-4">
        {view === "field" && <FieldOverview />}
        {view === "reservoir" && <Reservoir />}
        {view === "diagnostics" && <Diagnostics />}
        {view === "decision" && <Decision />}
        {view === "whatif" && <WhatIf />}
        {view === "evaluation" && <Evaluation />}
        {view === "about" && <About />}
      </main>
    </div>
  );
}

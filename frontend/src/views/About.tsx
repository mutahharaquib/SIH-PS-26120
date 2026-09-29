import { useEffect, useState } from "react";
import { get } from "../api/client";
import { Card } from "../components/ui";

export default function About() {
  const [about, setAbout] = useState<any>(null);
  const [cfg, setCfg] = useState<any>(null);
  useEffect(() => { get("/about").then(setAbout); get("/config").then(setCfg); }, []);
  return (
    <div className="grid md:grid-cols-2 gap-3">
      <Card title="About this twin">
        <p className="text-sm text-slate-300">Coupled well-to-surface digital twin for Cyclic Steam Stimulation (CSS) with sucker-rod pumps
          (SIH 26120, Baghewala field). Reservoir heating/cooling, wellbore heat loss, viscosity, rod dynamics, pump diagnostics,
          cycle optimization and SRP control share one well state. Problem-statement facts (Tier A): 17-19 degAPI crude,
          46-48 degC reservoir, Jodhpur Sandstone, low pressure, high viscosity and asphaltene content.</p>
        <h4 className="text-sm font-semibold mt-3 mb-1">Confidence tiers</h4>
        <ul className="text-sm list-disc ml-5">{about && Object.entries(about.tiers).map(([k, v]) => <li key={k}><b>{k}</b>: {v as string}</li>)}</ul>
        <h4 className="text-sm font-semibold mt-3 mb-1">Known limitations</h4>
        <ul className="text-sm list-disc ml-5 space-y-0.5">{about?.limitations.map((l: string) => <li key={l}>{l}</li>)}</ul>
        <h4 className="text-sm font-semibold mt-3 mb-1">Phase 2 (not built, excluded from metrics)</h4>
        <ul className="text-sm list-disc ml-5">{about?.phase2_not_built.map((l: string) => <li key={l}>{l}</li>)}</ul>
      </Card>
      <Card title={`Placeholder parameters (${cfg?.placeholders.length ?? 0}) - every value derived from these carries a PH badge`}>
        <div className="max-h-[70vh] overflow-y-auto text-xs font-mono">
          {cfg?.placeholders.map((p: string) => <div key={p}>{p}</div>)}
        </div>
      </Card>
    </div>
  );
}

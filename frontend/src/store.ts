import { create } from "zustand";
import { get, post, SimControl, WellRow } from "./api/client";

export type View = "field" | "reservoir" | "diagnostics" | "decision" | "whatif" | "evaluation" | "about";

interface State {
  view: View;
  wells: WellRow[];
  selected: string;
  sim: SimControl | null;
  wsOk: boolean;
  units: "si" | "field";
  setView: (v: View) => void;
  select: (w: string) => void;
  setUnits: (u: "si" | "field") => void;
  refresh: () => Promise<void>;
  simControl: (body: Record<string, unknown>) => Promise<void>;
  connect: () => void;
}

export const useStore = create<State>((set, getState) => ({
  view: "field",
  wells: [],
  selected: "W01",
  sim: null,
  wsOk: false,
  units: "si",
  setView: (view) => set({ view }),
  select: (selected) => set({ selected }),
  setUnits: (units) => set({ units }),
  refresh: async () => {
    const [w, s] = await Promise.all([get<{ wells: WellRow[] }>("/wells"), get<SimControl>("/sim")]);
    set({ wells: w.wells, sim: s });
  },
  simControl: async (body) => {
    const s = await post<SimControl>("/sim", body);
    set({ sim: s });
    await getState().refresh();
  },
  connect: () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
    ws.onopen = () => set({ wsOk: true });
    ws.onclose = () => {
      set({ wsOk: false });
      setTimeout(() => getState().connect(), 2000);
    };
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.wells) {
        const sorted = [...m.wells].sort((a: WellRow, b: WellRow) => a.well_id.localeCompare(b.well_id));
        set((st) => ({
          wells: sorted,
          sim: st.sim ? { ...st.sim, sim_time: m.t ?? st.sim.sim_time, tick: m.tick ?? st.sim.tick } : m.control ?? st.sim,
        }));
      }
    };
  },
}));

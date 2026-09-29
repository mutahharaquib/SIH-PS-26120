export const API = "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(API + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const get = <T = any>(p: string) => req<T>(p);
export const post = <T = any>(p: string, body: unknown) =>
  req<T>(p, { method: "POST", body: JSON.stringify(body) });

export type Tier = "A" | "B" | "C";

export interface Quantity {
  value: number;
  unit: string;
  uncertainty?: number | null;
  prov: { tier: Tier; source: string; extrapolated: boolean; placeholder: boolean; timestamp: string };
}

export interface WellRow {
  well_id: string;
  phase: string;
  cycle: number;
  q_oil_m3d: number;
  sor: number;
  kwh_per_bbl: number;
  rfi: number;
  spm: number;
  downstroke: number;
  fillage: number;
  fault: string | null;
  risk: number;
  risk_rank: number;
  mode: string;
  alarms: { code: string; severity: string; message: string; t: string }[];
  T_avg_K: number;
  g_star: number | null;
  stop_recommended: boolean;
  pending: number;
  tiers: Record<string, Tier>;
}

export interface SimControl {
  playing: boolean;
  seconds_per_day: number;
  sim_time: string;
  tick: number;
  scenario: string | null;
  n_wells: number;
}

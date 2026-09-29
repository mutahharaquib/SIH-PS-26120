"""Baseline-vs-twin evaluation harness (BUILD_SPEC §12).

The same simulated wells, seeds, noise and events are run under two policies:
the current-practice baseline (§8.4) and the twin (§8 + §9, autonomous in-band).
A shadow twin in advisory mode rides along the baseline runs to measure RFI lead time
on labeled float events (it never acts). Every metric is reported for
production_weight = 0 (value) and the configured production-weighted setting.

Output: reports/<run_id>/report.json + PNG charts.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import yaml

from core.config import get_config
from core.units import DAY
from diagnostics.rfi import lead_times
from evaluation.classifier_eval import ensure_model
from evaluation.metrics import mean_ci, paired_delta, run_metrics, spearman
from optimization.baseline import BaselinePolicy
from optimization.objective import Econ
from physics.pumping_unit import stroke_length
from simulation.field_sim import T0, FieldSimulator
from simulation.labels import ScheduledEvent

REPORTS = Path(__file__).resolve().parents[2] / "reports"
KEY_METRICS = ["oil_m3", "npv_usd", "sor", "kwh_per_bbl", "float_hours", "float_events", "impact_events",
               "pound_hours", "downtime_hours", "failures", "volumetric_efficiency", "peak_stress_ratio_p95"]


@dataclass
class HarnessConfig:
    n_wells: int = 8
    seeds: list[int] = field(default_factory=lambda: [1])
    horizon_days: float = 360.0
    weights: list[float] | None = None
    workers: int | None = None
    wells: list[str] | None = None
    extra_events: dict | None = None
    noise_scale: float = 1.0
    shadow: bool = True
    label: str = ""


def _events(extra: dict | None) -> dict | None:
    if not extra:
        return None
    return {wid: [ScheduledEvent(e["kind"], T0 + timedelta(days=e["start_day"]), T0 + timedelta(days=e["end_day"]),
                                 e.get("magnitude", 0.0)) for e in evs] for wid, evs in extra.items()}


def run_well(task: dict) -> dict:
    """One (seed, well, policy, weight) run. Top-level so it can run in a worker process."""
    from engine.twin import WellTwin  # local: keeps worker start-up light

    cfg = get_config()
    f = FieldSimulator(cfg, n_wells=task["n_wells"], seed=task["seed"], noise_scale=task.get("noise_scale", 1.0),
                       extra_events=_events(task.get("extra_events")))
    wid = task["well"]
    sim = f.wells[wid]
    meta = f.meta[wid]
    clf = ensure_model()
    econ = Econ.from_config(cfg, production_weight=task["weight"])
    shadow = None
    if task["policy"] == "baseline":
        pol = BaselinePolicy(meta, cfg=cfg)
        if task.get("shadow"):
            shadow = WellTwin(meta, cfg, clf, mode="advisory", auto_cycle=False, optimize=False, publish=False)
        step_policy = pol.on_telemetry
    else:
        tw = WellTwin(meta, cfg, clf, mode="autonomous", bo_fast=True, auto_cycle=True, econ=econ, publish=False)
        step_policy = tw.on_telemetry
    horizon = T0 + timedelta(days=task["horizon_days"])
    t_start = time.time()
    rfi_t, rfi_v, cls_pairs = [], [], []
    while sim.t < horizon:
        tel = sim.step()
        cmds = step_policy(tel)
        sim.apply(cmds)
        if shadow is not None:
            shadow.on_telemetry(tel)
            if tel.phase.value == "produce":
                rfi_t.append(sim.truth[-1].t)
                rfi_v.append(shadow.rfi)
                if tel.card is not None:
                    cls_pairs.append((sim.truth[-1].fault, shadow.fault_class))
    stroke = stroke_length(sim.wp.unit)
    m = run_metrics(sim.truth, sim.cycles, econ, cfg.v("economics.failure_cost"), sim.wp.A_p, stroke)
    out = {"task": task, "metrics": m, "runtime_s": time.time() - t_start}
    if task["policy"] == "twin":
        r = tw.risk.score()
        out["twin"] = {"risk": r["risk"], "cycles": tw.cycles, "refits": tw.refits[-5:],
                       "rls": {"A": tw.rls.A, "B": tw.rls.B}}
    if shadow is not None:
        onsets = [lab.onset for lab in sim.derived_labels() if lab.kind == "rod_float"]
        out["shadow"] = {"lead_times_h": lead_times(rfi_t, rfi_v, onsets, cfg.v("controller.rfi_warn")),
                         "n_onsets": len(onsets), "cls_pairs": cls_pairs,
                         "rfi_warn_hours": float(sum(v >= cfg.v("controller.rfi_warn") for v in rfi_v)),
                         "false_alarm_hours": _false_alarm_hours(rfi_t, rfi_v, onsets, cfg.v("controller.rfi_warn"))}
    out["truth_exposure"] = m.get("true_damage", 0) * 100 + m.get("float_hours", 0) + m.get("impact_events", 0) \
        + 0.1 * m.get("pound_hours", 0)
    return out


def _false_alarm_hours(times, values, onsets, warn, window_h: float = 72.0) -> float:
    ons = np.array([o.timestamp() for o in onsets])
    n = 0.0
    for t, v in zip(times, values):
        if v >= warn:
            ts = t.timestamp()
            if not np.any((ons >= ts) & (ons <= ts + window_h * 3600)) and not np.any(np.abs(ons - ts) < 3600 * 24):
                n += 1
    return n


def physics_verification() -> dict:
    """Wave-equation round-trip error table and energy-balance closure (wellbore, reservoir)."""
    from core.units import BAR
    from physics.params import build_well_params
    from physics.pump import DownholePump, PumpCondition
    from physics.pumping_unit import kinematics
    from physics.reservoir import marx_langenheim_area
    from physics.rod_string import diagnostic, predictive
    from physics.wellbore import produced_fluid_profile, steam_injection

    cfg = get_config()
    wp = build_well_params(cfg, "V01")
    grid = wp.rods.discretize(40)
    rows = []
    for c_level in (0.0, 0.5, 1.0, 2.0):
        kin = kinematics(wp.unit, 4.0, dt=grid.dt_cfl)
        c = np.full(len(grid.z), c_level)
        w = grid.node_weights(950.0)
        pump = DownholePump(wp.A_p, 14e3, cond=PumpCondition(0.8), S_p=kin.stroke)
        n_str = 1 if c_level == 0 else 8
        fwd = predictive(grid, kin.x, kin.t[1] - kin.t[0], c, w, pump, n_strokes=n_str, allow_float=False)
        d = diagnostic(grid, fwd.surface_pos, fwd.surface_load, kin.period, c, w, resample=False, periodic=False)
        ok = np.isfinite(d.pump_load)
        err = np.abs(d.pump_load[ok] - fwd.pump_load[ok]) / 14e3
        rows.append({"damping_1_s": c_level, "max_rel_err": float(err.max()), "median_rel_err": float(np.median(err)),
                     "mode": "same grid (a dt = dz), interior samples"})
    # realistic path: forward at CFL 0.9, 1 % load noise, Fourier resampling (20 harmonics) to a dt = dz
    rng = np.random.default_rng(0)
    for c_level in (0.5, 1.5):
        kin = kinematics(wp.unit, 4.0, dt=grid.dt_cfl * 0.9)
        c = np.full(len(grid.z), c_level)
        w = grid.node_weights(950.0)
        pump = DownholePump(wp.A_p, 14e3, cond=PumpCondition(0.8), S_p=kin.stroke)
        fwd = predictive(grid, kin.x, kin.t[1] - kin.t[0], c, w, pump, n_strokes=8, allow_float=False)
        noisy = fwd.surface_load + rng.normal(0, 0.01 * np.ptp(fwd.surface_load), len(fwd.surface_load))
        d = diagnostic(grid, fwd.surface_pos, noisy, kin.period, c, w, n_harmonics=20)
        ref = np.interp(np.linspace(0, 1, len(d.pump_load), endpoint=False),
                        np.linspace(0, 1, len(fwd.pump_load), endpoint=False), fwd.pump_load)
        err = np.abs(d.pump_load - ref) / 14e3
        rows.append({"damping_1_s": c_level, "max_rel_err": float(err.max()), "median_rel_err": float(np.median(err)),
                     "mode": "resampled + 1% noise, 20 harmonics (realistic)"})
    closure = []
    for U in (5.0, 12.0, 25.0):
        prof = steam_injection(wp.geom, 2.9, 90 * BAR, 0.8, 5 * DAY, U)
        closure.append({"case": f"injection U={U}", "rel_err": prof.closure_rel_err})
    for q in (5.0, 20.0):
        pp = produced_fluid_profile(wp.geom, q / DAY * 950, 3000.0, 480.0, 30 * DAY, wp.U_produce, wp.pump_depth)
        closure.append({"case": f"production q={q} m3/d", "rel_err": pp.closure_rel_err})
    Q, dT, t = 4e6, 250.0, 60.0
    A = marx_langenheim_area(Q, wp.res.M_R, wp.res.h, wp.res.lambda_ob, wp.res.M_ob, dT, t)
    closure.append({"case": "reservoir ML early-time energy", "rel_err": abs(A * wp.res.h * wp.res.M_R * dT / (Q * t) - 1)})
    return {"wave_roundtrip": rows, "energy_closure": closure, "tier": "B"}


def aggregate(results: list[dict], weights: list[float]) -> dict:
    out: dict = {"by_weight": {}}
    for wgt in weights:
        base = {(r["task"]["seed"], r["task"]["well"]): r for r in results
                if r["task"]["policy"] == "baseline"}
        twin = {(r["task"]["seed"], r["task"]["well"]): r for r in results
                if r["task"]["policy"] == "twin" and r["task"]["weight"] == wgt}
        keys = sorted(set(base) & set(twin))
        summary = {}
        for mname in KEY_METRICS:
            a = [base[k]["metrics"][mname] for k in keys]
            b = [twin[k]["metrics"][mname] for k in keys]
            summary[mname] = {"baseline": mean_ci(a), "twin": mean_ci(b), "delta": paired_delta(a, b)}
        risk = [twin[k]["twin"]["risk"] for k in keys]
        expo = [twin[k]["truth_exposure"] for k in keys]
        per_well = [{"seed": k[0], "well": k[1],
                     "baseline": {m: base[k]["metrics"][m] for m in KEY_METRICS},
                     "twin": {m: twin[k]["metrics"][m] for m in KEY_METRICS},
                     "twin_cycles": twin[k]["twin"]["cycles"], "baseline_cycles": base[k]["metrics"]["cycles"],
                     "twin_risk": twin[k]["twin"]["risk"]} for k in keys]
        out["by_weight"][str(wgt)] = {
            "summary": summary, "per_well": per_well,
            "risk_rank_spearman_vs_true_exposure": spearman(risk, expo),
            "cum_oil": {"baseline": _mean_curve([base[k]["metrics"]["daily_oil"] for k in keys]),
                        "twin": _mean_curve([twin[k]["metrics"]["daily_oil"] for k in keys])},
        }
    sh = [r["shadow"] for r in results if "shadow" in r]
    lts = [x for s in sh for x in s["lead_times_h"]]
    hit = [x for x in lts if x is not None]
    out["rfi_lead_time"] = {
        "n_onsets": len(lts), "detected": len(hit), "missed": len(lts) - len(hit),
        "lead_time_h": mean_ci(hit), "lead_times_h": hit,
        "p10": float(np.percentile(hit, 10)) if hit else None, "p50": float(np.percentile(hit, 50)) if hit else None,
        "p90": float(np.percentile(hit, 90)) if hit else None,
        "false_alarm_hours": float(sum(s["false_alarm_hours"] for s in sh)),
        "warn_hours": float(sum(s["rfi_warn_hours"] for s in sh)),
        "note": "RFI uses the minimum fall velocity over depth (conservative); measured on baseline runs by an "
                "advisory-only shadow twin.",
    }
    pairs = [p for s in sh for p in s["cls_pairs"]]
    if pairs:
        labels = ["normal", "fluid_pound", "gas_interference", "rod_float", "pump_unseated"]
        acc = float(np.mean([a == b for a, b in pairs]))
        per = {lab: {"n": sum(a == lab for a, _ in pairs),
                     "recall": (sum(a == lab and b == lab for a, b in pairs) / max(sum(a == lab for a, _ in pairs), 1))}
               for lab in labels}
        out["classifier_online"] = {"accuracy": acc, "per_class": per, "n_cards": len(pairs), "tier": "C"}
    return out


def _mean_curve(series: list[list[float]]) -> list[float]:
    if not series:
        return []
    n = min(len(s) for s in series)
    return np.cumsum(np.mean([s[:n] for s in series], axis=0)).tolist()


def charts(report: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    w0 = report["results"]["by_weight"][str(report["config"]["weights"][0])]
    s = w0["summary"]
    names = ["oil_m3", "npv_usd", "sor", "kwh_per_bbl", "float_hours", "downtime_hours"]
    rel = [100 * (s[n]["delta"]["rel"] or 0) for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, rel, color=["#2a7" if (v >= 0) == (n in ("oil_m3", "npv_usd")) else "#c54" for n, v in zip(names, rel)])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("twin vs baseline (%)")
    ax.set_title("Twin vs baseline (value objective, paired by well/seed)")
    fig.tight_layout()
    fig.savefig(out_dir / "deltas.png", dpi=110)
    files.append("deltas.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(w0["cum_oil"]["baseline"], label="baseline")
    ax.plot(w0["cum_oil"]["twin"], label="twin")
    ax.set_xlabel("day")
    ax.set_ylabel("cumulative oil per well (m3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cum_oil.png", dpi=110)
    files.append("cum_oil.png")
    plt.close(fig)
    lt = report["results"]["rfi_lead_time"]["lead_times_h"]
    if lt:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(lt, bins=15, color="#468")
        ax.set_xlabel("RFI warn -> float onset lead time (h)")
        ax.set_ylabel("events")
        fig.tight_layout()
        fig.savefig(out_dir / "rfi_lead_time.png", dpi=110)
        files.append("rfi_lead_time.png")
        plt.close(fig)
    cm = (report.get("classifier") or {}).get("held_out_by_well", {}).get("confusion")
    if cm:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        M = np.array(cm["matrix"])
        ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(len(cm["labels"])), cm["labels"], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(cm["labels"])), cm["labels"], fontsize=7)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7)
        ax.set_title("Classifier confusion (held-out wells)")
        fig.tight_layout()
        fig.savefig(out_dir / "confusion.png", dpi=110)
        files.append("confusion.png")
        plt.close(fig)
    return files


def run(hc: HarnessConfig, run_id: str | None = None, progress=None) -> dict:
    cfg = get_config()
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    out_dir = REPORTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = hc.weights or [cfg.v("economics.production_weight"), cfg.v("economics.production_weight_alt")]
    ensure_model()
    wells = hc.wells or [f"W{i + 1:02d}" for i in range(hc.n_wells)]
    tasks = []
    for seed in hc.seeds:
        for wid in wells:
            common = dict(seed=seed, well=wid, n_wells=max(hc.n_wells, len(wells)), horizon_days=hc.horizon_days,
                          extra_events=hc.extra_events, noise_scale=hc.noise_scale)
            tasks.append({**common, "policy": "baseline", "weight": weights[0], "shadow": hc.shadow})
            for wgt in weights:
                tasks.append({**common, "policy": "twin", "weight": wgt})
    t0 = time.time()
    results = []
    workers = hc.workers or max(1, min(len(tasks), (os.cpu_count() or 2) - 1))
    if workers == 1:
        for i, t in enumerate(tasks):
            results.append(run_well(t))
            if progress:
                progress(i + 1, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(run_well, tasks)):
                results.append(r)
                if progress:
                    progress(i + 1, len(tasks))
    clf_report_path = Path(__file__).resolve().parents[1] / "models" / "card_clf_report.json"
    report = {
        "run_id": run_id, "label": hc.label, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"n_wells": len(wells), "seeds": hc.seeds, "horizon_days": hc.horizon_days, "weights": weights,
                   "noise_scale": hc.noise_scale, "extra_events": hc.extra_events},
        "results": aggregate(results, weights),
        "physics_verification": physics_verification(),
        "classifier": json.loads(clf_report_path.read_text()) if clf_report_path.exists() else None,
        "runtime_s": time.time() - t0,
        "tiers": {"production/economics": "B (physics with placeholder parameters)", "classifier": "C",
                  "risk": "heuristic ranking"},
        "phase2_excluded": ["DP cut-off", "multi-cycle lookahead", "survival model", "field steam allocation"],
    }
    report["charts"] = charts(report, out_dir)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def run_scenario(path: str | Path, run_id: str | None = None, workers: int | None = None) -> dict:
    """Demo scenario (e.g. scenarios/demo_float.yaml): baseline vs twin on the same seeded events."""
    sc = yaml.safe_load(Path(path).read_text())
    hc = HarnessConfig(n_wells=sc.get("n_wells", 6), seeds=[sc["seed"]], horizon_days=sc["horizon_days"],
                       wells=[sc["well"]], extra_events={sc["well"]: sc["events"]}, weights=[0.0], workers=workers,
                       shadow=True, label=sc.get("name", "scenario"))
    rep = run(hc, run_id=run_id or f"scenario-{sc.get('name', 'demo')}")
    w = rep["results"]["by_weight"]["0.0"]["per_well"][0]
    b, t = w["baseline"], w["twin"]
    rep["scenario"] = {
        "name": sc.get("name"), "well": sc["well"], "events": sc["events"],
        "avoided": {"float_hours": b["float_hours"] - t["float_hours"],
                    "float_events": b["float_events"] - t["float_events"],
                    "impact_events": b["impact_events"] - t["impact_events"]},
        "impact": {"npv_usd": t["npv_usd"] - b["npv_usd"], "sor": t["sor"] - b["sor"],
                   "kwh_per_bbl": t["kwh_per_bbl"] - b["kwh_per_bbl"], "oil_m3": t["oil_m3"] - b["oil_m3"]},
    }
    (REPORTS / rep["run_id"] / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--days", type=float, default=360)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--scenario", type=str, default=None)
    a = ap.parse_args()
    if a.scenario:
        r = run_scenario(a.scenario, workers=a.workers)
        print(json.dumps(r["scenario"], indent=1))
    else:
        r = run(HarnessConfig(n_wells=a.wells, seeds=a.seeds, horizon_days=a.days, workers=a.workers),
                progress=lambda i, n: print(f"  {i}/{n} runs done", flush=True))
        s = r["results"]["by_weight"][str(r["config"]["weights"][0])]["summary"]
        for k in KEY_METRICS:
            print(f"{k:24s} baseline={s[k]['baseline']['mean']:.3g} twin={s[k]['twin']['mean']:.3g} "
                  f"delta={s[k]['delta']['mean']:.3g} rel={s[k]['delta']['rel']}")
        print("run_id", r["run_id"], "runtime", round(r["runtime_s"]), "s")

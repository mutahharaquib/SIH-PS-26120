"""Evaluation metrics from simulator ground truth (BUILD_SPEC §12)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from core.units import BBL
from optimization.objective import Econ
from simulation.truth import CycleTruth, HiddenTruth


def run_metrics(truth: list[HiddenTruth], cycles: list[CycleTruth], econ: Econ, failure_cost: float,
                A_p: float, stroke: float) -> dict:
    """Metrics for one well-run (fixed horizon)."""
    if not truth:
        return {}
    t0 = truth[0].t
    oil = water = steam = kwh = float_h = pound_h = down_h = 0.0
    impacts = 0
    npv = 0.0
    daily_oil: dict[int, float] = {}
    started = set()
    prod_liquid = theo = 0.0
    peak_sr: list[float] = []
    in_float = False
    float_events = 0
    for h in truth:
        dt_d = h.dt_h / 24.0
        day = (h.t - t0).total_seconds() / 86400.0
        disc = (1 + econ.discount_rate) ** (-day / 365.0)
        o, w = h.q_oil * dt_d, h.q_water * dt_d
        oil += o
        water += w
        steam += h.steam_t
        kwh += h.energy_kwh
        cf = econ.oil_price_m3 * o - econ.tariff_kwh * h.energy_kwh - econ.opex_d * dt_d - econ.water_m3 * w \
            - econ.steam_cost_t * h.steam_t
        if h.cycle not in started and h.phase == "inject":
            started.add(h.cycle)
            cf -= econ.fixed_cycle
        npv += cf * disc
        daily_oil[int(day)] = daily_oil.get(int(day), 0.0) + o
        if h.phase == "produce":
            fl = h.fault == "rod_float"
            float_h += h.dt_h if fl else 0.0
            if fl and not in_float:
                float_events += 1
            in_float = fl
            impacts += int(h.impact_load > 0 and fl)
            pound_h += h.dt_h if h.fault == "fluid_pound" else 0.0
            down_h += h.downtime_h
            if h.strokes > 0:
                prod_liquid += (h.q_oil + h.q_water) * dt_d
                theo += A_p * stroke * h.strokes
                peak_sr.append(h.stress_ratio)
    failures = sum(c.failures for c in cycles)
    npv -= failures * failure_cost
    days = (truth[-1].t - t0).total_seconds() / 86400.0 + truth[-1].dt_h / 24
    done = [c for c in cycles if c.end is not None]
    return {
        "days": days, "oil_m3": oil, "water_m3": water, "steam_t": steam, "energy_kwh": kwh,
        "sor": steam / max(oil, 1e-9), "kwh_per_bbl": kwh / max(oil / BBL, 1e-9), "npv_usd": npv,
        "float_hours": float_h, "float_events": float_events, "impact_events": impacts, "pound_hours": pound_h,
        "downtime_hours": down_h, "failures": failures, "cycles_completed": len(done),
        "volumetric_efficiency": prod_liquid / max(theo, 1e-9),
        "peak_stress_ratio_p50": float(np.percentile(peak_sr, 50)) if peak_sr else 0.0,
        "peak_stress_ratio_p95": float(np.percentile(peak_sr, 95)) if peak_sr else 0.0,
        "peak_stress_ratio_max": float(np.max(peak_sr)) if peak_sr else 0.0,
        "true_damage": max((h.damage_max for h in truth), default=0.0),
        "cycles": [{"cycle": c.cycle, "steam_t": c.steam_t, "p_inj_bar": c.inj_pressure_bar, "soak_d": c.soak_d,
                    "produce_d": c.produce_d, "oil_m3": c.oil_m3, "sor": c.steam_t / max(c.oil_m3, 1e-9),
                    "kwh_per_bbl": c.energy_kwh / max(c.oil_m3 / BBL, 1e-9), "float_h": c.float_h,
                    "complete": c.end is not None} for c in cycles],
        "daily_oil": [daily_oil.get(d, 0.0) for d in range(int(days) + 1)],
    }


def mean_ci(values: list[float], conf: float = 0.95) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = float(v.mean())
    if v.size < 2:
        return {"mean": m, "lo": m, "hi": m, "n": 1}
    h = float(stats.t.ppf(0.5 + conf / 2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return {"mean": m, "lo": m - h, "hi": m + h, "n": int(v.size)}


def paired_delta(a: list[float], b: list[float]) -> dict:
    """Mean (b - a) with CI on paired runs (same well/seed)."""
    d = [y - x for x, y in zip(a, b)]
    out = mean_ci(d)
    base = np.mean(a) if a else 0.0
    out["rel"] = (out["mean"] / base) if out["mean"] is not None and base else None
    return out


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(stats.spearmanr(x, y).statistic)

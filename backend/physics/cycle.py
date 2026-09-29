"""Fast one-cycle forecaster used by the optimizer and the what-if engine.

Simulates inject -> soak -> produce at daily resolution from a (cloned) reservoir
state, with lift matched to inflow at the target drawdown (reliability-first: the
pump never targets displacement above predicted inflow). Returns the production,
power and constraint quantities the nested optimizer needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.state import SpeedProfile
from core.units import BAR, DAY
from physics import srp_model as sm
from physics.inflow import inflow_rate, water_cut
from physics.injection import InjectionPlan, plan_injection
from physics.params import WellParams
from physics.pump import PumpCondition
from physics.pumping_unit import kinematics
from physics.reservoir import CSSReservoir, stimulation_ratio


@dataclass
class CycleForecast:
    t_d: np.ndarray                 # production days (grid)
    q_oil: np.ndarray               # m3/d
    q_water: np.ndarray
    T_avg: np.ndarray               # K
    mu_top: np.ndarray              # Pa.s at the top of the rod string
    p_motor_kw: np.ndarray
    spm_req: np.ndarray
    rfi_est: np.ndarray
    t_inject_d: float
    t_soak_d: float
    steam_t: float
    plan: InjectionPlan
    r_h: float
    peak_smax: np.ndarray           # Pa per taper section at peak rate (Goodman applied by caller)
    peak_smin: np.ndarray
    feasible: dict[str, bool] = field(default_factory=dict)
    end_state: CSSReservoir | None = None


@dataclass
class LiftLimits:
    spm_max: float = 7.0
    eta_v: float = 0.85
    p_wf_target: float = 6.0 * BAR


def simulate_cycle(wp: WellParams, res: CSSReservoir, steam_t: float, p_inj_bar: float, soak_d: float,
                   x_surface: float, capacity_kg_s: float, T_max_d: float = 150.0, dt_d: float = 1.0,
                   limits: LiftLimits | None = None, check_stress: bool = True,
                   frac_limit_bar: float | None = None, min_quality: float = 0.0) -> CycleForecast:
    lim = limits or LiftLimits()
    r = res.clone()
    plan = plan_injection(wp, steam_t * 1000.0, p_inj_bar * BAR, r.p_res, x_surface, capacity_kg_s)
    r.inject(plan.Q_sf, plan.t_inj, plan.T_s)
    r.advance(soak_d * DAY)
    n = int(round(T_max_d / dt_d))
    t = (np.arange(n) + 0.5) * dt_d
    q_o, q_w, Ta, mut, pk, spm, rf = (np.zeros(n) for _ in range(7))
    mu_c = float(wp.visc.mu_value(wp.T_R))
    grid = wp.rods.discretize(20)
    kin_cache: dict[float, object] = {}
    stroke_len = None
    for i in range(n):
        T = r.T_avg()
        sr = r.effective_sr(stimulation_ratio(wp.res.r_e, wp.res.r_w, r.r_h, float(wp.visc.mu_value(T)), mu_c))
        wc = float(water_cut(t[i] * DAY, wp.wc_early, wp.wc_late, wp.wc_tau))
        rho = wp.rho_liquid(T, wc)
        p_wf = lim.p_wf_target + rho * 9.80665 * (wp.depth - wp.pump_depth)
        ql = inflow_rate(wp.J_c, sr, r.p_res, p_wf)
        r.advance(dt_d * DAY, ql * (1 - wc), ql * wc)
        q_o[i], q_w[i], Ta[i] = ql * (1 - wc) * DAY, ql * wc * DAY, T
        if i % 3 == 0 or i == n - 1:
            z, Tz = sm.tubing_temperature(wp, ql, wc, T, t[i] * DAY, n_cells=20)
            mu = sm.mu_on_grid(wp, grid, z, Tz, wc)
            c, w, vfm = sm.rod_coefficients(wp, grid, mu, rho)
        if stroke_len is None:
            k0 = kinematics(wp.unit, 4.0)
            stroke_len = k0.stroke
        s_req = ql * DAY / (wp.A_p * stroke_len * 0.9 * lim.eta_v * 1440.0)
        s_req = float(np.clip(s_req, 1.0, lim.spm_max * 1.5))
        key = round(s_req * 4) / 4
        kin = kin_cache.get(key)
        if kin is None:
            kin = kinematics(wp.unit, max(key, 0.5), SpeedProfile())
            kin_cache[key] = kin
        dp = wp.p_discharge(rho) - (lim.p_wf_target)
        p_pr = sm.pr_power_quick(wp, grid, kin, mu, ql, dp)       # type: ignore[arg-type]
        pk[i] = sm.motor_power(wp, p_pr) / 1000.0
        spm[i] = s_req
        mut[i] = float(mu[0])
        rf[i] = sm.rfi_value(kin, vfm)                            # type: ignore[arg-type]
    feas = {
        "frac": frac_limit_bar is None or plan.p_sf / BAR < frac_limit_bar,
        "generator": plan.w <= capacity_kg_s * 1.0001,
        "quality": plan.x_sf >= min_quality,
        "injectable": plan.injectable,
        "lift_spm": bool(spm.max() <= lim.spm_max),
    }
    smax = smin = np.zeros(len(wp.rods.sections))
    if check_stress:
        i = int(np.argmax(q_o + q_w))
        g2 = wp.rods.discretize(30)
        z, Tz = sm.tubing_temperature(wp, (q_o[i] + q_w[i]) / DAY, 0.5, Ta[i], t[i] * DAY, n_cells=20)
        wc_i = q_w[i] / max(q_o[i] + q_w[i], 1e-9)
        rho_i = wp.rho_liquid(Ta[i], wc_i)
        mu2 = sm.mu_on_grid(wp, g2, z, Tz, wc_i)
        st = sm.simulate_stroke(wp, g2, float(min(spm[i], lim.spm_max)), SpeedProfile(), mu2, rho_i,
                                lim.p_wf_target, PumpCondition(0.95), n_strokes=2, allow_float=False)
        smax, smin = st.fwd.sec_max_stress, st.fwd.sec_min_stress
    return CycleForecast(t_d=t, q_oil=q_o, q_water=q_w, T_avg=Ta, mu_top=mut, p_motor_kw=pk, spm_req=spm, rfi_est=rf,
                         t_inject_d=plan.t_inj / DAY, t_soak_d=soak_d, steam_t=steam_t, plan=plan, r_h=r.r_h,
                         peak_smax=smax, peak_smin=smin, feasible=feas, end_state=r)

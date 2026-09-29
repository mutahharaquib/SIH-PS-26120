"""M2 — viscosity, wellbore, reservoir, inflow: limiting-behaviour tests."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate

from core import steam
from core.config import get_config
from core.units import BAR, DAY
from physics import inflow, reservoir as rsv, viscosity as visc, wellbore as wb

CFG = get_config()


# ---------------------------------------------------------------- viscosity
def test_walther_roundtrip_recovers_A_B():
    A, B = 9.1, 3.35
    T = np.linspace(300, 480, 8)
    nu = 10 ** (10 ** (A - B * np.log10(T))) - 0.7
    p = visc.fit_walther(T, nu)
    assert p.A == pytest.approx(A, rel=1e-6) and p.B == pytest.approx(B, rel=1e-6)


def test_viscosity_monotone_decreasing_and_extrapolation_flag():
    m = visc.model_for_api(CFG.v("field.crude.viscosity_lab_points"), 18.0, -0.12, 7e-4)
    T = np.linspace(300, 600, 200)
    assert np.all(np.diff(m.mu_value(T)) < 0)
    lo, hi = m.fit_range
    assert visc.mu(0.5 * (lo + hi), m).prov.extrapolated is False
    assert visc.mu(hi + 50, m).prov.extrapolated is True
    assert visc.mu(lo - 5, m).prov.extrapolated is True
    assert visc.mu(320.0, m).uncertainty is not None


def test_arrhenius_fallback_with_few_points():
    rho = visc.rho_oil_fn(18, 7e-4)
    m = visc.fit_viscosity([320.0, 373.0], [9000.0, 260.0], rho)
    assert m.walther is None and m.arrhenius is not None
    assert "Arrhenius" in m.source


def test_heavier_crude_is_more_viscous():
    pts = CFG.v("field.crude.viscosity_lab_points")
    m17 = visc.model_for_api(pts, 17.0, -0.12, 7e-4)
    m19 = visc.model_for_api(pts, 19.0, -0.12, 7e-4)
    assert m17.mu_value(330.0) > m19.mu_value(330.0)


# ---------------------------------------------------------------- wellbore
def geom(**kw):
    base = dict(depth=1000.0, r_ti=0.031, r_to=0.0365, r_ci=0.0785, r_cw=0.108, k_e=2.0, alpha_e=9e-7,
                T_surface=301.15, grad=0.019, n_cells=100)
    base.update(kw)
    return wb.WellboreGeom(**base)


def test_no_heat_loss_when_U_zero():
    prof = wb.steam_injection(geom(), w=2.9, p_wh=9e6, x_wh=0.8, t_s=5 * DAY, U=0.0)
    assert prof.total_loss_W == 0.0
    assert np.all(prof.q_loss == 0.0)


def test_quality_decreases_with_depth_and_energy_closes():
    prof = wb.steam_injection(geom(), w=2.9, p_wh=9e6, x_wh=0.8, t_s=5 * DAY, U=12.0)
    assert np.all(np.diff(prof.x) < 0)
    assert prof.closure_rel_err < 1e-3
    assert 0 < prof.x_sandface < 0.8
    assert np.all(np.isfinite(prof.p)) and prof.p_sandface > 0
    # at low rate friction is negligible, so the column head raises bottomhole pressure
    slow = wb.steam_injection(geom(), w=0.3, p_wh=9e6, x_wh=0.8, t_s=5 * DAY, U=12.0)
    assert slow.p_sandface > 9e6


def test_heat_loss_decreases_with_time():
    losses = [wb.steam_injection(geom(), 2.9, 9e6, 0.8, t * DAY, 12.0).total_loss_W for t in (0.5, 2, 10, 30)]
    assert all(a > b for a, b in zip(losses, losses[1:]))
    fs = [wb.ramey_f(t * DAY, 9e-7, 0.108) for t in (0.01, 0.5, 2, 10, 30)]
    assert all(a < b for a, b in zip(fs, fs[1:]))


def test_ramey_long_time_limit():
    t = 300 * DAY
    assert wb.ramey_f(t, 9e-7, 0.108) == pytest.approx(wb.ramey_long_time(t, 9e-7, 0.108), abs=0.02)


def test_production_profile_cools_toward_geotherm_and_closes():
    g = geom()
    hot = wb.produced_fluid_profile(g, w=0.25, cp=3000.0, T_bottom=450.0, t_s=20 * DAY, U=25.0, bottom_depth=975.0)
    assert hot.T[-1] == 450.0
    assert np.all(np.diff(hot.T) > 0)                     # hotter with depth
    assert hot.closure_rel_err < 1e-3
    slow = wb.produced_fluid_profile(g, w=0.05, cp=3000.0, T_bottom=450.0, t_s=20 * DAY, U=25.0, bottom_depth=975.0)
    assert slow.T[0] < hot.T[0]                           # low rate -> colder at surface
    shut = wb.produced_fluid_profile(g, w=0.0, cp=3000.0, T_bottom=450.0, t_s=20 * DAY, U=25.0, bottom_depth=975.0)
    assert np.allclose(shut.T, g.T_earth(shut.z))


# ---------------------------------------------------------------- reservoir
PRM = rsv.ReservoirParams(T_R=320.15, h=12.0, M_R=2.35e6, M_ob=2.3e6, lambda_ob=1.7, r_w=0.089, r_e=120.0,
                          p_init=65 * BAR, p_min=12 * BAR, compliance=240.0 / BAR)


def test_ml_dimensions_and_small_time_limit():
    # t_D is dimensionless: [W/m/K]*[J/m3/K]*[s] / ([J/m3/K]^2 [m]^2) -> (J/s/m/K * J/m3/K * s) / (J^2/m6/K^2 * m2) = 1
    tD = rsv.t_D_ml(1 * DAY, PRM.lambda_ob, PRM.M_ob, PRM.M_R, PRM.h)
    assert 0 < tD < 1
    # scaling: A is linear in Q_i and in 1/dT -> m^2 (check vs energy balance at small t: A*h*M_R*dT ~ Q*t)
    Q, dT, t = 4e6, 250.0, 60.0
    A = rsv.marx_langenheim_area(Q, PRM.M_R, PRM.h, PRM.lambda_ob, PRM.M_ob, dT, t)
    assert A * PRM.h * PRM.M_R * dT == pytest.approx(Q * t, rel=1e-2)
    A2 = rsv.marx_langenheim_area(2 * Q, PRM.M_R, PRM.h, PRM.lambda_ob, PRM.M_ob, dT, 10 * DAY)
    A1 = rsv.marx_langenheim_area(Q, PRM.M_R, PRM.h, PRM.lambda_ob, PRM.M_ob, dT, 10 * DAY)
    assert A2 == pytest.approx(2 * A1)


def test_f_VD_matches_numerical_heat_kernel():
    h, a, t = 10.0, 7e-7, 50 * DAY
    s = 2 * np.sqrt(a * t)
    Tz = lambda z: 0.5 * (visc.np.array(0) + rsv.erf((h / 2 - z) / s) + rsv.erf((h / 2 + z) / s))  # noqa: E731
    num = integrate.quad(Tz, -h / 2, h / 2)[0] / h
    assert rsv.f_VD(t, h, a) == pytest.approx(num, rel=1e-6)


def test_f_HD_matches_monte_carlo_heat_kernel():
    rng = np.random.default_rng(0)
    R, a, t = 12.0, 7e-7, 80 * DAY
    n = 400_000
    r = R * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    sig = np.sqrt(2 * a * t)
    x = r * np.cos(th) + rng.normal(0, sig, n)
    y = r * np.sin(th) + rng.normal(0, sig, n)
    frac_inside = np.mean(x * x + y * y < R * R)
    assert rsv.f_HD(t, R, a) == pytest.approx(frac_inside, abs=4e-3)


@settings(max_examples=60, deadline=None)
@given(t=st.floats(0.0, 3e8), r_h=st.floats(0.5, 60.0), fpd=st.floats(0.0, 2.0))
def test_T_avg_bounded(t, r_h, fpd):
    T = rsv.T_avg_bl(320.0, 560.0, rsv.f_HD(t, r_h, 7e-7), rsv.f_VD(t, 12.0, 7e-7), fpd)
    assert 320.0 <= T <= 560.0


def test_T_avg_tends_to_T_R():
    res = rsv.CSSReservoir(PRM)
    res.inject(4e6, 15 * DAY, 570.0)
    temps = []
    for _ in range(12):
        res.advance(1e9)
        temps.append(res.T_avg())
    assert temps[-1] - PRM.T_R < 1.0
    assert all(a >= b for a, b in zip(temps, temps[1:]))


def test_stimulation_ratio_limits():
    assert rsv.stimulation_ratio(120, 0.089, 15, 1.0, 1.0) == pytest.approx(1.0)
    assert rsv.stimulation_ratio(120, 0.089, 0.089, 0.01, 1.0) == pytest.approx(1.0)
    assert rsv.stimulation_ratio(120, 0.089, 15, 0.01, 1.0) > 1.0
    assert rsv.stimulation_ratio(120, 0.089, 15, 0.001, 1.0) > rsv.stimulation_ratio(120, 0.089, 15, 0.01, 1.0)


def test_inflow_and_water_cut():
    assert inflow.inflow_rate(1e-9, 2.0, 60 * BAR, 70 * BAR) == 0.0
    q = inflow.inflow_rate(0.08 / DAY / BAR, 3.0, 60 * BAR, 6 * BAR)
    assert q * DAY == pytest.approx(0.08 * 3 * 54)
    wc = inflow.water_cut(np.array([0, 10, 1000]) * DAY, 0.92, 0.4, 8 * DAY)
    assert wc[0] == pytest.approx(0.92) and wc[-1] == pytest.approx(0.4) and wc[0] > wc[1] > wc[2]


def run_fixed_cycles(n_cycles=4, carry_over=True):
    """Fixed recipe, fixed cut-off, fixed p_wf: oil per cycle must decline (depletion)."""
    res = rsv.CSSReservoir(PRM)
    vm = visc.model_for_api(CFG.v("field.crude.viscosity_lab_points"), 18.0, -0.12, 7e-4)
    mu_c = float(vm.mu_value(PRM.T_R))
    g = geom()
    J_c = 0.08 / DAY / BAR
    oils = []
    for _ in range(n_cycles):
        w = 3000e3 / (15 * DAY)
        prof = wb.steam_injection(g, w, 9e6, 0.8, 15 * DAY, 12.0)
        heat = w * (prof.x_sandface * float(steam.h_fg(prof.p_sandface)) + float(steam.h_f(prof.p_sandface))
                    - steam.C_P_WATER * (PRM.T_R - 273.15))
        res.inject(heat, 15 * DAY, prof.T_sandface)
        res.advance(5 * DAY)
        oil = 0.0
        for d in range(90):
            sr = rsv.stimulation_ratio(PRM.r_e, PRM.r_w, res.r_h, float(vm.mu_value(res.T_avg())), mu_c)
            ql = inflow.inflow_rate(J_c, sr, res.p_res, 6 * BAR)
            wc = float(inflow.water_cut(d * DAY, 0.92, 0.4, 8 * DAY))
            res.advance(DAY, ql * (1 - wc), ql * wc)
            oil += ql * (1 - wc) * DAY
        res.end_cycle()
        if not carry_over:
            res.E_residual = 0.0
        oils.append(oil)
    return oils, res


def test_cycle_oil_declines_with_fixed_recipe():
    # cycle 1 has no carried-over heat, so decline is asserted from cycle 2 on
    oils, res = run_fixed_cycles()
    assert all(a > b for a, b in zip(oils[1:], oils[2:])), oils
    # without carry-over, depletion alone makes every cycle weaker
    oils_nc, _ = run_fixed_cycles(carry_over=False)
    assert all(a > b for a, b in zip(oils_nc, oils_nc[1:])), oils_nc
    assert res.p_res < PRM.p_init
    assert res.E_residual > 0          # heat carried over

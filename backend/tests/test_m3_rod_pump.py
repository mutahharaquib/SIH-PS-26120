"""M3 — kinematics, drag, wave equation (both modes), pump BC, energy."""

import numpy as np
import pytest

from core.config import get_config
from core.state import SpeedProfile
from core.units import BAR, DAY, G
from physics import drag, energy
from physics import srp_model as sm
from physics.params import build_well_params
from physics.pump import DownholePump, PumpCondition
from physics.pumping_unit import kinematics, stroke_length
from physics.rod_string import check_cfl, diagnostic, predictive

CFG = get_config()
WP = build_well_params(CFG, "T01")
GRID = WP.rods.discretize(40)


# ---------------------------------------------------------------- kinematics
def test_kinematics_basic():
    k = kinematics(WP.unit, 4.0)
    assert k.period == pytest.approx(15.0, rel=1e-3)
    assert k.x.min() == pytest.approx(0.0, abs=1e-3) and k.x.max() == pytest.approx(stroke_length(WP.unit), rel=1e-3)
    assert 1.5 < k.stroke < 3.5
    sg = np.sign(k.v)
    assert np.sum(sg != np.roll(sg, 1)) == 2                 # one up, one down per stroke (periodic)
    assert k.upstroke[:10].all() and not k.upstroke[-10:].any()
    assert kinematics(WP.unit, 4.0, harmonic=True).mode == "harmonic-fallback"


def test_downstroke_shaping_slows_downstroke_only():
    base = kinematics(WP.unit, 4.0)
    shaped = kinematics(WP.unit, 4.0, SpeedProfile(downstroke=[1.0, 0.5, 0.5, 1.0]))
    assert shaped.max_down_speed < 0.8 * base.max_down_speed
    assert shaped.v.max() == pytest.approx(base.v.max(), rel=0.02)   # upstroke unchanged
    assert shaped.period > base.period                                 # time budget cost


def test_longer_crank_gives_longer_stroke():
    assert stroke_length(WP.with_unit_R(0.97).unit) > stroke_length(WP.with_unit_R(0.76).unit)


# ---------------------------------------------------------------- drag
def test_drag_relations():
    assert drag.damping_coeff(2.0, 7850, 3.9e-4, 0.031, 0.011) == pytest.approx(
        2 * drag.damping_coeff(1.0, 7850, 3.9e-4, 0.031, 0.011))
    v1, v2 = drag.v_fall([1.0, 2.0], 7850, 950, 3.9e-4, 0.031, 0.011)
    assert v1 == pytest.approx(2 * v2)
    # terminal velocity balances buoyant weight and drag
    vf = float(drag.v_fall(1.5, 7850, 950, 3.9e-4, 0.031, 0.011))
    assert float(drag.drag_per_length(1.5, vf, 0.031, 0.011)) == pytest.approx((7850 - 950) * G * 3.9e-4)
    assert drag.v_fall_min([1, 5, 2], 7850, 950, [3.9e-4] * 3, 0.031, [0.011] * 3) == pytest.approx(
        float(drag.v_fall(5, 7850, 950, 3.9e-4, 0.031, 0.011)))


# ---------------------------------------------------------------- wave equation
def _forward(c_level, n_strokes, cond=None, cfl=0.9, spm=4.0, mu=None, allow_float=False, record_energy=False):
    kin = kinematics(WP.unit, spm, dt=GRID.dt_cfl * cfl)
    c = np.full(len(GRID.z), c_level)
    w = GRID.node_weights(950.0)
    pump = DownholePump(WP.A_p, 14e3, cond=cond or PumpCondition(1.0), S_p=kin.stroke)
    fwd = predictive(GRID, kin.x, kin.t[1] - kin.t[0], c, w, pump, n_strokes=n_strokes, allow_float=allow_float,
                     record_energy=record_energy)
    return kin, c, w, fwd


def test_roundtrip_undamped_recovers_pump_load():
    # exact d'Alembert grid (a dt = dz) for both modes
    kin, c, w, fwd = _forward(0.0, 1, cfl=1.0)
    d = diagnostic(GRID, fwd.surface_pos, fwd.surface_load, kin.period, c, w, resample=False, periodic=False)
    ok = np.isfinite(d.pump_load)
    assert ok.sum() > 0.6 * len(ok)
    err = np.max(np.abs(d.pump_load[ok] - fwd.pump_load[ok]))
    assert err < 1e-6 * 14e3


def test_roundtrip_damped_periodic():
    kin, c, w, fwd = _forward(0.6, 8, cond=PumpCondition(0.7), cfl=1.0)
    d = diagnostic(GRID, fwd.surface_pos, fwd.surface_load, kin.period, c, w, resample=False)
    err = np.abs(d.pump_load - fwd.pump_load)
    assert np.median(err) < 0.01 * 14e3


def test_diagnostic_with_resampling_and_noise_is_stable():
    kin, c, w, fwd = _forward(0.6, 8, cond=PumpCondition(0.7))
    rng = np.random.default_rng(1)
    load = fwd.surface_load + rng.normal(0, 150.0, len(fwd.surface_load))
    d = diagnostic(GRID, fwd.surface_pos, load, kin.period, c, w, n_harmonics=20)
    assert np.all(np.isfinite(d.pump_load))
    assert np.max(d.pump_load) == pytest.approx(14e3, rel=0.25)
    assert d.pump_pos.max() == pytest.approx(fwd.pump_pos.max(), rel=0.1)


def test_stress_at_taper_is_load_over_local_area():
    x = np.zeros(400)
    c = np.full(len(GRID.z), 5.0)
    w = GRID.node_weights(950.0)
    pump = DownholePump(WP.A_p, 0.0, friction=0.0, cond=PumpCondition(unseated=True))
    fwd = predictive(GRID, x, GRID.dt_cfl * 0.9, c, w, pump, n_strokes=3)
    tops = GRID.section_top_elements()
    below = np.cumsum(w[::-1])[::-1]
    for j, e in enumerate(tops):
        expect = below[e + 1] / GRID.A[e]
        assert fwd.sec_max_stress[j] == pytest.approx(expect, rel=2e-3)


def test_energy_decays_with_damping():
    """Free vibration after a step of the polished rod: kinetic energy decays only with damping."""
    n = 3000
    x = np.full(n, 0.2)
    x[0] = 0.0
    w = GRID.node_weights(950.0)
    out = {}
    for c_level in (0.0, 1.0):
        pump = DownholePump(WP.A_p, 0.0, friction=0.0, cond=PumpCondition(unseated=True))
        f = predictive(GRID, x, GRID.dt_cfl * 0.9, np.full(len(GRID.z), c_level), w, pump, n_strokes=1,
                       allow_float=False, record_energy=True)
        out[c_level] = f.energy_hist
    early, late = slice(50, 600), slice(-600, -5)   # skip the periodic wrap-around step at the end
    assert out[1.0][late].max() < 0.05 * out[1.0][early].max()
    assert out[0.0][late].max() > 0.5 * out[0.0][early].max()


def test_cfl_violation_raises():
    with pytest.raises(ValueError, match="CFL"):
        check_cfl(GRID, GRID.dt_cfl * 1.05)


# ---------------------------------------------------------------- pump BC
def _pump_trace(cond, n=400):
    """Drive the pump through one ideal sinusoidal stroke (down +)."""
    t = np.linspace(0, 2 * np.pi, n)
    u = -1.0 * (1 - np.cos(t))            # starts at bottom (0), goes up to -2, back
    v = np.gradient(u)
    p = DownholePump(1.55e-3, 10e3, friction=0.0, cond=cond, S_p=2.0)
    return u, np.array([p.force(ui, vi) for ui, vi in zip(u, v)])


def test_pump_conditions():
    u, full = _pump_trace(PumpCondition(1.0))
    _, pound = _pump_trace(PumpCondition(0.6))
    _, gas = _pump_trace(PumpCondition(0.6, gas=True))
    _, unseat = _pump_trace(PumpCondition(unseated=True))
    half = len(u) // 2
    # release after the top is delayed by the valve hysteresis (~1.5 % of stroke) + 3 % ramp
    assert full[half + 60:].max() < 1.0 and full[20:half - 5].min() > 9.9e3
    # fluid pound: load held after the top, then a sharp drop
    drop = np.max(-np.diff(pound[half:]))
    assert pound[half + 40] > 9.9e3 and drop > 0.2 * 10e3
    # gas: gradual release (no single-step drop as large)
    assert np.max(-np.diff(gas[half:])) < drop
    assert np.all(np.abs(unseat) < 1.0)
    area = lambda F: energy.card_area(-u, F)  # noqa: E731
    assert area(unseat) < 0.01 * area(full)
    assert area(full) > area(pound) > 0


# ---------------------------------------------------------------- coupled float + energy
def test_rod_float_emerges_when_cold_and_is_mitigated_by_shaping():
    z, T = sm.tubing_temperature(WP, 4 / DAY, 0.4, 325.0, 40 * DAY)
    mu = sm.mu_on_grid(WP, GRID, z, T, 0.4)
    rho = WP.rho_liquid(325.0, 0.4)
    fast = sm.simulate_stroke(WP, GRID, 4.5, SpeedProfile(), mu, rho, 6 * BAR, PumpCondition(0.9))
    slow = sm.simulate_stroke(WP, GRID, 4.5, SpeedProfile(downstroke=[1.0, 0.45, 0.45, 1.0]), mu, rho, 6 * BAR,
                              PumpCondition(0.9))
    assert fast.rfi > 1.0 and fast.fwd.float_fraction > 0.02 and fast.fwd.surface_load.min() == 0.0
    assert slow.rfi < fast.rfi and slow.fwd.float_fraction < fast.fwd.float_fraction


def test_hot_well_does_not_float():
    z, T = sm.tubing_temperature(WP, 20 / DAY, 0.5, 450.0, 40 * DAY)
    mu = sm.mu_on_grid(WP, GRID, z, T, 0.5)
    r = sm.simulate_stroke(WP, GRID, 4.0, SpeedProfile(), mu, WP.rho_liquid(450, 0.5), 6 * BAR, PumpCondition(1.0))
    assert r.rfi < 0.8 and r.fwd.float_fraction == 0.0 and r.fwd.surface_load.min() > 5e3
    # card-area power >= hydraulic power (fluid work) and in a plausible range
    assert 1e3 < r.pr_power < 3e4


def test_energy_helpers():
    assert energy.card_area([0, 1, 1, 0], [0, 0, 2, 2]) == pytest.approx(2.0)
    assert energy.polished_rod_power([0, 1, 1, 0], [0, 0, 2, 2], 6.0) == pytest.approx(0.2)
    assert energy.motor_input_power(1000, 0.95, 0.96, 0.9, 0.97) == pytest.approx(1000 / (0.95 * 0.96 * 0.9 * 0.97))
    assert energy.sor(3000, 1000) == 3.0
    assert energy.kwh_per_bbl(100, 0.158987294928 * 50) == pytest.approx(2.0)

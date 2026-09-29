"""M7 — objective, cut-off (sim + live), Bayesian optimizer, baseline."""

import numpy as np
import pytest
from scipy.optimize import brentq

from core.config import get_config
from core.state import CyclePhase
from core.telemetry import Command
from optimization.baseline import BaselinePolicy
from optimization.cutoff import LiveCutoff, sim_cutoff
from optimization.cycle_bo import BOSettings, CycleOptimizer, evaluate
from optimization.objective import Econ, g_curve, q_net
from physics.params import build_well_params
from physics.reservoir import CSSReservoir
from simulation.field_sim import FieldSimulator

CFG = get_config()


def test_q_net_and_production_weight():
    e = Econ.from_config()
    v = float(q_net(10.0, 20.0, 5.0, e))
    assert v == pytest.approx(e.oil_price_m3 * 10 - e.tariff_kwh * 20 * 24 - e.opex_d - e.water_m3 * 5)
    e1 = Econ.from_config(production_weight=1.0)
    assert float(q_net(10.0, 20.0, 5.0, e1)) == pytest.approx(e.oil_price_m3 * 10)


def _unimodal(t, a=400.0, tau=40.0, c=150.0):
    return a * (t / tau) * np.exp(1 - t / tau) - c


def test_grid_argmax_matches_marginal_value_theorem():
    t = np.arange(0, 300, 0.1) + 0.05
    qn = _unimodal(t)
    ti, ts, cs, cf = 10.0, 4.0, 3000.0, 500.0
    cut = sim_cutoff(t, qn, ti, ts, cs, cf)
    # analytic optimum: q_net(T*) = g(T*) on the declining branch
    T, V, g = g_curve(t, qn, ti, ts, cs, cf)

    def h(Tq):
        return _unimodal(Tq) - np.interp(Tq, T, g)

    T_star = brentq(h, 45.0, 290.0)
    assert cut.T_star_d == pytest.approx(T_star, abs=0.3)
    assert _unimodal(cut.T_star_d) == pytest.approx(cut.g_star, rel=0.02)


def test_negative_early_profile_never_stops_at_early_crossing():
    t = np.arange(0, 200, 1.0) + 0.5
    qn = np.where(t < 8, -500.0 + 60 * t, _unimodal(t))     # negative, then rises
    cut = sim_cutoff(t, qn, 10, 4, 3000, 500)
    assert cut.T_star_d > 30
    lc = LiveCutoff(g_star=cut.g_star, alpha=1.0, hold_hours=0.0)
    for ti, qi in zip(t, qn):
        if lc.update(ti * 24, float(qi)):
            break
    assert ti > 30


def test_live_and_sim_rules_agree_noise_free():
    t = np.arange(0, 250, 1 / 24) + 1 / 48
    qn = _unimodal(t)
    cut = sim_cutoff(t, qn, 10, 4, 3000, 500)
    lc = LiveCutoff(g_star=cut.g_star, alpha=1.0, hold_hours=0.0)
    stop_t = None
    for ti, qi in zip(t, qn):
        if lc.update(ti * 24, float(qi)):
            stop_t = ti
            break
    assert stop_t == pytest.approx(cut.T_star_d, abs=0.2)


def test_live_cutoff_hold_and_ewma_resist_noise():
    rng = np.random.default_rng(0)
    t = np.arange(0, 250, 1 / 24)
    qn = _unimodal(t) + rng.normal(0, 60, len(t))
    cut = sim_cutoff(t, _unimodal(t), 10, 4, 3000, 500)
    lc = LiveCutoff(g_star=cut.g_star, alpha=0.05, hold_hours=12)
    stop = next((ti for ti, qi in zip(t, qn) if lc.update(ti * 24, float(qi))), None)
    assert stop is not None and abs(stop - cut.T_star_d) < 12


@pytest.fixture(scope="module")
def well():
    wp = build_well_params(CFG, "W01")
    return wp, CSSReservoir(wp.res)


def test_evaluate_constraints(well):
    wp, res = well
    s = BOSettings.from_config(fast=True)
    ok = evaluate(wp, res, np.array([3000.0, 90.0, 5.0]), Econ.from_config(), s)
    assert ok.feasible and 20 < ok.T_star_d < 150 and ok.oil_m3 > 100
    s.frac_limit_bar = 40.0                                       # force a violated frac constraint
    bad = evaluate(wp, res, np.array([3000.0, 120.0, 5.0]), Econ.from_config(), s)
    assert not bad.constraints["frac"] and bad.objective < bad.g_star


def test_bo_improves_on_initial_design_and_is_seeded(well):
    wp, res = well
    s = BOSettings.from_config(fast=True)
    s.n_param_samples = 3
    rec = CycleOptimizer(s).run(wp, res)
    init_best = max(h.g_star for h in rec.history[: s.n_init] if h.feasible)
    assert rec.best.feasible and rec.best.g_star >= init_best
    lo, hi = s.bounds[:, 0], s.bounds[:, 1]
    assert np.all(rec.best.theta >= lo - 1e-9) and np.all(rec.best.theta <= hi + 1e-9)
    summ = rec.summary()
    assert summ["uncertainty"]["param_std"] >= 0 and summ["tier"] == "B"
    rec2 = CycleOptimizer(s).run(wp, res)
    assert rec2.recipe.steam_mass_t == rec.recipe.steam_mass_t


def test_optimizer_uses_current_state(well):
    """Receding horizon: a depleted, pre-heated state changes the forecast."""
    wp, res = well
    s = BOSettings.from_config(fast=True)
    theta = np.array([3000.0, 90.0, 5.0])
    fresh = evaluate(wp, res, theta, Econ.from_config(), s, check_stress=False)
    depleted = res.clone()
    depleted.p_res *= 0.7
    later = evaluate(wp, depleted, theta, Econ.from_config(), s, check_stress=False)
    assert later.oil_m3 < fresh.oil_m3


def test_baseline_policy_fixed_recipe_and_days_cutoff():
    f = FieldSimulator(n_wells=1, seed=2, event_scale=0.0)
    w = f.wells["W01"]
    pol = BaselinePolicy(f.meta["W01"])
    pol.s.cutoff_days = 20
    ended = False
    for _ in range(3000):
        tel = w.step()
        cmds = pol.on_telemetry(tel)
        if any(c.kind == "stop_production" for c in cmds):
            ended = True
        w.apply(cmds)
        if ended:
            break
    assert ended and w.phase == CyclePhase.IDLE
    c = w.cycles[-1]
    assert c.steam_t == pol.s.recipe.steam_mass_t and 19.5 < c.produce_d < 20.6
    assert isinstance(cmds[0], Command)

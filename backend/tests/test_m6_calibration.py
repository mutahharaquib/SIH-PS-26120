"""M6 — RLS (Walther) and inflow refit with guardrails."""

import numpy as np
import pytest

from calibration.inflow_refit import InflowObs, refit
from calibration.rls import WaltherRLS
from physics.viscosity import WaltherParams, walther_x


def true_nu(T, A=8.6, B=3.30):
    return 10 ** (10 ** (A - B * np.log10(T))) - 0.7


def prior():
    return WaltherParams(A=8.45, B=3.24, T_min_fit=303, T_max_fit=473)


def test_rls_converges_to_true_params_with_noise():
    rng = np.random.default_rng(0)
    r = WaltherRLS.from_params(prior(), 0.02, 0.003, lam=0.995)
    for _ in range(150):
        T = rng.uniform(305, 420)
        r.update(T, true_nu(T) * (1 + rng.normal(0, 0.05)))
    for T in (310.0, 350.0, 400.0):
        assert r.params().nu(T) == pytest.approx(true_nu(T), rel=0.05)
    assert r.n_updates > 100
    # uncertainty shrinks from the prior
    assert r.P[0, 0] < 0.02 and r.P[1, 1] < 0.003


def test_rls_uncertainty_propagates_to_viscosity():
    r = WaltherRLS.from_params(prior(), 0.02, 0.003)
    sig0 = float(r.params().nu_rel_sigma(330.0))
    rng = np.random.default_rng(1)
    for _ in range(60):
        T = rng.uniform(305, 420)
        r.update(T, true_nu(T) * (1 + rng.normal(0, 0.03)))
    assert 0 < float(r.params().nu_rel_sigma(330.0)) < sig0


def test_rls_forgetting_tracks_drift():
    rng = np.random.default_rng(2)
    r = WaltherRLS.from_params(prior(), 0.02, 0.003, lam=0.97)
    for _ in range(80):
        T = rng.uniform(305, 420)
        r.update(T, true_nu(T))
    for _ in range(120):                         # crude changes (e.g. emulsion/asphaltene)
        T = rng.uniform(305, 420)
        r.update(T, true_nu(T, A=8.66))
    assert r.params().nu(330.0) == pytest.approx(true_nu(330.0, A=8.66), rel=0.08)


def test_rls_guardrails_reject_spikes():
    r = WaltherRLS.from_params(prior(), 0.02, 0.003)
    rng = np.random.default_rng(3)
    for _ in range(40):
        T = rng.uniform(305, 420)
        r.update(T, true_nu(T) * (1 + rng.normal(0, 0.03)))
    A, B = r.A, r.B
    u = r.update(330.0, true_nu(330.0) * 20.0)
    assert not u.accepted and u.reason in ("outlier", "step_limit")
    assert (r.A, r.B) == (A, B)
    x = float(walther_x(330.0))
    assert np.isfinite(x)


def _obs(J=0.08, wl=0.38, tau=9.0, wc0=0.92, n=80, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = i * 1.0
        drive = 3.3 * (60 - 6 - 0.05 * i)
        q = J * drive * (1 + rng.normal(0, noise))
        wc = wl + (wc0 - wl) * np.exp(-t / tau) + rng.normal(0, 0.02)
        out.append(InflowObs(t, q, wc, drive))
    return out


def test_inflow_refit_recovers_parameters():
    res = refit(_obs(), J_c0=0.06, wc_early=0.92, wc_late0=0.3, tau0=5.0)
    assert res.accepted
    assert res.J_c == pytest.approx(0.08, rel=0.03)
    assert res.wc_late == pytest.approx(0.38, abs=0.03)
    assert res.tau_d == pytest.approx(9.0, rel=0.25)
    assert res.J_c_std is not None and res.J_c_std > 0


def test_inflow_refit_guardrail_and_min_points():
    res = refit(_obs(J=0.30), J_c0=0.06, wc_early=0.92, wc_late0=0.3, tau0=5.0, max_rel=0.6)
    assert not res.accepted and res.reason == "guardrail_J_c" and res.J_c == 0.06
    short = refit(_obs(n=5), J_c0=0.06, wc_early=0.92, wc_late0=0.3, tau0=5.0)
    assert not short.accepted and short.reason == "insufficient_data"

"""Outer Bayesian optimizer over the CSS recipe (BUILD_SPEC §8.3).

Decision variables: steam mass (t), wellhead injection pressure (bar), soak time (d).
Objective: g*(theta) from the inner cut-off (§8.2), evaluated by simulating one cycle
from the CURRENT calibrated well state (carried-over heat, current reservoir pressure).
Constraints (penalty): injection pressure below fracture pressure x safety factor,
steam rate <= generator capacity, sandface quality >= minimum, required lift
achievable within max SPM and the Goodman limit.

Surrogate: BoTorch is not a hard dependency; this is the documented fallback, a
Gaussian process (scikit-learn, Matern 5/2 + white noise, seeded) with log expected
improvement maximised over Sobol candidates plus local perturbations of the incumbent.
Receding horizon: re-run before every new cycle from the updated state.
NOT built (Phase 2): multi-cycle lookahead / depletion-aware DP cut-off (see phase2.py).
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from control.goodman import stress_ratio
from core.config import Config, get_config
from core.telemetry import Recipe
from core.units import BBL
from optimization.cutoff import sim_cutoff
from optimization.objective import Econ, cycle_costs, q_net
from physics.cycle import LiftLimits, simulate_cycle
from physics.injection import gen_capacity_kg_s
from physics.params import WellParams
from physics.reservoir import CSSReservoir

VARS = ("steam_mass", "injection_pressure", "soak_time")


@dataclass
class BOSettings:
    bounds: np.ndarray                  # 3 x 2
    n_init: int
    n_iter: int
    seed: int
    frac_limit_bar: float
    capacity_kg_s: float
    min_quality: float
    x_surface: float
    T_max_d: float
    dt_d: float
    limits: LiftLimits
    tensile: float
    service_factor: float
    stress_ratio_max: float
    n_param_samples: int

    @classmethod
    def from_config(cls, cfg: Config | None = None, fast: bool = False) -> "BOSettings":
        c = cfg or get_config()
        o = "optimizer."
        grade = c.v("rods.grade")
        return cls(
            bounds=np.array([c.v(o + v) for v in VARS], dtype=float),
            n_init=int(c.v(o + ("n_init_fast" if fast else "n_init"))),
            n_iter=int(c.v(o + ("n_iter_fast" if fast else "n_iter"))),
            seed=int(c.v(o + "seed")),
            frac_limit_bar=c.v(o + "fracture_pressure") * c.v(o + "frac_safety_factor"),
            capacity_kg_s=gen_capacity_kg_s(c.v(o + "generator_capacity")),
            min_quality=c.v(o + "min_sandface_quality"), x_surface=c.v(o + "surface_quality"),
            T_max_d=c.v(o + "T_max"), dt_d=c.v(o + "dt_cycle"),
            limits=LiftLimits(spm_max=c.v("controller.spm_max"), eta_v=c.v("controller.eta_v_prior"),
                              p_wf_target=c.v(o + "p_wf_target") * 1e5),
            tensile=float(c.v(f"rods.grades.{grade}")), service_factor=c.v("rods.service_factor"),
            stress_ratio_max=c.v("controller.stress_ratio_max"), n_param_samples=int(c.v(o + "n_param_samples")),
        )


@dataclass
class Evaluation:
    theta: np.ndarray
    g_star: float
    T_star_d: float
    feasible: bool
    constraints: dict[str, bool]
    objective: float                    # g* with penalty (what the GP sees)
    oil_m3: float
    sor: float
    kwh_per_bbl: float
    peak_stress_ratio: float
    t_inject_d: float
    x_sf: float
    q_net_curve: list[float] = field(default_factory=list)
    g_curve: list[float] = field(default_factory=list)
    t_curve: list[float] = field(default_factory=list)
    q_oil_curve: list[float] = field(default_factory=list)


@dataclass
class Recommendation:
    recipe: Recipe
    best: Evaluation
    gp_std: float
    param_std: float
    g_samples: list[float]
    history: list[Evaluation]
    n_evals: int
    surrogate: str = "sklearn-GP + logEI (BoTorch fallback)"

    def summary(self) -> dict:
        b = self.best
        return {
            "recipe": {"steam_mass_t": self.recipe.steam_mass_t, "inj_pressure_bar": self.recipe.inj_pressure_bar,
                       "soak_d": self.recipe.soak_d, "recipe_id": self.recipe.recipe_id},
            "g_star": b.g_star, "T_star_d": b.T_star_d, "sor": b.sor, "kwh_per_bbl": b.kwh_per_bbl,
            "cycle_oil_m3": b.oil_m3, "feasible": b.feasible, "constraints": b.constraints,
            "peak_stress_ratio": b.peak_stress_ratio, "t_inject_d": b.t_inject_d, "x_sandface": b.x_sf,
            "uncertainty": {"gp_std": self.gp_std, "param_std": self.param_std,
                            "total_std": float(np.hypot(self.gp_std, self.param_std))},
            "n_evals": self.n_evals, "surrogate": self.surrogate, "tier": "B",
            "curves": {"t_d": b.t_curve, "q_net": b.q_net_curve, "g": b.g_curve, "q_oil": b.q_oil_curve},
        }


def evaluate(wp: WellParams, res: CSSReservoir, theta: np.ndarray, econ: Econ, s: BOSettings,
             check_stress: bool = True) -> Evaluation:
    steam, p_inj, soak = (float(v) for v in theta)
    fc = simulate_cycle(wp, res, steam, p_inj, soak, s.x_surface, s.capacity_kg_s, s.T_max_d, s.dt_d, s.limits,
                        check_stress=check_stress, frac_limit_bar=s.frac_limit_bar, min_quality=s.min_quality)
    qn = q_net(fc.q_oil, fc.p_motor_kw, fc.q_water, econ)
    c_steam, c_fixed = cycle_costs(steam, econ)
    cut = sim_cutoff(fc.t_d, qn, fc.t_inject_d, fc.t_soak_d, c_steam, c_fixed)
    k = int(np.searchsorted(cut.T, cut.T_star_d))
    oil = float(np.sum(fc.q_oil[: k + 1]) * s.dt_d)
    kwh = float(np.sum(fc.p_motor_kw[: k + 1]) * 24 * s.dt_d)
    cons = dict(fc.feasible)
    sr = float(np.max(stress_ratio(fc.peak_smax, fc.peak_smin, s.tensile, s.service_factor))) if check_stress else 0.0
    if check_stress:
        cons["goodman"] = sr <= s.stress_ratio_max
    feas = all(cons.values())
    violations = sum(not v for v in cons.values())
    obj = cut.g_star if feas else cut.g_star - (abs(cut.g_star) + 2000.0) * violations
    return Evaluation(theta=np.asarray(theta, dtype=float), g_star=cut.g_star, T_star_d=cut.T_star_d, feasible=feas,
                      constraints=cons, objective=float(obj), oil_m3=oil, sor=steam / max(oil, 1e-6),
                      kwh_per_bbl=kwh / max(oil / BBL, 1e-6), peak_stress_ratio=sr, t_inject_d=fc.t_inject_d,
                      x_sf=fc.plan.x_sf, q_net_curve=[float(v) for v in qn], g_curve=[float(v) for v in cut.g],
                      t_curve=[float(v) for v in cut.T], q_oil_curve=[float(v) for v in fc.q_oil])


def _log_ei(mu: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    """log expected improvement (maximisation), stable for very negative z."""
    sd = np.maximum(sd, 1e-9)
    z = (mu - best) / sd
    ei = z * norm.cdf(z) + norm.pdf(z)
    out = np.log(np.maximum(ei, 1e-300))
    tail = z < -6.0
    out[tail] = norm.logpdf(z[tail]) - 2.0 * np.log(-z[tail])     # EI ~ phi(z) / z^2
    return np.log(sd) + out


def default_param_sampler(wp: WellParams, res: CSSReservoir, rng: np.random.Generator
                          ) -> tuple[WellParams, CSSReservoir]:
    """Sample calibration parameters around the current estimate (J_c, viscosity, depletion)."""
    w = copy.copy(wp)
    w.J_c = wp.J_c * float(np.exp(rng.normal(0, 0.12)))
    w.visc = copy.copy(wp.visc)
    w.visc.scale = wp.visc.scale * float(np.exp(rng.normal(0, 0.2)))
    r = res.clone()
    r.p_res = res.p_res * float(np.exp(rng.normal(0, 0.03)))
    return w, r


class CycleOptimizer:
    def __init__(self, settings: BOSettings | None = None, econ: Econ | None = None):
        self.s = settings or BOSettings.from_config()
        self.econ = econ or Econ.from_config()

    def _to_x(self, u: np.ndarray) -> np.ndarray:
        b = self.s.bounds
        return b[:, 0] + u * (b[:, 1] - b[:, 0])

    def run(self, wp: WellParams, res: CSSReservoir, recipe_id: str = "", param_sampler=default_param_sampler
            ) -> Recommendation:
        s = self.s
        rng = np.random.default_rng(s.seed)
        sob = qmc.Sobol(3, scramble=True, seed=s.seed)
        U = sob.random_base2(int(np.ceil(np.log2(max(s.n_init, 2)))))[: s.n_init]
        hist: list[Evaluation] = [evaluate(wp, res, self._to_x(u), self.econ, s, check_stress=False) for u in U]
        Us = [u for u in U]
        kernel = ConstantKernel(1.0) * Matern(length_scale=[0.3, 0.3, 0.3], nu=2.5) + WhiteKernel(1e-4)
        for it in range(s.n_iter):
            y = np.array([h.objective for h in hist])
            ym, ys = y.mean(), y.std() or 1.0
            gp = GaussianProcessRegressor(kernel, normalize_y=False, random_state=s.seed + it, n_restarts_optimizer=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(np.array(Us), (y - ym) / ys)
            cand = qmc.Sobol(3, scramble=True, seed=s.seed + 100 + it).random_base2(10)
            ib = int(np.argmax(y))
            local = np.clip(Us[ib] + rng.normal(0, 0.06, (256, 3)), 0, 1)
            cand = np.vstack([cand, local])
            mu, sd = gp.predict(cand, return_std=True)
            acq = _log_ei(mu, sd, (y.max() - ym) / ys)
            u_next = cand[int(np.argmax(acq))]
            hist.append(evaluate(wp, res, self._to_x(u_next), self.econ, s, check_stress=False))
            Us.append(u_next)
        # re-check the top candidates with the full constraint set (Goodman via wave equation)
        order = np.argsort([-h.objective for h in hist])
        best = None
        for i in order[:5]:
            e = evaluate(wp, res, hist[i].theta, self.econ, s, check_stress=True)
            hist.append(e)
            Us.append(Us[i])
            if e.feasible and (best is None or e.g_star > best.g_star):
                best = e
        if best is None:
            best = max(hist, key=lambda h: h.objective)
        # uncertainty: GP posterior std at the recommendation + spread over calibration samples
        y = np.array([h.objective for h in hist])
        ym, ys = y.mean(), y.std() or 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(kernel, random_state=s.seed).fit(np.array(Us), (y - ym) / ys)
        ub = (best.theta - s.bounds[:, 0]) / (s.bounds[:, 1] - s.bounds[:, 0])
        _, sd = gp.predict(ub[None, :], return_std=True)
        samples = []
        for _ in range(s.n_param_samples):
            w2, r2 = param_sampler(wp, res, rng)
            samples.append(evaluate(w2, r2, best.theta, self.econ, s, check_stress=False).g_star)
        rec = Recipe(steam_mass_t=round(float(best.theta[0]), 0), inj_pressure_bar=round(float(best.theta[1]), 1),
                     soak_d=round(float(best.theta[2]), 2), recipe_id=recipe_id or "bo")
        return Recommendation(rec, best, float(sd[0] * ys), float(np.std(samples)), [float(v) for v in samples], hist,
                              len(hist))

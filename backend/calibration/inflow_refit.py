"""Inflow refit (§7): nonlinear least squares (SciPy least_squares) of J_c and the
water-cut schedule (wc_late, tau) against observed rates, at the end of each cycle or
every K days. Guardrail: reject a refit that moves J_c by more than `max_rel`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class InflowObs:
    t_prod_d: float          # days since production start
    q_liquid: float          # m3/d measured
    wc: float | None         # measured water cut (may be None)
    drive: float             # model SR * (p_res - p_wf) in bar (twin-predicted)


@dataclass
class RefitResult:
    J_c: float               # m3/d/bar
    wc_late: float
    tau_d: float
    accepted: bool
    reason: str
    rmse_rate: float
    rmse_wc: float
    n: int
    J_c_std: float | None = None


def refit(obs: list[InflowObs], J_c0: float, wc_early: float, wc_late0: float, tau0: float,
          max_rel: float = 0.6, min_points: int = 24) -> RefitResult:
    ok = [o for o in obs if o.q_liquid is not None and np.isfinite(o.q_liquid) and o.drive > 0.5 and o.q_liquid > 0.05]
    if len(ok) < min_points:
        return RefitResult(J_c0, wc_late0, tau0, False, "insufficient_data", np.nan, np.nan, len(ok))
    t = np.array([o.t_prod_d for o in ok])
    q = np.array([o.q_liquid for o in ok])
    d = np.array([o.drive for o in ok])
    wc_mask = np.array([o.wc is not None and np.isfinite(o.wc) for o in ok])
    wc = np.array([o.wc if o.wc is not None else np.nan for o in ok])
    q_scale = max(float(np.median(q)), 1e-3)

    def resid(p: np.ndarray) -> np.ndarray:
        lnJ, wl, lntau = p
        r_q = (np.exp(lnJ) * d - q) / q_scale
        wc_m = wl + (wc_early - wl) * np.exp(-t / np.exp(lntau))
        r_w = (wc_m[wc_mask] - wc[wc_mask]) / 0.05
        prior = np.array([(wl - wc_late0) / 0.15, (lntau - np.log(tau0)) / 0.7])   # weak priors
        return np.concatenate([r_q, r_w, prior])

    x0 = np.array([np.log(J_c0), wc_late0, np.log(tau0)])
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=1.0,
                        bounds=([np.log(1e-4), 0.05, np.log(0.5)], [np.log(10.0), 0.95, np.log(90.0)]))
    J = float(np.exp(sol.x[0]))
    std = None
    try:
        JtJ = sol.jac.T @ sol.jac
        cov = np.linalg.pinv(JtJ) * (2 * sol.cost / max(len(sol.fun) - 3, 1))
        std = float(J * np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        pass
    r = resid(sol.x)
    nq = len(q)
    rm_q = float(np.sqrt(np.mean((r[:nq] * q_scale) ** 2)))
    rm_w = float(np.sqrt(np.mean((r[nq:-2] * 0.05) ** 2))) if wc_mask.any() else float("nan")
    if abs(J / J_c0 - 1.0) > max_rel:
        return RefitResult(J_c0, wc_late0, tau0, False, "guardrail_J_c", rm_q, rm_w, nq, std)
    return RefitResult(J, float(sol.x[1]), float(np.exp(sol.x[2])), True, "ok", rm_q, rm_w, nq, std)

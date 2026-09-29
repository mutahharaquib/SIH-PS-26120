"""Oil viscosity vs temperature (BUILD_SPEC §4.1).

Walther / ASTM D341 (default):  log10(log10(nu + 0.7)) = A - B*log10(T)
Arrhenius fallback:             ln(mu) = a + b/T    (fewer than `min_points_walther` points)

nu in cSt, T in K, mu in Pa.s. The Walther form is linear in (A, B) in the
transformed variables, which is what makes the RLS calibration (§7) valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.state import Quantity, Tier, q
from core.units import api_to_rho

LN10 = float(np.log(10.0))


def walther_y(nu_cst: np.ndarray | float) -> np.ndarray:
    return np.log10(np.log10(np.asarray(nu_cst, dtype=float) + 0.7))


def walther_x(T_K: np.ndarray | float) -> np.ndarray:
    return np.log10(np.asarray(T_K, dtype=float))


@dataclass(frozen=True)
class WaltherParams:
    A: float
    B: float
    T_min_fit: float
    T_max_fit: float
    cov: tuple[tuple[float, float], tuple[float, float]] | None = None

    def nu(self, T_K: np.ndarray | float) -> np.ndarray:
        y = self.A - self.B * walther_x(T_K)
        return np.power(10.0, np.power(10.0, y)) - 0.7

    def nu_rel_sigma(self, T_K: np.ndarray | float) -> np.ndarray:
        """1-sigma relative uncertainty of nu from the (A, B) covariance."""
        if self.cov is None:
            return np.zeros_like(np.asarray(T_K, dtype=float))
        x = walther_x(T_K)
        c = np.asarray(self.cov)
        var_y = c[0, 0] - 2.0 * x * c[0, 1] + x * x * c[1, 1]
        y = self.A - self.B * x
        nu = self.nu(T_K)
        # d ln(nu+0.7)/dy = ln10^2 * 10^y
        return np.sqrt(np.maximum(var_y, 0.0)) * LN10 * LN10 * np.power(10.0, y) * (nu + 0.7) / nu


@dataclass(frozen=True)
class ArrheniusParams:
    a: float
    b: float
    T_min_fit: float
    T_max_fit: float

    def mu(self, T_K: np.ndarray | float) -> np.ndarray:
        return np.exp(self.a + self.b / np.asarray(T_K, dtype=float))


def fit_walther(T_K: np.ndarray, nu_cSt: np.ndarray) -> WaltherParams:
    T_K = np.asarray(T_K, dtype=float)
    x = walther_x(T_K)
    y = walther_y(nu_cSt)
    X = np.column_stack([np.ones_like(x), -x])
    coef, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    dof = max(len(x) - 2, 1)
    s2 = float(res[0] / dof) if res.size else 1e-8
    cov = s2 * np.linalg.inv(X.T @ X)
    return WaltherParams(
        A=float(coef[0]), B=float(coef[1]), T_min_fit=float(T_K.min()), T_max_fit=float(T_K.max()),
        cov=((float(cov[0, 0]), float(cov[0, 1])), (float(cov[1, 0]), float(cov[1, 1]))),
    )


def fit_arrhenius(T_K: np.ndarray, mu_pas: np.ndarray) -> ArrheniusParams:
    T_K = np.asarray(T_K, dtype=float)
    X = np.column_stack([np.ones_like(T_K), 1.0 / T_K])
    coef, *_ = np.linalg.lstsq(X, np.log(np.asarray(mu_pas, dtype=float)), rcond=None)
    return ArrheniusParams(a=float(coef[0]), b=float(coef[1]), T_min_fit=float(T_K.min()), T_max_fit=float(T_K.max()))


def rho_oil_fn(api: float, beta: float) -> Callable[[np.ndarray | float], np.ndarray]:
    """Linear thermal-expansion correction from the 15.6 degC density."""
    rho15 = api_to_rho(api)

    def rho(T_K: np.ndarray | float) -> np.ndarray:
        return rho15 * (1.0 - beta * (np.asarray(T_K, dtype=float) - 288.71))

    return rho


@dataclass
class ViscosityModel:
    """Dynamic viscosity model for one well's crude."""

    rho_fn: Callable[[np.ndarray | float], np.ndarray]
    walther: WaltherParams | None = None
    arrhenius: ArrheniusParams | None = None
    placeholder: bool = True
    source: str = "physics.viscosity(Walther/ASTM D341)"
    tags: dict[str, float] = field(default_factory=dict)

    @property
    def fit_range(self) -> tuple[float, float]:
        p = self.walther or self.arrhenius
        assert p is not None
        return p.T_min_fit, p.T_max_fit

    def mu_value(self, T_K: np.ndarray | float) -> np.ndarray:
        if self.walther is not None:
            return self.walther.nu(T_K) * 1e-6 * self.rho_fn(T_K)
        assert self.arrhenius is not None
        return self.arrhenius.mu(T_K)

    def nu_value(self, T_K: np.ndarray | float) -> np.ndarray:
        return self.mu_value(T_K) / self.rho_fn(T_K) * 1e6


def mu(T_K: float, model: ViscosityModel, tier: Tier = Tier.B) -> Quantity:
    """Dynamic viscosity (Pa.s) as a Quantity; flags extrapolation outside the fit range."""
    lo, hi = model.fit_range
    val = float(model.mu_value(T_K))
    sig = None
    if model.walther is not None and model.walther.cov is not None:
        sig = float(model.walther.nu_rel_sigma(T_K)) * val
    return q(val, "Pa.s", tier, model.source, uncertainty=sig, extrapolated=not (lo <= T_K <= hi),
             placeholder=model.placeholder, inputs={"T": round(T_K, 3)})


def fit_viscosity(T_K: np.ndarray, nu_cSt: np.ndarray, rho_fn: Callable, min_points_walther: int = 4,
                  placeholder: bool = True) -> ViscosityModel:
    T_K = np.asarray(T_K, dtype=float)
    nu_cSt = np.asarray(nu_cSt, dtype=float)
    if len(np.unique(np.round(T_K, 1))) >= min_points_walther:
        return ViscosityModel(rho_fn=rho_fn, walther=fit_walther(T_K, nu_cSt), placeholder=placeholder)
    mu_pas = nu_cSt * 1e-6 * rho_fn(T_K)
    return ViscosityModel(rho_fn=rho_fn, arrhenius=fit_arrhenius(T_K, mu_pas), placeholder=placeholder,
                          source="physics.viscosity(Arrhenius fallback)")


def model_for_api(lab_points: list[list[float]], api: float, api_slope: float, beta: float,
                  placeholder: bool = True) -> ViscosityModel:
    """Shift the reference lab curve (18 degAPI) to a well's API gravity, then fit Walther."""
    pts = np.asarray(lab_points, dtype=float)
    nu = pts[:, 1] * 10.0 ** (api_slope * (api - 18.0))
    return fit_viscosity(pts[:, 0], nu, rho_oil_fn(api, beta), placeholder=placeholder)

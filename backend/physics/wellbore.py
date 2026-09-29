"""Wellbore heat loss (BUILD_SPEC §4.2) — Ramey (1962) / Satter (1965) form.

Q'(z) = 2*pi*r_to*U*k_e*(T_f - T_e(z)) / (k_e + r_to*U*f(t))

[VERIFY resolved] f(t): Ramey's long-time approximation is
    f(t) = -ln(r_cw / (2*sqrt(alpha*t))) - 0.290.
We use the Hasan & Kabir (1991) full-range fit, which reproduces Ramey's
long-time form (0.4063 + 0.5 ln t_D  ==  ln(2 sqrt t_D) - 0.290 to 3e-3) and stays
positive at early time. Test `test_ramey_long_time_limit` checks the agreement.

Injection (saturated steam): the march is done on mixture enthalpy,
    w dh/dz = -Q'(z) + w g           (z down, potential energy -> enthalpy)
which reduces to the spec's w h_fg dx/dz = -Q' at constant pressure; x is then
recovered as (h - h_f(p)) / h_fg(p) and T = T_sat(p). Pressure: hydrostatic
(homogeneous mixture density) minus Darcy friction with a homogeneous two-phase
friction factor (documented simplification).

Production: single-phase liquid energy balance w c_p dT/dz_up = -Q', integrated
per cell with the exact exponential relaxation toward the local geotherm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core import steam
from core.units import G


@dataclass(frozen=True)
class WellboreGeom:
    depth: float          # m, to sandface (injection) / pump (production computed separately)
    r_ti: float
    r_to: float
    r_ci: float
    r_cw: float
    k_e: float
    alpha_e: float
    T_surface: float      # K
    grad: float           # K/m geothermal gradient
    n_cells: int = 100
    friction: float = 0.02

    def T_earth(self, z: np.ndarray | float) -> np.ndarray:
        return self.T_surface + self.grad * np.asarray(z, dtype=float)


def ramey_f(t_s: float, alpha_e: float, r_cw: float) -> float:
    """Transient heat-conduction time function f(t) (Hasan-Kabir full-range fit)."""
    tD = max(alpha_e * max(t_s, 1.0) / (r_cw * r_cw), 1e-9)
    if tD <= 1.5:
        s = np.sqrt(tD)
        return float(1.1281 * s * (1.0 - 0.3 * s))
    return float((0.4063 + 0.5 * np.log(tD)) * (1.0 + 0.6 / tD))


def ramey_long_time(t_s: float, alpha_e: float, r_cw: float) -> float:
    return float(-np.log(r_cw / (2.0 * np.sqrt(alpha_e * t_s))) - 0.290)


def heat_loss_per_length(T_f, T_e, r_to: float, U: float, k_e: float, f: float):
    """W/m lost from the tubing fluid to the formation (positive when T_f > T_e)."""
    return 2.0 * np.pi * r_to * U * k_e * (np.asarray(T_f) - np.asarray(T_e)) / (k_e + r_to * U * f)


@dataclass
class InjectionProfile:
    z: np.ndarray
    p: np.ndarray
    T: np.ndarray
    x: np.ndarray
    h: np.ndarray
    q_loss: np.ndarray        # W/m per cell
    total_loss_W: float
    x_sandface: float
    p_sandface: float
    T_sandface: float
    closure_rel_err: float


def steam_injection(geom: WellboreGeom, w: float, p_wh: float, x_wh: float, t_s: float, U: float) -> InjectionProfile:
    """March saturated steam down the tubing. w in kg/s, p_wh in Pa, t_s = injection time elapsed."""
    n = geom.n_cells
    dz = geom.depth / n
    z = np.linspace(0.0, geom.depth, n + 1)
    f = ramey_f(t_s, geom.alpha_e, geom.r_cw)
    area = np.pi * geom.r_ti ** 2
    D = 2.0 * geom.r_ti
    p = np.empty(n + 1)
    h = np.empty(n + 1)
    x = np.empty(n + 1)
    T = np.empty(n + 1)
    ql = np.zeros(n)
    p[0] = p_wh
    h[0] = float(steam.h_f(p_wh) + x_wh * steam.h_fg(p_wh))
    for i in range(n + 1):
        hf, hfg = float(steam.h_f(p[i])), float(steam.h_fg(p[i]))
        xi = (h[i] - hf) / hfg
        if xi >= 0.0:
            x[i] = min(xi, 1.0)
            T[i] = float(steam.T_sat(p[i]))
        else:  # fully condensed: subcooled liquid
            x[i] = 0.0
            T[i] = float(steam.T_sat(p[i])) + (h[i] - hf) / steam.C_P_WATER
        if i == n:
            break
        zm = z[i] + 0.5 * dz
        ql[i] = float(heat_loss_per_length(T[i], geom.T_earth(zm), geom.r_to, U, geom.k_e, f)) if U > 0 else 0.0
        h[i + 1] = h[i] + G * dz - ql[i] * dz / w
        rho_m = 1.0 / (x[i] / float(steam.rho_g(p[i])) + (1.0 - x[i]) / float(steam.rho_f(p[i])))
        v = w / (rho_m * area)
        dp = (rho_m * G - geom.friction * rho_m * v * v / (2.0 * D)) * dz
        p[i + 1] = p[i] + dp
    total = float(np.sum(ql) * dz)
    # Energy closure: enthalpy drop + potential energy gain = heat lost
    lhs = w * (h[0] - h[-1]) + w * G * geom.depth
    closure = abs(lhs - total) / max(abs(total), 1.0)
    return InjectionProfile(z=z, p=p, T=T, x=x, h=h, q_loss=ql, total_loss_W=total, x_sandface=float(x[-1]),
                            p_sandface=float(p[-1]), T_sandface=float(T[-1]), closure_rel_err=closure)


@dataclass
class ProductionProfile:
    z: np.ndarray          # m, 0 = surface
    T: np.ndarray          # K
    total_loss_W: float
    closure_rel_err: float


def produced_fluid_profile(geom: WellboreGeom, w: float, cp: float, T_bottom: float, t_s: float, U: float,
                           bottom_depth: float, n_cells: int | None = None) -> ProductionProfile:
    """Temperature of the produced liquid from `bottom_depth` (pump) up to the surface.

    w: mass rate kg/s (0 -> fluid at geotherm), cp: J/kg/K.
    """
    n = n_cells or geom.n_cells
    z = np.linspace(0.0, bottom_depth, n + 1)
    dz = bottom_depth / n
    T = np.empty(n + 1)
    T[-1] = T_bottom
    f = ramey_f(t_s, geom.alpha_e, geom.r_cw)
    coef = 2.0 * np.pi * geom.r_to * U * geom.k_e / (geom.k_e + geom.r_to * U * f)   # W/m/K
    if w <= 1e-9 or coef <= 0.0:
        if w <= 1e-9:
            return ProductionProfile(z=z, T=geom.T_earth(z), total_loss_W=0.0, closure_rel_err=0.0)
        T[:] = T_bottom
        return ProductionProfile(z=z, T=T, total_loss_W=0.0, closure_rel_err=0.0)
    L_R = w * cp / coef
    decay = np.exp(-dz / L_R)
    loss = 0.0
    for i in range(n, 0, -1):
        Te_mid = float(geom.T_earth(z[i] - 0.5 * dz))
        T[i - 1] = Te_mid + (T[i] - Te_mid) * decay
        loss += w * cp * (T[i] - T[i - 1])
    # closure: sum of local Q' (trapezoid in cell) vs enthalpy drop
    Te = geom.T_earth(z)
    ql = coef * (T - Te)
    integ = float(np.trapezoid(ql, z))
    closure = abs(integ - loss) / max(abs(loss), 1.0)
    return ProductionProfile(z=z, T=T, total_loss_W=loss, closure_rel_err=closure)

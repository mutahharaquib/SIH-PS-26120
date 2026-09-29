"""CSS reservoir model (BUILD_SPEC §4.3): Marx-Langenheim heated area during
injection + Boberg-Lantz average heated-zone temperature during soak/production.

[VERIFY resolved] f_HD and f_VD are computed from the exact 1-D conduction
solutions (the option the spec allows when textbook approximations are not
verified), with the overburden diffusivity alpha = lambda_ob / M_ob:

  f_VD  slab of thickness h, initially dT, cooling to both sides into an infinite
        medium; slab-average:   erf(X) + (exp(-X^2) - 1) / (X sqrt(pi)),
        X = h / (2 sqrt(alpha t)).
  f_HD  cylinder of radius r_h, initially dT, cooling radially; cylinder-average
        (Carslaw & Jaeger):     1 - exp(-y) [I0(y) + I1(y)],   y = r_h^2 / (2 alpha t).

Both are checked in tests against direct numerical integration of the heat kernel.
f_PD = (1/2) * E_removed / (pi r_h^2 h M_R dT)  (Boberg-Lantz energy-removal term).

Carry-over: the heat left at the end of a cycle, E_res = pi r_h^2 h M_R (T_avg - T_R),
is added to the next cycle's heated volume at the new steam temperature.
Depletion: tank model, dp = -(oil voidage) / compliance (documented choice:
injected water is assumed produced back, so only oil voidage depletes pressure).

Heated-zone oil depletion (extension beyond Boberg-Lantz, PLACEHOLDER parameters):
the stimulated inflow comes mostly from the heated zone, whose recoverable oil is
N_rec = pi r_h^2 h phi (S_oi - S_or_hot), reduced by oil already produced in earlier
cycles. The stimulation fades as it is produced:
    SR_eff = 1 + (SR - 1) * (1 - N_p,cycle / N_rec)^n
This gives the rise-then-decline cycle shape and the cycle-to-cycle decline that the
cut-off (§8.2) and recipe optimizer (§8.3) act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import erf, erfcx, i0e, i1e


def G_ml(tD: float) -> float:
    """Marx-Langenheim G(t_D) = e^tD erfc(sqrt tD) + 2 sqrt(tD/pi) - 1 (erfcx for stability)."""
    s = np.sqrt(max(tD, 0.0))
    return float(erfcx(s) + 2.0 * s / np.sqrt(np.pi) - 1.0)


def t_D_ml(t_s: float, lambda_ob: float, M_ob: float, M_R: float, h: float) -> float:
    return 4.0 * lambda_ob * M_ob * t_s / (M_R * M_R * h * h)


def marx_langenheim_area(Q_i: float, M_R: float, h: float, lambda_ob: float, M_ob: float, dT: float,
                         t_s: float) -> float:
    """Heated area (m^2) after injecting heat at rate Q_i (W, sandface) for t_s seconds."""
    if Q_i <= 0 or dT <= 0 or t_s <= 0:
        return 0.0
    tD = t_D_ml(t_s, lambda_ob, M_ob, M_R, h)
    return Q_i * M_R * h / (4.0 * lambda_ob * M_ob * dT) * G_ml(tD)


def f_VD(t_s: float, h: float, alpha: float) -> float:
    if t_s < 1e-3:
        return 1.0
    X = h / (2.0 * np.sqrt(alpha * t_s))
    if X > 50:
        return float(1.0 - 1.0 / (X * np.sqrt(np.pi)))
    return float(erf(X) + (np.exp(-X * X) - 1.0) / (X * np.sqrt(np.pi)))


def f_HD(t_s: float, r_h: float, alpha: float) -> float:
    if t_s < 1e-3:
        return 1.0
    if r_h <= 0:
        return 0.0
    y = r_h * r_h / (2.0 * alpha * t_s)
    return float(1.0 - i0e(y) - i1e(y))


def T_avg_bl(T_R: float, T_s: float, fhd: float, fvd: float, fpd: float) -> float:
    """Boberg-Lantz average heated-zone temperature, clamped to [T_R, T_s]."""
    T = T_R + (T_s - T_R) * (fhd * fvd * (1.0 - fpd) - fpd)
    return float(min(max(T, T_R), T_s))


def stimulation_ratio(r_e: float, r_w: float, r_h: float, mu_h: float, mu_c: float) -> float:
    """J_hot/J_cold for radial flow with a heated inner zone of radius r_h."""
    r_h = min(max(r_h, r_w), r_e)
    return float(np.log(r_e / r_w) / ((mu_h / mu_c) * np.log(r_h / r_w) + np.log(r_e / r_h)))


@dataclass
class ReservoirParams:
    T_R: float
    h: float
    M_R: float
    M_ob: float
    lambda_ob: float
    r_w: float
    r_e: float
    p_init: float
    p_min: float
    compliance: float     # m3 / Pa
    porosity: float = 0.28
    S_oi: float = 0.75
    S_or_hot: float = 0.30
    dep_exp: float = 2.0

    @property
    def alpha_ob(self) -> float:
        return self.lambda_ob / self.M_ob


@dataclass
class CSSReservoir:
    """Mutable reservoir state across cycles."""

    prm: ReservoirParams
    p_res: float = 0.0
    cycle_index: int = 0
    T_s: float = 0.0
    r_h: float = 0.0
    t_since_heat: float = 0.0         # s since end of injection
    E_removed: float = 0.0            # J removed with produced fluids this cycle
    E_residual: float = 0.0           # J carried over from the previous cycle
    cum_oil: float = 0.0              # m3, all cycles
    cycle_oil: float = 0.0            # m3, this cycle
    N_rec: float = 0.0                # m3 recoverable heated-zone oil this cycle
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.p_res == 0.0:
            self.p_res = self.prm.p_init
        if self.T_s == 0.0:
            self.T_s = self.prm.T_R

    # --- injection -----------------------------------------------------
    def inject(self, Q_sandface_W: float, t_inj_s: float, T_s: float) -> float:
        """Apply a completed injection; returns heated radius."""
        prm = self.prm
        dT = T_s - prm.T_R
        A = marx_langenheim_area(Q_sandface_W, prm.M_R, prm.h, prm.lambda_ob, prm.M_ob, dT, t_inj_s)
        A += self.E_residual / (prm.h * prm.M_R * dT) if dT > 0 else 0.0
        self.cycle_index += 1
        self.T_s = T_s
        self.r_h = float(np.sqrt(A / np.pi))
        self.t_since_heat = 0.0
        self.E_removed = 0.0
        pore = np.pi * self.r_h ** 2 * prm.h * prm.porosity
        gross = pore * (prm.S_oi - prm.S_or_hot)
        self.N_rec = float(max(gross - self.cum_oil, 0.15 * gross))
        self.cycle_oil = 0.0
        return self.r_h

    # --- soak / production ----------------------------------------------
    def heated_energy_nominal(self) -> float:
        return np.pi * self.r_h ** 2 * self.prm.h * self.prm.M_R * (self.T_s - self.prm.T_R)

    def f_PD(self) -> float:
        E0 = self.heated_energy_nominal()
        return 0.5 * self.E_removed / E0 if E0 > 0 else 0.0

    def T_avg(self) -> float:
        prm = self.prm
        if self.r_h <= 0:
            return prm.T_R
        a = prm.alpha_ob
        return T_avg_bl(prm.T_R, self.T_s, f_HD(self.t_since_heat, self.r_h, a),
                        f_VD(self.t_since_heat, prm.h, a), self.f_PD())

    def advance(self, dt_s: float, q_oil: float = 0.0, q_water: float = 0.0,
                rho_c_oil: float = 1.9e6, rho_c_water: float = 4.1e6) -> None:
        """Advance time; q in m3/s. Removes produced-fluid heat and depletes pressure."""
        T = self.T_avg()
        self.E_removed += (q_oil * rho_c_oil + q_water * rho_c_water) * (T - self.prm.T_R) * dt_s
        self.t_since_heat += dt_s
        vo = q_oil * dt_s
        self.cum_oil += vo
        self.cycle_oil += vo
        self.p_res = max(self.prm.p_min, self.p_res - vo / self.prm.compliance)

    def depletion_factor(self) -> float:
        if self.N_rec <= 0:
            return 1.0
        return float(max(0.0, 1.0 - self.cycle_oil / self.N_rec) ** self.prm.dep_exp)

    def effective_sr(self, sr: float) -> float:
        """Stimulation ratio after heated-zone oil depletion."""
        return 1.0 + (sr - 1.0) * self.depletion_factor()

    def end_cycle(self) -> float:
        """Close the cycle; the heat remaining is carried into the next one."""
        self.E_residual = np.pi * self.r_h ** 2 * self.prm.h * self.prm.M_R * (self.T_avg() - self.prm.T_R)
        self.history.append({"cycle": self.cycle_index, "E_residual": self.E_residual, "p_res": self.p_res})
        return self.E_residual

    def clone(self) -> "CSSReservoir":
        c = CSSReservoir(prm=self.prm)
        c.__dict__.update({k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()})
        return c

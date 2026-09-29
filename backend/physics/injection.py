"""Injection planning: couples the wellbore steam march (§4.2) to the reservoir.

Given a recipe (steam mass, wellhead injection pressure) the injection rate w is the
root of   w = min(capacity, injectivity * (p_sandface(w) - p_res)),
where p_sandface(w) falls with w through tubing friction. Heat delivered to the
reservoir uses the SANDFACE state (quality/enthalpy), never the surface quality.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from core import steam
from core.units import BAR, DAY
from physics.params import WellParams
from physics.wellbore import steam_injection


@dataclass
class InjectionPlan:
    w: float                 # kg/s
    t_inj: float             # s
    x_sf: float
    p_sf: float              # Pa
    T_s: float               # K (sandface steam temperature)
    Q_sf: float              # W heat delivered relative to reservoir-temperature liquid
    heat_loss: float         # W wellbore heat loss
    capped: bool             # rate limited by the generator
    injectable: bool         # sandface pressure exceeds reservoir pressure
    x_profile: list[tuple[float, float]]
    T_profile: list[tuple[float, float]]


def plan_injection(wp: WellParams, steam_mass_kg: float, p_wh: float, p_res: float, x_surface: float,
                   capacity_kg_s: float, n_cells: int = 40) -> InjectionPlan:
    geom = replace(wp.geom, n_cells=n_cells)

    def run(w: float, t: float):
        return steam_injection(geom, w, p_wh, x_surface, t, wp.U_inject)

    t_guess = 5 * DAY
    w = capacity_kg_s
    injectable = True
    for _ in range(2):  # heat loss depends on the (mid-)injection time, which depends on w
        lo, hi = 0.02, capacity_kg_s
        prof_hi = run(hi, t_guess)
        if wp.injectivity * (prof_hi.p_sandface - p_res) >= hi:
            w, capped = hi, True
        else:
            capped = False
            prof_lo = run(lo, t_guess)
            if wp.injectivity * (prof_lo.p_sandface - p_res) < lo:
                w, injectable = lo, False
            else:
                for _ in range(22):
                    mid = 0.5 * (lo + hi)
                    pm = run(mid, t_guess)
                    if wp.injectivity * (pm.p_sandface - p_res) >= mid:
                        lo = mid
                    else:
                        hi = mid
                w = 0.5 * (lo + hi)
        t_inj = steam_mass_kg / w
        t_guess = 0.5 * t_inj
    prof = run(w, t_guess)
    h_sf = float(prof.h[-1])
    h_ref = steam.C_P_WATER * (wp.T_R - 273.15)
    Q = w * max(h_sf - h_ref, 0.0)
    return InjectionPlan(
        w=w, t_inj=steam_mass_kg / w, x_sf=prof.x_sandface, p_sf=prof.p_sandface, T_s=prof.T_sandface, Q_sf=Q,
        heat_loss=prof.total_loss_W, capped=capped, injectable=injectable and prof.p_sandface > p_res,
        x_profile=[(float(z), float(x)) for z, x in zip(prof.z[::4], prof.x[::4])],
        T_profile=[(float(z), float(T)) for z, T in zip(prof.z[::4], prof.T[::4])],
    )


def gen_capacity_kg_s(t_per_day: float) -> float:
    return t_per_day * 1000.0 / DAY


def bar(p: float) -> float:
    return p * BAR

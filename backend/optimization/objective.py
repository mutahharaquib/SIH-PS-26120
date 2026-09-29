"""Cycle economics (BUILD_SPEC §8.1).

q_net(t) = P_oil q_oil(t) - tariff P_motor(t) - opex_rate      (+ water handling)
V(T)     = int_0^T q_net dt - C_steam(recipe) - C_fixed
g(T)     = V(T) / (t_inject + t_soak + T)
production_weight w in [0,1] blends in a volume term: (1-w) q_net + w P_oil q_oil.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import Config, get_config
from core.units import BBL


@dataclass(frozen=True)
class Econ:
    oil_price_m3: float
    steam_cost_t: float
    tariff_kwh: float
    opex_d: float
    fixed_cycle: float
    water_m3: float
    production_weight: float = 0.0
    discount_rate: float = 0.10

    @classmethod
    def from_config(cls, cfg: Config | None = None, production_weight: float | None = None) -> "Econ":
        c = cfg or get_config()
        e = "economics."
        return cls(
            oil_price_m3=c.v(e + "oil_price") / BBL, steam_cost_t=c.v(e + "steam_cost"), tariff_kwh=c.v(e + "tariff"),
            opex_d=c.v(e + "opex_rate"), fixed_cycle=c.v(e + "fixed_cost_cycle"), water_m3=c.v(e + "water_handling"),
            production_weight=c.v(e + "production_weight") if production_weight is None else production_weight,
            discount_rate=c.v(e + "discount_rate"),
        )


def q_net(q_oil_m3d, p_motor_kw, q_water_m3d, econ: Econ):
    """Net value rate (currency/day)."""
    q_oil_m3d = np.asarray(q_oil_m3d, dtype=float)
    value = (econ.oil_price_m3 * q_oil_m3d - econ.tariff_kwh * np.asarray(p_motor_kw) * 24.0 - econ.opex_d
             - econ.water_m3 * np.asarray(q_water_m3d))
    w = econ.production_weight
    return (1.0 - w) * value + w * econ.oil_price_m3 * q_oil_m3d


def cycle_costs(steam_t: float, econ: Econ) -> tuple[float, float]:
    return steam_t * econ.steam_cost_t, econ.fixed_cycle


def g_curve(t_d: np.ndarray, qn: np.ndarray, t_inject_d: float, t_soak_d: float, c_steam: float,
            c_fixed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T grid, V(T), g(T)); q_net sampled at cell centres t_d with uniform spacing."""
    t_d = np.asarray(t_d, dtype=float)
    dt = float(t_d[1] - t_d[0]) if len(t_d) > 1 else 1.0
    T = t_d + 0.5 * dt                              # end of each cell
    V = np.cumsum(np.asarray(qn) * dt) - c_steam - c_fixed
    g = V / (t_inject_d + t_soak_d + T)
    return T, V, g

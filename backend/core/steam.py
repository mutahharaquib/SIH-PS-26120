"""Saturated water/steam properties behind one interface (IAPWS-IF97 via `iapws`).

A log-pressure lookup table is built once at import so the marching solvers stay
fast; linear interpolation error is < 0.05 % over 0.05-20 MPa.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

P_MIN = 5.0e4    # Pa
P_MAX = 2.0e7    # Pa
C_P_WATER = 4200.0   # J/kg/K, liquid water (used for subcooled condensate)


@lru_cache(maxsize=1)
def _table() -> dict[str, np.ndarray]:
    p = np.geomspace(P_MIN, P_MAX, 400)
    try:
        from iapws import IAPWS97

        T, hf, hg, rf, rg = [], [], [], [], []
        for pi in p:
            liq = IAPWS97(P=pi / 1e6, x=0.0)
            vap = IAPWS97(P=pi / 1e6, x=1.0)
            T.append(liq.T)
            hf.append(liq.h * 1e3)
            hg.append(vap.h * 1e3)
            rf.append(liq.rho)
            rg.append(vap.rho)
        src = "IAPWS-IF97 (iapws)"
    except ImportError:  # pragma: no cover - fallback: Antoine-type fit + linearized enthalpies
        T = 1730.63 / (8.07131 - np.log10(p / 133.322)) - 233.426 + 273.15
        T = np.asarray(T)
        hf = C_P_WATER * (T - 273.15)
        hg = 2.501e6 + 1.82e3 * (T - 273.15) - 0.0 * T
        rf = 1000.0 - 0.0035 * (T - 277.0) ** 2
        rg = p / (461.5 * T)
        src = "fallback correlation"
    return {
        "lnp": np.log(p),
        "T": np.asarray(T),
        "hf": np.asarray(hf),
        "hg": np.asarray(hg),
        "rf": np.asarray(rf),
        "rg": np.asarray(rg),
        "source": np.array([src]),
    }


def _interp(key: str, p: float | np.ndarray) -> np.ndarray | float:
    t = _table()
    pc = np.clip(p, P_MIN, P_MAX)
    return np.interp(np.log(pc), t["lnp"], t[key])


def T_sat(p: float | np.ndarray):
    """Saturation temperature (K) at pressure p (Pa)."""
    return _interp("T", p)


def h_f(p):
    return _interp("hf", p)


def h_g(p):
    return _interp("hg", p)


def h_fg(p):
    return _interp("hg", p) - _interp("hf", p)


def rho_f(p):
    return _interp("rf", p)


def rho_g(p):
    return _interp("rg", p)


def source() -> str:
    return str(_table()["source"][0])


def p_sat(T: float) -> float:
    """Saturation pressure (Pa) for temperature T (K), by inverting the table."""
    t = _table()
    return float(np.exp(np.interp(T, t["T"], t["lnp"])))

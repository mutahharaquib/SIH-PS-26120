"""Unit conversions. SI internally; field units only at the API/UI boundary."""

from __future__ import annotations

BAR = 1.0e5                      # Pa
PSI = 6894.757293168             # Pa
BBL = 0.158987294928             # m3
FT = 0.3048                      # m
INCH = 0.0254                    # m
CP = 1.0e-3                      # Pa.s
DAY = 86400.0                    # s
HOUR = 3600.0                    # s
G = 9.80665                      # m/s2
RHO_WATER_STD = 999.0            # kg/m3 (60 degF)


def c_to_k(t_c: float) -> float:
    return t_c + 273.15


def k_to_c(t_k: float) -> float:
    return t_k - 273.15


def k_to_f(t_k: float) -> float:
    return (t_k - 273.15) * 9.0 / 5.0 + 32.0


def f_to_k(t_f: float) -> float:
    return (t_f - 32.0) * 5.0 / 9.0 + 273.15


def pa_to_psi(p: float) -> float:
    return p / PSI


def psi_to_pa(p: float) -> float:
    return p * PSI


def bar_to_pa(p: float) -> float:
    return p * BAR


def pa_to_bar(p: float) -> float:
    return p / BAR


def m3d_to_bbld(q: float) -> float:
    return q / BBL


def bbld_to_m3d(q: float) -> float:
    return q * BBL


def m_to_ft(x: float) -> float:
    return x / FT


def ft_to_m(x: float) -> float:
    return x * FT


def pas_to_cp(mu: float) -> float:
    return mu / CP


def cp_to_pas(mu: float) -> float:
    return mu * CP


def api_to_sg(api: float) -> float:
    return 141.5 / (131.5 + api)


def api_to_rho(api: float) -> float:
    """Oil density at 15.6 degC (60 degF), kg/m3."""
    return api_to_sg(api) * RHO_WATER_STD


# Field-unit conversion table used by the API layer: SI unit -> (field unit, fn)
TO_FIELD = {
    "Pa": ("psi", pa_to_psi),
    "bar": ("psi", lambda p: pa_to_psi(bar_to_pa(p))),
    "m3/d": ("bbl/d", m3d_to_bbld),
    "m3": ("bbl", lambda v: v / BBL),
    "K": ("degF", k_to_f),
    "m": ("ft", m_to_ft),
    "Pa.s": ("cP", pas_to_cp),
}


def to_field(value: float, unit: str) -> tuple[float, str]:
    if unit in TO_FIELD:
        new_unit, fn = TO_FIELD[unit]
        return fn(value), new_unit
    return value, unit

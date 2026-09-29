"""Energy and efficiency outputs (BUILD_SPEC §4.9)."""

from __future__ import annotations

import numpy as np

from core.units import BBL


def card_area(position, load) -> float:
    """Work per stroke (J): closed-contour area of load vs position (shoelace)."""
    x = np.asarray(position, dtype=float)
    y = np.asarray(load, dtype=float)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polished_rod_power(position, load, spm: float) -> float:
    """W = card area x strokes per second."""
    return card_area(position, load) * spm / 60.0


def motor_input_power(p_pr: float, eta_gearbox: float, eta_belt: float, eta_motor: float, eta_vfd: float) -> float:
    return p_pr / (eta_gearbox * eta_belt * eta_motor * eta_vfd)


def kwh_per_bbl(energy_kwh: float, oil_m3: float) -> float:
    bbl = oil_m3 / BBL
    return float(energy_kwh / bbl) if bbl > 1e-9 else float("nan")


def sor(steam_t: float, oil_m3: float) -> float:
    """Cycle steam-oil ratio: cold-water-equivalent m3 (1 t steam = 1 m3 CWE) per m3 oil."""
    return float(steam_t / oil_m3) if oil_m3 > 1e-9 else float("nan")


def pr_power_estimate(q_liquid: float, dp_lift: float, drag_coef_per_len, dz: float, v_rms: float,
                      friction_w: float = 50.0) -> float:
    """Between cards: hydraulic power + viscous rod-drag dissipation + friction (W).

    drag_coef_per_len: array of 2 pi mu / ln(r_t/r_r) * multiplier per rod segment (N s/m2).
    Mean drag power = sum(coef * <V^2>) * dz with <V^2> = v_rms^2.
    """
    p_drag = float(np.sum(np.asarray(drag_coef_per_len)) * dz * v_rms * v_rms)
    return q_liquid * dp_lift + p_drag + friction_w

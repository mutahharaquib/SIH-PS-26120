"""Viscous rod drag and damping (BUILD_SPEC §4.6).

Couette approximation (concentric rod in tubing, no net annular flow):
    F_drag / L = 2 pi mu V / ln(r_t / r_r)
Damping coefficient per unit rod mass (wave-equation form):
    c(z) = 2 pi mu(z) / (rho_steel A_r ln(r_t / r_r))
Terminal rod fall velocity (buoyant weight = drag):
    V_fall(z) = (rho_steel - rho_fluid) g A_r ln(r_t/r_r) / (2 pi mu(z))

Simplification (documented): couplings, guides and pressure-driven annular flow are
ignored in v1; `multiplier` (config drag.drag_multiplier) lumps couplings/guides.
"""

from __future__ import annotations

import numpy as np

from core.units import G


def log_ratio(r_t: float, r_r) -> np.ndarray:
    return np.log(r_t / np.asarray(r_r, dtype=float))


def drag_per_length(mu, V, r_t: float, r_r, multiplier: float = 1.0):
    return multiplier * 2 * np.pi * np.asarray(mu) * np.asarray(V) / log_ratio(r_t, r_r)


def damping_coeff(mu, rho_steel: float, A_r, r_t: float, r_r, multiplier: float = 1.0):
    return multiplier * 2 * np.pi * np.asarray(mu) / (rho_steel * np.asarray(A_r) * log_ratio(r_t, r_r))


def v_fall(mu, rho_steel: float, rho_fluid: float, A_r, r_t: float, r_r, multiplier: float = 1.0):
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    return (rho_steel - rho_fluid) * G * np.asarray(A_r) * log_ratio(r_t, r_r) / (multiplier * 2 * np.pi * mu)


def v_fall_min(mu_nodes, rho_steel: float, rho_fluid: float, A_nodes, r_t: float, r_nodes,
               multiplier: float = 1.0, plunger_extra: float = 0.0) -> float:
    """Slowest-falling segment governs. `plunger_extra` adds pump plunger drag as a
    fraction of the string's average drag (reduces the fall velocity uniformly)."""
    vf = v_fall(mu_nodes, rho_steel, rho_fluid, A_nodes, r_t, r_nodes, multiplier)
    return float(np.min(vf) / (1.0 + plunger_extra))

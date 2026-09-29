"""Inflow (BUILD_SPEC §4.4): q_in = J_c * (J_h/J_c) * (p_reservoir - p_wf).

Water cut follows a per-cycle schedule: condensed steam is produced back early,
so early-cycle production is water dominated (low / negative net value), which
is the shape the cut-off logic (§8) must handle.
"""

from __future__ import annotations

import numpy as np


def inflow_rate(J_c: float, sr: float, p_res: float, p_wf: float) -> float:
    """Total liquid rate (m3/s) for J_c in m3/s/Pa and pressures in Pa. Never negative."""
    return float(max(0.0, J_c * sr * (p_res - p_wf)))


def water_cut(t_prod_s: float, wc_early: float, wc_late: float, tau_s: float):
    """Water-cut schedule: exponential decay from wc_early to wc_late."""
    return wc_late + (wc_early - wc_late) * np.exp(-np.asarray(t_prod_s, dtype=float) / tau_s)


def p_wf_for_rate(J_c: float, sr: float, p_res: float, q: float) -> float:
    """Flowing bottomhole pressure that delivers liquid rate q."""
    return float(p_res - q / max(J_c * sr, 1e-30))

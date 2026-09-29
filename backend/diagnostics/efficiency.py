"""Pump efficiency (BUILD_SPEC §6.4)."""

from __future__ import annotations

from diagnostics.card_features import fillage_estimate


def fillage_from_card(dh_pos, dh_load) -> float:
    """Fillage = effective plunger stroke / gross plunger stroke (downhole card)."""
    return fillage_estimate(dh_pos, dh_load)


def theoretical_displacement_m3d(A_p: float, stroke_m: float, spm: float) -> float:
    return A_p * stroke_m * spm * 1440.0


def volumetric_efficiency(q_measured_m3d: float, A_p: float, stroke_m: float, spm: float) -> float:
    """Measured liquid rate / theoretical displacement (polished-rod stroke)."""
    d = theoretical_displacement_m3d(A_p, stroke_m, spm)
    return float(q_measured_m3d / d) if d > 1e-9 else 0.0

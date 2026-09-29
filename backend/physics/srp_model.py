"""Coupling helpers for the SRP chain: tubing temperature -> viscosity profile -> rod
damping / fall velocity -> wave equation -> cards, stresses, power.

This is the forward coupling point of the twin (§0): reservoir temperature reaches
the rods only through the wellbore production profile and the viscosity model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.state import SpeedProfile
from physics import drag, energy
from physics.params import WellParams
from physics.pump import DownholePump, PumpCondition
from physics.pumping_unit import Kinematics, kinematics
from physics.rod_string import ForwardResult, RodGrid, predictive
from physics.wellbore import produced_fluid_profile


FORWARD_CFL = 0.9   # the explicit scheme is marginal at exactly 1 with a nonlinear pump BC


def tubing_temperature(wp: WellParams, q_liquid: float, wc: float, T_pump: float, t_prod_s: float,
                       n_cells: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """(z, T) of the produced liquid from the pump to surface. q_liquid in m3/s."""
    rho = wp.rho_liquid(T_pump, wc)
    cp = wc * 4180.0 + (1 - wc) * 2000.0
    prof = produced_fluid_profile(wp.geom, q_liquid * rho, cp, T_pump, max(t_prod_s, 3600.0), wp.U_produce,
                                  wp.pump_depth, n_cells=n_cells)
    return prof.z, prof.T


def mu_at(wp: WellParams, T: np.ndarray, wc: float = 0.0) -> np.ndarray:
    """Effective annulus (tubing) liquid viscosity. Oil-continuous below the inversion
    water cut; beyond it the brine-continuous emulsion is taken at a fraction of oil
    viscosity (documented simplification, PLACEHOLDER)."""
    mu_o = wp.visc.mu_value(T)
    if wc <= 0.6:
        return mu_o * (1.0 + 2.5 * wc)            # Einstein-type dispersed-water correction
    return np.maximum(mu_o * 0.08, 1e-3)


def rod_coefficients(wp: WellParams, grid: RodGrid, mu_nodes: np.ndarray, rho_fluid: float
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """Damping per node (1/s), buoyant node weights (N), and V_fall_min (m/s)."""
    r_t = wp.geom.r_ti
    c = drag.damping_coeff(mu_nodes, grid.rod.sections[0].rho, grid.node_A, r_t, grid.node_r, wp.drag_multiplier)
    c = c + wp.damping_floor
    w = grid.node_weights(rho_fluid)
    vfm = drag.v_fall_min(mu_nodes, grid.rod.sections[0].rho, rho_fluid, grid.node_A, r_t, grid.node_r,
                          wp.drag_multiplier, wp.plunger_drag)
    return c, w, vfm


def mu_on_grid(wp: WellParams, grid: RodGrid, z_T: np.ndarray, T: np.ndarray, wc: float) -> np.ndarray:
    return mu_at(wp, np.interp(grid.z, z_T, T), wc)


def rfi_value(kin: Kinematics, v_fall_min: float) -> float:
    return kin.max_down_speed / max(v_fall_min, 1e-9)


@dataclass
class StrokeResult:
    fwd: ForwardResult
    kin: Kinematics
    F_fo: float
    pr_power: float          # W from card area
    stress_ratio: np.ndarray  # per section (filled by caller with Goodman)
    rfi: float
    v_fall_min: float


def simulate_stroke(wp: WellParams, grid: RodGrid, spm: float, profile: SpeedProfile, mu_nodes: np.ndarray,
                    rho_fluid: float, p_intake: float, cond: PumpCondition, n_strokes: int = 3,
                    allow_float: bool = True) -> StrokeResult:
    c, w, vfm = rod_coefficients(wp, grid, mu_nodes, rho_fluid)
    kin = kinematics(wp.unit, spm, profile, dt=grid.dt_cfl * FORWARD_CFL)
    F_fo = max(0.0, (wp.p_discharge(rho_fluid) - p_intake) * wp.A_p)
    pump = DownholePump(A_p=wp.A_p, F_fo=F_fo, cond=cond, S_p=kin.stroke, p_intake=p_intake)
    fwd = predictive(grid, kin.x, kin.t[1] - kin.t[0], c, w, pump, n_strokes=n_strokes, allow_float=allow_float)
    p_pr = energy.polished_rod_power(fwd.surface_pos, fwd.surface_load, kin.spm_effective)
    return StrokeResult(fwd=fwd, kin=kin, F_fo=F_fo, pr_power=p_pr, stress_ratio=np.zeros(len(wp.rods.sections)),
                        rfi=rfi_value(kin, vfm), v_fall_min=vfm)


def pr_power_quick(wp: WellParams, grid: RodGrid, kin: Kinematics, mu_nodes: np.ndarray, q_liquid: float,
                   dp_lift: float) -> float:
    """Polished-rod power between cards (hydraulic + viscous drag), W."""
    coef = drag.drag_per_length(mu_nodes[:-1], 1.0, wp.geom.r_ti, grid.node_r[:-1], wp.drag_multiplier)
    v_rms = float(np.sqrt(np.mean(kin.v ** 2)))
    return energy.pr_power_estimate(q_liquid, dp_lift, coef, grid.dz, v_rms)


def motor_power(wp: WellParams, p_pr: float) -> float:
    return energy.motor_input_power(p_pr, *wp.etas)


def vfd_hz(wp: WellParams, spm: float, speed_factor: float = 1.0) -> float:
    """Motor frequency for a crank speed (SPM x relative profile speed)."""
    return 60.0 * spm * speed_factor * wp.reduction / wp.motor_rpm_60hz

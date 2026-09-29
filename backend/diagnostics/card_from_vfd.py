"""STRETCH (M12, Tier C): estimate the surface card from VFD torque/speed (BUILD_SPEC §6.5).

Net gearbox torque = TF(theta) * (F_PR - B) - M_cb * sin(theta + tau)
where TF = dx/dtheta is the torque factor from pumping-unit kinematics, B the structural
unbalance and M_cb the counterbalance moment. Motor torque (VFD) x reduction x efficiency
gives the crank torque, so F_PR is recovered wherever TF is not near zero; points near the
dead centres are interpolated. Every output is Tier C.
"""

from __future__ import annotations

import numpy as np

from physics.pumping_unit import Kinematics


def torque_factor(kin: Kinematics) -> np.ndarray:
    """dx/dtheta (m/rad) on the kinematics grid."""
    dth = np.gradient(np.unwrap(kin.theta))
    dth[dth == 0] = 1e-9
    return np.gradient(kin.x) / dth


def crank_torque_from_load(kin: Kinematics, load: np.ndarray, M_cb: float, tau: float = 0.0, B: float = 0.0
                           ) -> np.ndarray:
    """Forward model (used to synthesise VFD torque in tests/simulation)."""
    return torque_factor(kin) * (np.asarray(load) - B) - M_cb * np.sin(kin.theta + tau)


def card_from_vfd(kin: Kinematics, crank_torque: np.ndarray, M_cb: float, tau: float = 0.0, B: float = 0.0,
                  tf_min_frac: float = 0.15) -> dict:
    tf = torque_factor(kin)
    good = np.abs(tf) > tf_min_frac * np.abs(tf).max()
    F = np.full(len(tf), np.nan)
    F[good] = (np.asarray(crank_torque)[good] + M_cb * np.sin(kin.theta[good] + tau)) / tf[good] + B
    idx = np.arange(len(F))
    F = np.interp(idx, idx[good], F[good], period=len(F))
    return {"position": kin.x.tolist(), "load": F.tolist(), "tier": "C", "interpolated_fraction": float(1 - good.mean()),
            "source": "diagnostics.card_from_vfd (torque-factor inversion)"}


def counterbalance_estimate(kin: Kinematics, crank_torque: np.ndarray, load_guess: float) -> float:
    """Least-squares counterbalance moment given a mean-load guess (for commissioning)."""
    tf = torque_factor(kin)
    s = np.sin(kin.theta)
    return float(np.dot(tf * load_guess - crank_torque, s) / max(np.dot(s, s), 1e-9))

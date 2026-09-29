"""M12 (stretch) — card from VFD torque; Tier C."""

import numpy as np

from core.config import get_config
from diagnostics.card_from_vfd import card_from_vfd, crank_torque_from_load
from physics.params import build_well_params
from physics.pumping_unit import kinematics


def test_card_from_vfd_recovers_load_away_from_dead_centres():
    wp = build_well_params(get_config(), "V01")
    kin = kinematics(wp.unit, 4.0)
    load = 30e3 + 12e3 * (kin.v > 0) + 2e3 * np.sin(4 * np.pi * kin.t / kin.period)
    M_cb = 35e3
    T = crank_torque_from_load(kin, load, M_cb)
    T_noisy = T + np.random.default_rng(0).normal(0, 0.01 * np.ptp(T), len(T))
    est = card_from_vfd(kin, T_noisy, M_cb)
    F = np.array(est["load"])
    mid = np.abs(kin.v) > 0.5 * np.abs(kin.v).max()
    assert np.median(np.abs(F[mid] - load[mid]) / load[mid]) < 0.05
    assert est["tier"] == "C" and 0 < est["interpolated_fraction"] < 0.5

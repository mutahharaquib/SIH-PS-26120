"""Labeled synthetic card dataset for the classifier (Tier C training data).

Surface cards come from the TRUE well physics; downhole cards are reconstructed the way
the twin would (diagnostic wave equation with a perturbed viscosity/damping estimate),
so the classifier sees realistic model mismatch. Labels are ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import Config, get_config
from core.state import SpeedProfile
from core.units import BAR, DAY
from diagnostics.card_features import features, vectorize
from diagnostics.downhole import downhole_card
from physics import srp_model as sm
from physics.params import build_well_params
from physics.pump import PumpCondition
from physics.rod_string import downsample_card
from simulation.field_sim import sample_choice
from simulation.labels import FAULT_CLASSES, fault_label
from simulation.noise import NoiseModel

POUND_LABEL_FILLAGE = 0.7


@dataclass
class CardSample:
    x: np.ndarray
    label: str
    well: int
    surf: tuple[list[float], list[float]]
    dh: tuple[list[float], list[float]]


def _conditions(target: str, rng: np.random.Generator) -> dict:
    c = dict(fill=rng.uniform(0.72, 1.0), gas=False, unseated=False, T_pump=rng.uniform(360, 540),
             q=rng.uniform(4, 30), wc=rng.uniform(0.3, 0.9), spm=rng.uniform(1.5, 7.0),
             ds=float(rng.choice([1.0, 1.0, rng.uniform(0.45, 1.0)])), p_int=float(np.exp(rng.uniform(np.log(3), np.log(55)))))
    if target == "fluid_pound":
        c["fill"] = rng.uniform(0.3, 0.66)
    elif target == "gas_interference":
        c.update(gas=True, fill=rng.uniform(0.35, 0.85))
    elif target == "pump_unseated":
        c["unseated"] = True
    elif target == "rod_float":
        c.update(T_pump=rng.uniform(318, 345), q=rng.uniform(1.5, 5), wc=rng.uniform(0.25, 0.5),
                 spm=rng.uniform(5.0, 7.5), fill=rng.uniform(0.9, 1.0))
    return c


def generate(n_wells: int = 24, per_class: int = 8, seed: int = 0, noise_scale: float = 1.0,
             cfg: Config | None = None) -> list[CardSample]:
    cfg = cfg or get_config()
    rng = np.random.default_rng(seed)
    noise = NoiseModel(cfg, rng, noise_scale)
    out: list[CardSample] = []
    for wi in range(n_wells):
        wp = build_well_params(cfg, f"D{wi:03d}", sample_choice(cfg, rng))
        grid = wp.rods.discretize(40)
        for target in FAULT_CLASSES:
            made, tries = 0, 0
            while made < per_class and tries < per_class * 6:
                tries += 1
                c = _conditions(target, rng)
                z, T = sm.tubing_temperature(wp, c["q"] / DAY, c["wc"], c["T_pump"], 30 * DAY, n_cells=40)
                mu = sm.mu_on_grid(wp, grid, z, T, c["wc"])
                rho = wp.rho_liquid(c["T_pump"], c["wc"])
                cond = PumpCondition(c["fill"], gas=c["gas"], unseated=c["unseated"])
                prof = SpeedProfile(downstroke=[1.0, c["ds"], c["ds"], 1.0])
                r = sm.simulate_stroke(wp, grid, c["spm"], prof, mu, rho, c["p_int"] * BAR, cond)
                label = fault_label(c["unseated"], r.fwd.float_fraction, c["gas"], c["fill"], POUND_LABEL_FILLAGE)
                if label != target:
                    continue
                sp, sl = downsample_card(r.fwd.surface_pos, r.fwd.surface_load)
                npos, nload = noise.card(np.array(sp), np.array(sl))
                mu_est = mu * np.exp(rng.normal(0, 0.3))           # twin's imperfect viscosity estimate
                c_est, w_est, _ = sm.rod_coefficients(wp, grid, mu_est, rho)
                dp, dl = downhole_card(grid, npos, nload, r.kin.period, c_est, w_est)
                x = vectorize(features(dp, dl, npos, nload, F_ref=max(r.F_fo, 1.0)))
                out.append(CardSample(x, label, wi, (npos.tolist(), nload.tolist()), (dp, dl)))
                made += 1
    return out

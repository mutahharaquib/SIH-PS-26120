"""Measurement noise: Gaussian noise, sensor dropout, spikes, clock jitter."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from core.config import Config


class NoiseModel:
    def __init__(self, cfg: Config, rng: np.random.Generator, scale: float = 1.0):
        n = "field.noise."
        self.rng = rng
        self.scale = scale
        self.sig = {
            "rate": cfg.v(n + "rate_sigma"),        # relative
            "pressure": cfg.v(n + "pressure_sigma"),  # bar
            "temperature": cfg.v(n + "temperature_sigma"),  # K
            "power": cfg.v(n + "power_sigma"),      # relative
            "wc": 0.02,                              # absolute
            "lab": cfg.v(n + "visc_lab_sigma"),     # relative
        }
        self.load_sigma = cfg.v(n + "load_sigma")
        self.pos_sigma = cfg.v(n + "position_sigma")
        self.dropout = cfg.v(n + "dropout_prob")
        self.spike = cfg.v(n + "spike_prob")
        self.jitter = cfg.v(n + "clock_jitter")

    def measure(self, value: float | None, kind: str) -> float | None:
        if value is None:
            return None
        if self.rng.random() < self.dropout * self.scale:
            return None
        s = self.sig[kind] * self.scale
        if kind in ("rate", "power", "lab"):
            v = value * (1.0 + self.rng.normal(0, s))
        else:
            v = value + self.rng.normal(0, s)
        if self.rng.random() < self.spike * self.scale:
            v *= 3.0 if self.rng.random() < 0.5 else 0.3
        if kind == "wc":
            v = float(np.clip(v, 0.0, 1.0))
        return float(v)

    def card(self, pos: np.ndarray, load: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng_l = float(np.ptp(load)) or 1.0
        p = pos + self.rng.normal(0, self.pos_sigma * self.scale, len(pos))
        f = load + self.rng.normal(0, self.load_sigma * self.scale * rng_l, len(load))
        return p, f

    def timestamp(self, t: datetime) -> datetime:
        return t + timedelta(seconds=float(self.rng.normal(0, self.jitter)))

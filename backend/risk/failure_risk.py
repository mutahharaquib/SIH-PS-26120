"""Failure-risk score (BUILD_SPEC §10). A heuristic RANKING, not a failure prediction.

N(stress_ratio) = N_ref * stress_ratio^(-m)          (uncalibrated S-N proxy)
D += strokes / N(stress_ratio)                       (Miner's rule, per taper section)
risk = w1 D + w2 float_rate + w3 impact_rate + w4 pound_rate   (rolling-window event rates)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from core.config import Config, get_config

LABEL = "Heuristic ranking only (uncalibrated S-N proxy) - not a failure prediction"


@dataclass
class RiskSettings:
    N_ref: float
    m: float
    weights: tuple[float, float, float, float] = (1.0, 2.0, 2.0, 0.5)
    window_h: float = 168.0

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "RiskSettings":
        c = cfg or get_config()
        return cls(c.v("rods.sn_N_ref"), c.v("rods.sn_m"))


@dataclass
class FailureRisk:
    n_sections: int
    s: RiskSettings = field(default_factory=RiskSettings.from_config)
    damage: np.ndarray = field(default_factory=lambda: np.zeros(0))
    events: deque = field(default_factory=lambda: deque())   # (t_h, float, impact, pound, dt_h)

    def __post_init__(self) -> None:
        if self.damage.size != self.n_sections:
            self.damage = np.zeros(self.n_sections)

    def cycles_to_failure(self, stress_ratio) -> np.ndarray:
        return self.s.N_ref * np.maximum(np.asarray(stress_ratio, dtype=float), 1e-3) ** (-self.s.m)

    def update(self, t_h: float, strokes: float, stress_ratio: np.ndarray, floating: bool, impact: bool,
               pound: bool, dt_h: float) -> None:
        self.damage += strokes / self.cycles_to_failure(stress_ratio)
        self.events.append((t_h, floating, impact, pound, dt_h))
        while self.events and t_h - self.events[0][0] > self.s.window_h:
            self.events.popleft()

    def rates(self) -> tuple[float, float, float]:
        tot = sum(e[4] for e in self.events) or 1.0
        f = sum(e[4] for e in self.events if e[1]) / tot
        i = sum(e[4] for e in self.events if e[2]) / tot
        p = sum(e[4] for e in self.events if e[3]) / tot
        return f, i, p

    def score(self) -> dict:
        f, i, p = self.rates()
        w = self.s.weights
        D = float(self.damage.max()) if self.damage.size else 0.0
        return {"risk": w[0] * D + w[1] * f + w[2] * i + w[3] * p, "damage_max": D, "float_rate": f,
                "impact_rate": i, "pound_rate": p, "damage_by_section": self.damage.tolist(), "label": LABEL}

    def reset_section(self, k: int) -> None:
        self.damage[k] = 0.0

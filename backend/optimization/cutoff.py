"""Production cut-off (BUILD_SPEC §8.2).

Simulation mode: T* = argmax_T g(T) on a time grid (NOT the first crossing of q_net = g,
because early-cycle q_net is low/negative, then rises, then declines).
Live mode: stop when q_net is falling (after its peak) and q_net <= g*, on an
EWMA-smoothed q_net, sustained for `hold_hours`.
(Marginal-value theorem / renewal-reward: at the optimum q_net(T*) = g*.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optimization.objective import g_curve


@dataclass
class CutoffResult:
    T_star_d: float
    g_star: float
    V_star: float
    T: np.ndarray
    g: np.ndarray


def sim_cutoff(t_d, qn, t_inject_d: float, t_soak_d: float, c_steam: float, c_fixed: float) -> CutoffResult:
    T, V, g = g_curve(np.asarray(t_d), np.asarray(qn), t_inject_d, t_soak_d, c_steam, c_fixed)
    k = int(np.argmax(g))
    return CutoffResult(float(T[k]), float(g[k]), float(V[k]), T, g)


@dataclass
class LiveCutoff:
    g_star: float
    alpha: float = 0.08
    hold_hours: float = 12.0
    ewma: float | None = None
    peak: float = -np.inf
    below_since_h: float | None = None
    history: list[tuple[float, float, float]] = field(default_factory=list)   # (t_h, raw, ewma)

    def update(self, t_h: float, q_net_value: float | None) -> bool:
        """Feed one observation (hours since production start). Returns True = stop now."""
        if q_net_value is None or not np.isfinite(q_net_value):
            return False
        self.ewma = q_net_value if self.ewma is None else self.alpha * q_net_value + (1 - self.alpha) * self.ewma
        self.history.append((t_h, q_net_value, self.ewma))
        self.peak = max(self.peak, self.ewma)
        falling = self._falling()
        if falling and self.ewma <= self.g_star:
            if self.below_since_h is None:
                self.below_since_h = t_h
            return (t_h - self.below_since_h) >= self.hold_hours
        self.below_since_h = None
        return False

    def _falling(self, lookback: int = 24) -> bool:
        if len(self.history) < 3 or self.ewma is None:
            return False
        past = self.history[max(0, len(self.history) - lookback)][2]
        return self.ewma < self.peak and self.ewma < past

    def state(self) -> dict:
        return {"g_star": self.g_star, "q_net_ewma": self.ewma, "peak": self.peak,
                "falling": self._falling(), "below_since_h": self.below_since_h}

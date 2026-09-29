"""Downhole pump boundary condition with standing/traveling valve logic (BUILD_SPEC §4.8).

Sign convention: rod displacement u is positive DOWN; plunger velocity v > 0 means
the plunger moves down. The returned force is the downward load on the plunger.

- Upstroke: TV closed, plunger carries F_fo = (p_discharge - p_intake) * A_p
  (gas interference: gradual pick-up while the barrel gas expands).
- Downstroke: TV opens once barrel pressure exceeds discharge. With fillage < 1 the
  plunger falls through the void first and hits liquid partway down -> sudden load
  drop (fluid pound). With gas the transfer is gradual (compression).
- Unseated: fluid load never transferred -> only plunger friction.
- Rod float is not a pump condition: it emerges at the carrier bar in rod_string.
Displacement per stroke = A_p * S_p * fillage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):  # type: ignore[no-redef]
        return a[0] if a and callable(a[0]) else (lambda f: f)

# pump state vector layout
DIR, U_TOP, U_BOT, U_TURN, S_P, HAS_TOP, HAS_BOT, U_EXT, HYST = range(9)


def new_pump_state(S_p: float, u0: float = 0.0) -> np.ndarray:
    st = np.zeros(9)
    st[DIR] = -1.0
    st[S_P] = S_p
    st[U_TURN] = u0
    st[U_EXT] = u0
    st[HYST] = 0.015 * S_p
    return st


@njit(cache=True)
def pump_force(st, F_fo, friction, fillage, gas, unseated, u, v):  # pragma: no cover - jit
    """Valve logic. Stroke reversals are detected with position hysteresis (the plunger
    must travel HYST back from its running extreme), mimicking pressure-driven valves
    and preventing chatter from high-frequency rod vibration."""
    d = st[0]
    if d < 0:  # upstroke: plunger moving up (u decreasing); track min u
        if u < st[7]:
            st[7] = u
        if u > st[7] + st[8]:
            st[1] = st[7]      # top of stroke reached at the extreme
            st[5] = 1.0
            st[3] = st[7]
            st[0] = 1.0
            st[7] = u
    else:      # downstroke: track max u
        if u > st[7]:
            st[7] = u
        if u < st[7] - st[8]:
            st[2] = st[7]
            st[6] = 1.0
            st[3] = st[7]
            st[0] = -1.0
            st[7] = u
    if st[5] > 0 and st[6] > 0:
        st[4] = max(st[2] - st[1], 1e-3)
    d = st[0]
    fric = friction if d < 0 else -friction
    if unseated:
        return fric
    S_p = max(st[4], 1e-3)
    travel = abs(u - st[3]) / S_p
    if d < 0:
        ramp = 0.35 * (1.0 - fillage) if gas else 0.02
        if ramp < 0.02:
            ramp = 0.02
        k = travel / ramp
        if k > 1.0:
            k = 1.0
        return F_fo * k + fric
    void = 1.0 - fillage
    if void < 0.0:
        void = 0.0
    if travel < void:
        if gas:
            s = travel / void
            return F_fo * (1.0 - s ** 2.5) + fric
        return F_fo + fric
    if gas:  # gas fully compressed at the liquid level: TV already open, load released
        return fric
    rel = (travel - void) / 0.03
    k = 1.0 - rel
    if k < 0.0:
        k = 0.0
    return F_fo * k + fric


@dataclass
class PumpCondition:
    fillage: float = 1.0          # liquid fraction of the barrel at top of upstroke
    gas: bool = False             # compressible gas (gradual transfer) instead of vacuum void
    unseated: bool = False


@dataclass
class DownholePump:
    A_p: float
    F_fo: float                   # N, differential fluid load
    friction: float = 400.0       # N, plunger friction
    cond: PumpCondition | None = None
    S_p: float = 2.0              # m, initial estimate of the gross plunger stroke

    def __post_init__(self) -> None:
        self.cond = self.cond or PumpCondition()
        self.state = new_pump_state(self.S_p)

    def displacement_per_stroke(self) -> float:
        assert self.cond is not None
        return self.A_p * float(self.state[S_P]) * self.cond.fillage

    def force(self, u: float, v: float) -> float:
        c = self.cond
        assert c is not None
        return float(pump_force(self.state, self.F_fo, self.friction, c.fillage, c.gas, c.unseated, u, v))

"""Downhole pump boundary condition with standing/traveling valve logic (BUILD_SPEC §4.8).

Barrel-pressure model. Valves open on pressure, not on a direction relay:
  - barrel pressure p_b follows plunger motion: liquid compresses with bulk modulus K,
    a void (vapour, or gas when gas interference is active) takes up volume first;
  - TV opens when p_b >= p_discharge, SV opens when p_b <= p_intake;
  - plunger load = (p_discharge - p_b) * A_p  (+ smooth plunger friction).
This yields, without special cases:
  - Upstroke: fluid load F_fo = (p_d - p_i) A_p after a short liquid-expansion pick-up.
  - Downstroke with fillage < 1: plunger falls through the vapour void with the load
    still on, then hits liquid -> sudden load drop (fluid pound).
  - Gas interference: isothermal gas compression -> gradual load transfer, and a
    gradual pick-up on the upstroke (gas re-expansion).
  - Unseated pump: fluid load never transferred (friction only).
Rod float is not a pump condition: it emerges at the carrier bar in rod_string.
Displacement per stroke = A_p * S_p * fillage.

Sign convention: u (rod displacement) positive DOWN; v > 0 = plunger moving down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):  # type: ignore[no-redef]
        return a[0] if a and callable(a[0]) else (lambda f: f)

K_LIQUID = 1.2e9        # Pa, effective bulk modulus of the barrel liquid (PLACEHOLDER)

# state layout
P_B, V_V, ADM, V_V0, LAST_X, HAS_LAST, TV, SV, S_P, X_MIN, X_MAX = range(11)


def new_pump_state(S_p: float, p_d: float) -> np.ndarray:
    st = np.zeros(11)
    st[P_B] = p_d        # bottom of stroke: TV open, barrel liquid-full
    st[TV] = 1.0
    st[S_P] = S_p
    return st


@njit(cache=True)
def pump_force(st, F_fo, friction, fillage, gas, unseated, u, v, A_p, p_i):  # pragma: no cover - jit
    x = -u
    fric = friction * np.tanh(-v / 0.02)
    if unseated:
        st[4] = x
        st[5] = 1.0
        return fric
    p_d = p_i + F_fo / A_p
    if st[5] == 0.0:
        st[4] = x
        st[5] = 1.0
        st[9] = x
        st[10] = x
    dV = A_p * (x - st[4])
    st[4] = x
    S_p = st[8] if st[8] > 1e-3 else 1e-3
    V_b = A_p * S_p * 1.08
    # --- open valves: pressure pinned, flow through the valve
    if st[6] > 0.0:                       # TV open
        if dV <= 0.0:
            st[0] = p_d
        else:
            st[6] = 0.0
    if st[7] > 0.0:                       # SV open (intake)
        if dV >= 0.0:
            room = fillage * A_p * S_p - st[2]
            take = dV if dV < room else (room if room > 0.0 else 0.0)
            st[2] += take
            st[1] += dV - take            # remainder becomes void (vapour/gas)
            st[0] = p_i
            if x > st[10]:
                st[10] = x
            return (p_d - st[0]) * A_p + fric
        else:
            st[7] = 0.0                   # top of stroke: SV closes
            st[3] = st[1]                 # void volume at closure (gas reference)
            if st[10] - st[9] > 0.05 * S_p:
                st[8] = st[10] - st[9]
    if st[6] > 0.0:
        return (p_d - st[0]) * A_p + fric
    # --- both valves closed: compress / expand barrel contents
    if st[1] > 0.0:
        Vn = st[1] + dV
        if Vn > 1e-9:
            st[1] = Vn
            if gas and st[3] > 0.0:
                st[0] = p_i * st[3] / Vn
            else:
                st[0] = p_i
        else:
            st[1] = 0.0
            st[0] = p_i + K_LIQUID * (-Vn) / V_b
    else:
        st[0] = st[0] - K_LIQUID * dV / V_b
    if st[0] >= p_d:
        st[0] = p_d
        st[6] = 1.0
        if not gas:
            st[1] = 0.0
    elif st[0] <= p_i and dV > 0.0:
        st[0] = p_i
        st[7] = 1.0                       # bottom of stroke: SV opens, new intake
        st[2] = 0.0
        st[9] = x
        st[10] = x
    return (p_d - st[0]) * A_p + fric


@dataclass
class PumpCondition:
    fillage: float = 1.0          # liquid fraction of the barrel at top of upstroke
    gas: bool = False             # compressible gas (gradual transfer) instead of vapour void
    unseated: bool = False


@dataclass
class DownholePump:
    A_p: float
    F_fo: float                   # N, differential fluid load (p_d - p_i) * A_p
    friction: float = 400.0       # N, plunger friction
    cond: PumpCondition | None = None
    S_p: float = 2.0              # m, initial estimate of the gross plunger stroke
    p_intake: float = 6.0e5       # Pa (absolute level matters only for gas compression)

    def __post_init__(self) -> None:
        self.cond = self.cond or PumpCondition()
        self.state = new_pump_state(self.S_p, self.p_intake + self.F_fo / self.A_p)

    def displacement_per_stroke(self) -> float:
        assert self.cond is not None
        return self.A_p * float(self.state[S_P]) * self.cond.fillage

    def force(self, u: float, v: float) -> float:
        c = self.cond
        assert c is not None
        return float(pump_force(self.state, self.F_fo, self.friction, c.fillage, c.gas, c.unseated, u, v,
                                self.A_p, self.p_intake))

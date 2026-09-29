"""Damped 1-D wave equation for the rod string (BUILD_SPEC §4.7).

    d2u/dt2 = a^2 d2u/dz2 - c(z) du/dt,   a = sqrt(E / rho_steel)

[VERIFY resolved] Discretisation is a lumped-mass form of Gibbs (1963): node masses
m_i, element stiffnesses k = E A / dz, central differences in time for both inertia
and damping. Tapers are handled exactly by element-wise E, A, rho (displacement and
force continuity at joints hold by construction). The diagnostic mode is the
Everitt & Jennings (1992) style downward march of the same discrete equations with
Cauchy data (position AND load) at z = 0; with dz = a dt it reduces to the exact
d'Alembert shift for an undamped uniform rod, which is what keeps the march stable.
The round-trip test (predictive -> diagnostic) recovers the pump load to machine
precision on the same grid.

Sign convention: u is displacement, positive DOWN. Tension positive.
Rod float: if the polished-rod load would go negative on the downstroke, the rod
clamp separates from the carrier bar (load = 0, top node free). On re-contact the
carrier bar re-clamps the rod, producing the upstroke impact load.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.units import G
from physics.pump import DownholePump, PumpCondition, njit, pump_force


@dataclass(frozen=True)
class RodSection:
    length: float
    diameter: float
    E: float = 2.05e11
    rho: float = 7850.0
    tensile: float = 793e6

    @property
    def area(self) -> float:
        return float(np.pi * self.diameter ** 2 / 4.0)


@dataclass(frozen=True)
class RodString:
    sections: tuple[RodSection, ...]

    @property
    def length(self) -> float:
        return float(sum(s.length for s in self.sections))

    def weight_in_air(self) -> float:
        return float(sum(s.rho * s.area * s.length * G for s in self.sections))

    def discretize(self, n_elem: int) -> "RodGrid":
        L = self.length
        dz = L / n_elem
        mid = (np.arange(n_elem) + 0.5) * dz
        bounds = np.cumsum([s.length for s in self.sections])
        sec = np.minimum(np.searchsorted(bounds, mid), len(self.sections) - 1)
        A = np.array([self.sections[j].area for j in sec])
        E = np.array([self.sections[j].E for j in sec])
        rho = np.array([self.sections[j].rho for j in sec])
        m = np.zeros(n_elem + 1)
        m[:-1] += 0.5 * rho * A * dz
        m[1:] += 0.5 * rho * A * dz
        node_A = np.zeros(n_elem + 1)
        node_A[:-1] += 0.5 * A
        node_A[1:] += 0.5 * A
        node_A[0] *= 2
        node_A[-1] *= 2
        r = np.array([self.sections[j].diameter / 2 for j in sec])
        node_r = np.concatenate([[r[0]], 0.5 * (r[:-1] + r[1:]), [r[-1]]])
        return RodGrid(z=np.linspace(0, L, n_elem + 1), dz=dz, A=A, E=E, rho=rho, k=E * A / dz, m=m,
                       node_A=node_A, node_r=node_r, sec=sec, rod=self)


@dataclass
class RodGrid:
    z: np.ndarray
    dz: float
    A: np.ndarray        # per element
    E: np.ndarray
    rho: np.ndarray
    k: np.ndarray
    m: np.ndarray        # per node
    node_A: np.ndarray
    node_r: np.ndarray
    sec: np.ndarray      # element -> section index
    rod: RodString

    @property
    def wave_speed(self) -> float:
        return float(np.max(np.sqrt(self.E / self.rho)))

    @property
    def dt_cfl(self) -> float:
        return self.dz / self.wave_speed

    def node_weights(self, rho_fluid: float) -> np.ndarray:
        """Buoyant weight lumped at nodes (N)."""
        w = (self.rho - rho_fluid) * G * self.A * self.dz
        out = np.zeros(len(self.z))
        out[:-1] += 0.5 * w
        out[1:] += 0.5 * w
        return out

    def section_top_elements(self) -> list[int]:
        return [int(np.argmax(self.sec == j)) for j in range(len(self.rod.sections))]


def check_cfl(grid: RodGrid, dt: float) -> float:
    ratio = grid.wave_speed * dt / grid.dz
    if ratio > 1.0 + 1e-9:
        raise ValueError(f"CFL violated: a*dt/dz = {ratio:.3f} > 1")
    return ratio


@dataclass
class ForwardResult:
    t: np.ndarray
    surface_pos: np.ndarray      # m, up positive, 0 = bottom
    surface_load: np.ndarray     # N (polished-rod load)
    pump_pos: np.ndarray         # m, plunger position, up positive, 0 = bottom
    pump_load: np.ndarray        # N
    u_top: np.ndarray            # raw displacement of node 0 (down +)
    float_fraction: float        # fraction of the last stroke with carrier-bar separation
    impacts: int                 # re-contacts in the last stroke
    impact_load: float           # N, peak surface load within the re-contact window (0 if none)
    sec_max_stress: np.ndarray   # Pa, per section (top element)
    sec_min_stress: np.ndarray
    min_tension: float           # N, most compressive element force (negative -> compression)
    energy_hist: np.ndarray = field(default_factory=lambda: np.zeros(0))


@njit(cache=True)
def _forward_kernel(uc, dt, m, k, c, w, tops, tops_A, F_fo, friction, fillage, gas, unseated, S_p0,
                    n_strokes, allow_float, u_init, record_energy, A_p, p_i):  # pragma: no cover - jit
    n_t = uc.shape[0]
    nn = m.shape[0]
    cdt = 0.5 * c * dt
    u = u_init.copy()
    u_prev = u_init.copy()
    st = np.zeros(11)
    st[0] = p_i + F_fo / A_p     # bottom of stroke: TV open
    st[6] = 1.0
    st[8] = S_p0
    clamped = True
    total = n_strokes * n_t
    last0 = (n_strokes - 1) * n_t
    rec_load = np.zeros(n_t)
    rec_pump = np.zeros(n_t)
    rec_uN = np.zeros(n_t)
    rec_u0 = np.zeros(n_t)
    sep = np.zeros(n_t)
    ns = tops.shape[0]
    smax = np.full(ns, -1e30)
    smin = np.full(ns, 1e30)
    min_T = 1e30
    impacts = 0
    impact_js = np.full(64, -1)
    energy = np.zeros(total if record_energy else 1)
    T = np.zeros(nn - 1)
    F = np.zeros(nn)
    u_new = np.zeros(nn)
    for step in range(total):
        j = step % n_t
        uc_np1 = uc[(j + 1) % n_t]
        for e in range(nn - 1):
            T[e] = k[e] * (u[e + 1] - u[e])
        for i in range(nn):
            F[i] = w[i]
        for e in range(nn - 1):
            F[e] += T[e]
            F[e + 1] -= T[e]
        vN = (u[nn - 1] - u_prev[nn - 1]) / dt
        Fp = pump_force(st, F_fo, friction, fillage, gas, unseated, u[nn - 1], vN, A_p, p_i)
        F[nn - 1] += Fp
        for i in range(nn):
            u_new[i] = (2.0 * u[i] - (1.0 - cdt[i]) * u_prev[i] + dt * dt * F[i] / m[i]) / (1.0 + cdt[i])
        if clamped:
            acc0 = (uc_np1 - 2.0 * u[0] + u_prev[0]) / (dt * dt)
            vel0 = (uc_np1 - u_prev[0]) / (2.0 * dt)
            F_pr = F[0] - m[0] * acc0 - c[0] * m[0] * vel0
            if allow_float and F_pr < 0.0:
                clamped = False
                F_pr = 0.0
            else:
                u_new[0] = uc_np1
        else:
            F_pr = 0.0
            if u_new[0] >= uc_np1:
                u_new[0] = uc_np1
                clamped = True
                if step >= last0:
                    if impacts < 64:
                        impact_js[impacts] = j
                    impacts += 1
        if record_energy:
            ke = 0.0
            for i in range(nn):
                v = (u_new[i] - u_prev[i]) / (2.0 * dt)
                ke += 0.5 * m[i] * v * v
            energy[step] = ke
        if step >= last0:
            rec_load[j] = F_pr
            rec_pump[j] = Fp
            rec_uN[j] = u[nn - 1]
            rec_u0[j] = u[0]
            sep[j] = 0.0 if clamped else 1.0
            for s_ in range(ns):
                sv = T[tops[s_]] / tops_A[s_]
                if sv > smax[s_]:
                    smax[s_] = sv
                if sv < smin[s_]:
                    smin[s_] = sv
            for e in range(nn - 1):
                if T[e] < min_T:
                    min_T = T[e]
        for i in range(nn):
            u_prev[i] = u[i]
            u[i] = u_new[i]
    return rec_load, rec_pump, rec_uN, rec_u0, sep, smax, smin, min_T, impacts, impact_js, energy


def predictive(grid: RodGrid, x_surface: np.ndarray, dt: float, c_nodes: np.ndarray, w_nodes: np.ndarray,
               pump: DownholePump, n_strokes: int = 3, allow_float: bool = True,
               record_energy: bool = False) -> ForwardResult:
    """Forward simulation: prescribed polished-rod position (one periodic stroke sampled at dt,
    up positive, starting at the bottom) and the pump boundary condition -> surface card."""
    check_cfl(grid, dt)
    x = np.asarray(x_surface, dtype=float)
    uc = -x                                         # carrier position, down positive
    n_t = len(uc)
    # static initial state at the bottom of the stroke (TV open, no fluid load)
    w_nodes = np.asarray(w_nodes, dtype=float)
    below = np.cumsum(w_nodes[::-1])[::-1]
    u0 = np.empty(len(grid.z))
    u0[0] = uc[0]
    u0[1:] = uc[0] + np.cumsum(below[1:] / grid.k)
    tops = np.array(grid.section_top_elements(), dtype=np.int64)
    cond = pump.cond or PumpCondition()
    out = _forward_kernel(uc, float(dt), grid.m, grid.k, np.asarray(c_nodes, dtype=float), w_nodes, tops,
                          grid.A[tops].astype(float), float(pump.F_fo), float(pump.friction), float(cond.fillage),
                          bool(cond.gas), bool(cond.unseated), float(pump.S_p), int(n_strokes), bool(allow_float),
                          u0, bool(record_energy), float(pump.A_p), float(pump.p_intake))
    rec_load, rec_pump, rec_uN, rec_u0, sep, smax, smin, min_T, impacts, impact_js, energy = out
    impact_load = 0.0
    for j in impact_js[: min(int(impacts), 64)]:
        win = rec_load[j: min(n_t, j + max(3, n_t // 25))]
        if win.size:
            impact_load = max(impact_load, float(win.max()))
    return ForwardResult(t=np.arange(n_t) * dt, surface_pos=x - x.min(), surface_load=rec_load,
                         pump_pos=rec_uN.max() - rec_uN, pump_load=rec_pump, u_top=rec_u0,
                         float_fraction=float(sep.mean()), impacts=int(impacts), impact_load=impact_load,
                         sec_max_stress=smax, sec_min_stress=smin, min_tension=float(min_T),
                         energy_hist=energy if record_energy else np.zeros(0))


def fourier_resample(y: np.ndarray, n_out: int, n_harmonics: int | None = None) -> np.ndarray:
    """Periodic resampling with optional low-pass (truncated Fourier series, as in Gibbs)."""
    Y = np.fft.rfft(np.asarray(y, dtype=float))
    if n_harmonics is not None:
        Y[n_harmonics + 1:] = 0.0
    n_in = len(y)
    Yo = np.zeros(n_out // 2 + 1, dtype=complex)
    m = min(len(Y), len(Yo))
    Yo[:m] = Y[:m]
    return np.fft.irfft(Yo, n_out) * (n_out / n_in)


@dataclass
class DiagnosticResult:
    t: np.ndarray
    pump_pos: np.ndarray
    pump_load: np.ndarray
    dt: float


def diagnostic(grid: RodGrid, x_surface: np.ndarray, load_surface: np.ndarray, period: float, c_nodes: np.ndarray,
               w_nodes: np.ndarray, n_harmonics: int | None = None, resample: bool = True,
               periodic: bool = True) -> DiagnosticResult:
    """Surface card (one periodic stroke) -> downhole card, by marching the discrete wave
    equation down the string with Cauchy data at the surface."""
    x = np.asarray(x_surface, dtype=float)
    F = np.asarray(load_surface, dtype=float)
    if resample:
        n_t = int(np.ceil(period * grid.wave_speed / grid.dz))
        x = fourier_resample(x, n_t, n_harmonics)
        F = fourier_resample(F, n_t, n_harmonics)
    n_t = len(x)
    dt = period / n_t
    check_cfl(grid, dt)
    c = np.asarray(c_nodes, dtype=float)

    def ddt(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        yp, ym = np.roll(y, -1), np.roll(y, 1)
        if not periodic:  # invalid end samples propagate inward one sample per depth level
            yp[-1] = np.nan
            ym[0] = np.nan
        return (yp - 2 * y + ym) / (dt * dt), (yp - ym) / (2 * dt)

    u_i = -x
    acc, vel = ddt(u_i)
    T = F - w_nodes[0] + grid.m[0] * acc + c[0] * grid.m[0] * vel
    n = len(grid.z) - 1
    for i in range(1, n + 1):
        u_i = u_i + T / grid.k[i - 1]
        acc, vel = ddt(u_i)
        if i < n:
            T = grid.m[i] * acc + c[i] * grid.m[i] * vel - w_nodes[i] + T
    Fp = grid.m[n] * acc + c[n] * grid.m[n] * vel - w_nodes[n] + T
    pos = np.nanmax(u_i) - u_i
    return DiagnosticResult(t=np.arange(n_t) * dt, pump_pos=pos, pump_load=Fp, dt=dt)


def downsample_card(pos: np.ndarray, load: np.ndarray, n: int = 200) -> tuple[list[float], list[float]]:
    idx = np.linspace(0, len(pos) - 1, n).astype(int)
    return [float(v) for v in pos[idx]], [float(v) for v in load[idx]]

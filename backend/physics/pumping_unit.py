"""Conventional crank-balanced beam pumping unit kinematics (BUILD_SPEC §4.5).

Four-bar linkage: crank centre at the origin, saddle bearing at (I, H), back arm C,
pitman P, crank radius R, front arm A. The horsehead is a circular arc of radius A,
so polished-rod displacement = A * (beam rotation) exactly.

The VFD speed profile is a piecewise-linear crank angular speed omega(theta)
split into upstroke and downstroke segments (core.state.SpeedProfile).
A simple-harmonic fallback exists and is labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.state import SpeedProfile


@dataclass(frozen=True)
class UnitGeometry:
    A: float
    C: float
    I: float
    H: float
    P: float
    R: float


@dataclass
class Kinematics:
    t: np.ndarray            # s, uniform grid over one stroke
    theta: np.ndarray        # rad
    x: np.ndarray            # m, polished-rod position (0 = bottom of stroke, up positive)
    v: np.ndarray            # m/s (up positive)
    a: np.ndarray            # m/s2
    period: float            # s
    stroke: float            # m
    upstroke: np.ndarray     # bool mask
    mode: str = "four-bar"

    @property
    def spm_effective(self) -> float:
        return 60.0 / self.period

    @property
    def max_down_speed(self) -> float:
        """Max polished-rod downward speed (m/s, positive)."""
        return float(max(0.0, -self.v.min()))


def _beam_psi(theta: np.ndarray, g: UnitGeometry) -> np.ndarray:
    """Back-arm elevation angle (rad) vs crank angle via circle-circle intersection."""
    pc = np.stack([g.R * np.cos(theta), g.R * np.sin(theta)], axis=-1)
    S = np.array([g.I, g.H])
    d_vec = pc - S
    d = np.linalg.norm(d_vec, axis=-1)
    if np.any(d > g.C + g.P) or np.any(d < abs(g.C - g.P)):
        raise ValueError("pumping-unit geometry is not assemblable")
    a = (g.C ** 2 - g.P ** 2 + d ** 2) / (2 * d)
    hh = np.sqrt(np.maximum(g.C ** 2 - a ** 2, 0.0))
    u = d_vec / d[:, None]
    base = S + a[:, None] * u
    perp = np.stack([-u[:, 1], u[:, 0]], axis=-1)
    b1 = base + hh[:, None] * perp
    b2 = base - hh[:, None] * perp
    B = np.where((b1[:, 1] >= b2[:, 1])[:, None], b1, b2)
    return np.arcsin(np.clip((B[:, 1] - g.H) / g.C, -1.0, 1.0))


def position_vs_crank(g: UnitGeometry, n: int = 720) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    psi = _beam_psi(theta, g)
    x = g.A * (psi.max() - psi)
    return theta, x


def stroke_length(g: UnitGeometry) -> float:
    _, x = position_vs_crank(g)
    return float(x.max() - x.min())


def _omega_multiplier(theta: np.ndarray, th_bot: float, th_top: float, prof: SpeedProfile) -> np.ndarray:
    """Relative speed at each crank angle; upstroke = bottom->top in crank rotation."""
    two_pi = 2 * np.pi
    up_len = (th_top - th_bot) % two_pi
    rel = (theta - th_bot) % two_pi
    out = np.empty_like(theta)
    up = rel < up_len
    ku = np.asarray(prof.upstroke, dtype=float)
    kd = np.asarray(prof.downstroke, dtype=float)
    out[up] = np.interp(rel[up] / up_len, np.linspace(0, 1, len(ku)), ku)
    out[~up] = np.interp((rel[~up] - up_len) / (two_pi - up_len), np.linspace(0, 1, len(kd)), kd)
    return out


def kinematics(g: UnitGeometry, spm: float, profile: SpeedProfile | None = None, dt: float | None = None,
               n_theta: int = 1440, harmonic: bool = False) -> Kinematics:
    """Polished-rod position/velocity/acceleration over one stroke on a uniform time grid.

    Starts at the bottom of the stroke. `dt` sets the output time step (default: 400 samples).
    """
    profile = profile or SpeedProfile()
    if harmonic:  # labelled fallback
        S = stroke_length(g)
        period = 60.0 / spm
        n = int(round(period / dt)) if dt else 400
        t = np.arange(n) * (period / n)
        w = 2 * np.pi / period
        x = 0.5 * S * (1 - np.cos(w * t))
        v = 0.5 * S * w * np.sin(w * t)
        a = 0.5 * S * w * w * np.cos(w * t)
        return Kinematics(t, w * t, x, v, a, period, S, v >= 0, mode="harmonic-fallback")

    theta, x_th = position_vs_crank(g, n_theta)
    i_bot, i_top = int(np.argmin(x_th)), int(np.argmax(x_th))
    th_bot, th_top = theta[i_bot], theta[i_top]
    omega0 = 2 * np.pi * spm / 60.0
    mult = np.maximum(_omega_multiplier(theta, th_bot, th_top, profile), 0.05)
    # time from bottom of stroke to each crank angle (crank turns in +theta)
    order = (np.arange(n_theta) + i_bot) % n_theta
    th_seq = np.unwrap(theta[order])
    th_seq = np.append(th_seq, th_seq[0] + 2 * np.pi)
    m_seq = np.append(mult[order], mult[order][0])
    dtheta = np.diff(th_seq)
    dt_seg = dtheta / (omega0 * 0.5 * (m_seq[:-1] + m_seq[1:]))
    t_seq = np.concatenate([[0.0], np.cumsum(dt_seg)])
    period = float(t_seq[-1])
    x_seq = np.append(x_th[order], x_th[order][0])
    n = int(round(period / dt)) if dt else 400
    t = np.arange(n) * (period / n)
    x = np.interp(t, t_seq, x_seq)
    th = np.interp(t, t_seq, th_seq) % (2 * np.pi)
    h = period / n
    v = (np.roll(x, -1) - np.roll(x, 1)) / (2 * h)
    a = (np.roll(x, -1) - 2 * x + np.roll(x, 1)) / (h * h)
    stroke = float(x_th.max() - x_th.min())
    up_time = float(np.interp(th_seq[0] + ((th_top - th_bot) % (2 * np.pi)), th_seq, t_seq))
    return Kinematics(t, th, x, v, a, period, stroke, t < up_time)

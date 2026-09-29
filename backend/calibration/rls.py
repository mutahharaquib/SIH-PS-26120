"""Recursive least squares with forgetting factor on the Walther linear form (§7):

    y = A - B x,   y = log10(log10(nu + 0.7)),  x = log10(T)

Each (T, nu) pair updates (A, B) and their covariance. Parameter uncertainty is
propagated to viscosity through WaltherParams.cov. Guardrails reject updates that are
outliers (innovation > k sigma) or that would move a parameter by more than a
configured amount in one step; rejected updates are returned so the caller can
raise alarms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from physics.viscosity import WaltherParams, walther_x, walther_y


@dataclass
class RLSUpdate:
    accepted: bool
    reason: str
    innovation: float
    dA: float
    dB: float


@dataclass
class WaltherRLS:
    A: float
    B: float
    P: np.ndarray                       # 2x2, parameter covariance (in y-units^2)
    lam: float = 0.995
    max_dA: float = 0.08
    max_dB: float = 0.03
    outlier_sigma: float = 6.0
    sigma_y: float = 3e-3               # measurement noise in y (running estimate)
    T_min: float = 1e9
    T_max: float = 0.0
    n_updates: int = 0
    change_after: int = 5               # consecutive same-sign outliers -> regime change
    P0: np.ndarray | None = None
    log: list[RLSUpdate] = field(default_factory=list)
    _consec: int = 0
    _sign: float = 0.0

    @classmethod
    def from_params(cls, p: WaltherParams, P0_A: float, P0_B: float, **kw) -> "WaltherRLS":
        r = cls(A=p.A, B=p.B, P=np.diag([P0_A, P0_B]).astype(float), P0=np.diag([P0_A, P0_B]).astype(float), **kw)
        r.T_min, r.T_max = p.T_min_fit, p.T_max_fit
        return r

    def update(self, T_K: float, nu_cSt: float) -> RLSUpdate:
        x = np.array([1.0, -float(walther_x(T_K))])
        y = float(walther_y(nu_cSt))
        theta = np.array([self.A, self.B])
        innov = y - float(x @ theta)
        S = float(x @ self.P @ x) + self.sigma_y ** 2
        if abs(innov) > self.outlier_sigma * np.sqrt(S):
            sgn = float(np.sign(innov))
            self._consec = self._consec + 1 if sgn == self._sign else 1
            self._sign = sgn
            self.P = self.P / self.lam          # time still passes: forgetting inflates P
            if self._consec < self.change_after or self.P0 is None:
                u = RLSUpdate(False, "outlier", innov, 0.0, 0.0)
                self.log.append(u)
                return u
            # persistent same-sign deviation: a real change (e.g. emulsion/asphaltene shift)
            self.P = self.P + self.P0
            self._consec = 0
        else:
            self._consec = 0
        Px = self.P @ x
        K = Px / (self.sigma_y ** 2 + float(x @ Px))
        d = K * innov
        if abs(d[0]) > self.max_dA or abs(d[1]) > self.max_dB:
            self.P = self.P / self.lam
            u = RLSUpdate(False, "step_limit", innov, float(d[0]), float(d[1]))
            self.log.append(u)
            return u
        self.A, self.B = float(theta[0] + d[0]), float(theta[1] + d[1])
        self.P = (self.P - np.outer(K, Px)) / self.lam
        self.P = 0.5 * (self.P + self.P.T)
        self.sigma_y = float(np.sqrt(0.98 * self.sigma_y ** 2 + 0.02 * innov ** 2))
        self.T_min, self.T_max = min(self.T_min, T_K), max(self.T_max, T_K)
        self.n_updates += 1
        u = RLSUpdate(True, "ok", innov, float(d[0]), float(d[1]))
        self.log.append(u)
        return u

    def params(self) -> WaltherParams:
        c = self.P
        return WaltherParams(A=self.A, B=self.B, T_min_fit=self.T_min, T_max_fit=self.T_max,
                             cov=((float(c[0, 0]), float(c[0, 1])), (float(c[1, 0]), float(c[1, 1]))))

"""SRP controller (BUILD_SPEC §9). Runs every control interval; layers in order:

1. Feedforward   SPM_ff = q_in_pred / (A_p * S_p_eff * eta_v_est * 1440)
2. Fillage PI    SPM = clamp(SPM_ff + PI, SPM_min, SPM_max), anti-windup
                 (error = fillage_measured - fillage_target: under-filling lowers SPM)
3. Float guard   if RFI >= rfi_warn: reshape the downstroke VFD profile first
                 (slow the high-velocity downstroke segment, keep the upstroke),
                 check the per-stroke time budget; reduce SPM only if RFI is still
                 >= rfi_warn once shaping is at its limit.
4. Stress limit  predicted peak stress (predictive wave equation) vs Goodman allowable;
                 back SPM off until stress_ratio <= limit.
Special actions: pump_unseated -> SPM_min + alarm; sustained fluid pound -> SPM step down.
Reliability-first: in steady state the setpoint never targets displacement above the
predicted inflow, except that a full pump (fillage >= 0.97, level building) is taken
as evidence that inflow is under-predicted. `transient_weight` > 0 allows a short,
capped over-displacement.
Every decision is logged with inputs, active constraints and reason codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np

from core.config import Config, get_config
from core.state import SpeedProfile


@dataclass
class ControllerSettings:
    fillage_target: float
    Kp: float
    Ki: float
    i_limit: float
    spm_min: float
    spm_max: float
    max_change: float
    rfi_warn: float
    rfi_alarm: float
    ds_min: float
    shaping_step: float
    stress_max: float
    pound_fillage: float
    pound_sustain: int
    spm_step_pound: float
    transient_weight: float
    interval_h: float

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "ControllerSettings":
        c = cfg or get_config()
        k = "controller."
        return cls(
            fillage_target=c.v(k + "fillage_target"), Kp=c.v(k + "Kp"), Ki=c.v(k + "Ki"),
            i_limit=c.v(k + "integral_limit"), spm_min=c.v(k + "spm_min"), spm_max=c.v(k + "spm_max"),
            max_change=c.v(k + "max_spm_change"), rfi_warn=c.v(k + "rfi_warn"), rfi_alarm=c.v(k + "rfi_alarm"),
            ds_min=c.v(k + "downstroke_min_factor"), shaping_step=c.v(k + "shaping_step"),
            stress_max=c.v(k + "stress_ratio_max"), pound_fillage=c.v(k + "pound_fillage"),
            pound_sustain=int(c.v(k + "pound_sustain")), spm_step_pound=c.v(k + "spm_step_on_pound"),
            transient_weight=c.v(k + "transient_weight"), interval_h=c.v(k + "interval"),
        )


@dataclass
class ControlInput:
    t: datetime
    spm_current: float
    profile_current: SpeedProfile
    q_in_pred_m3d: float
    fillage_meas: float | None
    eta_v_est: float
    S_p_eff: float
    A_p: float
    rfi_fn: Callable[[float, SpeedProfile], float]
    stress_fn: Callable[[float, SpeedProfile], float]
    spm_eff_fn: Callable[[float, SpeedProfile], float]
    fault_class: str | None = None
    pound_streak: int = 0
    ds_floor_fn: Callable[[float], float] | None = None   # min downstroke factor allowed by VFD Hz limit


@dataclass
class ControlDecision:
    t: datetime
    spm: float
    profile: SpeedProfile
    spm_ff: float
    pi_term: float
    rfi_pred: float
    stress_ratio_pred: float
    reason_codes: list[str]
    active_constraints: list[str]
    alarms: list[str]
    inputs: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"t": self.t.isoformat(), "spm": self.spm, "profile": self.profile.model_dump(), "spm_ff": self.spm_ff,
                "pi_term": self.pi_term, "rfi_pred": self.rfi_pred, "stress_ratio_pred": self.stress_ratio_pred,
                "reason_codes": self.reason_codes, "active_constraints": self.active_constraints,
                "alarms": self.alarms, "inputs": self.inputs}


def shaped(ds: float) -> SpeedProfile:
    """Downstroke profile slowing the mid (high-velocity) segment to factor ds."""
    return SpeedProfile(upstroke=[1.0, 1.0, 1.0, 1.0], downstroke=[1.0, ds, ds, 1.0])


class SRPController:
    def __init__(self, settings: ControllerSettings | None = None):
        self.s = settings or ControllerSettings.from_config()
        self.integral = 0.0
        self.ds = 1.0
        self.log: list[ControlDecision] = []

    def reset(self) -> None:
        self.integral = 0.0
        self.ds = 1.0

    def step(self, x: ControlInput) -> ControlDecision:
        s = self.s
        reasons: list[str] = []
        active: list[str] = []
        alarms: list[str] = []
        # 1. feedforward
        denom = max(x.A_p * x.S_p_eff * max(x.eta_v_est, 0.2) * 1440.0, 1e-9)
        spm_ff = x.q_in_pred_m3d / denom
        reasons.append("FF_INFLOW_MATCH")
        # 2. fillage PI with anti-windup
        pi = 0.0
        if x.fillage_meas is not None:
            e = x.fillage_meas - s.fillage_target
            new_int = float(np.clip(self.integral + s.Ki * e * s.interval_h, -s.i_limit, s.i_limit))
            pi_try = s.Kp * e + new_int
            unsat = s.spm_min < spm_ff + pi_try < s.spm_max
            if unsat or np.sign(e) != np.sign(self.integral):
                self.integral = new_int                     # conditional integration (anti-windup)
            pi = s.Kp * e + self.integral
            reasons.append("PI_FILLAGE")
        spm = spm_ff + pi
        # reliability-first cap
        cap = spm_ff * (1.0 + s.transient_weight)
        if x.fillage_meas is not None and x.fillage_meas >= 0.97:
            cap = max(cap, x.spm_current + s.max_change)     # full pump: inflow under-predicted
            reasons.append("FULL_PUMP_RAISE_ALLOWED")
        if spm > cap:
            spm = cap
            active.append("RELIABILITY_CAP")
        # special actions
        if x.fault_class == "pump_unseated":
            spm = s.spm_min
            reasons.append("PUMP_UNSEATED_MIN_SPM")
            alarms.append("pump_unseated")
        elif x.pound_streak >= s.pound_sustain:
            spm = min(spm, x.spm_current - s.spm_step_pound)
            reasons.append("SUSTAINED_FLUID_POUND_STEP_DOWN")
        spm = float(np.clip(spm, s.spm_min, s.spm_max))
        # rate limit
        if abs(spm - x.spm_current) > s.max_change:
            spm = x.spm_current + np.sign(spm - x.spm_current) * s.max_change
            active.append("RATE_LIMIT")
        # 3. float protection: shape first, SPM last
        ds_now = x.profile_current.downstroke_factor()        # shape from what is actually applied
        ds = ds_now
        # shaping ramps at most 2 steps per interval (keeps actions inside autonomous bands)
        ds_min = max(s.ds_min, x.ds_floor_fn(spm) if x.ds_floor_fn else 0.0, ds_now - 2 * s.shaping_step)
        if ds < ds_min:
            ds = round(min(1.0, ds_min), 3)
            active.append("VFD_MIN_FREQUENCY")
        rfi = x.rfi_fn(spm, shaped(ds))
        if rfi >= s.rfi_warn:
            reasons.append("RFI_WARN")
            while rfi >= s.rfi_warn and ds - s.shaping_step >= ds_min - 1e-9:
                ds = round(ds - s.shaping_step, 3)
                rfi = x.rfi_fn(spm, shaped(ds))
            if ds < ds_now - 1e-9:
                reasons.append("DOWNSTROKE_SHAPED")
            q_req = x.q_in_pred_m3d
            disp = x.A_p * x.S_p_eff * max(x.eta_v_est, 0.2) * 1440.0 * x.spm_eff_fn(spm, shaped(ds))
            if disp < 0.95 * q_req:
                active.append("TIME_BUDGET_SHORT")
            spm_floor = max(s.spm_min, x.spm_current - s.max_change)
            while rfi >= s.rfi_warn and spm - 0.1 >= spm_floor - 1e-9:
                spm = round(spm - 0.1, 3)
                if x.ds_floor_fn and ds < x.ds_floor_fn(spm):
                    ds = round(min(1.0, x.ds_floor_fn(spm)), 3)
                rfi = x.rfi_fn(spm, shaped(ds))
                if "SPM_REDUCED_FOR_FLOAT" not in reasons:
                    reasons.append("SPM_REDUCED_FOR_FLOAT")
            if rfi >= s.rfi_alarm:
                alarms.append("rfi_alarm")
        elif ds < 1.0 and x.rfi_fn(spm, shaped(min(1.0, ds + s.shaping_step))) < s.rfi_warn - 0.15:
            ds = round(min(1.0, ds + s.shaping_step), 3)                  # relax shaping with hysteresis
            rfi = x.rfi_fn(spm, shaped(ds))
            reasons.append("SHAPING_RELAXED")
        if ds < 1.0:
            active.append("FLOAT_SHAPING")
        # 4. stress limit
        sr = x.stress_fn(spm, shaped(ds))
        n = 0
        while sr > s.stress_max and spm - 0.2 >= max(s.spm_min, x.spm_current - s.max_change) and n < 20:
            spm = round(spm - 0.2, 3)
            sr = x.stress_fn(spm, shaped(ds))
            n += 1
            if "STRESS_BACKOFF" not in reasons:
                reasons.append("STRESS_BACKOFF")
                active.append("GOODMAN_LIMIT")
        # final rate limit (every layer must respect the envelope; the rest happens next interval)
        lo, hi = x.spm_current - s.max_change, x.spm_current + s.max_change
        if not lo - 1e-9 <= spm <= hi + 1e-9:
            spm = float(np.clip(spm, lo, hi))
            if "RATE_LIMIT" not in active:
                active.append("RATE_LIMIT")
        spm = float(np.clip(spm, s.spm_min, s.spm_max))
        if x.ds_floor_fn and ds < x.ds_floor_fn(spm):          # VFD minimum frequency after final clamp
            ds = round(min(1.0, x.ds_floor_fn(spm) + 1e-3), 3)
            rfi = x.rfi_fn(spm, shaped(ds))
        self.ds = ds
        d = ControlDecision(
            t=x.t, spm=float(spm), profile=shaped(ds), spm_ff=float(spm_ff), pi_term=float(pi), rfi_pred=float(rfi),
            stress_ratio_pred=float(sr), reason_codes=reasons, active_constraints=active, alarms=alarms,
            inputs={"q_in_pred_m3d": x.q_in_pred_m3d, "fillage_meas": x.fillage_meas, "eta_v_est": x.eta_v_est,
                    "S_p_eff": x.S_p_eff, "spm_current": x.spm_current, "fault_class": x.fault_class,
                    "pound_streak": x.pound_streak},
        )
        self.log.append(d)
        if len(self.log) > 2000:
            self.log = self.log[-1000:]
        return d


def recommend_stroke(R_options: tuple[float, ...], peak_q_m3d: float, disp_per_spm_fn: Callable[[float], float],
                     stress_fn_R: Callable[[float, float], float], spm_max: float, stress_max: float) -> dict:
    """Per-cycle stroke-length (crank hole) recommendation — advisory only, no live actuation.
    Prefer the longest stroke (slowest SPM for the same displacement) that keeps the
    predicted peak stress ratio within limits at the SPM needed for the peak inflow."""
    best = None
    rows = []
    for R in sorted(R_options):
        spm_needed = peak_q_m3d / max(disp_per_spm_fn(R), 1e-9)
        sr = stress_fn_R(R, min(max(spm_needed, 1.0), spm_max))
        ok = spm_needed <= spm_max and sr <= 0.9 * stress_max
        rows.append({"R": R, "spm_needed": spm_needed, "stress_ratio": sr, "ok": ok})
        if ok:
            best = R
    return {"R": best if best is not None else min(R_options), "options": rows, "advisory_only": True}

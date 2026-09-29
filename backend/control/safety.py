"""Safety envelope and fallback (BUILD_SPEC §9.2). Runs after every controller output,
in every mode. On any violation, stale data, classifier `unknown` on N consecutive
cards, or an internal exception -> fallback to the last known-safe setting (or the
configured safe default) and raise an alarm."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from core.config import Config, get_config
from core.state import SpeedProfile


@dataclass
class SafetySettings:
    spm_min: float
    spm_max: float
    max_change: float
    hz_min: float
    hz_max: float
    stress_max: float
    max_data_age_h: float
    unknown_consecutive: int
    safe_spm: float

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "SafetySettings":
        c = cfg or get_config()
        k = "controller."
        return cls(c.v(k + "spm_min"), c.v(k + "spm_max"), c.v(k + "max_spm_change"), c.v(k + "vfd_hz_min"),
                   c.v(k + "vfd_hz_max"), c.v(k + "stress_ratio_max"), c.v(k + "max_data_age"),
                   int(c.v(k + "unknown_consecutive")), c.v(k + "safe_default_spm"))


@dataclass
class SafetyResult:
    spm: float
    profile: SpeedProfile
    ok: bool
    fallback: bool
    violations: list[str]


@dataclass
class SafetyGuard:
    s: SafetySettings = field(default_factory=SafetySettings.from_config)
    last_safe_spm: float | None = None
    last_safe_profile: SpeedProfile | None = None
    log: list[dict] = field(default_factory=list)

    def check(self, t: datetime, spm: float, profile: SpeedProfile, spm_current: float, hz_fn,
              stress_ratio: float, data_age_h: float, unknown_streak: int, error: str | None = None) -> SafetyResult:
        s = self.s
        v: list[str] = []
        if error:
            v.append(f"exception:{error}")
        if not (s.spm_min - 1e-9 <= spm <= s.spm_max + 1e-9):
            v.append("spm_out_of_envelope")
        if abs(spm - spm_current) > s.max_change + 1e-9:
            v.append("spm_rate_of_change")
        mult = [*profile.upstroke, *profile.downstroke]
        hz_hi, hz_lo = hz_fn(spm * max(mult)), hz_fn(spm * min(mult))
        if hz_hi > s.hz_max or hz_lo < s.hz_min:
            v.append("vfd_frequency_limit")
        if stress_ratio > s.stress_max:
            v.append("stress_ratio")
        if data_age_h > s.max_data_age_h:
            v.append("stale_data")
        if unknown_streak >= s.unknown_consecutive:
            v.append("classifier_unknown_streak")
        if v:
            if self.last_safe_spm is not None:
                fs, fp = self.last_safe_spm, self.last_safe_profile or SpeedProfile()
            else:
                fs, fp = s.safe_spm, SpeedProfile()
            # the fallback itself must respect the rate limit from the current setting
            fs = min(max(fs, spm_current - s.max_change), spm_current + s.max_change)
            fs = min(max(fs, s.spm_min), s.spm_max)
            res = SafetyResult(fs, fp, False, True, v)
        else:
            self.last_safe_spm, self.last_safe_profile = spm, profile
            res = SafetyResult(spm, profile, True, False, [])
        self.log.append({"t": t.isoformat(), "ok": res.ok, "violations": v, "spm": res.spm})
        if len(self.log) > 2000:
            self.log = self.log[-1000:]
        return res

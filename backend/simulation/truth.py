"""Hidden ground truth. ONLY simulation/* and evaluation/* may import this module;
the twin sees measured Telemetry only (import-lint test in test_m4_simulator.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from physics.params import WellParams


@dataclass
class HiddenTruth:
    t: datetime
    well_id: str
    phase: str
    cycle: int
    dt_h: float
    T_avg: float
    r_h: float
    p_res: float
    sr: float
    mu_top: float
    mu_pump: float
    q_in: float              # m3/d liquid inflow
    q_oil: float             # m3/d produced
    q_water: float
    fillage: float
    float_fraction: float
    impact_load: float
    rfi_true: float
    fault: str
    damage_max: float
    stress_ratio: float
    motor_kw: float
    energy_kwh: float        # this tick
    failed: bool
    downtime_h: float
    level_m: float
    steam_t: float           # steam injected this tick (t)
    spm_eff: float
    strokes: float


@dataclass
class CycleTruth:
    well_id: str
    cycle: int
    start: datetime
    end: datetime | None = None
    steam_t: float = 0.0
    inj_pressure_bar: float = 0.0
    soak_d: float = 0.0
    inject_d: float = 0.0
    produce_d: float = 0.0
    oil_m3: float = 0.0
    water_m3: float = 0.0
    energy_kwh: float = 0.0
    float_h: float = 0.0
    float_events: int = 0
    impact_events: int = 0
    pound_h: float = 0.0
    downtime_h: float = 0.0
    failures: int = 0
    peak_stress_ratio: float = 0.0
    x_sf: float = 0.0
    r_h: float = 0.0
    daily: list[dict] = field(default_factory=list)


@dataclass
class TrueWell:
    params: WellParams
    hidden: dict = field(default_factory=dict)   # e.g. visc_scale, true drag multiplier

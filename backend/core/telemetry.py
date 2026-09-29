"""Measured telemetry and commands: the ONLY data contract between the field
(simulated or real) and the twin. The twin never sees simulator hidden truth
(enforced by tests/test_m4_simulator.py::test_twin_never_imports_hidden_truth).

The same schema is what a real SCADA/historian replay (CSV/Parquet) must provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from core.state import CyclePhase, SpeedProfile


class MeasuredCard(BaseModel):
    position: list[float]        # m (polished rod, up positive)
    load: list[float]            # N
    period: float                # s
    spm: float                   # effective strokes/min


class LabSample(BaseModel):
    T_K: float
    nu_cSt: float


class Telemetry(BaseModel):
    well_id: str
    t: datetime
    phase: CyclePhase
    cycle_index: int
    phase_elapsed_h: float
    # actuator read-backs
    spm_set: float
    spm_eff: float | None = None
    profile: SpeedProfile = Field(default_factory=SpeedProfile)
    crank_R: float
    running: bool = True
    # injection
    steam_rate_tph: float | None = None
    steam_cum_t: float | None = None
    inj_pressure_bar: float | None = None
    # production
    liquid_rate_m3d: float | None = None
    water_cut: float | None = None
    p_intake_bar: float | None = None
    T_wellhead_K: float | None = None
    motor_power_kw: float | None = None
    vfd_hz: float | None = None
    card: MeasuredCard | None = None
    lab: list[LabSample] = Field(default_factory=list)
    recipe_id: str | None = None


class WellMetadata(BaseModel):
    """Static well-record data the twin is allowed to know."""

    well_id: str
    depth_m: float
    pump_depth_m: float
    api: float
    T_R_K: float
    net_thickness_m: float
    p_initial_bar: float
    plunger_d_m: float
    crank_R_m: float
    rod_taper: list[list[float]]     # [[diameter_m, length_m], ...]


@dataclass
class Recipe:
    steam_mass_t: float
    inj_pressure_bar: float          # wellhead injection pressure
    soak_d: float
    recipe_id: str = ""


@dataclass
class Command:
    kind: Literal["set_spm", "set_profile", "start_cycle", "stop_production", "set_crank", "run", "stop"]
    value: object = None
    source: str = "twin"
    reason: str = ""
    meta: dict = field(default_factory=dict)

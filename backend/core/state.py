"""Shared well state (BUILD_SPEC §3). Every module reads the latest WellState and
publishes updates through core.bus. Every numeric output is a Quantity carrying
a confidence tier and a provenance record."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

MODEL_VERSION = "0.1.0"


class Tier(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class Provenance(BaseModel):
    tier: Tier
    source: str
    model_version: str = MODEL_VERSION
    inputs_hash: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extrapolated: bool = False
    placeholder: bool = False


class Quantity(BaseModel):
    value: float
    unit: str
    uncertainty: float | None = None
    prov: Provenance


def inputs_hash(*items: Any) -> str:
    blob = json.dumps(items, default=str, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def q(
    value: float,
    unit: str,
    tier: Tier | str,
    source: str,
    *,
    uncertainty: float | None = None,
    extrapolated: bool = False,
    placeholder: bool = False,
    t: datetime | None = None,
    inputs: Any = None,
) -> Quantity:
    """Convenience constructor for a Quantity with provenance."""
    return Quantity(
        value=float(value),
        unit=unit,
        uncertainty=None if uncertainty is None else float(uncertainty),
        prov=Provenance(
            tier=Tier(tier),
            source=source,
            inputs_hash=inputs_hash(inputs) if inputs is not None else "",
            timestamp=t or datetime.now(timezone.utc),
            extrapolated=extrapolated,
            placeholder=placeholder,
        ),
    )


class CyclePhase(str, Enum):
    INJECT = "inject"
    SOAK = "soak"
    PRODUCE = "produce"
    IDLE = "idle"


class SpeedProfile(BaseModel):
    """Piecewise-linear relative crank speed over one revolution.

    `upstroke` / `downstroke` are relative speed multipliers at breakpoints spread
    evenly over each half-stroke (crank-angle domain). The absolute speed is scaled
    so that, with all multipliers = 1, one revolution takes 60/SPM seconds.
    """

    upstroke: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    downstroke: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    def downstroke_factor(self) -> float:
        return min(self.downstroke)


class Card(BaseModel):
    position: list[float]   # m
    load: list[float]       # N
    kind: Literal["surface", "downhole"]
    t: datetime
    spm: float
    tier: Tier = Tier.B
    source: str = ""


class Alarm(BaseModel):
    t: datetime
    well_id: str
    code: str
    severity: Literal["info", "warning", "alarm", "critical"]
    message: str


class CycleState(BaseModel):
    cycle_index: int
    phase: CyclePhase
    phase_start: datetime
    steam_injected_kg: Quantity
    injection_pressure: Quantity
    soak_duration: Quantity
    cumulative_oil_m3: Quantity
    cumulative_water_m3: Quantity
    recipe_id: str | None = None


class ReservoirState(BaseModel):
    T_avg_heated: Quantity
    r_heated: Quantity
    p_reservoir: Quantity
    stimulation_ratio: Quantity


class WellboreState(BaseModel):
    sandface_steam_quality: Quantity
    heat_loss_rate: Quantity
    T_tubing_profile: list[tuple[float, float]]
    mu_at_pump: Quantity
    mu_profile: list[tuple[float, float]]


class PumpState(BaseModel):
    spm: Quantity
    stroke_length: Quantity
    vfd_profile: SpeedProfile
    intake_pressure: Quantity
    fillage: Quantity
    volumetric_efficiency: Quantity
    surface_card: Card | None = None
    downhole_card: Card | None = None
    rfi: Quantity
    fault_class: str | None = None
    fault_probs: dict[str, float] = Field(default_factory=dict)
    peak_rod_stress_ratio: Quantity


class EconomicState(BaseModel):
    q_oil: Quantity
    q_net_value_rate: Quantity
    kwh_per_bbl: Quantity
    sor_cycle: Quantity
    g_star: Quantity | None = None


class WellState(BaseModel):
    well_id: str
    t: datetime
    cycle: CycleState
    reservoir: ReservoirState
    wellbore: WellboreState
    pump: PumpState
    econ: EconomicState
    risk_score: Quantity
    control_mode: Literal["advisory", "supervised", "autonomous"] = "advisory"
    alarms: list[Alarm] = Field(default_factory=list)


def iter_quantities(obj: Any, prefix: str = ""):
    """Yield (path, Quantity) for every Quantity nested in a model (used by tests/UI)."""
    if isinstance(obj, Quantity):
        yield prefix, obj
    elif isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            yield from iter_quantities(getattr(obj, name), f"{prefix}.{name}" if prefix else name)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_quantities(v, f"{prefix}.{k}")

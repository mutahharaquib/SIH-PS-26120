"""Labeled event injection (ground truth for diagnostics evaluation).

Scheduled (random) events: gas interference windows, pump unseating, cooling
(ambient cold snap / tubing cooling -> viscosity rise). Derived events (from the
simulated physics, not scheduled): fluid pound (over-pumping), rod float (cooling
driven viscosity rise + fast downstroke), rod failure (Miner damage threshold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from core.config import Config

FAULT_CLASSES = ["normal", "fluid_pound", "gas_interference", "rod_float", "pump_unseated"]


@dataclass
class ScheduledEvent:
    kind: str                 # gas | unseat | cooling
    start: datetime
    end: datetime
    magnitude: float = 0.0    # gas void fraction | cooling delta-T (K)


@dataclass
class Label:
    well_id: str
    kind: str
    onset: datetime
    end: datetime | None = None
    meta: dict = field(default_factory=dict)


class EventSchedule:
    def __init__(self, events: list[ScheduledEvent] | None = None):
        self.events = sorted(events or [], key=lambda e: e.start)

    @classmethod
    def random(cls, cfg: Config, rng: np.random.Generator, t0: datetime, horizon_days: float,
               scale: float = 1.0) -> "EventSchedule":
        e = "field.events."
        out: list[ScheduledEvent] = []

        def poisson(rate: float, kind: str, dur_rng, mag_rng=None) -> None:
            t = 0.0
            while rate > 0 and scale > 0:
                t += rng.exponential(1.0 / (rate * scale))
                if t > horizon_days:
                    break
                dur = rng.uniform(*dur_rng)
                mag = float(rng.uniform(*mag_rng)) if mag_rng else 0.0
                out.append(ScheduledEvent(kind, t0 + timedelta(days=t), t0 + timedelta(days=t + dur), mag))

        poisson(cfg.v(e + "gas_interference_rate"), "gas", cfg.v(e + "gas_duration"), cfg.v(e + "gas_void_fraction"))
        poisson(cfg.v(e + "unseat_rate"), "unseat", cfg.v(e + "unseat_duration"))
        return cls(out)

    def active(self, t: datetime, kind: str) -> ScheduledEvent | None:
        for ev in self.events:
            if ev.kind == kind and ev.start <= t < ev.end:
                return ev
        return None

    def add(self, ev: ScheduledEvent) -> None:
        self.events.append(ev)
        self.events.sort(key=lambda e: e.start)


def fault_label(unseated: bool, float_fraction: float, gas: bool, fillage: float, pound_fillage: float) -> str:
    """Ground-truth card class for a stroke (priority order)."""
    if unseated:
        return "pump_unseated"
    if float_fraction > 0.02:
        return "rod_float"
    if gas and fillage < 0.95:
        return "gas_interference"
    if fillage < pound_fillage:
        return "fluid_pound"
    return "normal"


def intervals(times: list[datetime], flags: list[bool], well_id: str, kind: str) -> list[Label]:
    """Contiguous True runs -> labels with onset/end."""
    out: list[Label] = []
    start = None
    for t, f in zip(times, flags):
        if f and start is None:
            start = t
        elif not f and start is not None:
            out.append(Label(well_id, kind, start, t))
            start = None
    if start is not None:
        out.append(Label(well_id, kind, start, times[-1]))
    return out

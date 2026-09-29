"""Edge controller simulator (BUILD_SPEC §9.3).

- advisory (default): recommendations only, nothing written;
- supervised: a recommendation creates a pending action that must be approved in the UI
  before it is written to the (simulated) VFD;
- autonomous-within-band: actions inside operator bands are applied automatically;
  anything outside the band becomes a supervised pending action.
Every mode change, approval and write is audit-logged.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from core.state import SpeedProfile
from core.telemetry import Command

Mode = Literal["advisory", "supervised", "autonomous"]
_ids = itertools.count(1)


@dataclass
class Bands:
    spm: float = 0.5                  # max |delta SPM| auto-applied per action
    downstroke: float = 0.2           # max |delta downstroke factor| auto-applied


@dataclass
class PendingAction:
    id: int
    t: datetime
    spm: float
    profile: SpeedProfile
    reason_codes: list[str]
    status: Literal["pending", "approved", "rejected", "superseded"] = "pending"


@dataclass
class EdgeController:
    well_id: str
    mode: Mode = "advisory"
    bands: Bands = field(default_factory=Bands)
    pending: dict[int, PendingAction] = field(default_factory=dict)
    audit: list[dict] = field(default_factory=list)
    writes: int = 0

    def _audit(self, event: str, **kw) -> None:
        self.audit.append({"t": datetime.now(timezone.utc).isoformat(), "well_id": self.well_id, "event": event, **kw})
        if len(self.audit) > 5000:
            self.audit = self.audit[-2500:]

    def set_mode(self, mode: Mode, bands: Bands | None = None, user: str = "operator") -> None:
        old = self.mode
        self.mode = mode
        if bands is not None:
            self.bands = bands
        self._audit("mode_change", old=old, new=mode, bands=vars(self.bands), user=user)

    def _write(self, spm: float, profile: SpeedProfile, why: str) -> list[Command]:
        self.writes += 1
        self._audit("vfd_write", spm=spm, profile=profile.model_dump(), why=why)
        return [Command("set_spm", spm, "twin", why), Command("set_profile", profile, "twin", why)]

    def process(self, t: datetime, spm: float, profile: SpeedProfile, spm_current: float,
                profile_current: SpeedProfile, reason_codes: list[str]) -> list[Command]:
        """Route a (safety-checked) recommendation according to the mode."""
        changed = abs(spm - spm_current) > 1e-3 or profile != profile_current
        if not changed:
            return []
        if self.mode == "advisory":
            return []
        in_band = (abs(spm - spm_current) <= self.bands.spm + 1e-9 and
                   abs(profile.downstroke_factor() - profile_current.downstroke_factor()) <= self.bands.downstroke + 1e-9)
        if self.mode == "autonomous" and in_band:
            return self._write(spm, profile, "autonomous_in_band:" + ",".join(reason_codes))
        for p in self.pending.values():
            if p.status == "pending":
                p.status = "superseded"
        a = PendingAction(next(_ids), t, spm, profile, reason_codes)
        self.pending[a.id] = a
        self._audit("pending_created", id=a.id, spm=spm, profile=profile.model_dump(), reasons=reason_codes,
                    out_of_band=self.mode == "autonomous")
        return []

    def approve(self, action_id: int, user: str = "operator") -> list[Command]:
        a = self.pending.get(action_id)
        if a is None or a.status != "pending":
            self._audit("approve_failed", id=action_id, user=user)
            return []
        a.status = "approved"
        self._audit("approved", id=action_id, user=user)
        return self._write(a.spm, a.profile, f"approved:{action_id}")

    def reject(self, action_id: int, user: str = "operator") -> None:
        a = self.pending.get(action_id)
        if a is not None and a.status == "pending":
            a.status = "rejected"
            self._audit("rejected", id=action_id, user=user)

    def pending_list(self) -> list[PendingAction]:
        return [a for a in self.pending.values() if a.status == "pending"]

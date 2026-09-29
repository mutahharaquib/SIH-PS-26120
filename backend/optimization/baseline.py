"""Current-practice baseline policy (BUILD_SPEC §8.4).

- fixed steam recipe (config);
- cut-off at a fixed day count OR a fixed cumulative oil volume (selectable);
- fixed SPM set at the start of production, reduced only when a volumetric-efficiency
  (fillage proxy) reading stays below a manual threshold for `adjust_delay` days,
  emulating reactive manual adjustment. No downstroke shaping, no float protection.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import Config, get_config
from core.state import CyclePhase, SpeedProfile
from core.telemetry import Command, Recipe, Telemetry, WellMetadata
from diagnostics.efficiency import volumetric_efficiency
from physics.pumping_unit import UnitGeometry, stroke_length


@dataclass
class BaselineSettings:
    recipe: Recipe
    cutoff_mode: str
    cutoff_days: float
    cutoff_oil_m3: float
    spm: float
    fillage_threshold: float
    adjust_delay_h: float
    spm_step: float
    spm_min: float

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "BaselineSettings":
        c = cfg or get_config()
        b = "optimizer.baseline."
        return cls(
            recipe=Recipe(c.v(b + "steam_mass"), c.v(b + "injection_pressure"), c.v(b + "soak_time"), "baseline"),
            cutoff_mode=c.v(b + "cutoff_mode"), cutoff_days=c.v(b + "cutoff_days"),
            cutoff_oil_m3=c.v(b + "cutoff_tonnage"), spm=c.v(b + "spm"),
            fillage_threshold=c.v(b + "fillage_threshold"), adjust_delay_h=c.v(b + "adjust_delay") * 24.0,
            spm_step=c.v(b + "spm_step"), spm_min=c.v("controller.spm_min"),
        )


class BaselinePolicy:
    name = "baseline"

    def __init__(self, meta: WellMetadata, settings: BaselineSettings | None = None, cfg: Config | None = None):
        c = cfg or get_config()
        self.meta = meta
        self.s = settings or BaselineSettings.from_config(c)
        f = "field.pumping_unit."
        self.stroke = stroke_length(UnitGeometry(c.v(f + "A"), c.v(f + "C"), c.v(f + "I"), c.v(f + "H"), c.v(f + "P"),
                                                 meta.crank_R_m))
        self.A_p = 3.141592653589793 * meta.plunger_d_m ** 2 / 4
        self._spm = self.s.spm
        self._low_since: float | None = None
        self._oil = 0.0
        self._last_phase: CyclePhase | None = None
        self.log: list[dict] = []

    def on_telemetry(self, tel: Telemetry) -> list[Command]:
        cmds: list[Command] = []
        if tel.phase == CyclePhase.IDLE:
            cmds.append(Command("start_cycle", self.s.recipe, "baseline", "fixed recipe"))
        elif tel.phase == CyclePhase.PRODUCE:
            if self._last_phase != CyclePhase.PRODUCE:
                self._spm, self._low_since, self._oil = self.s.spm, None, 0.0
                cmds += [Command("set_spm", self._spm, "baseline", "fixed SPM at start of production"),
                         Command("set_profile", SpeedProfile(), "baseline", "constant speed")]
            h = tel.phase_elapsed_h
            if tel.liquid_rate_m3d is not None and tel.water_cut is not None:
                self._oil += tel.liquid_rate_m3d * (1 - tel.water_cut) / 24.0
                eta = volumetric_efficiency(tel.liquid_rate_m3d, self.A_p, self.stroke, tel.spm_eff or self._spm)
                if eta < self.s.fillage_threshold:
                    self._low_since = h if self._low_since is None else self._low_since
                    if h - self._low_since >= self.s.adjust_delay_h and self._spm - self.s.spm_step >= self.s.spm_min:
                        self._spm -= self.s.spm_step
                        self._low_since = None
                        cmds.append(Command("set_spm", self._spm, "baseline", "manual SPM reduction (low fillage)"))
                else:
                    self._low_since = None
            stop = (h >= self.s.cutoff_days * 24.0) if self.s.cutoff_mode == "days" else (self._oil >= self.s.cutoff_oil_m3)
            if stop:
                cmds.append(Command("stop_production", None, "baseline", f"fixed cut-off ({self.s.cutoff_mode})"))
        self._last_phase = tel.phase
        return cmds

"""Per-well physical parameters (SI). The simulator samples a "true" WellParams per
well; the twin holds its own estimated copy (static metadata from well records +
calibrated reservoir/fluid parameters)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.config import Config
from core.units import BAR, DAY, G, c_to_k
from physics.pumping_unit import UnitGeometry
from physics.reservoir import ReservoirParams
from physics.rod_string import RodSection, RodString
from physics.viscosity import ViscosityModel, model_for_api
from physics.wellbore import WellboreGeom


def _mid(v: Any) -> float:
    return float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)


@dataclass
class WellParams:
    well_id: str
    depth: float                 # m, mid-perforation
    pump_depth: float            # m
    api: float
    res: ReservoirParams
    visc: ViscosityModel
    J_c: float                   # m3/s/Pa
    injectivity: float           # kg/s/Pa
    wc_early: float
    wc_late: float
    wc_tau: float                # s
    geom: WellboreGeom
    U_inject: float
    U_produce: float
    rods: RodString
    plunger_d: float
    unit: UnitGeometry
    R_options: tuple[float, ...]
    etas: tuple[float, float, float, float]
    motor_rpm_60hz: float
    reduction: float
    drag_multiplier: float
    damping_floor: float
    plunger_drag: float
    leak_fraction: float
    p_wellhead: float
    p_casing: float
    ann_area: float              # m2 annulus area for fluid level
    placeholder: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def A_p(self) -> float:
        return float(np.pi * self.plunger_d ** 2 / 4)

    @property
    def T_R(self) -> float:
        return self.res.T_R

    def rho_oil(self, T: float) -> float:
        return float(self.visc.rho_fn(T))

    def rho_liquid(self, T: float, wc: float) -> float:
        return float(wc * 990.0 + (1 - wc) * self.rho_oil(T))

    def p_discharge(self, rho_l: float) -> float:
        return self.p_wellhead + rho_l * G * self.pump_depth

    def with_unit_R(self, R: float) -> "WellParams":
        c = copy.copy(self)
        u = self.unit
        c.unit = UnitGeometry(u.A, u.C, u.I, u.H, u.P, R)
        return c

    def copy(self) -> "WellParams":
        return copy.deepcopy(self)


def build_well_params(cfg: Config, well_id: str, choice: dict[str, Any] | None = None) -> WellParams:
    """Build WellParams from config; `choice` pins sampled values (field-unit keys as in config).

    Without `choice`, range-valued parameters take their mid-point (the twin's prior).
    """
    ch = choice or {}
    f = "field."

    def pick(path: str) -> float:
        key = path.split(".")[-1]
        return float(ch[key]) if key in ch else _mid(cfg.v(f + path))

    depth = pick("reservoir.depth")
    pump_depth = depth - float(cfg.v(f + "pump.depth_above_perfs"))
    T_R = c_to_k(pick("reservoir.T_R"))
    api = pick("crude.api_gravity")
    T_surf = c_to_k(float(cfg.v(f + "wellbore.T_surface")))
    geom = WellboreGeom(
        depth=depth, r_ti=cfg.v(f + "wellbore.r_ti"), r_to=cfg.v(f + "wellbore.r_to"),
        r_ci=cfg.v(f + "wellbore.r_ci"), r_cw=cfg.v(f + "wellbore.r_cw"), k_e=cfg.v(f + "wellbore.k_e"),
        alpha_e=cfg.v(f + "wellbore.alpha_e"), T_surface=T_surf, grad=(T_R - T_surf) / depth,
        n_cells=int(cfg.v(f + "wellbore.n_cells")), friction=cfg.v(f + "wellbore.roughness_friction"),
    )
    res = ReservoirParams(
        T_R=T_R, h=pick("reservoir.net_thickness"), M_R=cfg.v(f + "reservoir.M_R"), M_ob=cfg.v(f + "reservoir.M_ob"),
        lambda_ob=cfg.v(f + "reservoir.lambda_ob"), r_w=cfg.v(f + "reservoir.r_w"), r_e=cfg.v(f + "reservoir.r_e"),
        p_init=pick("reservoir.p_initial") * BAR, p_min=cfg.v(f + "reservoir.p_min") * BAR,
        compliance=pick("reservoir.tank_compliance") / BAR,
    )
    visc = model_for_api(cfg.v(f + "crude.viscosity_lab_points"), api, cfg.v(f + "crude.api_log_visc_slope"),
                         cfg.v(f + "crude.thermal_expansion"))
    if "visc_scale" in ch:  # true crude differs from the lab reference (hidden in the sim)
        visc = model_for_api([[t, n * ch["visc_scale"]] for t, n in cfg.v(f + "crude.viscosity_lab_points")], api,
                             cfg.v(f + "crude.api_log_visc_slope"), cfg.v(f + "crude.thermal_expansion"))
    grade = cfg.v("rods.grade")
    tensile = float(cfg.v(f"rods.grades.{grade}"))
    rod_len = pump_depth
    rods = RodString(tuple(
        RodSection(length=rod_len * frac, diameter=d, E=cfg.v("rods.E"), rho=cfg.v("rods.rho_steel"), tensile=tensile)
        for d, frac in cfg.v("rods.taper")
    ))
    plungers = cfg.v(f + "pump.plunger_diameter")
    plunger_d = float(ch.get("plunger_diameter", plungers[len(plungers) // 2]))
    R_opts = tuple(float(r) for r in cfg.v(f + "pumping_unit.R_options"))
    R = float(ch.get("R", R_opts[int(cfg.v(f + "pumping_unit.R_default_index"))]))
    unit = UnitGeometry(A=cfg.v(f + "pumping_unit.A"), C=cfg.v(f + "pumping_unit.C"), I=cfg.v(f + "pumping_unit.I"),
                        H=cfg.v(f + "pumping_unit.H"), P=cfg.v(f + "pumping_unit.P"), R=R)
    dt = f + "drivetrain."
    etas = (cfg.v(dt + "eta_gearbox"), cfg.v(dt + "eta_belt"), cfg.v(dt + "eta_motor"), cfg.v(dt + "eta_vfd"))
    r_ci, r_to = cfg.v(f + "wellbore.r_ci"), cfg.v(f + "wellbore.r_to")
    return WellParams(
        well_id=well_id, depth=depth, pump_depth=pump_depth, api=api, res=res, visc=visc,
        J_c=pick("reservoir.J_cold") / DAY / BAR, injectivity=pick("reservoir.injectivity") * 1000 / DAY / BAR,
        wc_early=pick("water_cut.wc_early"), wc_late=pick("water_cut.wc_late"), wc_tau=pick("water_cut.tau") * DAY,
        geom=geom, U_inject=cfg.v(f + "wellbore.U_inject"), U_produce=cfg.v(f + "wellbore.U_produce"),
        rods=rods, plunger_d=plunger_d, unit=unit, R_options=R_opts, etas=etas,
        motor_rpm_60hz=cfg.v(dt + "motor_rpm_at_60hz"), reduction=cfg.v(dt + "total_reduction"),
        drag_multiplier=float(ch.get("drag_multiplier", cfg.v(f + "drag.drag_multiplier"))),
        damping_floor=cfg.v(f + "drag.damping_floor"), plunger_drag=cfg.v(f + "pump.plunger_drag"), leak_fraction=cfg.v(f + "pump.leak_fraction"),
        p_wellhead=cfg.v(f + "wellbore.p_wellhead") * BAR, p_casing=cfg.v(f + "wellbore.p_casing") * BAR,
        ann_area=float(np.pi * (r_ci ** 2 - r_to ** 2)),
        meta={"choice": dict(ch)},
    )

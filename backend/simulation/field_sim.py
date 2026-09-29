"""Synthetic multi-well field simulator — the ground truth (BUILD_SPEC §5).

Each well runs full CSS cycles (inject -> soak -> produce), coupling §4.1-4.9 at
every tick. It emits noisy measured Telemetry for the twin and keeps HiddenTruth
(true states, labels, damage) for evaluation only.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

from control.goodman import stress_ratio
from core.config import Config, get_config
from core.state import CyclePhase, SpeedProfile
from core.telemetry import Command, LabSample, MeasuredCard, Recipe, Telemetry, WellMetadata
from core.units import BAR, DAY, G, HOUR
from physics import srp_model as sm
from physics.inflow import inflow_rate, water_cut
from physics.injection import gen_capacity_kg_s, plan_injection
from physics.params import WellParams, build_well_params
from physics.pump import PumpCondition
from physics.pumping_unit import kinematics
from physics.reservoir import CSSReservoir, stimulation_ratio
from physics.rod_string import downsample_card
from simulation.labels import EventSchedule, Label, ScheduledEvent, fault_label, intervals
from simulation.noise import NoiseModel
from simulation.truth import CycleTruth, HiddenTruth, TrueWell

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class WellSimulator:
    def __init__(self, cfg: Config, true: TrueWell, meta: WellMetadata, rng: np.random.Generator,
                 events: EventSchedule, noise: NoiseModel, t0: datetime = T0):
        self.cfg = cfg
        self.true = true
        self.wp: WellParams = true.params
        self.meta = meta
        self.rng = rng
        self.events = events
        self.noise = noise
        self.t = t0
        self.phase = CyclePhase.IDLE
        self.phase_start = t0
        self.cycle = 0
        self.res = CSSReservoir(self.wp.res)
        self.grid = self.wp.rods.discretize(int(cfg.v("field.simulation.wave_nodes")))
        self.n_strokes = int(cfg.v("field.simulation.wave_strokes"))
        self.spm = float(cfg.v("optimizer.baseline.spm"))
        self.profile = SpeedProfile()
        self.level = 150.0                 # m of liquid above the pump
        self.S_p = 2.0
        self.q_last = 5.0 / DAY
        self.damage = np.zeros(len(self.wp.rods.sections))
        self.failed_until: datetime | None = None
        self.recipe: Recipe | None = None
        self.plan = None
        self.steam_done = 0.0
        self.t_prod = 0.0
        self.cycles: list[CycleTruth] = []
        self.truth: list[HiddenTruth] = []
        self.labels: list[Label] = []
        self.last_card_t: datetime | None = None
        self.last_lab_t: datetime | None = None
        self._cache: dict = {}
        self._kin_cache: dict = {}
        self._in_float = False
        s = "field.simulation."
        self.dt_inject = cfg.v(s + "dt_inject") * HOUR
        self.dt_produce = cfg.v(s + "dt_produce") * HOUR
        self.card_every = cfg.v(s + "card_every_hours") * HOUR
        self.lab_every = cfg.v(s + "lab_sample_every_days") * DAY
        self.T_max = cfg.v("optimizer.T_max") * DAY
        self.x_surface = cfg.v("optimizer.surface_quality")
        self.capacity = gen_capacity_kg_s(cfg.v("optimizer.generator_capacity"))
        self.pound_fillage = cfg.v("controller.pound_fillage")
        rods = cfg.v("rods.grade")
        self.tensile = float(cfg.v(f"rods.grades.{rods}"))
        self.sf = cfg.v("rods.service_factor")
        self.N_ref = cfg.v("rods.sn_N_ref")
        self.sn_m = cfg.v("rods.sn_m")
        self.fail_thr = cfg.v("field.events.failure_damage_threshold")
        self.fail_down = cfg.v("field.events.failure_downtime") * DAY
        self.eta = float(np.prod(self.wp.etas))

    # ------------------------------------------------------------------ commands
    def apply(self, cmds: list[Command]) -> None:
        for c in cmds:
            if c.kind == "set_spm":
                self.spm = float(np.clip(float(c.value), 0.5, 12.0))  # type: ignore[arg-type]
            elif c.kind == "set_profile":
                self.profile = c.value if isinstance(c.value, SpeedProfile) else SpeedProfile(**c.value)  # type: ignore
            elif c.kind == "start_cycle" and self.phase == CyclePhase.IDLE:
                self._start_cycle(c.value)  # type: ignore[arg-type]
            elif c.kind == "stop_production" and self.phase == CyclePhase.PRODUCE:
                self._end_production()
            elif c.kind == "set_crank" and self.phase != CyclePhase.PRODUCE:
                self.wp = self.wp.with_unit_R(float(c.value))  # type: ignore[arg-type]
                self._kin_cache.clear()
                self._cache.clear()

    def _start_cycle(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.cycle += 1
        self.plan = plan_injection(self.wp, recipe.steam_mass_t * 1000, recipe.inj_pressure_bar * BAR,
                                   self.res.p_res, self.x_surface, self.capacity)
        self.steam_done = 0.0
        self._set_phase(CyclePhase.INJECT)
        self.cycles.append(CycleTruth(self.wp.well_id, self.cycle, self.t, steam_t=recipe.steam_mass_t,
                                      inj_pressure_bar=recipe.inj_pressure_bar, soak_d=recipe.soak_d,
                                      x_sf=self.plan.x_sf))

    def _set_phase(self, ph: CyclePhase) -> None:
        self.phase = ph
        self.phase_start = self.t

    def _end_production(self) -> None:
        self.res.end_cycle()
        if self.cycles:
            self.cycles[-1].end = self.t
        self._set_phase(CyclePhase.IDLE)

    # ------------------------------------------------------------------ helpers
    def _kin(self):
        key = (round(self.spm, 3), tuple(np.round(self.profile.upstroke, 3)), tuple(np.round(self.profile.downstroke, 3)),
               round(self.wp.unit.R, 4))
        k = self._kin_cache.get(key)
        if k is None:
            k = kinematics(self.wp.unit, self.spm, self.profile, dt=self.grid.dt_cfl * sm.FORWARD_CFL)
            if len(self._kin_cache) > 500:
                self._kin_cache.clear()
            self._kin_cache[key] = k
        return k

    def _wp_now(self) -> WellParams:
        cool = self.events.active(self.t, "cooling")
        if not cool:
            return self.wp
        g = self.wp.geom
        Ts = g.T_surface - cool.magnitude
        wp = copy.copy(self.wp)
        wp.geom = replace(g, T_surface=Ts, grad=(self.wp.T_R - Ts) / g.depth)
        return wp

    def _stroke(self, wp: WellParams, mu: np.ndarray, rho: float, p_intake: float, cond: PumpCondition) -> dict:
        lm = np.log10(np.maximum(mu, 1e-4))
        key = (round(self.spm, 2), tuple(np.round(self.profile.downstroke, 2)), tuple(np.round(self.profile.upstroke, 2)),
               round(wp.unit.R, 3), round(lm[0] * 20) / 20, round(float(lm.mean()) * 20) / 20, round(lm[-1] * 10) / 10,
               round(cond.fillage * 20) / 20, cond.gas, cond.unseated, round(p_intake / BAR), round(rho, -1))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        r = sm.simulate_stroke(wp, self.grid, self.spm, self.profile, mu, rho, p_intake, cond, self.n_strokes)
        f = r.fwd
        sp, sl = downsample_card(f.surface_pos, f.surface_load)
        dp, dl = downsample_card(f.pump_pos, f.pump_load)
        sr = stress_ratio(f.sec_max_stress, f.sec_min_stress, self.tensile, self.sf)
        out = {
            "float_fraction": f.float_fraction, "impacts": f.impacts, "impact_load": f.impact_load,
            "pr_power": r.pr_power, "S_p": float(f.pump_pos.max()), "rfi": r.rfi, "vfall": r.v_fall_min,
            "sr": np.asarray(sr), "surface": (np.array(sp), np.array(sl)), "downhole": (dp, dl),
            "period": r.kin.period, "spm_eff": r.kin.spm_effective, "min_tension": f.min_tension,
        }
        if len(self._cache) > 3000:
            self._cache.clear()
        self._cache[key] = out
        return out

    # ------------------------------------------------------------------ main step
    def step(self, dt: float | None = None) -> Telemetry:
        """Advance one tick. `dt` (s) overrides the phase default (the live runner uses a
        uniform clock across wells)."""
        if dt is None:
            dt = self.dt_produce if self.phase in (CyclePhase.PRODUCE, CyclePhase.IDLE) else self.dt_inject
        tel_extra: dict = {}
        truth_kw: dict = dict(q_in=0.0, q_oil=0.0, q_water=0.0, fillage=0.0, float_fraction=0.0, impact_load=0.0,
                              rfi_true=0.0, fault="normal", motor_kw=0.0, energy_kwh=0.0, failed=False,
                              downtime_h=0.0, steam_t=0.0, spm_eff=0.0, strokes=0.0, mu_top=0.0, mu_pump=0.0,
                              sr=1.0, stress_ratio=0.0)
        wp = self.wp
        running = False
        if self.phase == CyclePhase.INJECT:
            assert self.plan is not None and self.recipe is not None
            target = self.recipe.steam_mass_t * 1000
            m = min(self.plan.w * dt, target - self.steam_done)
            self.steam_done += m
            truth_kw["steam_t"] = m / 1000
            tel_extra.update(steam_rate_tph=self.plan.w * 3.6, steam_cum_t=self.steam_done / 1000,
                             inj_pressure_bar=self.recipe.inj_pressure_bar)
            if self.steam_done >= target - 1e-6:
                self.res.inject(self.plan.Q_sf, self.plan.t_inj, self.plan.T_s)
                self.cycles[-1].inject_d = self.plan.t_inj / DAY
                self.cycles[-1].r_h = self.res.r_h
                self._set_phase(CyclePhase.SOAK)
        elif self.phase == CyclePhase.SOAK:
            self.res.advance(dt)
            assert self.recipe is not None
            if (self.t + timedelta(seconds=dt) - self.phase_start).total_seconds() >= self.recipe.soak_d * DAY:
                self._set_phase(CyclePhase.PRODUCE)
                self.t_prod = 0.0
                self.level = 250.0
                self._in_float = False
        elif self.phase == CyclePhase.IDLE:
            self.res.advance(dt)
        else:
            running, tel_extra = self._produce(dt, truth_kw)
        # truth record
        T_avg = self.res.T_avg()
        mu_c = float(wp.visc.mu_value(wp.T_R))
        sr_ = stimulation_ratio(wp.res.r_e, wp.res.r_w, self.res.r_h, float(wp.visc.mu_value(T_avg)), mu_c)
        truth_kw["sr"] = sr_
        self.truth.append(HiddenTruth(
            t=self.t, well_id=wp.well_id, phase=self.phase.value, cycle=self.cycle, dt_h=dt / HOUR, T_avg=T_avg,
            r_h=self.res.r_h, p_res=self.res.p_res / BAR, damage_max=float(self.damage.max()), level_m=self.level,
            **truth_kw,
        ))
        tel = self._telemetry(running, tel_extra)
        self.t = self.t + timedelta(seconds=dt)
        return tel

    def _produce(self, dt: float, tk: dict) -> tuple[bool, dict]:
        wp_true = self.wp
        wp = self._wp_now()
        res = self.res
        T_avg = res.T_avg()
        mu_c = float(wp.visc.mu_value(wp.T_R))
        sr = stimulation_ratio(wp.res.r_e, wp.res.r_w, res.r_h, float(wp.visc.mu_value(T_avg)), mu_c)
        wc = float(water_cut(self.t_prod, wp.wc_early, wp.wc_late, wp.wc_tau))
        rho = wp.rho_liquid(T_avg, wc)
        p_intake = wp.p_casing + rho * G * self.level
        p_wf = p_intake + rho * G * (wp.depth - wp.pump_depth)
        q_in = inflow_rate(wp.J_c, sr, res.p_res, p_wf)
        failed = self.failed_until is not None and self.t < self.failed_until
        if self.failed_until is not None and not failed:
            self.failed_until = None
        z, T = sm.tubing_temperature(wp, self.q_last, wc, T_avg, self.t_prod, n_cells=40)
        mu = sm.mu_on_grid(wp, self.grid, z, T, wc)
        gas = self.events.active(self.t, "gas")
        unseat = self.events.active(self.t, "unseat") is not None
        gv = gas.magnitude if gas else 0.0
        kin = self._kin()
        cap = wp.A_p * self.S_p * kin.spm_effective / 60.0 * (1 - wp.leak_fraction)
        avail = q_in * dt + self.level * wp.ann_area
        fill = float(np.clip(min(1.0 - gv, avail / max(cap * dt, 1e-12)), 0.05, 1.0))
        cond = PumpCondition(fill, gas=gv > 0, unseated=unseat)
        st = self._stroke(wp, mu, rho, p_intake, cond)
        self.S_p = max(st["S_p"], 0.3)
        spm_eff = st["spm_eff"]
        running = not failed
        if running and not unseat:
            pumped = wp.A_p * self.S_p * fill * (1 - wp.leak_fraction) * spm_eff / 60.0
            pumped = min(pumped, avail / dt)
        else:
            pumped = 0.0
        self.level = float(np.clip(self.level + (q_in - pumped) * dt / wp.ann_area, 0.0, wp.pump_depth))
        q_oil, q_water = pumped * (1 - wc), pumped * wc
        res.advance(dt, q_oil, q_water)
        self.q_last = max(pumped, 1e-6)
        self.t_prod += dt
        dt_h = dt / HOUR
        motor_kw = st["pr_power"] / self.eta / 1000.0 if running else 0.0
        strokes = spm_eff * 60 * dt_h if running else 0.0
        srs = st["sr"]
        if running:
            N = self.N_ref * np.maximum(srs, 1e-3) ** (-self.sn_m)
            impact_boost = 1.0 + (st["impact_load"] / max(np.max(st["surface"][1]), 1.0) if st["impacts"] else 0.0)
            self.damage += strokes / N * impact_boost
        cyc = self.cycles[-1]
        floating = running and st["float_fraction"] > 0.02
        fault = fault_label(unseat, st["float_fraction"] if running else 0.0, gv > 0, fill, self.pound_fillage)
        if floating and not self._in_float:
            cyc.float_events += 1
            self.labels.append(Label(wp.well_id, "rod_float_onset", self.t))
        self._in_float = floating
        if running and st["impacts"] > 0:
            cyc.impact_events += 1
        downtime = 0.0
        if failed or unseat:
            downtime = dt_h
        if running and self.damage.max() >= self.fail_thr:
            self.failed_until = self.t + timedelta(seconds=self.fail_down)
            self.damage[int(np.argmax(self.damage))] = 0.0
            cyc.failures += 1
            self.labels.append(Label(wp.well_id, "rod_failure", self.t))
        cyc.oil_m3 += q_oil * dt
        cyc.water_m3 += q_water * dt
        cyc.energy_kwh += motor_kw * dt_h
        cyc.produce_d += dt / DAY
        cyc.float_h += dt_h if floating else 0.0
        cyc.pound_h += dt_h if fault == "fluid_pound" else 0.0
        cyc.downtime_h += downtime
        cyc.peak_stress_ratio = max(cyc.peak_stress_ratio, float(np.max(srs)) if running else 0.0)
        tk.update(q_in=q_in * DAY, q_oil=q_oil * DAY, q_water=q_water * DAY, fillage=fill,
                  float_fraction=st["float_fraction"] if running else 0.0,
                  impact_load=st["impact_load"] if running else 0.0, rfi_true=st["rfi"], fault=fault,
                  motor_kw=motor_kw, energy_kwh=motor_kw * dt_h, failed=failed, downtime_h=downtime,
                  spm_eff=spm_eff, strokes=strokes, mu_top=float(mu[0]), mu_pump=float(mu[-1]),
                  stress_ratio=float(np.max(srs)))
        n = self.noise
        tel: dict = dict(
            spm_eff=spm_eff, liquid_rate_m3d=n.measure(pumped * DAY, "rate") if running else 0.0,
            water_cut=n.measure(wc, "wc"), p_intake_bar=n.measure(p_intake / BAR, "pressure"),
            T_wellhead_K=n.measure(float(T[0]), "temperature"),
            motor_power_kw=n.measure(motor_kw, "power") if running else 0.0,
            vfd_hz=sm.vfd_hz(wp, self.spm) if running else 0.0,
        )
        if running and (self.last_card_t is None or (self.t - self.last_card_t).total_seconds() >= self.card_every - 1):
            pos, load = n.card(*st["surface"])
            tel["card"] = MeasuredCard(position=pos.tolist(), load=load.tolist(), period=st["period"], spm=spm_eff)
            self.last_card_t = self.t
        if self.last_lab_t is None or (self.t - self.last_lab_t).total_seconds() >= self.lab_every:
            temps = self.rng.uniform(305.0, 400.0, 3)
            nu = wp_true.visc.nu_value(temps)
            tel["lab"] = [LabSample(T_K=float(Tk), nu_cSt=float(n.measure(float(v), "lab") or v))
                          for Tk, v in zip(temps, nu)]
            self.last_lab_t = self.t
        if self.t_prod >= self.T_max:
            self._end_production()
        return running, tel

    def _telemetry(self, running: bool, extra: dict) -> Telemetry:
        return Telemetry(
            well_id=self.wp.well_id, t=self.noise.timestamp(self.t), phase=self.phase, cycle_index=self.cycle,
            phase_elapsed_h=(self.t - self.phase_start).total_seconds() / HOUR, spm_set=self.spm,
            profile=self.profile, crank_R=self.wp.unit.R, running=running,
            recipe_id=self.recipe.recipe_id if self.recipe else None, **extra,
        )

    # ------------------------------------------------------------------ labels
    def derived_labels(self) -> list[Label]:
        prod = [h for h in self.truth if h.phase == "produce"]
        times = [h.t for h in prod]
        out = list(self.labels)
        out += intervals(times, [h.float_fraction > 0.02 for h in prod], self.wp.well_id, "rod_float")
        out += intervals(times, [h.fault == "fluid_pound" for h in prod], self.wp.well_id, "fluid_pound")
        out += intervals(times, [h.fault == "gas_interference" for h in prod], self.wp.well_id, "gas_interference")
        out += intervals(times, [h.fault == "pump_unseated" for h in prod], self.wp.well_id, "pump_unseated")
        return out


def sample_choice(cfg: Config, rng: np.random.Generator) -> dict:
    f = "field."

    def u(path: str) -> float:
        lo, hi = cfg.v(f + path)
        return float(rng.uniform(lo, hi))

    return {
        "depth": u("reservoir.depth"), "net_thickness": u("reservoir.net_thickness"),
        "p_initial": u("reservoir.p_initial"), "J_cold": u("reservoir.J_cold"), "T_R": u("reservoir.T_R"),
        "api_gravity": u("crude.api_gravity"), "tank_compliance": u("reservoir.tank_compliance"),
        "wc_late": u("water_cut.wc_late"), "tau": u("water_cut.tau"),
        "plunger_diameter": float(rng.choice(cfg.v(f + "pump.plunger_diameter"))),
        "visc_scale": float(np.exp(rng.normal(0.0, 0.25))),
        "drag_multiplier": float(cfg.v(f + "drag.drag_multiplier") * np.exp(rng.normal(0.0, 0.15))),
    }


def metadata_for(wp: WellParams, choice: dict) -> WellMetadata:
    return WellMetadata(
        well_id=wp.well_id, depth_m=wp.depth, pump_depth_m=wp.pump_depth, api=wp.api, T_R_K=wp.T_R,
        net_thickness_m=wp.res.h, p_initial_bar=wp.res.p_init / BAR, plunger_d_m=wp.plunger_d,
        crank_R_m=wp.unit.R, rod_taper=[[s.diameter, s.length] for s in wp.rods.sections],
    )


class FieldSimulator:
    def __init__(self, cfg: Config | None = None, n_wells: int | None = None, seed: int | None = None,
                 noise_scale: float = 1.0, event_scale: float = 1.0, horizon_days: float = 1500.0,
                 extra_events: dict[str, list[ScheduledEvent]] | None = None, t0: datetime = T0):
        self.cfg = cfg or get_config()
        n = n_wells or int(self.cfg.v("field.simulation.n_wells"))
        self.seed = int(seed if seed is not None else self.cfg.v("field.simulation.seed"))
        root = np.random.default_rng(self.seed)
        self.wells: dict[str, WellSimulator] = {}
        self.meta: dict[str, WellMetadata] = {}
        for i in range(n):
            wid = f"W{i + 1:02d}"
            rng = np.random.default_rng(root.integers(0, 2 ** 32))
            choice = sample_choice(self.cfg, rng)
            wp = build_well_params(self.cfg, wid, choice)
            wp.placeholder = True
            true = TrueWell(params=wp, hidden={k: choice[k] for k in ("visc_scale", "drag_multiplier", "J_cold",
                                                                      "tank_compliance", "wc_late", "tau")})
            meta = metadata_for(wp, choice)
            ev = EventSchedule.random(self.cfg, rng, t0, horizon_days, event_scale)
            for e in (extra_events or {}).get(wid, []):
                ev.add(e)
            self.wells[wid] = WellSimulator(self.cfg, true, meta, rng, ev, NoiseModel(self.cfg, rng, noise_scale), t0)
            self.meta[wid] = meta

    def step_all(self, dt: float | None = None) -> dict[str, Telemetry]:
        return {wid: w.step(dt) for wid, w in self.wells.items()}

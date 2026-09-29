"""Per-well digital twin. Consumes measured Telemetry ONLY (never simulator truth) and
publishes a WellState with tiers/provenance on every tick.

Forward chain:  steam/soak -> heated zone (Boberg-Lantz) -> T -> viscosity (RLS Walther)
                -> tubing profile (anchored to measured wellhead T) -> rod drag/damping
                -> RFI, cards (diagnostic wave equation), stresses.
Feedback chain: SPM/profile -> displacement -> intake pressure/drawdown -> inflow ->
                cooling/decline -> live cut-off & next-cycle recipe (BO).
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from calibration.inflow_refit import InflowObs, refit
from calibration.rls import WaltherRLS
from control.edge_sim import Bands, EdgeController, Mode
from control.goodman import stress_ratio
from control.safety import SafetyGuard
from control.srp_controller import ControlInput, SRPController, recommend_stroke, shaped
from core import steam
from core.bus import bus
from core.config import Config, get_config
from core.state import (
    Alarm, Card, CycleState, CyclePhase, EconomicState, PumpState, ReservoirState, SpeedProfile, Tier, WellboreState,
    WellState, q,
)
from core.telemetry import Command, Recipe, Telemetry, WellMetadata
from core.units import BAR, BBL, DAY, G, HOUR, k_to_c
from diagnostics.card_features import fillage_estimate
from diagnostics.classifier import CardClassifier
from diagnostics.downhole import downhole_card
from diagnostics.efficiency import volumetric_efficiency
from diagnostics.rfi import RFITracker
from optimization.cutoff import LiveCutoff
from optimization.cycle_bo import BOSettings, CycleOptimizer, Recommendation
from optimization.objective import Econ, q_net
from physics import srp_model as sm
from physics.cycle import simulate_cycle
from physics.inflow import inflow_rate, water_cut
from physics.injection import gen_capacity_kg_s
from physics.params import WellParams, build_well_params
from physics.pump import PumpCondition
from physics.pumping_unit import kinematics, stroke_length
from physics.reservoir import CSSReservoir, stimulation_ratio
from physics.viscosity import ViscosityModel
from physics.wellbore import steam_injection
from risk.failure_risk import FailureRisk

SRC = "engine.twin"


def twin_params(cfg: Config, meta: WellMetadata) -> WellParams:
    """Twin prior: static well-record metadata + config mid-points for unknowns."""
    choice = {"depth": meta.depth_m, "api_gravity": meta.api, "T_R": k_to_c(meta.T_R_K),
              "net_thickness": meta.net_thickness_m, "p_initial": meta.p_initial_bar,
              "plunger_diameter": meta.plunger_d_m, "R": meta.crank_R_m}
    return build_well_params(cfg, meta.well_id, choice)


class WellTwin:
    def __init__(self, meta: WellMetadata, cfg: Config | None = None, classifier: CardClassifier | None = None,
                 mode: Mode = "advisory", bo_fast: bool = False, auto_cycle: bool = False,
                 econ: Econ | None = None, publish: bool = True, optimize: bool = True):
        self.cfg = c = cfg or get_config()
        self.meta = meta
        self.id = meta.well_id
        self.wp = twin_params(c, meta)
        self.res = CSSReservoir(self.wp.res)
        self.grid = self.wp.rods.discretize(30)
        k = "controller.calibration."
        assert self.wp.visc.walther is not None
        self.rls = WaltherRLS.from_params(self.wp.visc.walther, c.v(k + "rls_P0_A"), c.v(k + "rls_P0_B"),
                                          lam=c.v(k + "rls_lambda"), max_dA=c.v(k + "rls_max_dA"),
                                          max_dB=c.v(k + "rls_max_dB"), outlier_sigma=c.v(k + "rls_outlier_sigma"))
        self.refit_every_h = c.v(k + "refit_every") * 24
        self.refit_max_rel = c.v(k + "refit_max_rel")
        self.refit_min = int(c.v(k + "refit_min_points"))
        self.rfi_tr = RFITracker(c.v("controller.rfi_warn"), c.v("controller.rfi_alarm"))
        self.ctrl = SRPController()
        self.safety = SafetyGuard()
        self.edge = EdgeController(self.id, mode, Bands(c.v("controller.band_spm"), c.v("controller.band_downstroke")))
        self.risk = FailureRisk(len(self.wp.rods.sections))
        self.econ = econ or Econ.from_config(c)
        self.optimizer = CycleOptimizer(BOSettings.from_config(c, fast=bo_fast), self.econ)
        self.clf = classifier
        self.clf_min_conf = c.v("controller.clf_min_conf")
        self.auto_cycle = auto_cycle
        self.optimize = optimize
        self.publish = publish
        self.p_wf_target = c.v("optimizer.p_wf_target") * BAR
        self.x_surface = c.v("optimizer.surface_quality")
        self.capacity = gen_capacity_kg_s(c.v("optimizer.generator_capacity"))
        self.ewma_alpha = c.v("controller.ewma_alpha")
        self.hold_h = c.v("controller.hold_hours")
        self.tensile = float(c.v(f"rods.grades.{c.v('rods.grade')}"))
        self.sf = c.v("rods.service_factor")
        self.eta_total = float(np.prod(self.wp.etas))
        # dynamic state
        self.phase = CyclePhase.IDLE
        self.cycle = 0
        self.phase_start: datetime | None = None
        self.t: datetime | None = None
        self.last_valid_t: datetime | None = None
        self.spm = c.v("optimizer.baseline.spm")
        self.profile = SpeedProfile()
        self.R = meta.crank_R_m
        self.inj_rates: list[float] = []
        self.inj_p: list[float] = []
        self.steam_cum_t = 0.0
        self.x_sf = 0.0
        self.heat_loss_W = 0.0
        self.recipe_id: str | None = None
        self.cycle_oil = self.cycle_water = self.cycle_kwh = self.cycle_steam_t = 0.0
        self.cycle_value = 0.0
        self.t_prod_h = 0.0
        self.rec: Recommendation | None = None
        self.rec_summary: dict | None = None
        self.stroke_rec: dict | None = None
        self.g_star: float | None = None
        self.live: LiveCutoff | None = None
        self.stop_recommended = False
        self.fault_class: str | None = None
        self.fault_probs: dict[str, float] = {}
        self.pound_streak = 0
        self.unknown_streak = 0
        self.fillage_card: float | None = None
        self.eta_v = c.v("controller.eta_v_prior")
        self.obs: list[InflowObs] = []
        self.last_refit_h = 0.0
        self.refits: list[dict] = []
        self.calib_hist: deque = deque(maxlen=2000)
        self.hist: deque = deque(maxlen=24 * 420)
        self.cycles: list[dict] = []
        self.cards: deque = deque(maxlen=12)
        self.alarms: deque = deque(maxlen=200)
        self.decisions: deque = deque(maxlen=500)
        self.state: WellState | None = None
        self.mu_nodes = np.full(len(self.grid.z), 1.0)
        self.T_profile: tuple[np.ndarray, np.ndarray] = (np.array([0.0, self.wp.pump_depth]),
                                                         np.array([self.wp.geom.T_surface, self.wp.T_R]))
        self.rho = 950.0
        self.p_intake = 10 * BAR
        self.q_liquid = 0.0
        self.wc = self.wp.wc_early
        self._stress_cache: dict = {}
        self._kin_cache: dict = {}
        self.last_sr = np.zeros(len(self.wp.rods.sections))
        self.rfi = 0.0
        self.vfall = 1.0
        self.q_in_pred = 0.0
        self.qnet_now = 0.0
        self.qnet_potential = 0.0
        self.last_decision: dict | None = None

    # ================================================================ helpers
    def _visc(self) -> ViscosityModel:
        vm = copy.copy(self.wp.visc)
        vm.walther = self.rls.params()
        return vm

    def _kin(self, spm: float, prof: SpeedProfile):
        key = (round(spm, 2), tuple(np.round(prof.downstroke, 3)), tuple(np.round(prof.upstroke, 3)), round(self.R, 3))
        k = self._kin_cache.get(key)
        if k is None:
            k = kinematics(self.wp.unit, max(spm, 0.3), prof, dt=self.grid.dt_cfl * sm.FORWARD_CFL)
            if len(self._kin_cache) > 400:
                self._kin_cache.clear()
            self._kin_cache[key] = k
        return k

    def _rfi(self, spm: float, prof: SpeedProfile) -> float:
        return sm.rfi_value(self._kin(spm, prof), self.vfall)

    def _spm_eff(self, spm: float, prof: SpeedProfile) -> float:
        return self._kin(spm, prof).spm_effective

    def _stress(self, spm: float, prof: SpeedProfile) -> float:
        lm = np.log10(np.maximum(self.mu_nodes, 1e-4))
        key = (round(spm, 1), round(prof.downstroke_factor(), 2), round(lm[0] * 10) / 10, round(lm.mean() * 10) / 10,
               round((self.fillage_card or 0.9) * 10) / 10, round(self.p_intake / BAR / 2), round(self.R, 3))
        v = self._stress_cache.get(key)
        if v is None:
            wp = self.wp
            r = sm.simulate_stroke(wp, self.grid, spm, prof, self.mu_nodes, self.rho, self.p_intake,
                                   PumpCondition(float(np.clip(self.fillage_card or 0.9, 0.3, 1.0))), n_strokes=2)
            sr = stress_ratio(r.fwd.sec_max_stress, r.fwd.sec_min_stress, self.tensile, self.sf)
            v = (float(np.max(sr)), sr)
            if len(self._stress_cache) > 3000:
                self._stress_cache.clear()
            self._stress_cache[key] = v
        self.last_sr = v[1]
        return v[0]

    def _S_p_eff(self) -> float:
        """Plunger stroke = polished-rod stroke - static rod stretch under fluid load."""
        F_fo = max(0.0, (self.wp.p_discharge(self.rho) - self.p_intake) * self.wp.A_p)
        stretch = sum(F_fo * s.length / (s.E * s.area) for s in self.wp.rods.sections)
        return max(stroke_length(self.wp.unit) - stretch, 0.3)

    def _alarm(self, t: datetime, code: str, severity: str, msg: str) -> None:
        a = Alarm(t=t, well_id=self.id, code=code, severity=severity, message=msg)  # type: ignore[arg-type]
        if self.alarms and self.alarms[-1].code == code and (t - self.alarms[-1].t) < timedelta(hours=6):
            return
        self.alarms.append(a)

    def set_mode(self, mode: Mode, bands: Bands | None = None, user: str = "operator") -> None:
        self.edge.set_mode(mode, bands, user)

    # ================================================================ main entry
    def on_telemetry(self, tel: Telemetry) -> list[Command]:
        cmds: list[Command] = []
        dt_h = 1.0 if self.t is None else max((tel.t - self.t).total_seconds() / HOUR, 1e-3)
        dt_h = min(dt_h, 24.0)
        self.t = tel.t
        self.spm, self.profile, self.R = tel.spm_set, tel.profile, tel.crank_R
        if abs(self.wp.unit.R - self.R) > 1e-6:
            self.wp = self.wp.with_unit_R(self.R)
            self._kin_cache.clear()
            self._stress_cache.clear()
        for s in tel.lab:
            u = self.rls.update(s.T_K, s.nu_cSt)
            if not u.accepted:
                self._alarm(tel.t, "calibration_rejected", "info", f"RLS update rejected ({u.reason})")
        if tel.lab:
            p = self.rls.params()
            self.calib_hist.append({"t": tel.t.isoformat(), "A": p.A, "B": p.B, "sA": float(np.sqrt(self.rls.P[0, 0])),
                                    "sB": float(np.sqrt(self.rls.P[1, 1])), "J_c": self.wp.J_c * DAY * BAR})
        prev = self.phase
        if tel.phase != prev:
            self._transition(prev, tel)
        self.phase = tel.phase
        if tel.phase == CyclePhase.INJECT:
            if tel.steam_rate_tph is not None:
                self.inj_rates.append(tel.steam_rate_tph)
            if tel.inj_pressure_bar is not None:
                self.inj_p.append(tel.inj_pressure_bar)
            if tel.steam_cum_t is not None:
                self.steam_cum_t = tel.steam_cum_t
        elif tel.phase == CyclePhase.SOAK:
            self.res.advance(dt_h * HOUR)
        elif tel.phase == CyclePhase.IDLE:
            self.res.advance(dt_h * HOUR)
            cmds += self._idle(tel)
        else:
            cmds += self._produce(tel, dt_h)
        self._build_state(tel)
        if self.publish and self.state is not None:
            bus.publish(f"well/{self.id}", self.state)
        return cmds

    # ================================================================ phases
    def _transition(self, prev: CyclePhase, tel: Telemetry) -> None:
        self.phase_start = tel.t
        if tel.phase == CyclePhase.INJECT:
            self.cycle = tel.cycle_index
            self.inj_rates, self.inj_p, self.steam_cum_t = [], [], 0.0
            self.recipe_id = tel.recipe_id
            self.cycle_oil = self.cycle_water = self.cycle_kwh = 0.0
            self.cycle_value = 0.0
        if prev == CyclePhase.INJECT and tel.phase in (CyclePhase.SOAK, CyclePhase.PRODUCE):
            self._finish_injection(tel)
        if tel.phase == CyclePhase.PRODUCE:
            self.t_prod_h = 0.0
            self.obs = []
            self.last_refit_h = 0.0
            self.ctrl.reset()
            self.last_valid_t = tel.t
            self.pound_streak = self.unknown_streak = 0
            self.fillage_card = None
            t_min = 0.0
            curves = (self.rec_summary or {}).get("curves") or {}
            if curves.get("q_net"):
                t_min = float(curves["t_d"][int(np.argmax(curves["q_net"]))]) * 24.0
            self.live = LiveCutoff(self.g_star, self.ewma_alpha, self.hold_h, t_min) if self.g_star is not None else None
            self.stop_recommended = False
        if prev == CyclePhase.PRODUCE and tel.phase == CyclePhase.IDLE:
            self._end_cycle(tel)

    def _finish_injection(self, tel: Telemetry) -> None:
        w = (np.mean(self.inj_rates) if self.inj_rates else 8.0) / 3.6
        p_wh = (np.mean(self.inj_p) if self.inj_p else 90.0) * BAR
        steam_kg = max(self.steam_cum_t, 1.0) * 1000
        t_inj = steam_kg / max(w, 1e-3)
        prof = steam_injection(replace(self.wp.geom, n_cells=40), w, p_wh, self.x_surface, 0.5 * t_inj,
                               self.wp.U_inject)
        h_ref = steam.C_P_WATER * (self.wp.T_R - 273.15)
        Q = w * max(float(prof.h[-1]) - h_ref, 0.0)
        self.res.inject(Q, t_inj, prof.T_sandface)
        self.x_sf, self.heat_loss_W = prof.x_sandface, prof.total_loss_W
        self.cycle_steam_t = self.steam_cum_t

    def _end_cycle(self, tel: Telemetry) -> None:
        self.res.end_cycle()
        self._do_refit(tel, force=True)
        pred = self.rec_summary or {}
        self.cycles.append({
            "cycle": self.cycle, "recipe_id": self.recipe_id, "steam_t": self.cycle_steam_t,
            "oil_m3": self.cycle_oil, "water_m3": self.cycle_water, "energy_kwh": self.cycle_kwh,
            "sor": self.cycle_steam_t / max(self.cycle_oil, 1e-6),
            "kwh_per_bbl": self.cycle_kwh / max(self.cycle_oil / BBL, 1e-6), "produce_d": self.t_prod_h / 24,
            "value": self.cycle_value, "pred_oil_m3": pred.get("cycle_oil_m3"), "pred_g_star": pred.get("g_star"),
            "pred_T_star_d": pred.get("T_star_d"), "x_sandface": self.x_sf, "r_h": self.res.r_h,
        })
        self.rec, self.rec_summary, self.g_star = None, None, None

    def optimize_cycle(self, recipe_id: str | None = None) -> dict:
        """Run the outer BO from the current calibrated state (receding horizon)."""
        wp = copy.copy(self.wp)
        wp.visc = self._visc()
        self.rec = self.optimizer.run(wp, self.res, recipe_id or f"{self.id}-c{self.cycle + 1}")
        self.rec_summary = self.rec.summary()
        self.g_star = self.rec.best.g_star
        # stroke-length recommendation for the coming cycle (advisory)
        peak_q = float(max(np.add(self.rec.best.q_oil_curve, 0.0))) / max(1 - self.wp.wc_late, 0.3)

        def disp_per_spm(R: float) -> float:
            return self.wp.A_p * stroke_length(self.wp.with_unit_R(R).unit) * 0.8 * 1440

        def stress_R(R: float, spm: float) -> float:
            old = self.wp
            self.wp = self.wp.with_unit_R(R)
            try:
                self._stress_cache.clear()
                return self._stress(spm, SpeedProfile())
            finally:
                self.wp = old
                self._stress_cache.clear()

        self.stroke_rec = recommend_stroke(self.wp.R_options, peak_q, disp_per_spm, stress_R,
                                           self.ctrl.s.spm_max, self.ctrl.s.stress_max)
        self.rec_summary["stroke_recommendation"] = self.stroke_rec
        return self.rec_summary

    def _idle(self, tel: Telemetry) -> list[Command]:
        if self.rec is None and self.optimize:
            self.optimize_cycle()
        cmds: list[Command] = []
        if self.auto_cycle and self.rec is not None:
            if self.stroke_rec and abs(self.stroke_rec["R"] - self.R) > 1e-6:
                cmds.append(Command("set_crank", self.stroke_rec["R"], "twin", "per-cycle stroke recommendation"))
            cmds.append(Command("start_cycle", self.rec.recipe, "twin", "BO recipe"))
        return cmds

    def _produce(self, tel: Telemetry, dt_h: float) -> list[Command]:
        cmds: list[Command] = []
        wp = self.wp
        self.t_prod_h += dt_h
        valid = tel.liquid_rate_m3d is not None and tel.running
        if valid:
            self.last_valid_t = tel.t
        wc = tel.water_cut if tel.water_cut is not None else float(
            water_cut(self.t_prod_h * HOUR, wp.wc_early, wp.wc_late, wp.wc_tau))
        self.wc = float(np.clip(wc, 0.0, 1.0))
        T_avg = self.res.T_avg()
        vm = self._visc()
        mu_c = float(vm.mu_value(wp.T_R))
        sr = self.res.effective_sr(stimulation_ratio(wp.res.r_e, wp.res.r_w, self.res.r_h, float(vm.mu_value(T_avg)),
                                                     mu_c))
        self.rho = wp.rho_liquid(T_avg, self.wc)
        if tel.p_intake_bar is not None and 0 < tel.p_intake_bar < 200:
            self.p_intake = tel.p_intake_bar * BAR
        p_wf = self.p_intake + self.rho * G * (wp.depth - wp.pump_depth)
        q_in_now = inflow_rate(wp.J_c, sr, self.res.p_res, p_wf) * DAY
        # feedforward target: inflow at the target drawdown (the steady state once the annulus
        # is drawn down); the fillage PI corrects model error around it.
        p_wf_tgt = self.p_wf_target + self.rho * G * (wp.depth - wp.pump_depth)
        self.q_in_pred = inflow_rate(wp.J_c, sr, self.res.p_res, p_wf_tgt) * DAY
        q_liq = tel.liquid_rate_m3d if valid else (q_in_now if tel.running else 0.0)
        q_liq = float(q_liq or 0.0)
        if valid and self.q_in_pred > 0 and q_liq > 3 * self.q_in_pred + 5:   # spike guard
            q_liq = q_in_now
        self.q_liquid = q_liq
        q_o, q_w = q_liq * (1 - self.wc), q_liq * self.wc
        self.res.advance(dt_h * HOUR, q_o / DAY, q_w / DAY)
        # tubing temperature (anchored to measured wellhead T) -> viscosity profile
        z, T = sm.tubing_temperature(wp, max(q_liq, 0.05) / DAY, self.wc, T_avg, self.t_prod_h * HOUR, n_cells=30)
        if tel.T_wellhead_K is not None and 250 < tel.T_wellhead_K < 600:
            T = T + (tel.T_wellhead_K - T[0]) * (1 - z / z[-1])
        self.T_profile = (z, T)
        wp_c = copy.copy(wp)
        wp_c.visc = vm                                  # calibrated (RLS) viscosity
        self.mu_nodes = sm.mu_at(wp_c, np.interp(self.grid.z, z, T), self.wc)
        _, _, self.vfall = sm.rod_coefficients(wp, self.grid, self.mu_nodes, self.rho)
        spm_eff = tel.spm_eff or self._spm_eff(self.spm, self.profile)
        self.rfi = self._rfi(self.spm, self.profile)
        rinfo = self.rfi_tr.update(tel.t, self.rfi, spm_eff)
        if rinfo["level"] == "alarm":
            self._alarm(tel.t, "rfi_alarm", "alarm", f"RFI {self.rfi:.2f} >= alarm: rod float expected")
        elif rinfo["level"] == "warn":
            self._alarm(tel.t, "rfi_warn", "warning", f"RFI {self.rfi:.2f} >= warn")
        # cards -> downhole -> classifier, fillage
        if tel.card is not None:
            self._process_card(tel)
        if tel.liquid_rate_m3d is not None and self.fillage_card and tel.running:
            eta = volumetric_efficiency(tel.liquid_rate_m3d, wp.A_p, stroke_length(wp.unit), spm_eff)
            self.eta_v = float(np.clip(0.95 * self.eta_v + 0.05 * eta / max(self.fillage_card, 0.3), 0.3, 1.0))
        # economics
        p_kw = tel.motor_power_kw if tel.motor_power_kw is not None else 0.0
        self.qnet_now = float(q_net(q_o, p_kw, q_w, self.econ))
        self.cycle_oil += q_o * dt_h / 24
        self.cycle_water += q_w * dt_h / 24
        self.cycle_kwh += p_kw * dt_h
        self.cycle_value += self.qnet_now * dt_h / 24
        # live cut-off on the twin's calibrated inflow potential (refit weekly against measured
        # rates) so pump-side transients (gas lock, unseat, downtime) are not read as reservoir decline
        q_pot = self.q_in_pred
        self.qnet_potential = float(q_net(q_pot * (1 - self.wc), p_kw, q_pot * self.wc, self.econ))
        if self.live is not None:
            if self.live.update(self.t_prod_h, self.qnet_potential) and not self.stop_recommended:
                self.stop_recommended = True
                self._alarm(tel.t, "cutoff", "info", f"Cut-off: q_net fell below g*={self.g_star:.0f} after peak")
        if self.stop_recommended and self.auto_cycle:
            cmds.append(Command("stop_production", None, "twin", "live cut-off q_net <= g* after peak"))
        # inflow calibration data + periodic refit
        self.obs.append(InflowObs(self.t_prod_h / 24, tel.liquid_rate_m3d if valid else None,  # type: ignore[arg-type]
                                  tel.water_cut, sr * (self.res.p_res - p_wf) / BAR))
        if self.t_prod_h - self.last_refit_h >= self.refit_every_h:
            self._do_refit(tel)
        # SRP control
        data_age = (tel.t - self.last_valid_t).total_seconds() / HOUR if self.last_valid_t else 0.0
        err = None
        try:
            dec = self.ctrl.step(ControlInput(
                t=tel.t, spm_current=self.spm, profile_current=self.profile, q_in_pred_m3d=self.q_in_pred,
                fillage_meas=self.fillage_card, eta_v_est=self.eta_v, S_p_eff=self._S_p_eff(), A_p=wp.A_p,
                rfi_fn=self._rfi, stress_fn=self._stress, spm_eff_fn=self._spm_eff, fault_class=self.fault_class,
                pound_streak=self.pound_streak,
                ds_floor_fn=lambda s_: self.safety.s.hz_min / max(sm.vfd_hz(wp, s_), 1e-6)))
            new_spm, new_prof, srp = dec.spm, dec.profile, dec.stress_ratio_pred
        except Exception as e:  # noqa: BLE001 - any internal failure -> safety fallback
            err, dec = repr(e)[:80], None
            new_spm, new_prof, srp = self.spm, self.profile, 0.0
        safe = self.safety.check(tel.t, new_spm, new_prof, self.spm, lambda s: sm.vfd_hz(wp, s), srp, data_age,
                                 self.unknown_streak, err)
        if safe.fallback:
            self._alarm(tel.t, "safety_fallback", "alarm", "Safety fallback: " + ",".join(safe.violations))
        for a in (dec.alarms if dec else []):
            if a == "pump_unseated":
                self._alarm(tel.t, "pump_unseated", "critical", "Pump unseated: SPM to minimum")
        codes = (dec.reason_codes if dec else []) + (["SAFETY_FALLBACK"] if safe.fallback else [])
        self.last_decision = {**(dec.as_dict() if dec else {}), "safe_spm": safe.spm,
                              "safe_profile": safe.profile.model_dump(), "safety_ok": safe.ok,
                              "violations": safe.violations, "mode": self.edge.mode, "reason_codes": codes}
        self.decisions.append(self.last_decision)
        if tel.running:
            cmds += self.edge.process(tel.t, safe.spm, safe.profile, self.spm, self.profile, codes)
        # risk
        strokes = spm_eff * 60 * dt_h if tel.running else 0.0
        self._stress(self.spm, self.profile)
        floating = self.fault_class == "rod_float" or self.rfi >= self.rfi_tr.alarm
        self.risk.update(self.t_prod_h, strokes, self.last_sr, floating, self.fault_class == "rod_float",
                         self.fault_class == "fluid_pound", dt_h)
        self.hist.append({
            "t": tel.t.isoformat(), "phase": tel.phase.value, "cycle": self.cycle, "q_oil": q_o,
            "q_oil_pred": self.q_in_pred * (1 - self.wc), "q_liquid": q_liq, "wc": self.wc, "T_avg_C": k_to_c(T_avg),
            "r_h": self.res.r_h, "mu_pump_cp": float(self.mu_nodes[-1] * 1000), "mu_top_cp": float(self.mu_nodes[0] * 1000),
            "rfi": self.rfi, "fillage": self.fillage_card, "eta_v": self.eta_v, "q_net": self.qnet_now,
            "q_net_potential": self.qnet_potential,
            "q_net_ewma": self.live.ewma if self.live else None, "g_star": self.g_star, "spm": self.spm,
            "ds": self.profile.downstroke_factor(), "stress_ratio": float(np.max(self.last_sr)),
            "risk": self.risk.score()["risk"], "p_res_bar": self.res.p_res / BAR, "motor_kw": p_kw,
            "T_wh_C": k_to_c(float(T[0])),
        })
        return cmds

    def _process_card(self, tel: Telemetry) -> None:
        card = tel.card
        assert card is not None
        c, w, _ = sm.rod_coefficients(self.wp, self.grid, self.mu_nodes, self.rho)
        dp, dl = downhole_card(self.grid, card.position, card.load, card.period, c, w)
        self.fillage_card = fillage_estimate(dp, dl)
        F_fo = max((self.wp.p_discharge(self.rho) - self.p_intake) * self.wp.A_p, 1.0)
        if self.clf is not None:
            pred = self.clf.predict(dp, dl, card.position, card.load, F_ref=F_fo)
            self.fault_probs = pred.probs
            self.fault_class = pred.label
            # physics cross-check: an unseated pump cannot deliver liquid
            if pred.label == "pump_unseated" and tel.liquid_rate_m3d is not None:
                ev = volumetric_efficiency(tel.liquid_rate_m3d, self.wp.A_p, stroke_length(self.wp.unit),
                                           card.spm or self.spm)
                if ev > 0.3:
                    self.fault_class = "unknown"
                    self._alarm(tel.t, "classifier_conflict", "info",
                                f"Card looks unseated but volumetric efficiency is {ev:.2f}; label withheld")
            self.unknown_streak = self.unknown_streak + 1 if pred.label == "unknown" else 0
            if self.fault_class == "pump_unseated":
                self._alarm(tel.t, "pump_unseated", "critical", "Classifier: pump unseated")
            elif self.fault_class == "rod_float":
                self._alarm(tel.t, "rod_float", "alarm", "Classifier: rod float on card")
        self.pound_streak = self.pound_streak + 1 if (self.fault_class == "fluid_pound" or
                                                       (self.fillage_card or 1) < self.ctrl.s.pound_fillage) else 0
        self.cards.append({"t": tel.t.isoformat(), "surface": {"position": card.position, "load": card.load},
                           "downhole": {"position": dp, "load": dl}, "spm": card.spm, "fault": self.fault_class,
                           "probs": self.fault_probs, "fillage": self.fillage_card})

    def _do_refit(self, tel: Telemetry, force: bool = False) -> None:
        if not self.obs:
            return
        self.last_refit_h = self.t_prod_h
        r = refit(self.obs, self.wp.J_c * DAY * BAR, self.wp.wc_early, self.wp.wc_late, self.wp.wc_tau / DAY,
                  max_rel=self.refit_max_rel, min_points=self.refit_min)
        self.refits.append({"t": tel.t.isoformat(), "accepted": r.accepted, "reason": r.reason, "J_c": r.J_c,
                            "J_c_std": r.J_c_std, "wc_late": r.wc_late, "tau_d": r.tau_d, "rmse_rate": r.rmse_rate})
        if r.accepted:
            self.wp.J_c = r.J_c / DAY / BAR
            self.wp.wc_late, self.wp.wc_tau = r.wc_late, r.tau_d * DAY
        elif r.reason not in ("insufficient_data",):
            self._alarm(tel.t, "calibration_rejected", "warning", f"Inflow refit rejected ({r.reason})")

    # ================================================================ what-if
    def whatif(self, steam_t: float | None = None, p_inj_bar: float | None = None, soak_d: float | None = None,
               spm: float | None = None, downstroke: float | None = None) -> dict:
        """Scenario on a cloned state; no side effects."""
        base = self.rec_summary["recipe"] if self.rec_summary else {"steam_mass_t": 3000.0, "inj_pressure_bar": 90.0,
                                                                    "soak_d": 5.0}
        wp = copy.copy(self.wp)
        wp.visc = self._visc()
        from optimization.cycle_bo import evaluate

        th0 = np.array([base["steam_mass_t"], base["inj_pressure_bar"], base["soak_d"]], dtype=float)
        th1 = np.array([steam_t or th0[0], p_inj_bar or th0[1], soak_d or th0[2]], dtype=float)
        s = self.optimizer.s
        e0 = evaluate(wp, self.res, th0, self.econ, s, check_stress=False)
        e1 = evaluate(wp, self.res, th1, self.econ, s, check_stress=False)
        spm0, prof0 = self.spm, self.profile
        spm1 = spm if spm is not None else spm0
        prof1 = shaped(downstroke) if downstroke is not None else prof0
        rfi0, rfi1 = self._rfi(spm0, prof0), self._rfi(spm1, prof1)
        sr0, sr1 = self._stress(spm0, prof0), self._stress(spm1, prof1)

        def pack(e, rfi_, sr_):
            return {"cycle_oil_m3": e.oil_m3, "sor": e.sor, "kwh_per_bbl": e.kwh_per_bbl, "g_star": e.g_star,
                    "T_star_d": e.T_star_d, "rfi": rfi_, "stress_ratio": sr_}

        a, b = pack(e0, rfi0, sr0), pack(e1, rfi1, sr1)
        return {"plan": a, "scenario": b, "delta": {k: b[k] - a[k] for k in a}, "tier": "B",
                "inputs": {"steam_t": float(th1[0]), "p_inj_bar": float(th1[1]), "soak_d": float(th1[2]),
                           "spm": spm1, "downstroke": prof1.downstroke_factor()}}

    # ================================================================ state
    def _build_state(self, tel: Telemetry) -> None:
        t = tel.t
        ph = bool(self.wp.placeholder)

        def B(v, unit, src=SRC, unc=None, ext=False, tier=Tier.B):
            return q(float(v) if v is not None and np.isfinite(v) else 0.0, unit, tier, src, uncertainty=unc,
                     extrapolated=ext, placeholder=ph, t=t)

        T_avg = self.res.T_avg()
        vm = self._visc()
        lo, hi = vm.fit_range
        mu_p = float(self.mu_nodes[-1])
        rel = float(vm.walther.nu_rel_sigma(float(self.T_profile[1][-1]))) if vm.walther else 0.0
        z, T = self.T_profile
        risk = self.risk.score()
        cy_oil = max(self.cycle_oil, 1e-9)
        card_s = card_d = None
        if self.cards:
            last = self.cards[-1]
            ct = datetime.fromisoformat(last["t"])
            card_s = Card(position=last["surface"]["position"], load=last["surface"]["load"], kind="surface", t=ct,
                          spm=last["spm"], tier=Tier.A, source="measured")
            card_d = Card(position=last["downhole"]["position"], load=last["downhole"]["load"], kind="downhole", t=ct,
                          spm=last["spm"], tier=Tier.B, source="diagnostic wave equation")
        stroke = stroke_length(self.wp.unit)
        self.state = WellState(
            well_id=self.id, t=t,
            cycle=CycleState(
                cycle_index=self.cycle, phase=tel.phase, phase_start=self.phase_start or t,
                steam_injected_kg=q(self.steam_cum_t * 1000, "kg", Tier.A, "measured", t=t),
                injection_pressure=q((np.mean(self.inj_p) if self.inj_p else 0.0) * BAR, "Pa", Tier.A, "measured", t=t),
                soak_duration=B((self.rec_summary or {}).get("recipe", {}).get("soak_d", 0.0), "d", "recipe"),
                cumulative_oil_m3=q(self.cycle_oil, "m3", Tier.A, "measured (integrated)", t=t),
                cumulative_water_m3=q(self.cycle_water, "m3", Tier.A, "measured (integrated)", t=t),
                recipe_id=self.recipe_id,
            ),
            reservoir=ReservoirState(
                T_avg_heated=B(T_avg, "K", "physics.reservoir (Boberg-Lantz)"),
                r_heated=B(self.res.r_h, "m", "physics.reservoir (Marx-Langenheim)"),
                p_reservoir=B(self.res.p_res, "Pa", "physics.reservoir (tank)"),
                stimulation_ratio=B(self.res.effective_sr(stimulation_ratio(
                    self.wp.res.r_e, self.wp.res.r_w, self.res.r_h, float(vm.mu_value(T_avg)),
                    float(vm.mu_value(self.wp.T_R)))), "-", "physics.reservoir"),
            ),
            wellbore=WellboreState(
                sandface_steam_quality=B(self.x_sf, "-", "physics.wellbore (Ramey)"),
                heat_loss_rate=B(self.heat_loss_W, "W", "physics.wellbore (Ramey)"),
                T_tubing_profile=[(float(a), float(b)) for a, b in zip(z[::3], T[::3])],
                mu_at_pump=B(mu_p, "Pa.s", "physics.viscosity (RLS Walther)", unc=rel * mu_p,
                             ext=not (lo <= float(T[-1]) <= hi)),
                mu_profile=[(float(a), float(b)) for a, b in zip(self.grid.z[::3], self.mu_nodes[::3])],
            ),
            pump=PumpState(
                spm=q(self.spm, "1/min", Tier.A, "VFD read-back", t=t),
                stroke_length=q(stroke, "m", Tier.A, "unit geometry", t=t),
                vfd_profile=self.profile,
                intake_pressure=q(self.p_intake, "Pa", Tier.A, "downhole gauge", t=t),
                fillage=B(self.fillage_card if self.fillage_card is not None else 0.0, "-", "diagnostics.efficiency"),
                volumetric_efficiency=B(volumetric_efficiency(self.q_liquid, self.wp.A_p, stroke,
                                                              tel.spm_eff or self.spm) if tel.running else 0.0, "-",
                                        "diagnostics.efficiency"),
                surface_card=card_s, downhole_card=card_d,
                rfi=B(self.rfi, "-", "diagnostics.rfi"),
                fault_class=self.fault_class, fault_probs=self.fault_probs,
                peak_rod_stress_ratio=B(float(np.max(self.last_sr)), "-", "control.goodman + wave equation"),
            ),
            econ=EconomicState(
                q_oil=q(self.q_liquid * (1 - self.wc), "m3/d", Tier.A, "measured", t=t),
                q_net_value_rate=B(self.qnet_now, "USD/d", "optimization.objective"),
                kwh_per_bbl=B(self.cycle_kwh / max(cy_oil / BBL, 1e-9) if self.cycle_oil > 0.5 else 0.0, "kWh/bbl",
                              "physics.energy"),
                sor_cycle=B(self.cycle_steam_t / cy_oil if self.cycle_oil > 0.5 else 0.0, "-", "physics.energy"),
                g_star=B(self.g_star, "USD/d", "optimization.cycle_bo",
                         unc=(self.rec_summary or {}).get("uncertainty", {}).get("total_std"))
                if self.g_star is not None else None,
            ),
            risk_score=B(risk["risk"], "-", "risk.failure_risk (heuristic ranking)"),
            control_mode=self.edge.mode,
            alarms=list(self.alarms)[-10:],
        )

    # ================================================================ views for the API
    def cutoff_view(self) -> dict:
        return {"g_star": self.g_star, "stop_recommended": self.stop_recommended,
                "live": self.live.state() if self.live else None,
                "trend": [{"t": h["t"], "q_net": h["q_net"], "ewma": h["q_net_ewma"]} for h in list(self.hist)[-24 * 60:]],
                "plan_curves": (self.rec_summary or {}).get("curves"), "tier": "B"}

    def recommendation_view(self) -> dict:
        return {"decision": self.last_decision, "pending": [
            {"id": a.id, "t": a.t.isoformat(), "spm": a.spm, "profile": a.profile.model_dump(),
             "reason_codes": a.reason_codes} for a in self.edge.pending_list()],
            "mode": self.edge.mode, "bands": vars(self.edge.bands), "audit": self.edge.audit[-50:],
            "stroke_recommendation": self.stroke_rec}

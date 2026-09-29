"""M8 — SRP controller, Goodman, safety, edge modes, twin closed loop."""

from datetime import datetime, timezone

import numpy as np
import pytest

from control.edge_sim import Bands, EdgeController
from control.goodman import s_allow, stress_ratio
from control.safety import SafetyGuard
from control.srp_controller import ControlInput, SRPController, recommend_stroke, shaped
from core.state import CyclePhase, SpeedProfile

T = datetime(2026, 3, 1, tzinfo=timezone.utc)


def test_goodman_textbook():
    # grade D, T = 115 ksi = 793 MPa, S_min = 100 MPa, SF = 0.9
    assert s_allow(793e6, 100e6, 0.9) == pytest.approx((793e6 / 4 + 0.5625 * 100e6) * 0.9)
    assert float(stress_ratio(229.05e6, 100e6, 793e6, 0.9)) == pytest.approx(1.0, rel=1e-3)
    assert s_allow(793e6, -50e6, 1.0) == pytest.approx(793e6 / 4)       # compressive S_min clipped


def ci(**kw):
    base = dict(t=T, spm_current=4.0, profile_current=SpeedProfile(), q_in_pred_m3d=15.0, fillage_meas=0.87,
                eta_v_est=0.85, S_p_eff=2.0, A_p=1.55e-3, rfi_fn=lambda s, p: 0.1 * s,
                stress_fn=lambda s, p: 0.1 * s, spm_eff_fn=lambda s, p: s * (0.7 + 0.3 * p.downstroke_factor()))
    base.update(kw)
    return ControlInput(**base)


def test_feedforward_matches_inflow():
    c = SRPController()
    d = c.step(ci())
    assert d.spm_ff == pytest.approx(15.0 / (1.55e-3 * 2.0 * 0.85 * 1440))
    assert "FF_INFLOW_MATCH" in d.reason_codes


def test_fillage_pi_lowers_spm_when_underfilled_and_antiwindup():
    c = SRPController()
    lows = [c.step(ci(fillage_meas=0.5, spm_current=4.0)).spm for _ in range(50)]
    assert lows[-1] < 4.0
    assert abs(c.integral) <= c.s.i_limit + 1e-9


def test_reliability_cap_never_exceeds_inflow_unless_pump_full():
    c = SRPController()
    d = c.step(ci(fillage_meas=0.92, spm_current=3.0))
    assert d.spm <= d.spm_ff * (1 + c.s.transient_weight) + 1e-9
    c2 = SRPController()
    d2 = c2.step(ci(fillage_meas=0.99, spm_current=d.spm_ff))
    assert "FULL_PUMP_RAISE_ALLOWED" in d2.reason_codes


def test_float_protection_shapes_before_reducing_spm():
    c = SRPController()
    rfi = lambda s, p: 0.25 * s * p.downstroke_factor()     # noqa: E731  (RFI 1.0 at 4 SPM unshaped)
    d = c.step(ci(rfi_fn=rfi, q_in_pred_m3d=15.0, spm_current=4.0, fillage_meas=0.9))
    assert "DOWNSTROKE_SHAPED" in d.reason_codes
    assert d.profile.downstroke_factor() < 1.0
    assert d.profile.upstroke == [1.0, 1.0, 1.0, 1.0]         # upstroke kept
    assert d.rfi_pred < c.s.rfi_warn or "SPM_REDUCED_FOR_FLOAT" in d.reason_codes


def test_spm_reduced_only_when_shaping_exhausted():
    c = SRPController()
    rfi = lambda s, p: 0.5 * s                               # noqa: E731  shaping cannot help
    prof = shaped(c.s.ds_min)
    d = c.step(ci(rfi_fn=rfi, spm_current=4.0, profile_current=prof))
    assert "SPM_REDUCED_FOR_FLOAT" in d.reason_codes and d.spm < 4.0
    assert abs(d.spm - 4.0) <= c.s.max_change + 1e-9


def test_stress_backoff():
    c = SRPController()
    d = c.step(ci(stress_fn=lambda s, p: 0.3 * s, fillage_meas=0.95, spm_current=4.0, q_in_pred_m3d=30))
    assert d.stress_ratio_pred <= c.s.stress_max + 1e-9 or d.spm == pytest.approx(4.0 - c.s.max_change)
    assert "GOODMAN_LIMIT" in d.active_constraints


def test_unseated_and_pound_special_actions():
    c = SRPController()
    d = c.step(ci(fault_class="pump_unseated", spm_current=2.0))
    assert d.spm == pytest.approx(c.s.spm_min) and "pump_unseated" in d.alarms
    d2 = SRPController().step(ci(pound_streak=5, fillage_meas=0.6))
    assert "SUSTAINED_FLUID_POUND_STEP_DOWN" in d2.reason_codes and d2.spm < 4.0


def test_safety_envelope_and_fallback():
    g = SafetyGuard()
    ok = g.check(T, 4.0, SpeedProfile(), 3.8, lambda s: 8.5 * s, 0.5, 0.5, 0)
    assert ok.ok and not ok.fallback
    bad = g.check(T, 9.5, SpeedProfile(), 4.0, lambda s: 8.5 * s, 0.5, 0.5, 0)
    assert bad.fallback and "spm_out_of_envelope" in bad.violations and bad.spm == pytest.approx(4.0)
    stale = g.check(T, 4.2, SpeedProfile(), 4.0, lambda s: 8.5 * s, 0.5, 10.0, 0)
    assert stale.fallback and "stale_data" in stale.violations
    unk = g.check(T, 4.2, SpeedProfile(), 4.0, lambda s: 8.5 * s, 0.5, 0.5, 3)
    assert "classifier_unknown_streak" in unk.violations
    exc = g.check(T, 4.2, SpeedProfile(), 4.0, lambda s: 8.5 * s, 0.5, 0.5, 0, error="boom")
    assert exc.fallback and exc.violations[0].startswith("exception")


def test_edge_modes_and_audit():
    e = EdgeController("W01")
    assert e.process(T, 4.3, SpeedProfile(), 4.0, SpeedProfile(), ["X"]) == []      # advisory: nothing written
    e.set_mode("supervised")
    assert e.process(T, 4.3, SpeedProfile(), 4.0, SpeedProfile(), ["X"]) == []
    pend = e.pending_list()
    assert len(pend) == 1
    cmds = e.approve(pend[0].id)
    assert {c.kind for c in cmds} == {"set_spm", "set_profile"}
    e.set_mode("autonomous", Bands(spm=0.5, downstroke=0.2))
    assert e.process(T, 4.3, SpeedProfile(), 4.0, SpeedProfile(), ["X"])               # in band -> written
    assert e.process(T, 5.5, SpeedProfile(), 4.0, SpeedProfile(), ["X"]) == []         # out of band -> pending
    assert e.pending_list()
    events = [a["event"] for a in e.audit]
    assert events.count("mode_change") == 2 and "approved" in events and "vfd_write" in events


def test_stroke_recommendation_prefers_longest_safe_stroke():
    rec = recommend_stroke((0.76, 0.86, 0.97), 20.0, lambda R: 5.0 * R, lambda R, s: 0.5 + 0.1 * R, 7.0, 1.0)
    assert rec["R"] == 0.97 and rec["advisory_only"]
    rec2 = recommend_stroke((0.76, 0.86, 0.97), 20.0, lambda R: 5.0 * R, lambda R, s: 1.125 * R, 7.0, 1.0)
    assert rec2["R"] == 0.76


@pytest.mark.slow
def test_twin_closed_loop_one_cycle():
    from engine.twin import WellTwin
    from evaluation.classifier_eval import ensure_model
    from simulation.field_sim import FieldSimulator

    f = FieldSimulator(n_wells=1, seed=1)
    w = f.wells["W01"]
    tw = WellTwin(f.meta["W01"], classifier=ensure_model(quick=True), mode="autonomous", bo_fast=True,
                  auto_cycle=True, publish=False)
    for _ in range(6000):
        tel = w.step()
        w.apply(tw.on_telemetry(tel))
        if w.cycles and w.cycles[-1].end is not None:
            tw.on_telemetry(w.step())            # twin sees the IDLE transition
            break
    c = w.cycles[-1]
    assert c.end is not None and c.oil_m3 > 200
    assert c.pound_h < 0.2 * c.produce_d * 24
    assert tw.state is not None and tw.state.pump.rfi.prov.tier.value == "B"
    assert tw.cycles and tw.cycles[0]["pred_oil_m3"] is not None
    assert np.isfinite(tw.state.econ.kwh_per_bbl.value)
    assert tw.phase in (CyclePhase.IDLE, CyclePhase.INJECT)

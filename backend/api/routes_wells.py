"""Well routes (BUILD_SPEC §13)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.deps import convert_units, runner, twin_or_404
from control.edge_sim import Bands

router = APIRouter()


@router.get("/wells")
def wells():
    r = runner()
    return {"sim_time": r.sim_t.isoformat(), "wells": sorted(r.summary(), key=lambda x: x["well_id"])}


@router.get("/wells/{wid}/state")
def state(wid: str, units: Literal["si", "field"] = "si"):
    tw = twin_or_404(wid)
    if tw.state is None:
        raise HTTPException(409, "no state yet")
    d = tw.state.model_dump(mode="json")
    return convert_units(d) if units == "field" else d


@router.get("/wells/{wid}/history")
def history(wid: str, t_from: str | None = None, t_to: str | None = None, fields: str | None = None,
            every: int = 1):
    tw = twin_or_404(wid)
    rows = list(tw.hist)
    if t_from:
        rows = [x for x in rows if x["t"] >= datetime.fromisoformat(t_from).isoformat()]
    if t_to:
        rows = [x for x in rows if x["t"] <= datetime.fromisoformat(t_to).isoformat()]
    rows = rows[:: max(every, 1)]
    if fields:
        keep = set(fields.split(",")) | {"t"}
        rows = [{k: v for k, v in x.items() if k in keep} for x in rows]
    return {"well_id": wid, "rows": rows, "tiers": {"q_oil": "A", "q_oil_pred": "B", "T_avg_C": "B", "rfi": "B",
                                                    "fillage": "B", "q_net": "B", "risk": "B", "mu_pump_cp": "B"}}


@router.get("/wells/{wid}/cards")
def cards(wid: str, limit: int = 6):
    tw = twin_or_404(wid)
    return {"well_id": wid, "cards": list(tw.cards)[-limit:],
            "tiers": {"surface": "A (measured)", "downhole": "B (diagnostic wave equation)", "fault": "C"}}


@router.get("/wells/{wid}/reservoir")
def reservoir(wid: str):
    tw = twin_or_404(wid)
    return {"cycles": tw.cycles, "calibration": list(tw.calib_hist), "refits": tw.refits[-20:],
            "recommendation": tw.rec_summary, "rls": {"A": tw.rls.A, "B": tw.rls.B, "P": tw.rls.P.tolist()},
            "tier": "B"}


@router.get("/wells/{wid}/diagnostics")
def diagnostics(wid: str):
    tw = twin_or_404(wid)
    r = tw.risk.score()
    return {"risk": r, "stress_ratio_by_section": tw.last_sr.tolist(), "fault_class": tw.fault_class,
            "fault_probs": tw.fault_probs, "fillage": tw.fillage_card, "eta_v": tw.eta_v, "rfi": tw.rfi,
            "rfi_slope_per_stroke": tw.rfi_tr.slope_per_stroke(), "rfi_warn": tw.rfi_tr.warn,
            "rfi_alarm": tw.rfi_tr.alarm, "sections": [{"d_m": s.diameter, "L_m": s.length} for s in tw.wp.rods.sections]}


@router.post("/wells/{wid}/optimize-cycle")
def optimize_cycle(wid: str):
    tw = twin_or_404(wid)
    with runner().lock:
        return tw.optimize_cycle()


@router.get("/wells/{wid}/cutoff")
def cutoff(wid: str):
    return twin_or_404(wid).cutoff_view()


@router.get("/wells/{wid}/controller/recommendation")
def recommendation(wid: str):
    return twin_or_404(wid).recommendation_view()


class Approve(BaseModel):
    action_id: int
    approve: bool = True
    user: str = "operator"


@router.post("/wells/{wid}/controller/approve")
def approve(wid: str, body: Approve):
    r = runner()
    tw = twin_or_404(wid)
    with r.lock:
        if body.approve:
            cmds = tw.edge.approve(body.action_id, body.user)
            r.field.wells[wid].apply(cmds)
            return {"written": [c.kind for c in cmds]}
        tw.edge.reject(body.action_id, body.user)
        return {"written": []}


class ModeBody(BaseModel):
    mode: Literal["advisory", "supervised", "autonomous"]
    band_spm: float | None = None
    band_downstroke: float | None = None
    user: str = "operator"


@router.post("/wells/{wid}/mode")
def set_mode(wid: str, body: ModeBody):
    tw = twin_or_404(wid)
    bands = None
    if body.band_spm is not None or body.band_downstroke is not None:
        bands = Bands(body.band_spm if body.band_spm is not None else tw.edge.bands.spm,
                      body.band_downstroke if body.band_downstroke is not None else tw.edge.bands.downstroke)
    with runner().lock:
        tw.set_mode(body.mode, bands, body.user)
    return {"mode": tw.edge.mode, "bands": vars(tw.edge.bands)}


class WhatIf(BaseModel):
    well_id: str
    steam_t: float | None = None
    p_inj_bar: float | None = None
    soak_d: float | None = None
    spm: float | None = None
    downstroke: float | None = None


@router.post("/whatif")
def whatif(body: WhatIf):
    tw = twin_or_404(body.well_id)
    with runner().lock:
        return tw.whatif(body.steam_t, body.p_inj_bar, body.soak_d, body.spm, body.downstroke)

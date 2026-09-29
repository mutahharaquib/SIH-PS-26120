"""Evaluation, config, simulation-control and about routes."""

from __future__ import annotations

import json
import threading
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.deps import runner
from core.config import get_config
from evaluation.harness import REPORTS, HarnessConfig, run, run_scenario

router = APIRouter()
_jobs: dict[str, dict] = {}

LIMITATIONS = [
    "Validation uses physics and synthetic data only (no Oil India data bundled).",
    "Boberg-Lantz assumes radial flow and a simple heated zone; heterogeneity is not captured.",
    "Asphaltene effects may take viscosity outside the fitted Walther range at low temperature; extrapolation is flagged.",
    "Rod drag ignores couplings/guides (apart from a multiplier) and pressure-driven annular flow.",
    "The failure-risk score is an uncalibrated heuristic ranking, not a failure prediction.",
    "Per-well optimization is not field-optimal under a shared steam supply (Phase 2).",
    "Dynamometer-card availability is not confirmed by the problem statement; cycle optimization runs without "
    "cards, diagnostics require them.",
    "RFI uses the minimum fall velocity over depth (conservative): expect warn-level alarms well before float.",
    "Heated-zone oil depletion and emulsion viscosity rules are placeholder extensions beyond Boberg-Lantz.",
    "BoTorch is optional; the optimizer uses a scikit-learn GP + log-EI fallback.",
]


class EvalBody(BaseModel):
    n_wells: int = 4
    seeds: list[int] = [1]
    horizon_days: float = 300.0
    scenario: str | None = None


@router.post("/evaluation/run")
def evaluation_run(body: EvalBody):
    run_id = ("scenario-" if body.scenario else "run-") + uuid.uuid4().hex[:8]
    _jobs[run_id] = {"status": "running", "progress": [0, 0]}

    def work() -> None:
        try:
            if body.scenario:
                run_scenario(REPORTS.parent / "scenarios" / f"{body.scenario}.yaml", run_id=run_id)
            else:
                run(HarnessConfig(n_wells=body.n_wells, seeds=body.seeds, horizon_days=body.horizon_days), run_id,
                    progress=lambda i, n: _jobs[run_id].update(progress=[i, n]))
            _jobs[run_id]["status"] = "done"
        except Exception as e:  # noqa: BLE001
            _jobs[run_id].update(status="error", error=repr(e))

    threading.Thread(target=work, daemon=True).start()
    return {"run_id": run_id}


@router.get("/evaluation")
def evaluation_list():
    runs = []
    for p in sorted(REPORTS.glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            r = json.loads(p.read_text())
            runs.append({"run_id": r["run_id"], "created": r.get("created"), "label": r.get("label"),
                         "config": r.get("config"), "scenario": bool(r.get("scenario"))})
        except (json.JSONDecodeError, KeyError):
            continue
    running = [{"run_id": k, **v} for k, v in _jobs.items() if v["status"] == "running"]
    return {"runs": runs, "running": running}


@router.get("/evaluation/{run_id}")
def evaluation_get(run_id: str):
    p = REPORTS / run_id / "report.json"
    if p.exists():
        return json.loads(p.read_text())
    if run_id in _jobs:
        return _jobs[run_id]
    raise HTTPException(404, "unknown run")


@router.get("/config")
def config():
    cfg = get_config()
    return {"config": cfg.as_dict(), "placeholders": cfg.placeholders()}


class SimControl(BaseModel):
    playing: bool | None = None
    seconds_per_day: float | None = None
    step_hours: int | None = None
    reset: bool = False
    scenario: str | None = None


@router.get("/sim")
def sim_state():
    return runner().control()


@router.post("/sim")
def sim_control(body: SimControl):
    r = runner()
    if body.scenario:
        r.load_scenario(body.scenario)
    elif body.reset:
        r.reset()
    if body.seconds_per_day is not None:
        r.seconds_per_day = max(0.2, body.seconds_per_day)
    if body.step_hours:
        r.step(min(body.step_hours, 24 * 60))
    if body.playing is not None:
        r.playing = body.playing
    return r.control()


@router.get("/about")
def about():
    return {"limitations": LIMITATIONS, "tiers": {
        "A": "Directly bounded by problem-statement values or a cited standard/model (T_R, API, Goodman formula).",
        "B": "Output of first-principles physics modules with calibrated or placeholder parameters.",
        "C": "ML outputs (card classifier) and cards estimated from VFD data."},
        "phase2_not_built": ["Depletion-aware DP cut-off and multi-cycle lookahead", "Survival-model failure prediction",
                             "Field-level optimization across wells sharing a steam generator",
                             "Real Oil India data ingestion beyond the replay/OPC UA mapping"]}

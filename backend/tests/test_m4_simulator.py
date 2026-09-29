"""M4 — multi-well simulator, noise, labeled events, replay, hidden-truth separation."""

import ast
import asyncio
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from core.state import CyclePhase
from core.telemetry import Command, Recipe, Telemetry
from simulation.field_sim import FieldSimulator, T0
from simulation.labels import ScheduledEvent
from simulation.replay import load_history, save_csv, stream

BACKEND = Path(__file__).resolve().parents[1]
TWIN_PACKAGES = ["core", "physics", "diagnostics", "calibration", "optimization", "control", "risk", "engine"]


def _imports(py: Path) -> set[str]:
    tree = ast.parse(py.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_twin_never_imports_hidden_truth():
    """Import-lint: twin packages never import the simulator; only simulation/ and
    evaluation/ may import simulation.truth."""
    for pkg in TWIN_PACKAGES:
        for py in (BACKEND / pkg).rglob("*.py"):
            bad = [m for m in _imports(py) if m == "simulation" or m.startswith("simulation.")]
            assert not bad, f"{py.relative_to(BACKEND)} imports {bad}"
    for py in BACKEND.rglob("*.py"):
        rel = py.relative_to(BACKEND)
        if rel.parts[0] in ("simulation", "evaluation", "tests"):
            continue
        assert "simulation.truth" not in _imports(py), f"{rel} imports hidden truth"


def test_telemetry_schema_has_no_truth_fields():
    forbidden = {"T_avg", "r_h", "damage", "fault", "float_fraction", "J_c", "visc_scale", "p_res"}
    assert not forbidden & set(Telemetry.model_fields)


def run_cycle(w, recipe=Recipe(2500, 90, 4, "t"), days=60):
    w.apply([Command("start_cycle", recipe)])
    tels = []
    while True:
        tels.append(w.step())
        if w.phase == CyclePhase.PRODUCE and w.t_prod >= days * 86400:
            w.apply([Command("stop_production")])
        if w.phase == CyclePhase.IDLE:
            return tels


@pytest.fixture(scope="module")
def field():
    f = FieldSimulator(n_wells=4, seed=11)
    tels = {wid: run_cycle(w) for wid, w in f.wells.items()}
    return f, tels


def test_wells_differ_and_ranges_respected(field):
    f, _ = field
    depths = [w.wp.depth for w in f.wells.values()]
    assert len(set(np.round(depths, 3))) == 4
    for w in f.wells.values():
        assert 17.0 <= w.wp.api <= 19.0
        assert 319.15 <= w.wp.T_R <= 321.15


def test_full_cycle_phases_and_outputs(field):
    f, tels = field
    for wid, w in f.wells.items():
        phases = {t.phase for t in tels[wid]}
        assert {CyclePhase.INJECT, CyclePhase.SOAK, CyclePhase.PRODUCE} <= phases
        assert w.phase == CyclePhase.IDLE
        c = w.cycles[-1]
        assert c.oil_m3 > 100 and c.water_m3 > 50 and c.energy_kwh > 100
        assert 1.0 < c.steam_t / c.oil_m3 < 15.0             # SOR in a plausible band
        cards = [t.card for t in tels[wid] if t.card is not None]
        assert len(cards) >= 100
        assert w.res.E_residual > 0


def test_noise_and_dropout_present(field):
    f, tels = field
    prod = [t for t in tels["W01"] if t.phase == CyclePhase.PRODUCE]
    truth = [h for h in f.wells["W01"].truth if h.phase == "produce"]
    rates = np.array([t.liquid_rate_m3d if t.liquid_rate_m3d is not None else np.nan for t in prod])
    assert np.isnan(rates).sum() > 0                                   # dropouts
    true_rates = np.array([h.q_oil + h.q_water for h in truth])
    ok = ~np.isnan(rates) & (true_rates > 1)
    rel = np.abs(rates[ok] / true_rates[ok] - 1)
    assert 0.01 < np.median(rel) < 0.2                                 # noisy but close
    jitter = [abs((t.t - h.t).total_seconds()) for t, h in zip(prod, truth)]
    assert max(jitter) > 0


def test_labeled_events():
    ev = {"W01": [ScheduledEvent("unseat", T0 + timedelta(days=30), T0 + timedelta(days=31)),
                  ScheduledEvent("gas", T0 + timedelta(days=40), T0 + timedelta(days=43), 0.4)]}
    f = FieldSimulator(n_wells=1, seed=3, extra_events=ev, event_scale=0.0)
    w = f.wells["W01"]
    run_cycle(w, days=60)
    kinds = {lab.kind for lab in w.derived_labels()}
    assert "pump_unseated" in kinds and "gas_interference" in kinds
    faults = {h.fault for h in w.truth if h.phase == "produce"}
    assert {"pump_unseated", "gas_interference", "normal"} <= faults


def test_cooling_event_raises_viscosity_and_float_risk():
    ev = {"W01": [ScheduledEvent("cooling", T0 + timedelta(days=45), T0 + timedelta(days=70), 20.0)]}
    f = FieldSimulator(n_wells=1, seed=3, extra_events=ev, event_scale=0.0)
    w = f.wells["W01"]
    run_cycle(w, days=60)
    prod = [h for h in w.truth if h.phase == "produce"]
    before = np.mean([h.mu_top for h in prod if h.t < T0 + timedelta(days=44)][-48:])
    during = np.mean([h.mu_top for h in prod if h.t > T0 + timedelta(days=46)][:48])
    assert during > 1.5 * before
    rfi_b = np.mean([h.rfi_true for h in prod if h.t < T0 + timedelta(days=44)][-48:])
    rfi_d = np.mean([h.rfi_true for h in prod if h.t > T0 + timedelta(days=46)][:48])
    assert rfi_d > rfi_b


def test_seeded_determinism():
    a = FieldSimulator(n_wells=2, seed=5)
    b = FieldSimulator(n_wells=2, seed=5)
    ta = run_cycle(a.wells["W02"], days=10)
    tb = run_cycle(b.wells["W02"], days=10)
    assert [x.liquid_rate_m3d for x in ta] == [x.liquid_rate_m3d for x in tb]


def test_replay_roundtrip_csv_and_stream(field, tmp_path):
    _, tels = field
    recs = tels["W02"][:200]
    p = tmp_path / "hist.csv"
    save_csv(recs, p)
    back = load_history(p)
    assert len(back) == len(recs)
    assert back[150].liquid_rate_m3d == recs[150].liquid_rate_m3d
    assert (back[150].card is None) == (recs[150].card is None)

    async def collect():
        return [r async for r in stream(back[:20], seconds_per_day=0.0)]

    assert len(asyncio.run(collect())) == 20

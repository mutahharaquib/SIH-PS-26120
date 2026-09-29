"""Live field runner: simulated field (replayed telemetry) + one twin per well, on a
uniform simulated clock, with play/pause/speed control. Twins see Telemetry only."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml

from core.config import get_config
from core.state import WellState
from core.store import Store
from core.units import HOUR
from engine.twin import WellTwin
from evaluation.classifier_eval import ensure_model
from simulation.field_sim import T0, FieldSimulator
from simulation.labels import ScheduledEvent

SCENARIOS = Path(__file__).resolve().parents[2] / "scenarios"


class FieldRunner:
    def __init__(self, n_wells: int | None = None, seed: int | None = None, db_url: str = "sqlite:///:memory:"):
        self.cfg = get_config()
        self.n_wells = n_wells or int(self.cfg.v("field.simulation.n_wells"))
        self.seed = seed
        self.store = Store(db_url)
        self.lock = threading.RLock()
        self.playing = False
        self.seconds_per_day = float(self.cfg.v("field.simulation.replay_seconds_per_day"))
        self.dt_h = 1.0
        self.tick = 0
        self.subscribers: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.scenario: dict | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.clf = ensure_model(quick=True)
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, scenario: dict | None = None, mode: str = "advisory") -> None:
        with self.lock:
            self.scenario = scenario
            extra = None
            n, seed = self.n_wells, self.seed
            if scenario:
                n, seed = scenario.get("n_wells", n), scenario.get("seed", seed)
                extra = {scenario["well"]: [ScheduledEvent(e["kind"], T0 + timedelta(days=e["start_day"]),
                                                           T0 + timedelta(days=e["end_day"]), e.get("magnitude", 0.0))
                                            for e in scenario["events"]]}
                mode = scenario.get("live", {}).get("mode", mode)
                self.seconds_per_day = scenario.get("live", {}).get("seconds_per_day", self.seconds_per_day)
            self.field = FieldSimulator(self.cfg, n_wells=n, seed=seed, extra_events=extra)
            self.twins = {wid: WellTwin(meta, self.cfg, self.clf, mode=mode, bo_fast=True, auto_cycle=True)  # type: ignore[arg-type]
                          for wid, meta in self.field.meta.items()}
            self.tick = 0
            self.sim_t = T0
            self.step(1)

    def load_scenario(self, name: str = "demo_float") -> dict:
        sc = yaml.safe_load((SCENARIOS / f"{name}.yaml").read_text())
        self.reset(sc)
        return sc

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="field-runner")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.playing:
                time.sleep(0.1)
                continue
            t0 = time.time()
            self.step(1)
            budget = self.seconds_per_day * self.dt_h / 24.0
            time.sleep(max(0.0, budget - (time.time() - t0)))

    # ------------------------------------------------------------------ stepping
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            with self.lock:
                for wid, sim in self.field.wells.items():
                    tel = sim.step(self.dt_h * HOUR)
                    cmds = self.twins[wid].on_telemetry(tel)
                    sim.apply(cmds)
                self.tick += 1
                self.sim_t = next(iter(self.field.wells.values())).t
                if self.tick % 6 == 0:
                    for tw in self.twins.values():
                        if tw.state is not None:
                            self.store.save_state(tw.state)
            self._broadcast()

    def _broadcast(self) -> None:
        if not self.subscribers or self.loop is None:
            return
        msg = {"type": "tick", "t": self.sim_t.isoformat(), "tick": self.tick, "wells": self.summary()}
        for q in list(self.subscribers):
            try:
                self.loop.call_soon_threadsafe(q.put_nowait, msg)
            except RuntimeError:
                pass

    # ------------------------------------------------------------------ views
    def state(self, wid: str) -> WellState | None:
        return self.twins[wid].state

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        with self.lock:
            for wid, tw in self.twins.items():
                s = tw.state
                if s is None:
                    continue
                rows.append({
                    "well_id": wid, "phase": s.cycle.phase.value, "cycle": s.cycle.cycle_index,
                    "q_oil_m3d": s.econ.q_oil.value, "sor": s.econ.sor_cycle.value,
                    "kwh_per_bbl": s.econ.kwh_per_bbl.value, "rfi": s.pump.rfi.value, "spm": s.pump.spm.value,
                    "downstroke": s.pump.vfd_profile.downstroke_factor(), "fillage": s.pump.fillage.value,
                    "fault": s.pump.fault_class, "risk": s.risk_score.value, "mode": s.control_mode,
                    "alarms": [a.model_dump(mode="json") for a in s.alarms[-3:]],
                    "T_avg_K": s.reservoir.T_avg_heated.value, "g_star": s.econ.g_star.value if s.econ.g_star else None,
                    "stop_recommended": tw.stop_recommended, "pending": len(tw.edge.pending_list()),
                    "tiers": {"q_oil": "A", "sor": "B", "kwh_per_bbl": "B", "rfi": "B", "risk": "B", "fault": "C"},
                })
        order = sorted(rows, key=lambda r: -r["risk"])
        for i, r in enumerate(order):
            r["risk_rank"] = i + 1
        return rows

    def control(self) -> dict:
        return {"playing": self.playing, "seconds_per_day": self.seconds_per_day, "sim_time": self.sim_t.isoformat(),
                "tick": self.tick, "scenario": (self.scenario or {}).get("name"), "n_wells": len(self.twins)}

"""Simulated OPC UA server (BUILD_SPEC §15): per-well telemetry nodes (read) and VFD
setpoint nodes (write: SPM, downstroke speed profile, run/stop), backed by the field
simulator. Run: python -m integration.opcua_server_sim --wells 4 --port 4840
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml
from asyncua import Server, ua

from core.state import SpeedProfile
from core.telemetry import Command, Recipe
from core.units import HOUR
from simulation.field_sim import FieldSimulator

MAP_PATH = Path(__file__).resolve().parents[2] / "config" / "opcua_map.yaml"


def load_map(path: Path = MAP_PATH) -> dict:
    return yaml.safe_load(path.read_text())


class OPCUAFieldServer:
    def __init__(self, n_wells: int = 4, seed: int = 1, port: int = 4840, seconds_per_hour: float = 0.5):
        self.m = load_map()
        self.field = FieldSimulator(n_wells=n_wells, seed=seed)
        self.url = f"opc.tcp://127.0.0.1:{port}/sih26120/"
        self.seconds_per_hour = seconds_per_hour
        self.server = Server()
        self.nodes: dict[str, dict[str, object]] = {}
        self.ns = 0
        self.ticks = 0

    async def init(self) -> None:
        await self.server.init()
        self.server.set_endpoint(self.url)
        self.server.set_server_name("SIH26120 CSS-SRP field simulator")
        self.ns = await self.server.register_namespace(self.m["namespace"])
        objs = self.server.nodes.objects
        for wid in self.field.wells:
            o = await objs.add_object(self.ns, self.m["well_object"].format(well_id=wid))
            d: dict[str, object] = {}
            for key, tag in self.m["telemetry"].items():
                init = "idle" if key == "phase" else (False if key == "running" else 0.0)
                d[key] = await o.add_variable(self.ns, tag, init)
            for key, tag in self.m["setpoints"].items():
                init = True if key == "run" else (4.0 if key == "spm" else 1.0)
                v = await o.add_variable(self.ns, tag, init)
                await v.set_writable()
                d["sp_" + key] = v
            self.nodes[wid] = d

    async def tick(self) -> None:
        for wid, sim in self.field.wells.items():
            nd = self.nodes[wid]
            spm = float(await nd["sp_spm"].read_value())            # type: ignore[attr-defined]
            ds = float(await nd["sp_downstroke"].read_value())      # type: ignore[attr-defined]
            cmds = [Command("set_spm", spm, "opcua"),
                    Command("set_profile", SpeedProfile(downstroke=[1.0, ds, ds, 1.0]), "opcua")]
            if sim.phase.value == "idle":
                cmds.append(Command("start_cycle", Recipe(3000, 90, 5, "opcua-default"), "opcua"))
            sim.apply(cmds)
            tel = sim.step(HOUR)
            for key in self.m["telemetry"]:
                val = getattr(tel, key, None)
                if key == "phase":
                    val = tel.phase.value
                if val is None:
                    continue
                if isinstance(val, bool):
                    await nd[key].write_value(val)                   # type: ignore[attr-defined]
                elif isinstance(val, (int, float)):
                    await nd[key].write_value(float(val), ua.VariantType.Double)  # type: ignore[attr-defined]
                else:
                    await nd[key].write_value(str(val))              # type: ignore[attr-defined]
        self.ticks += 1

    async def run(self, max_ticks: int | None = None) -> None:
        async with self.server:
            while max_ticks is None or self.ticks < max_ticks:
                await self.tick()
                await asyncio.sleep(self.seconds_per_hour)


async def _main(n: int, port: int, sph: float) -> None:
    s = OPCUAFieldServer(n, port=port, seconds_per_hour=sph)
    await s.init()
    print("OPC UA field simulator at", s.url, flush=True)
    await s.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", type=int, default=4)
    ap.add_argument("--port", type=int, default=4840)
    ap.add_argument("--seconds-per-hour", type=float, default=0.5)
    a = ap.parse_args()
    asyncio.run(_main(a.wells, a.port, a.seconds_per_hour))

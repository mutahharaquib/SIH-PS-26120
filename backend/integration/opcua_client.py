"""Twin-side OPC UA client (BUILD_SPEC §15). Advisory mode is read-only; supervised and
autonomous modes may write VFD setpoints. The tag mapping comes from config/opcua_map.yaml
so a real SCADA tag list can be dropped in. Modbus (optional) would reuse the same map."""

from __future__ import annotations

from pathlib import Path

import yaml
from asyncua import Client, ua

MAP_PATH = Path(__file__).resolve().parents[2] / "config" / "opcua_map.yaml"


class TwinOPCUAClient:
    def __init__(self, url: str, mode: str = "advisory", map_path: Path = MAP_PATH):
        self.url = url
        self.mode = mode
        self.m = yaml.safe_load(map_path.read_text())
        self.client = Client(url)
        self.ns = 0

    async def __aenter__(self) -> "TwinOPCUAClient":
        await self.client.connect()
        self.ns = await self.client.get_namespace_index(self.m["namespace"])
        return self

    async def __aexit__(self, *exc) -> None:
        await self.client.disconnect()

    async def _node(self, well_id: str, tag: str):
        obj = self.m["well_object"].format(well_id=well_id)
        return await self.client.nodes.objects.get_child([f"{self.ns}:{obj}", f"{self.ns}:{tag}"])

    async def read_telemetry(self, well_id: str) -> dict:
        out = {}
        for key, tag in self.m["telemetry"].items():
            out[key] = await (await self._node(well_id, tag)).read_value()
        return out

    async def read_setpoints(self, well_id: str) -> dict:
        return {k: await (await self._node(well_id, tag)).read_value() for k, tag in self.m["setpoints"].items()}

    async def write_setpoints(self, well_id: str, spm: float | None = None, downstroke: float | None = None,
                              run: bool | None = None) -> None:
        if self.mode == "advisory":
            raise PermissionError("advisory mode is read-only: no setpoint writes")
        if spm is not None:
            await (await self._node(well_id, self.m["setpoints"]["spm"])).write_value(float(spm), ua.VariantType.Double)
        if downstroke is not None:
            await (await self._node(well_id, self.m["setpoints"]["downstroke"])).write_value(
                float(downstroke), ua.VariantType.Double)
        if run is not None:
            await (await self._node(well_id, self.m["setpoints"]["run"])).write_value(bool(run))

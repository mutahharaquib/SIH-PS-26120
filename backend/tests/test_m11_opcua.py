"""M11 — OPC UA simulated integration: read telemetry, write setpoints, advisory is read-only."""

import asyncio
import socket

import pytest

from integration.opcua_client import TwinOPCUAClient
from integration.opcua_server_sim import OPCUAFieldServer


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_opcua_roundtrip():
    async def scenario():
        srv = OPCUAFieldServer(n_wells=1, seed=2, port=free_port(), seconds_per_hour=0.0)
        await srv.init()
        async with srv.server:
            for _ in range(3):
                await srv.tick()
            async with TwinOPCUAClient(srv.url, mode="advisory") as ro:
                tel = await ro.read_telemetry("W01")
                assert tel["phase"] in ("inject", "soak", "produce", "idle")
                with pytest.raises(PermissionError):
                    await ro.write_setpoints("W01", spm=3.0)
            async with TwinOPCUAClient(srv.url, mode="supervised") as rw:
                await rw.write_setpoints("W01", spm=3.3, downstroke=0.7)
                sp = await rw.read_setpoints("W01")
                assert sp["spm"] == pytest.approx(3.3) and sp["downstroke"] == pytest.approx(0.7)
            await srv.tick()
            sim = srv.field.wells["W01"]
            assert sim.spm == pytest.approx(3.3)
            assert sim.profile.downstroke_factor() == pytest.approx(0.7)

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))

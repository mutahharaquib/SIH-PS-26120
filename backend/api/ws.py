"""WebSocket: live state + alarm stream."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.deps import runner

router = APIRouter()


@router.websocket("/ws/telemetry")
async def telemetry(ws: WebSocket) -> None:
    await ws.accept()
    r = runner()
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    r.subscribers.add(q)
    try:
        await ws.send_json({"type": "hello", "control": r.control(), "wells": r.summary()})
        while True:
            msg = await q.get()
            while q.qsize() > 5:            # drop stale frames if the client is slow
                msg = q.get_nowait()
            await ws.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        r.subscribers.discard(q)

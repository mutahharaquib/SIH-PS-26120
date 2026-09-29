"""FastAPI app. Run: uvicorn api.main:app --port 8000 (from backend/)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import routes_eval, routes_wells, ws
from api.deps import runner
from evaluation.harness import REPORTS


@asynccontextmanager
async def lifespan(app: FastAPI):
    r = runner()
    r.start(asyncio.get_running_loop())
    yield
    r.stop()


app = FastAPI(title="CSS + SRP Digital Twin (SIH 26120)", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(routes_wells.router)
app.include_router(routes_eval.router)
app.include_router(ws.router)
REPORTS.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(REPORTS)), name="reports")

FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="ui")

"""Telemetry replay (BUILD_SPEC §5): stream recorded or live-simulated telemetry at a
configurable speed (e.g. 1 simulated day per 10 s). Real historical data in the same
schema (CSV or Parquet, one row per well-tick) can be replayed through the same path.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import AsyncIterator, Iterable

from core.telemetry import Telemetry

JSON_FIELDS = ("profile", "card", "lab")


def to_row(t: Telemetry) -> dict:
    d = t.model_dump(mode="json")
    for k in JSON_FIELDS:
        d[k] = json.dumps(d[k])
    return d


def from_row(row: dict) -> Telemetry:
    d = {k: (None if v in ("", "None") else v) for k, v in row.items()}
    for k in JSON_FIELDS:
        if d.get(k) is not None:
            d[k] = json.loads(d[k])
    if d.get("lab") is None:
        d["lab"] = []
    if d.get("profile") is None:
        d.pop("profile", None)
    return Telemetry.model_validate(d)


def save_csv(records: Iterable[Telemetry], path: str | Path) -> None:
    rows = [to_row(r) for r in records]
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_history(path: str | Path) -> list[Telemetry]:
    """Load historical telemetry (CSV or Parquet) in the Telemetry schema."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":  # optional dependency
        import pandas as pd  # type: ignore[import-untyped]

        return [from_row({k: ("" if v is None else v) for k, v in r.items()})
                for r in pd.read_parquet(p).astype(str).to_dict("records")]
    with open(p, encoding="utf-8") as fh:
        return [from_row(r) for r in csv.DictReader(fh)]


async def stream(records: Iterable[Telemetry], seconds_per_day: float = 10.0) -> AsyncIterator[Telemetry]:
    """Yield telemetry, sleeping in proportion to simulated time between records."""
    last = None
    for r in records:
        if last is not None and seconds_per_day > 0:
            dt_days = max((r.t - last).total_seconds(), 0.0) / 86400.0
            await asyncio.sleep(dt_days * seconds_per_day)
        last = r.t
        yield r

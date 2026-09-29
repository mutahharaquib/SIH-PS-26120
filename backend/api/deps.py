"""Shared API state and helpers (unit conversion at the boundary)."""

from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException

from api.runner import FieldRunner
from core.units import to_field

_runner: FieldRunner | None = None


def runner() -> FieldRunner:
    global _runner
    if _runner is None:
        n = int(os.environ.get("TWIN_N_WELLS", "0")) or None
        _runner = FieldRunner(n_wells=n, db_url=os.environ.get("TWIN_DB", "sqlite:///:memory:"))
    return _runner


def twin_or_404(wid: str):
    r = runner()
    if wid not in r.twins:
        raise HTTPException(404, f"unknown well {wid}")
    return r.twins[wid]


def convert_units(obj: Any) -> Any:
    """Walk a JSON-able structure and convert every {value, unit} Quantity to field units."""
    if isinstance(obj, dict):
        if "value" in obj and "unit" in obj and isinstance(obj.get("value"), (int, float)):
            v, u = to_field(float(obj["value"]), obj["unit"])
            out = dict(obj, value=v, unit=u)
            if isinstance(obj.get("uncertainty"), (int, float)):
                out["uncertainty"] = to_field(float(obj["uncertainty"]), obj["unit"])[0] - to_field(0.0, obj["unit"])[0]
            return out
        return {k: convert_units(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_units(v) for v in obj]
    return obj

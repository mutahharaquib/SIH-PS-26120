"""Rod Float Index (BUILD_SPEC §6.1).

RFI = max over the downstroke of V_polished_rod(t) / V_fall_min
RFI >= 1 -> float expected. V_fall_min uses the minimum over depth (slowest-falling
segment governs), which makes the index conservative; evaluation reports both the
lead time to labeled float onset and the false-alarm rate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


def rfi(max_down_speed: float, v_fall_min: float) -> float:
    return float(max_down_speed / max(v_fall_min, 1e-9))


def level(value: float, warn: float, alarm: float) -> str:
    return "alarm" if value >= alarm else ("warn" if value >= warn else "ok")


@dataclass
class RFITracker:
    warn: float = 0.8
    alarm: float = 1.0
    window: int = 24
    hist: deque = field(default_factory=lambda: deque(maxlen=2000))

    def update(self, t: datetime, value: float, spm: float) -> dict:
        self.hist.append((t, value, spm))
        return {"rfi": value, "slope_per_stroke": self.slope_per_stroke(), "level": level(value, self.warn, self.alarm)}

    def slope_per_stroke(self) -> float:
        """Rate of change of RFI per stroke over the recent window (least squares)."""
        pts = list(self.hist)[-self.window:]
        if len(pts) < 3:
            return 0.0
        t0 = pts[0][0]
        x = np.array([(p[0] - t0).total_seconds() / 60.0 for p in pts])   # minutes
        y = np.array([p[1] for p in pts])
        if np.ptp(x) <= 0:
            return 0.0
        slope_per_min = float(np.polyfit(x, y, 1)[0])
        spm = max(float(np.mean([p[2] for p in pts])), 1e-3)
        return slope_per_min / spm

    def series(self) -> list[tuple[datetime, float]]:
        return [(t, v) for t, v, _ in self.hist]


def lead_times(times: list[datetime], values: list[float], onsets: list[datetime], warn: float,
               lookback_h: float = 72.0) -> list[float | None]:
    """For each labeled float onset, hours between the RFI first crossing `warn` (within the
    preceding lookback window, staying above until onset) and the onset. None = missed."""
    out: list[float | None] = []
    ts = np.array([t.timestamp() for t in times])
    vs = np.asarray(values)
    for on in onsets:
        o = on.timestamp()
        mask = (ts <= o) & (ts >= o - lookback_h * 3600)
        if not mask.any():
            out.append(None)
            continue
        idx = np.where(mask)[0]
        above = vs[idx] >= warn
        if not above[-1]:
            out.append(None)
            continue
        k = len(above) - 1
        while k > 0 and above[k - 1]:
            k -= 1
        out.append((o - ts[idx[k]]) / 3600.0)
    return out

"""Card features (BUILD_SPEC §6.2): downhole card normalised to [0,1] in position and
load; area, fillage estimate, load range, corner angles, Fourier descriptors (first 16
of the closed-contour complex series) and a 32x32 occupancy grid. A few surface-card
features (normalised minimum load, zero-load fraction) are appended because rod float
is visible at the carrier bar before it is visible downhole.
"""

from __future__ import annotations

import numpy as np

from physics.energy import card_area

N_FD = 16
GRID = 32


def normalize(pos, load) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(pos, dtype=float)
    y = np.asarray(load, dtype=float)
    x = (x - x.min()) / max(np.ptp(x), 1e-9)
    y = (y - y.min()) / max(np.ptp(y), 1e-9)
    return x, y


def fillage_estimate(pos, load) -> float:
    """Effective / gross plunger stroke: position (from the bottom) where the downstroke
    load first falls below mid-range, divided by the gross stroke."""
    x, y = normalize(pos, load)
    top = int(np.argmax(x))
    xs = np.roll(x, -top)
    ys = np.roll(y, -top)
    bottom = int(np.argmin(xs))
    down_x, down_y = xs[: bottom + 1], ys[: bottom + 1]
    below = np.where(down_y < 0.5)[0]
    if below.size == 0:
        return 0.0
    return float(np.clip(down_x[below[0]], 0.0, 1.0))


def _resample_contour(x: np.ndarray, y: np.ndarray, n: int = 64) -> np.ndarray:
    z = x + 1j * y
    z = np.append(z, z[0])
    seg = np.abs(np.diff(z))
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.zeros(n, dtype=complex)
    si = np.linspace(0, s[-1], n, endpoint=False)
    return np.interp(si, s, z.real) + 1j * np.interp(si, s, z.imag)


def fourier_descriptors(x: np.ndarray, y: np.ndarray, n: int = N_FD) -> np.ndarray:
    z = _resample_contour(x, y)
    F = np.fft.fft(z - z.mean())
    mag = np.abs(np.concatenate([F[1: n // 2 + 1], F[-(n // 2):]]))
    return mag / max(np.abs(F[1]), 1e-9)


def occupancy_grid(x: np.ndarray, y: np.ndarray, n: int = GRID) -> np.ndarray:
    z = _resample_contour(x, y, 256)
    g = np.zeros((n, n))
    ix = np.clip((z.real * (n - 1)).round().astype(int), 0, n - 1)
    iy = np.clip((z.imag * (n - 1)).round().astype(int), 0, n - 1)
    g[iy, ix] = 1.0
    return g


def corner_angles(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interior angles (rad) at the contour points nearest the four unit-square corners."""
    pts = np.column_stack([x, y])
    out = []
    n = len(x)
    for cx, cy in ((0, 0), (1, 0), (1, 1), (0, 1)):
        i = int(np.argmin((x - cx) ** 2 + (y - cy) ** 2))
        k = max(1, n // 25)
        a = pts[(i - k) % n] - pts[i]
        b = pts[(i + k) % n] - pts[i]
        cosang = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
        out.append(float(np.arccos(np.clip(cosang, -1, 1))))
    return np.array(out)


def features(dh_pos, dh_load, surf_pos=None, surf_load=None, F_ref: float | None = None) -> dict[str, float | np.ndarray]:
    x, y = normalize(dh_pos, dh_load)
    load = np.asarray(dh_load, dtype=float)
    ref = F_ref or max(np.ptp(load), 1.0)
    f: dict[str, float | np.ndarray] = {
        "area": card_area(x, y),
        "fillage": fillage_estimate(dh_pos, dh_load),
        "load_range": float(np.ptp(load) / ref),
        "angles": corner_angles(x, y),
        "fd": fourier_descriptors(x, y),
        "grid": occupancy_grid(x, y),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }
    if surf_pos is not None and surf_load is not None:
        sl = np.asarray(surf_load, dtype=float)
        f["surf_min_ratio"] = float(sl.min() / max(sl.max(), 1.0))
        f["surf_zero_frac"] = float(np.mean(sl < 0.03 * max(sl.max(), 1.0)))
        f["surf_area_ratio"] = card_area(*normalize(surf_pos, surf_load))
    else:
        f["surf_min_ratio"] = f["surf_zero_frac"] = f["surf_area_ratio"] = 0.0
    return f


def pooled_grid(g: np.ndarray, k: int = 4) -> np.ndarray:
    n = g.shape[0] // k
    return g[: n * k, : n * k].reshape(n, k, n, k).mean(axis=(1, 3))


def vectorize(f: dict) -> np.ndarray:
    return np.concatenate([
        [f["area"], f["fillage"], f["load_range"], f["y_mean"], f["y_std"], f["surf_min_ratio"],
         f["surf_zero_frac"], f["surf_area_ratio"]],
        np.asarray(f["angles"]), np.asarray(f["fd"]), pooled_grid(np.asarray(f["grid"])).ravel(),
    ])

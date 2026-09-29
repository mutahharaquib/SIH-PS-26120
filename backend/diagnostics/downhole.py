"""Surface card -> downhole card with the twin's rod model (diagnostic wave equation)."""

from __future__ import annotations

import numpy as np

from physics.rod_string import RodGrid, diagnostic, downsample_card


def downhole_card(grid: RodGrid, surf_pos, surf_load, period: float, c_nodes: np.ndarray, w_nodes: np.ndarray,
                  n_harmonics: int = 20, n_out: int = 200) -> tuple[list[float], list[float]]:
    d = diagnostic(grid, np.asarray(surf_pos, dtype=float), np.asarray(surf_load, dtype=float), period, c_nodes,
                   w_nodes, n_harmonics=n_harmonics)
    return downsample_card(d.pump_pos, d.pump_load, n_out)

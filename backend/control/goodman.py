"""Modified Goodman allowable stress (BUILD_SPEC §9.1).

[VERIFY resolved] API RP 11BR modified Goodman diagram:
    S_allow = (T/4 + 0.5625 * S_min) * SF
T = minimum tensile strength of the rod grade, S_min = minimum stress in the stroke,
SF = service factor. Checked in tests against the textbook grade-D example.
"""

from __future__ import annotations

import numpy as np


def s_allow(T_tensile: float, s_min, sf: float):
    """Allowable max stress. The modified Goodman diagram is defined for tensile S_min;
    compressive minima are clipped to 0 here and reported separately (buckling risk)."""
    return (T_tensile / 4.0 + 0.5625 * np.maximum(np.asarray(s_min, dtype=float), 0.0)) * sf


def stress_ratio(s_max, s_min, T_tensile: float, sf: float):
    """S_max / S_allow per taper section."""
    return np.asarray(s_max, dtype=float) / s_allow(T_tensile, s_min, sf)

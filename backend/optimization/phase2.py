"""Phase 2 — NOT built (BUILD_SPEC §17). Interfaces only; excluded from all metrics.

- Depletion-aware dynamic-programming cut-off and multi-cycle lookahead.
- Survival-model failure prediction.
- Field-level optimization across wells sharing a steam generator.
"""

from __future__ import annotations

from typing import Any, Protocol

PHASE2 = True


class MultiCycleLookahead(Protocol):
    def plan(self, well_state: Any, horizon_cycles: int) -> list[Any]: ...


class DepletionAwareDPCutoff(Protocol):
    def cutoff(self, well_state: Any, value_to_go: Any) -> float: ...


class SurvivalFailureModel(Protocol):
    def hazard(self, well_history: Any) -> float: ...


class FieldSteamAllocator(Protocol):
    def allocate(self, wells: list[Any], generator_capacity_t_d: float) -> dict[str, Any]: ...


def not_built(*_: Any, **__: Any) -> None:
    raise NotImplementedError("Phase 2 feature (BUILD_SPEC §17) — interface only, not built")

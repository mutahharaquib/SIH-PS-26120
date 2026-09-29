from datetime import datetime, timezone

import pytest

from core.state import (
    CycleState,
    CyclePhase,
    EconomicState,
    PumpState,
    ReservoirState,
    SpeedProfile,
    Tier,
    WellboreState,
    WellState,
    q,
)


@pytest.fixture
def sample_state() -> WellState:
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    b = lambda v, u: q(v, u, Tier.B, "test")  # noqa: E731
    return WellState(
        well_id="W01",
        t=t,
        cycle=CycleState(
            cycle_index=1, phase=CyclePhase.PRODUCE, phase_start=t,
            steam_injected_kg=b(3e6, "kg"), injection_pressure=b(9e6, "Pa"), soak_duration=b(5, "d"),
            cumulative_oil_m3=b(100, "m3"), cumulative_water_m3=b(200, "m3"),
        ),
        reservoir=ReservoirState(T_avg_heated=b(400, "K"), r_heated=b(10, "m"), p_reservoir=b(6e6, "Pa"),
                                 stimulation_ratio=b(3, "-")),
        wellbore=WellboreState(sandface_steam_quality=b(0.6, "-"), heat_loss_rate=b(1e5, "W"),
                               T_tubing_profile=[(0.0, 330.0)], mu_at_pump=b(0.1, "Pa.s"), mu_profile=[(0.0, 1.0)]),
        pump=PumpState(spm=b(4, "1/min"), stroke_length=b(2.5, "m"), vfd_profile=SpeedProfile(),
                       intake_pressure=b(6e5, "Pa"), fillage=b(0.9, "-"), volumetric_efficiency=b(0.8, "-"),
                       rfi=b(0.3, "-"), peak_rod_stress_ratio=b(0.7, "-")),
        econ=EconomicState(q_oil=b(10, "m3/d"), q_net_value_rate=b(3000, "USD/d"), kwh_per_bbl=b(3, "kWh/bbl"),
                           sor_cycle=b(3, "-")),
        risk_score=b(0.1, "-"),
    )

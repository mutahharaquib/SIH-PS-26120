# Coupled Well-to-Surface Digital Twin — CSS + SRP (SIH 26120, Baghewala)

A working prototype of a digital twin for heavy-oil wells produced by **Cyclic Steam
Stimulation (CSS)** and lifted by **Sucker Rod Pumps (SRP)**, optimizing both as one coupled
system. The build follows `claude.md` (the build specification), milestones M1–M11.

- **Forward chain:** steam & soak → heated zone (Marx-Langenheim + Boberg-Lantz) → temperature →
  viscosity (Walther, RLS-calibrated) → tubing temperature profile → rod drag / damping →
  wave-equation cards, Rod Float Index, stresses, pump fillage.
- **Feedback chain:** SPM / VFD downstroke profile → displacement → intake pressure & drawdown →
  inflow → cooling & decline → live cut-off → next-cycle recipe (Bayesian optimizer).

Every module reads and writes one shared `WellState`. Every number carries a confidence
tier (A/B/C) and a provenance record, plus placeholder and extrapolation flags.

> **Validation is physics- and synthetic-data only.** No Oil India data is bundled. All
> non-problem-statement parameters are `PLACEHOLDER - calibrate` in `config/` and the UI marks
> every value derived from them with a **PH** badge. Treat results as relative comparisons.

## Quick start

```bash
# backend (Python 3.11+)
cd backend
pip install -r requirements.txt numba          # numba JIT-compiles the wave-equation kernel
python -m pytest -q                             # all milestone tests
python -m evaluation.classifier_eval           # trains the card classifier (also done lazily)
uvicorn api.main:app --port 8000               # REST + WebSocket, live simulated field

# frontend
cd ../frontend
npm install && npm run dev                      # http://localhost:5173 (proxies /api, /ws to :8000)

# evaluation (baseline vs twin, both objective settings) -> reports/<run_id>/
cd ../backend
python -m evaluation.harness --wells 6 --days 450
python -m evaluation.harness --scenario ../scenarios/demo_float.yaml

# OPC UA field simulator (M11)
python -m integration.opcua_server_sim --wells 4 --port 4840
```

Or run everything with `docker compose up`: backend on :8000, UI on :8080 and the OPC UA simulator
on :4840. Docker was not available on the build machine, so the compose files are untested.

In the UI, press **Demo scenario** to load `scenarios/demo_float.yaml` (a seeded cold snap on W01),
switch W01 to *autonomous* or *supervised* in the **Decision console**, and press **Play**.
The **Evaluation** tab runs the same scenario under the baseline and the twin and shows the
avoided-event impact.

## Repository layout

```
config/        field, economics, rods, controller, optimizer (every leaf: value, unit, source)
               opcua_map.yaml (SCADA tag mapping)
scenarios/     demo_float.yaml (seeded demo)
backend/
  core/        state (WellState/Quantity/Provenance/Tier), config, units, steam (IAPWS-IF97),
               bus, store (SQLAlchemy), telemetry (the only field<->twin data contract)
  physics/     viscosity, wellbore, reservoir, inflow, pumping_unit, drag, rod_string (numba),
               pump, energy, injection, cycle (fast cycle forecaster), srp_model, params
  diagnostics/ rfi, card_features, classifier, efficiency, downhole
  calibration/ rls (Walther RLS), inflow_refit (NLS)
  optimization/objective, cutoff (sim + live), cycle_bo (GP/logEI), baseline, phase2 (stubs)
  control/     srp_controller, goodman, safety, edge_sim
  risk/        failure_risk (Miner + event rates; ranking only)
  engine/      twin.py - per-well twin (consumes Telemetry only)
  simulation/  field_sim (ground truth), noise, labels, replay, truth (hidden), card_dataset
  evaluation/  harness, metrics, classifier_eval
  integration/ opcua_server_sim, opcua_client
  api/         main, routes_wells, routes_eval, ws, runner
  tests/       test_m1 ... test_m11
frontend/      React + TS + Vite + Tailwind + Recharts + Zustand
```

## [VERIFY] items and how they were resolved

| Item | Resolution | Guarding test |
|---|---|---|
| Ramey f(t) (§4.2) | Hasan-Kabir (1991) full-range fit. Its long-time branch equals Ramey's `ln(2√(αt)/r_cw) − 0.290` to 3e-3, and it stays positive at early time. | `test_ramey_long_time_limit`, `test_heat_loss_decreases_with_time` |
| Steam march (§4.2) | Marched on mixture enthalpy, `w dh/dz = −Q' + w g`. This reduces to `w h_fg dx/dz = −Q'` at constant pressure. Pressure is hydrostatic (homogeneous density) minus Darcy friction. | `test_quality_decreases_with_depth_and_energy_closes` (closure < 0.1 %), `test_no_heat_loss_when_U_zero` |
| Marx-Langenheim (§4.3) | Implemented as specified; `G(t_D)` uses `erfcx` for stability. | `test_ml_dimensions_and_small_time_limit` (t_D dimensionless; small-t energy balance) |
| Boberg-Lantz f_HD, f_VD (§4.3) | Computed from the exact 1-D conduction solutions (the option the spec allows). **f_VD** = slab average `erf(X) + (e^{−X²} − 1)/(X√π)`; **f_HD** = cylinder average `1 − e^{−y}[I0(y) + I1(y)]` (Carslaw & Jaeger). f_PD = ½ E_removed / E_heated. | `test_f_VD_matches_numerical_heat_kernel` (quadrature), `test_f_HD_matches_monte_carlo_heat_kernel`, `test_T_avg_bounded` (hypothesis), `test_T_avg_tends_to_T_R` |
| Wave equation (§4.7) | Lumped-mass Gibbs form with central differences for inertia and damping, and tapers by element. The diagnostic mode is an Everitt-Jennings style downward march of the same discrete equations, which is exact d'Alembert when `a·dt = dz`. The forward model runs at CFL 0.9: at exactly 1 it is marginal with a nonlinear pump boundary. | `test_roundtrip_undamped_recovers_pump_load` (1e-6), `test_roundtrip_damped_periodic`, `test_stress_at_taper_is_load_over_local_area`, `test_energy_decays_with_damping`, `test_cfl_violation_raises` |
| Goodman (§9.1) | `S_allow = (T/4 + 0.5625·S_min)·SF` (API RP 11BR). Compressive S_min is clipped to 0, since the diagram is defined for tension. | `test_goodman_textbook` |

## Modelling choices beyond the spec (all documented, placeholder-parameterised)

- **Pump boundary condition:** a barrel-pressure model. Valves open on pressure, the liquid is compressible, and the gas is isothermal. A first direction-relay model caused limit-cycle chatter against the elastic rod, which this removes. Fluid pound, gas interference and the unseated pump emerge without special cases.
- **Rod float** emerges at the carrier bar: the clamp separates when the polished-rod load would go negative, and re-contact produces the impact load.
- **Damping floor** of 0.8 s⁻¹ (Gibbs dimensionless damping D ≈ 0.025), added to viscous rod drag.
- **Heated-zone oil depletion:** `SR_eff = 1 + (SR − 1)·(1 − N_p/N_rec)^n`. Without it, the Boberg-Lantz stimulation ratio saturates and the cycle never declines, so the cut-off would be meaningless.
- **Depletion:** a tank model on oil voidage (injected water assumed produced back).
- **Emulsion viscosity:** Einstein-type factor below 60 % water cut, and a brine-continuous fraction above it.
- **Optimizer:** BoTorch is optional. The default is a scikit-learn GP (Matern 5/2) with log-EI over Sobol candidates, seeded (the spec permits a fallback).
- **SRP feedforward** uses inflow at the *target* drawdown, i.e. the steady state once the annulus is drawn down.
- **Live cut-off** is evaluated on the calibrated inflow potential, so pump-side transients (gas lock, unseat, downtime) are not mistaken for reservoir decline. It also waits for the forecast's main peak.
- **Controller** shapes the downstroke relative to the *applied* profile and ramps within the autonomous bands. SPM is reduced for float only after shaping is exhausted, and the VFD minimum frequency bounds the shaping.
- **Classifier cross-check:** an `pump_unseated` label is withheld when the measured volumetric efficiency is above 0.3 (an unseated pump cannot deliver liquid).

## Placeholders

Run `GET /config` or open the **About** tab for the full list. Placeholders include all
reservoir, wellbore, drivetrain, pump, noise, event, economic and controller-tuning values; the
lab viscosity curve; rod grade and taper; and the S-N proxy. Tier A inputs are the API gravity
range, T_R, the formation and the rod minimum tensile strengths (API Spec 11B).

## Known limitations

- Validation uses physics and synthetic data only.
- Boberg-Lantz assumes radial flow and a simple heated zone; heterogeneity is not captured.
- Asphaltene effects may take viscosity outside the fitted Walther range at low temperature. Extrapolation is flagged.
- The rod-drag model ignores couplings/guides (apart from a multiplier) and pressure-driven annular flow.
- The failure-risk score is an uncalibrated heuristic ranking.
- Per-well optimization is not field-optimal under a shared steam supply.
- Dynamometer-card availability is not confirmed by the problem statement. Cycle optimization runs without cards; diagnostics need them.
- The RFI uses the minimum fall velocity over depth, as specified, which makes it very conservative. Evaluation reports lead time *and* false-alarm hours.

## Phase 2 (not built; stubs in `optimization/phase2.py`, excluded from metrics)

Depletion-aware DP cut-off and multi-cycle lookahead; a survival-model failure prediction;
field-level steam allocation across wells; and real Oil India data ingestion beyond the
replay/OPC UA mapping. The harness already accepts CSV/Parquet in the `Telemetry` schema
(`simulation/replay.py`).

# BUILD_SPEC — Coupled Well-to-Surface Digital Twin for CSS + SRP Optimization (Baghewala Field, SIH 26120)

> **Instructions for Claude Code (read first)**
>
> 1. This is a full build specification. Build in the milestone order in §16. Each milestone must pass its tests before the next starts.
> 2. **Never invent field data.** Every physical constant, field parameter or economic value lives in `config/` and carries a `source` note. Values not taken from a cited source are marked `source: "PLACEHOLDER - calibrate"`, and the UI shows them as such.
> 3. Equations marked **[VERIFY]** must be checked against the cited reference before being finalized. Write a unit test that encodes a known limiting behavior (listed with each one) so a wrong implementation fails loudly.
> 4. SI units internally. Convert to field units (bbl/d, psi, °F, ft, cP) only at the API/UI boundary, through `core/units.py`.
> 5. Every numeric output the system produces carries a confidence tier (A/B/C, §11) and a provenance record.
> 6. Anything listed under "Phase 2" (§17) is **not** built. Only stubs and interfaces are allowed, and they must be excluded from all reported metrics.

---

## 0. What we are building

A digital twin for heavy-oil wells produced by **Cyclic Steam Stimulation (CSS)** and lifted by **Sucker Rod Pumps (SRP)**. CSS and SRP are optimized as **one coupled system**:

- **Forward chain (reservoir → pump):** steam volume & soak → heated zone & cooling → temperature → viscosity → rod drag & fluid load → rod-float risk & pump fillage.
- **Feedback chain (pump → reservoir):** SPM / stroke / speed profile → pump displacement → intake pressure & drawdown → inflow → cooling & decline → cut-off & next-cycle design.

Every module reads and writes one shared **WellState** (§3).

Problem-statement facts to use (Tier A):
- Crude gravity: 17–19 °API.
- Native reservoir temperature: 46–48 °C.
- Jodhpur Sandstone reservoir, low pressure, high viscosity, high asphaltene content.

Everything else is configurable and starts as a placeholder.

### Product capabilities (must all exist in the prototype)
1. Real-time monitoring of well state, driven by a replayed telemetry stream.
2. Prediction of reservoir heating, cooling and production per cycle.
3. CSS cycle optimization: steam volume, injection pressure, soak time and production cut-off, solved as one nested problem.
4. Inflow-matched SRP control of SPM and the VFD speed profile, plus a per-cycle stroke-length recommendation.
5. Rod-float detection (Rod Float Index) and impact-load minimization via downstroke shaping.
6. Card-based fault classification: normal, fluid pound, gas interference, rod float, pump unseated.
7. Pump efficiency metrics (volumetric efficiency, fillage).
8. Rule-based failure-risk score that ranks wells.
9. Energy per barrel and steam-oil ratio (SOR) as first-class outputs.
10. Confidence tier on every output.
11. Edge controller simulator with three modes and a hard safety fallback.
12. Baseline-vs-twin evaluation harness.

---

## 1. Tech stack

| Layer | Choice |
|---|---|
| Language (backend/physics) | Python 3.11+ |
| Numerics | NumPy, SciPy |
| Steam/water properties | `iapws` (IAPWS-IF97) or CoolProp, behind one interface in `core/steam.py` |
| Bayesian optimization | BoTorch + GPyTorch (fallback: `scikit-optimize`) |
| ML classifier | scikit-learn (HistGradientBoosting) first; optional PyTorch 1-D/2-D CNN |
| Data models | Pydantic v2 |
| API | FastAPI, with WebSocket for live telemetry |
| Storage | SQLite (dev) behind SQLAlchemy; schema compatible with TimescaleDB/Postgres |
| Industrial protocols (simulated) | `asyncua` (OPC UA server/client), `pymodbus` (optional) |
| Frontend | React + TypeScript + Vite, Tailwind, Recharts or Plotly, Zustand for state |
| Packaging | Docker Compose (backend, frontend, optional opcua-sim) |
| Tests | pytest, hypothesis (property tests), Playwright (optional UI smoke tests) |
| Lint/format | ruff, black, mypy (strict on `core/` and `physics/`) |

---

## 2. Repository layout

```
twin/
├── README.md
├── BUILD_SPEC.md
├── docker-compose.yml
├── config/
│   ├── field.yaml            # field/well parameters (placeholders + sources)
│   ├── economics.yaml        # oil price, steam cost, tariff, fixed costs
│   ├── rods.yaml             # rod string grades, tensile strengths, S-N params
│   ├── controller.yaml       # setpoints, limits, gains, safety envelope
│   └── optimizer.yaml        # bounds, constraints, BO settings
├── backend/
│   ├── core/
│   │   ├── state.py          # WellState, CycleState, Provenance, Tier
│   │   ├── units.py
│   │   ├── steam.py
│   │   ├── config.py
│   │   └── bus.py            # in-process pub/sub for state updates
│   ├── physics/
│   │   ├── viscosity.py      # Walther + Arrhenius
│   │   ├── wellbore.py       # Ramey-type heat loss + steam quality
│   │   ├── reservoir.py      # Boberg-Lantz CSS model (+ Marx-Langenheim term)
│   │   ├── inflow.py         # IPR with stimulation ratio
│   │   ├── pumping_unit.py   # surface kinematics (polished-rod position/velocity)
│   │   ├── rod_string.py     # wave equation: predictive + diagnostic
│   │   ├── pump.py           # downhole pump boundary condition, valve logic
│   │   ├── drag.py           # viscous rod drag, damping coefficient
│   │   └── energy.py         # polished-rod power, motor/gearbox, kWh/bbl
│   ├── diagnostics/
│   │   ├── rfi.py            # Rod Float Index
│   │   ├── card_features.py
│   │   ├── classifier.py
│   │   ├── efficiency.py     # fillage, volumetric efficiency
│   │   └── card_from_vfd.py  # STRETCH: estimate cards from VFD torque/speed
│   ├── calibration/
│   │   ├── rls.py            # recursive least squares w/ forgetting factor
│   │   └── inflow_refit.py
│   ├── optimization/
│   │   ├── objective.py      # net value, g(T)
│   │   ├── cutoff.py         # inner g* rule (sim + live)
│   │   ├── cycle_bo.py       # outer Bayesian optimizer
│   │   └── baseline.py       # current-practice policy
│   ├── control/
│   │   ├── srp_controller.py # feedforward + fillage feedback + float + stress
│   │   ├── goodman.py
│   │   ├── safety.py         # envelope, fallback
│   │   └── edge_sim.py       # modes: advisory / supervised / autonomous-in-band
│   ├── risk/
│   │   └── failure_risk.py   # Miner's-rule score + event counts
│   ├── simulation/
│   │   ├── field_sim.py      # synthetic multi-well field generator (ground truth)
│   │   ├── noise.py
│   │   ├── labels.py         # labeled fault/float event injection
│   │   └── replay.py         # telemetry replay stream
│   ├── evaluation/
│   │   ├── harness.py        # baseline vs twin experiments
│   │   └── metrics.py
│   ├── integration/
│   │   ├── opcua_server_sim.py
│   │   └── opcua_client.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routes_*.py
│   │   └── ws.py
│   └── tests/
└── frontend/
    └── src/ (views, components, api client, store)
```

---

## 3. Shared state (the backbone)

Implement in `core/state.py` with Pydantic. Every module reads the latest `WellState` and publishes updates through `core/bus.py`. There are no hidden module-to-module calls; coupling goes through state.

```python
class Tier(str, Enum): A="A"; B="B"; C="C"

class Provenance(BaseModel):
    tier: Tier
    source: str              # module name or citation
    model_version: str
    inputs_hash: str
    timestamp: datetime
    extrapolated: bool = False   # True if outside calibrated range

class Quantity(BaseModel):
    value: float
    unit: str
    uncertainty: float | None = None   # 1-sigma or interval half-width
    prov: Provenance

class CyclePhase(str, Enum): INJECT="inject"; SOAK="soak"; PRODUCE="produce"; IDLE="idle"

class CycleState(BaseModel):
    cycle_index: int
    phase: CyclePhase
    phase_start: datetime
    steam_injected_kg: Quantity
    injection_pressure: Quantity
    soak_duration: Quantity
    cumulative_oil_m3: Quantity
    cumulative_water_m3: Quantity
    recipe_id: str | None

class ReservoirState(BaseModel):
    T_avg_heated: Quantity        # average heated-zone temperature
    r_heated: Quantity            # heated radius
    p_reservoir: Quantity
    stimulation_ratio: Quantity   # J_hot / J_cold

class WellboreState(BaseModel):
    sandface_steam_quality: Quantity
    heat_loss_rate: Quantity
    T_tubing_profile: list[tuple[float, float]]   # (depth_m, T_K)
    mu_at_pump: Quantity
    mu_profile: list[tuple[float, float]]         # (depth_m, Pa·s)

class PumpState(BaseModel):
    spm: Quantity
    stroke_length: Quantity
    vfd_profile: "SpeedProfile"
    intake_pressure: Quantity
    fillage: Quantity
    volumetric_efficiency: Quantity
    surface_card: "Card | None"
    downhole_card: "Card | None"
    rfi: Quantity
    fault_class: str | None
    fault_probs: dict[str, float]
    peak_rod_stress_ratio: Quantity   # max stress / Goodman allowable

class EconomicState(BaseModel):
    q_oil: Quantity
    q_net_value_rate: Quantity   # currency/time
    kwh_per_bbl: Quantity
    sor_cycle: Quantity
    g_star: Quantity | None

class WellState(BaseModel):
    well_id: str
    t: datetime
    cycle: CycleState
    reservoir: ReservoirState
    wellbore: WellboreState
    pump: PumpState
    econ: EconomicState
    risk_score: Quantity
    control_mode: Literal["advisory","supervised","autonomous"]
    alarms: list["Alarm"]
```

Persist a `WellState` snapshot per tick (configurable, default 1 per simulated minute during production) and every card.

---

## 4. Physics modules

### 4.1 Viscosity — `physics/viscosity.py`

**Walther / ASTM D341 (default):**

```
log10( log10( ν + 0.7 ) ) = A − B · log10( T )
```
- ν is kinematic viscosity in cSt (mm²/s); T in kelvin.
- Dynamic viscosity: μ = ν · ρ(T), where ρ(T) comes from API gravity with a linear thermal-expansion correction (coefficient in config, PLACEHOLDER).
- The model is **linear in (A, B)** in the transformed variables, which is what makes RLS calibration (§7) valid.

**Arrhenius fallback:** `ln μ = a + b / T`. Used only if a well has fewer than `min_points_walther` (config, default 4) distinct temperature points.

API:
```python
fit_walther(T_K: array, nu_cSt: array) -> WaltherParams(A, B, T_min_fit, T_max_fit)
mu(T_K, params, rho_fn) -> Quantity   # sets prov.extrapolated if T outside fit range
```

Tests:
- A round-trip fit on synthetic data recovers A and B.
- Viscosity is monotone decreasing in T.
- Out-of-range T sets `extrapolated=True`.

### 4.2 Wellbore heat loss — `physics/wellbore.py` [VERIFY: Ramey 1962; Satter 1965]

The tubing is discretized into N depth cells (config, default 100).

Geothermal gradient: `T_e(z) = T_surface + G·z` (config).

Heat loss per unit length (Ramey form):
```
Q'(z) = 2π r_to U k_e (T_f − T_e(z)) / ( k_e + r_to U f(t) )
```
- `r_to`: tubing outer radius.
- `U`: overall heat transfer coefficient (config; optionally computed from tubing, annulus and cement resistances).
- `k_e`: earth thermal conductivity.
- `f(t)`: Ramey transient time function. Long-time approximation: `f(t) ≈ −ln( r_cw / (2√(α_e t)) ) − 0.290` [VERIFY].

**Saturated steam** (injection phase): temperature is fixed at `T_sat(p(z))`. Heat loss reduces quality:
```
w · h_fg(p) · dx/dz = −Q'(z)
```
Pressure drop is hydrostatic plus friction (simple two-phase approximation acceptable; document the choice).

Output: `sandface_steam_quality`, `heat_loss_rate`, tubing temperature profile.

**Production phase:** the same ODE is solved for the produced liquid (single-phase energy balance, `w·c_p·dT/dz = −Q'`). This gives the temperature profile up the tubing, and from it the **viscosity profile along the rod string**, which feeds rod drag (§4.6).

Tests:
- No heat loss when U = 0.
- Quality decreases monotonically with depth.
- Heat loss decreases with time (f(t) grows).
- The energy balance closes to within 0.1 %.

### 4.3 Reservoir — `physics/reservoir.py` [VERIFY: Boberg & Lantz 1966; Marx & Langenheim 1959]

**Injection-phase heated area (Marx-Langenheim term, sub-component only):**
```
A(t) = ( Q_i · M_R · h ) / ( 4 · λ_ob · M_ob · ΔT ) · G(t_D)
t_D  = 4 · λ_ob · M_ob · t / ( M_R² · h² )
G(t_D) = e^{t_D} · erfc(√t_D) + 2·√(t_D/π) − 1
```
- `Q_i`: heat injection rate at the sandface. It must use sandface quality from §4.2, not surface quality.
- `M_R`, `M_ob`: volumetric heat capacities of reservoir and overburden.
- `λ_ob`: overburden conductivity.
- `h`: net thickness.
- `ΔT`: steam temperature minus reservoir temperature.
- Heated radius: `r_h = sqrt(A/π)`.

Unit test: `A` has units of m², and `t_D` is dimensionless (write a dimensional check).

**Boberg-Lantz cycle model (backbone):**
- Average heated-zone temperature during soak/production:
  ```
  T_avg = T_R + (T_s − T_R) · [ f_HD · f_VD · (1 − f_PD) − f_PD ]
  ```
  - `f_HD`, `f_VD`: dimensionless radial and vertical conduction-loss functions.
  - `f_PD`: energy removed with produced fluids.
  - **[VERIFY]** the exact definitions and any closed-form approximations of f_HD and f_VD against the 1966 paper or a thermal-recovery textbook. If the approximations are not verifiable, compute them from the underlying 1-D conduction solutions numerically. Document which option was used.
- Carry-over of heat remaining from the previous cycle into the next cycle's initial state (required for multi-cycle behavior).
- Stimulation ratio (radial flow, heated inner zone):
  ```
  J_h / J_c = ln(r_e/r_w) / [ (μ_h/μ_c) · ln(r_h/r_w) + ln(r_e/r_h) ]
  ```
  `μ_h` is evaluated at `T_avg` (§4.1) and `μ_c` at `T_R`.

Tests (limiting behaviors):
- `T_R ≤ T_avg ≤ T_s` always.
- `T_avg → T_R` as t → ∞.
- `J_h/J_c = 1` when `μ_h = μ_c` or `r_h = r_w`.
- `J_h/J_c > 1` when `μ_h < μ_c`.
- Cycle-to-cycle oil response declines when the recipe is held fixed (depletion).

**Depletion:** `p_reservoir` declines with cumulative voidage through a simple material-balance or tank model (config). This is what makes cycles non-stationary.

### 4.4 Inflow — `physics/inflow.py`

```
q_in = J_c · (J_h/J_c) · ( p_reservoir − p_wf )
p_wf ≈ pump intake pressure (from §4.7 / §5)
```
Water cut per cycle follows a configurable schedule, since condensed steam is produced back early in the cycle. Early-cycle production is water-dominant, so net oil rate starts low or negative in value terms. That shape is required for the cut-off logic in §8.

### 4.5 Pumping-unit kinematics — `physics/pumping_unit.py`

- Conventional crank-balanced beam unit geometry, with parameters in config (API RP 11L-style dimensions).
- Output: polished-rod position `x(θ)`, velocity and acceleration versus crank angle, given the motor speed profile.
- The VFD speed profile is a piecewise-linear crank angular speed `ω(θ)` over one revolution, split into upstroke and downstroke segments.
- A simple harmonic approximation is acceptable **only** as a fallback mode, and must be labeled as such.

### 4.6 Rod drag & damping — `physics/drag.py`

Viscous drag on a rod moving concentrically in tubing (Couette approximation, no net flow):
```
F_drag / L = 2π μ(z) V / ln( r_t / r_r )
```
Equivalent damping coefficient in the wave equation, per unit rod mass:
```
c(z) = 2π μ(z) / ( ρ_steel · A_r · ln(r_t/r_r) )
```
- `μ(z)` is the tubing viscosity profile from §4.2 (this is the coupling point).
- Document the simplification: couplings, guides and pressure-driven annular flow are ignored in v1. An optional `drag_multiplier` in config accounts for couplings and guides.

**Terminal rod fall velocity** (used by the RFI in §6.1). The buoyant weight per length equals drag at terminal velocity:
```
V_fall(z) = ( ρ_steel − ρ_fluid ) · g · A_r · ln(r_t/r_r) / ( 2π μ(z) )
```
Use the minimum over depth (the slowest-falling segment governs), and optionally include pump plunger drag as an extra resistance.

### 4.7 Rod string wave equation — `physics/rod_string.py` [VERIFY: Gibbs 1963; Everitt & Jennings 1992]

Damped 1-D wave equation for rod displacement u(z, t):
```
∂²u/∂t² = a² ∂²u/∂z² − c(z) ∂u/∂t
a = sqrt(E / ρ_steel)
```
Support a tapered string: piecewise E, A_r and ρ per section, with continuity of displacement and force at taper joints.

**Two modes are required:**

1. **Predictive (forward) mode.** Used by the simulator to create synthetic surface cards.
   - Boundary condition at the top: prescribed polished-rod position from §4.5.
   - Boundary condition at the bottom: the pump force from §4.8.
   - Explicit finite-difference time stepping, with the CFL condition `a·Δt/Δz ≤ 1` enforced by assertion.
   - Output: surface load vs. position (surface card).
2. **Diagnostic mode.** Used on measured or replayed data: surface card to downhole card.
   - Given measured surface position AND load (Cauchy data at z = 0), march down the string with a finite-difference scheme in the style of Everitt & Jennings.
   - Output: downhole pump load vs. plunger position (downhole card).

Tests:
- With damping = 0 and a rigid pump, the predictive → diagnostic round trip recovers the imposed pump load to within tolerance.
- Stress at a taper equals load divided by the local area.
- Energy decays with damping.
- CFL violations raise errors.

### 4.8 Downhole pump — `physics/pump.py`

Pump boundary-condition logic with standing and traveling valves:
- **Upstroke:** traveling valve closed → the plunger carries fluid load `F = (p_discharge − p_intake) · A_p`.
- **Downstroke:** the traveling valve opens only when barrel pressure exceeds discharge pressure. With incomplete fillage, the plunger hits the fluid level partway down, which produces **fluid pound** (a sudden load drop).
- **Gas interference:** compressible gas in the barrel → gradual load transfer.
- **Pump unseated:** the pump is not held down, so fluid load is never transferred and the downhole card collapses to near-zero area.
- **Rod float:** the rods cannot fall as fast as the polished rod descends. Polished-rod load goes to ~0 (carrier-bar separation) and compressive/impact loading appears at the start of the upstroke.
- Pump displacement per stroke: `A_p · S_p · fillage`, where `S_p` is the effective plunger stroke from the downhole card.

### 4.9 Energy — `physics/energy.py`

- Polished-rod power = (surface card area) × SPM.
- Motor input power = polished-rod power / (η_gearbox · η_belt · η_motor · η_vfd), all from config (PLACEHOLDER).
- `kWh_per_bbl = motor_energy / oil_produced`.
- Cycle SOR = steam injected (cold-water-equivalent m³) / oil produced (m³).

---

## 5. Synthetic field simulator — `simulation/field_sim.py`

This is the ground truth, because no Oil India data is available yet.

- Generate `n_wells` (config, default 12). Parameters are sampled from configured ranges: depth, h, permeability-related J_c, T_R ∈ [46, 48] °C (Tier A), API ∈ [17, 19] (Tier A), rod taper, pump size, pumping unit.
- Each well runs full CSS cycles (inject → soak → produce), coupling §4.1–4.9 at each tick.
- Card generation: predictive wave-equation mode, once per N strokes (config).
- **Noise** (`noise.py`): Gaussian load/position noise, sensor dropout, clock jitter, and occasional spikes. Levels come from config.
- **Labeled events** (`labels.py`): inject fluid pound (over-pumping), gas interference, rod float (a cooling-driven viscosity rise with a fast downstroke) and pump unseating (a random event). Store labels with timestamps and onset times.
- The simulator exposes **its own hidden truth** for evaluation only. The twin must never read hidden truth, only noisy "measured" telemetry. Enforce this with separate data classes and an import-lint test.

`replay.py` streams simulated telemetry at configurable speed (e.g. 1 simulated day per 10 s) over WebSocket and, optionally, the OPC UA sim server.

---

## 6. Diagnostics

### 6.1 Rod Float Index — `diagnostics/rfi.py`
```
RFI = max over downstroke of V_polished_rod(t) / V_fall_min
```
- RFI ≥ 1 means float is expected.
- The warning threshold `rfi_warn` (config, default 0.8) and the alarm threshold `rfi_alarm` (default 1.0) are configurable.
- Output a trend and a slope (rate of change per stroke).
- **Evaluation target (not an assumed property):** lead time between RFI crossing `rfi_warn` and labeled float onset on replayed events. Report its distribution.

### 6.2 Card features — `diagnostics/card_features.py`
- Normalize the downhole card to [0,1] in both position and load.
- Features: area, fillage estimate, load range, corner angles, Fourier descriptors (first 16 of the closed-contour complex series), and a 32×32 occupancy-grid image.

### 6.3 Classifier — `diagnostics/classifier.py`
- Classes: `normal, fluid_pound, gas_interference, rod_float, pump_unseated`. Optional extras: `tv_leak, sv_leak`.
- v1 model: HistGradientBoosting on the features from §6.2. v2 (optional): a small CNN on the grid image.
- Training data: the synthetic simulator (Tier C). Splits are **by well** (no leakage across wells), plus a separate noise-stressed test set.
- Report per-class precision, recall and F1, and a confusion matrix.
- Output probabilities. An `unknown` label is used when max probability < `clf_min_conf` (config).
- **On `pump_unseated`:** raise an alarm and request an SPM reduction (the controller acts on it, §9).

### 6.4 Efficiency — `diagnostics/efficiency.py`
- Fillage = effective plunger stroke / gross plunger stroke (from the downhole card).
- Volumetric efficiency = measured liquid rate / theoretical displacement.

### 6.5 Card from VFD (STRETCH, Tier C) — `diagnostics/card_from_vfd.py`
- Estimate polished-rod load from motor torque via the gearbox torque factor from pumping-unit kinematics, and position from crank angle.
- Build only after all core milestones pass. Label every output Tier C.

---

## 7. Calibration — `calibration/`

- **RLS with forgetting factor λ** (config, default 0.995) on the Walther linear form `y = A − B·x`, where `y = log10(log10(ν+0.7))` and `x = log10(T)`.
  - Each new (T, ν) pair updates (A, B) and their covariance.
  - Parameter uncertainty is propagated into μ outputs.
- **Inflow refit** (`inflow_refit.py`): at the end of each cycle, or every K days, run a nonlinear least-squares refit (SciPy `least_squares`) of `J_c` and the water-cut schedule parameters against observed rates.
- **Guardrails:** reject updates that move parameters more than a configured amount per step, and log rejected updates as alarms.

---

## 8. Cycle optimization — `optimization/`

### 8.1 Objective — `objective.py`

Net value rate at time t in the production phase:
```
q_net(t) = P_oil · q_oil(t) − tariff · P_motor(t) − opex_rate
```
Cycle net value for a production duration T:
```
V(T) = ∫_0^T q_net(t) dt − C_steam(recipe) − C_fixed
g(T) = V(T) / ( t_inject + t_soak + T )
```
- `C_steam = steam_mass · cost_per_tonne_steam`.
- `C_fixed` covers per-cycle switching/workover cost.
- All prices come from `economics.yaml` (PLACEHOLDER).
- A `production_weight` option (0–1) blends in a volume term, so a production-weighted objective can be run alongside the value-weighted one.

### 8.2 Inner cut-off — `cutoff.py`

**Simulation mode:**
- Simulate the production phase to `T_max`.
- Evaluate g(T) on a time grid and take `T* = argmax g(T)` and `g* = g(T*)`.
- **Do not** use the first crossing of `q_net = g`. The early-cycle rate is low or negative, then rises and then declines.

**Live mode:**
- Given the current `g*` (from the latest optimization), stop production when **q_net(t) is falling (after its peak) and q_net(t) ≤ g*.**
- Smooth `q_net` with an EWMA before evaluating, and require the condition to hold for `hold_hours` (config) to avoid noise triggers.

Tests:
- On a synthetic unimodal q_net, the grid argmax matches the analytical optimum.
- A profile that is negative early never stops at the early crossing.
- The live rule and the sim rule agree on a noise-free profile.

### 8.3 Outer Bayesian optimizer — `cycle_bo.py`

- Decision variables: steam mass, injection pressure and soak time. Bounds come from `optimizer.yaml`.
- Objective: `g*(θ)` from §8.2, evaluated by simulating one cycle from the **current calibrated well state** (including carried-over heat and current reservoir pressure).
- Constraints (feasibility as a GP classifier, or a penalty):
  - injection pressure < fracture pressure × safety factor;
  - steam rate ≤ generator capacity;
  - sandface steam quality ≥ minimum;
  - the required lift (peak predicted inflow) is achievable within max SPM and the Goodman limit.
- BoTorch settings: SingleTaskGP, qLogExpectedImprovement (or qLogNEI), 8–12 initial Sobol points, 30–60 iterations (config). Seed everything.
- Output: the recommended recipe, predicted g*, T*, SOR, kWh/bbl, cycle oil, and uncertainty (posterior std, plus the spread across calibration-parameter samples).
- **Receding horizon (built):** re-run before every new cycle from the updated state.
- **NOT built (Phase 2):** lookahead over future cycles, and the depletion-aware DP cut-off.

### 8.4 Baseline policy — `baseline.py`

Current-practice emulation:
- a fixed steam recipe (config);
- cut-off at a fixed day count OR a fixed cumulative tonnage (both selectable);
- a fixed SPM set at the start of production and adjusted only when fillage drops below a manual threshold, with a delay (config) to emulate reactive manual adjustment.

---

## 9. SRP controller — `control/srp_controller.py`

The controller runs every control interval (config, e.g. every 15 simulated minutes). The four layers apply in this order:

1. **Feedforward:**
   ```
   SPM_ff = q_in_pred / ( A_p · S_p_eff · η_v_est · 1440 )    # q in m³/day, SPM in strokes/min
   ```
   `q_in_pred` comes from the reservoir twin and inflow model (§4.4) at the current drawdown.
2. **Feedback on fillage:** a PI controller on `fillage_target − fillage_measured` (target from config, e.g. 0.85–0.90 PLACEHOLDER), with anti-windup. `SPM = clamp(SPM_ff + PI, SPM_min, SPM_max)`.
3. **Float protection:** if `RFI ≥ rfi_warn`, first **reshape the downstroke VFD speed profile** (reduce downstroke crank speed in the high-velocity segment, keep the upstroke speed, and check the per-stroke time budget). Reduce SPM only if RFI is still ≥ `rfi_warn` after shaping reaches its limit.
4. **Stress limit:** predict peak and minimum rod stress for the proposed setting using the predictive wave equation. Reject or back off any setting where `max_stress > S_allow` (§9.1).

**Special actions:**
- `pump_unseated` detected → SPM to minimum + alarm.
- Sustained fluid pound → drop SPM by a step.

**Stroke length:** computed as a per-cycle **recommendation** only (beam units change stroke mechanically at the crank). No live actuation.

**Reliability-first default:** the setpoint never targets displacement above predicted inflow in steady state. The `transient_weight` option (config, default 0) allows a short, time-limited over-displacement, capped by fillage and RFI limits.

Every decision is logged with its inputs, active constraints and reason code (for explainability in the UI).

### 9.1 Goodman — `control/goodman.py` [VERIFY: API RP 11BR]
Modified Goodman allowable maximum stress:
```
S_allow = ( T/4 + 0.5625 · S_min ) · SF
```
- `T` is the minimum tensile strength of the rod grade; `S_min` is the minimum stress in the stroke; `SF` is a service factor (config).
- Output `stress_ratio = S_max / S_allow` per taper section.

### 9.2 Safety — `control/safety.py`
- A hard envelope on SPM, stroke rate of change, VFD frequency limits and stress ratio.
- On any violation, stale data (older than `max_data_age`), classifier `unknown` on 3 consecutive cards, or an internal exception → **fallback** to the last known-safe setting, or to the configured safe default. Alarm raised.
- Safety checks run **after** every controller output, in every mode.

### 9.3 Edge controller simulator — `control/edge_sim.py`
- **Advisory** (default): recommendations only. Nothing is written.
- **Supervised:** a recommendation creates a pending action. The UI must approve it before it is written to the simulated VFD (OPC UA write).
- **Autonomous-within-band:** actions within operator-set bands are auto-applied; anything outside the band becomes supervised.
- Every mode change and approval is audit-logged.

---

## 10. Failure-risk score — `risk/failure_risk.py`

This is a heuristic ranking, **not** a failure prediction. Label it as such in the UI.

- Per taper section per stroke: `stress_ratio` from §9.1.
- Cycles-to-failure proxy: `N(stress_ratio) = N_ref · stress_ratio^(−m)`, with `N_ref` and `m` from config (PLACEHOLDER; document as uncalibrated).
- Miner's damage accumulates: `D += 1 / N(stress_ratio)` per stroke.
- Event terms: counts of float events, impact events (compressive load at upstroke start) and fluid-pound strokes, over a rolling window.
- `risk_score = w1·D + w2·float_rate + w3·impact_rate + w4·pound_rate` (weights in config).
- The output is used **only to rank wells**. Evaluation reports ranking quality (e.g. rank correlation with simulated failure events, if the simulator injects failures by damage threshold).

---

## 11. Confidence tiers

Every `Quantity` carries a `Provenance.tier`:

- **A:** directly bounded by the problem-statement values or a cited standard/model (e.g. T_R, API, the Goodman formula).
- **B:** the output of first-principles physics modules (§4) with calibrated or placeholder parameters.
- **C:** ML outputs (classifier) and cards estimated from VFD data.

Rules:
- Any value computed from a placeholder config parameter shows a "placeholder" badge.
- `extrapolated=True` shows a warning badge.
- The UI never shows a number without its tier badge.

---

## 12. Evaluation harness — `evaluation/`

Experiments run the **same simulated wells, seeds and noise** under two policies: the baseline (§8.4) and the twin (§8 + §9).

Metrics (`metrics.py`), each with mean ± CI across wells and seeds:
- oil production and recovery per cycle and cumulative;
- SOR;
- kWh/bbl;
- NPV (discount rate in config);
- float events, impact events, downtime hours, cumulative production alongside downtime;
- peak stress ratio distribution;
- volumetric efficiency;
- simulated failure events (if enabled) and risk-score rank correlation.

Also produced:
- **Physics verification:** a wave-equation round-trip error table and energy-balance closure for wellbore and reservoir.
- **Classifier:** per-class precision/recall on the held-out-by-well and noise-stressed sets.
- **RFI lead time:** the distribution on labeled float events.

Every metric is reported under both `production_weight = 0` (value) and a configured production-weighted setting.

The output is a JSON report plus auto-generated charts (PNG) in `reports/<run_id>/`, and an `evaluation` view in the UI.

---

## 13. API (FastAPI) — `api/`

| Method | Path | Purpose |
|---|---|---|
| GET | `/wells` | list wells with summary + risk rank |
| GET | `/wells/{id}/state` | latest WellState |
| GET | `/wells/{id}/history?from&to&fields` | time series |
| GET | `/wells/{id}/cards?limit` | recent surface + downhole cards |
| POST | `/wells/{id}/optimize-cycle` | run outer BO from current state → recipe |
| GET | `/wells/{id}/cutoff` | current g*, q_net trend, stop recommendation |
| GET | `/wells/{id}/controller/recommendation` | latest SRP recommendation + reason codes |
| POST | `/wells/{id}/controller/approve` | approve pending action (supervised) |
| POST | `/wells/{id}/mode` | set advisory/supervised/autonomous + bands |
| POST | `/whatif` | run a scenario (recipe/SPM/profile changes) on a cloned state, no side effects |
| POST | `/evaluation/run` | start baseline-vs-twin experiment |
| GET | `/evaluation/{run_id}` | results |
| WS | `/ws/telemetry` | live state + alarm stream |
| GET | `/config` | current config with placeholder flags |

---

## 14. Frontend — five views (+ evaluation)

Shared across all views: a well selector, a tier badges component, an alarm tray, a control-mode indicator, and a simulated-time control (play/pause/speed).

1. **Field overview:** well table (status, phase, q_oil, SOR, kWh/bbl, RFI, risk rank, active alarms), field totals, sortable by risk.
2. **Reservoir view:**
   - heated radius and T_avg over the cycle;
   - viscosity at the pump over time;
   - predicted vs. observed rate;
   - cycle history (oil, SOR per cycle);
   - calibration parameter trends with uncertainty.
3. **Diagnostics view:**
   - live surface and downhole cards (overlay of the last N);
   - classifier label and probabilities;
   - RFI trend with thresholds;
   - fillage and volumetric efficiency;
   - stress ratio per taper;
   - failure-risk components.
4. **Decision console:**
   - current SRP recommendation with reason codes and active constraints;
   - approve/reject (supervised);
   - cut-off panel: q_net trend, g* line and stop recommendation;
   - next-cycle recipe card with uncertainty;
   - mode and band settings;
   - audit log.
5. **What-if simulator:** sliders for steam mass, pressure, soak, SPM and downstroke profile. It runs `/whatif` and shows deltas in production, SOR, kWh/bbl, RFI, stress ratio and g* against the current plan.
6. **Evaluation** (supporting view): baseline-vs-twin charts and tables from §12, with the physics verification and classifier reports.

The demo scenario must be reproducible from a seeded config (`scenarios/demo_float.yaml`):
- a cooling event raises viscosity;
- RFI rises;
- the controller reshapes the downstroke;
- float is avoided;
- the UI shows the avoided-event impact in energy, SOR and currency against the baseline run of the same scenario.

---

## 15. Integration (simulated) — `integration/`

- The OPC UA sim server exposes per-well nodes: telemetry (read) and VFD setpoints (write), with SPM, speed profile and run/stop.
- The twin connects as a client. Advisory mode uses read-only access; supervised and autonomous modes write setpoints.
- Modbus is optional and uses the same mapping table.
- The mapping lives in config so a real SCADA tag list can be dropped in later.

---

## 16. Milestones (build order + cut order)

Each milestone has its own acceptance tests.

1. **M1 — Core:** state, config (with placeholder flags), units, steam properties, bus, persistence.
2. **M2 — Physics I:** viscosity, wellbore, reservoir, inflow, including all limiting-behavior tests.
3. **M3 — Rod/pump:** kinematics, drag, the wave equation in both modes, pump boundary conditions, energy. Round-trip tests pass.
4. **M4 — Simulator:** a multi-well field, noise, labeled events, replay stream, and the hidden-truth separation test.
5. **M5 — Diagnostics:** RFI, features, classifier (with report), efficiency.
6. **M6 — Calibration:** RLS and inflow refit with guardrails.
7. **M7 — Optimization:** objective, cut-off (sim + live), BO, baseline.
8. **M8 — Control:** SRP controller, Goodman, safety, edge modes.
9. **M9 — Risk + evaluation harness:** the first full baseline-vs-twin report.
10. **M10 — API + frontend:** the five views plus evaluation, and the demo scenario.
11. **M11 — OPC UA sim integration.**
12. **M12 (stretch)** — card-from-VFD estimation.

**If time runs short, cut in this order:** M12 → M11 → the what-if view → autonomous mode (keep advisory and supervised) → RLS (fall back to per-cycle batch refit).

**Never cut:** the wave equation, the nested optimizer with the correct cut-off, the SRP controller, or the evaluation harness. The proposal's claims depend on them.

---

## 17. Phase 2 — NOT built (interfaces/stubs only, excluded from metrics)

- The depletion-aware dynamic-programming cut-off and multi-cycle lookahead in the recipe optimizer.
- Survival-model failure prediction.
- Field-level optimization across wells that share a steam generator.
- Real Oil India data ingestion beyond the replay/OPC UA mapping (the harness must accept real historical CSV/Parquet in the same schema, but no real data is bundled).

---

## 18. Known limitations (surface these in the README and the UI "About" panel)

- Validation uses physics and synthetic data only.
- Boberg-Lantz assumes radial flow and a simple heated zone; heterogeneity is not captured.
- Asphaltene effects may take viscosity outside the fitted Walther range at low temperature. Extrapolation is flagged.
- The rod-drag model ignores couplings/guides (apart from a multiplier) and pressure-driven annular flow.
- The failure-risk score is an uncalibrated heuristic ranking.
- Per-well optimization is not field-optimal under shared steam supply.
- Dynamometer-card availability is not confirmed by the problem statement. Cycle optimization runs without cards (using production, CSS and VFD/power data), but the diagnostics require them.

---

## 19. References (verify details before citing externally)

1. Boberg, T.C., Lantz, R.B. (1966). Calculation of the production rate of a thermally stimulated well. *JPT*.
2. Marx, J.W., Langenheim, R.H. (1959). Reservoir heating by hot fluid injection. *Trans. AIME*.
3. Ramey, H.J. (1962). Wellbore heat transmission. *JPT*.
4. Satter, A. (1965). Heat losses during flow of steam down a wellbore. *JPT*.
5. Gibbs, S.G. (1963). Predicting the behavior of sucker-rod pumping systems. *JPT*.
6. Everitt, T.A., Jennings, J.W. (1992). An improved finite-difference calculation of downhole dynamometer cards for sucker-rod pumps. *SPE Production Engineering*.
7. ASTM D341: Viscosity-temperature equations/charts for liquid petroleum products.
8. API RP 11L: Design calculations for sucker rod pumping systems.
9. API RP 11BR: Care and handling of sucker rods (modified Goodman diagram).
10. Charnov, E.L. (1976). Optimal foraging, the marginal value theorem. *Theoretical Population Biology*.
11. Ross, S.M. *Stochastic Processes* (renewal-reward theorem).

---

## 20. Definition of done

- All milestone tests pass in CI (`pytest -q`); mypy is clean on `core/` and `physics/`.
- `docker compose up` brings up the backend, the frontend and the simulator. The demo scenario plays end-to-end.
- The evaluation report is generated with both objective settings, and every number has a tier badge and uncertainty.
- The README documents every placeholder parameter, every [VERIFY] item and how it was resolved, and all limitations.

# Optional performance and race-event inputs

Existing snapshots, CLI/API calls and Transformer checkpoints continue to work.
Every new measurement is optional; omitted or null values preserve the original
ratings. No historical telemetry is invented or automatically fetched. Add evidence
to `drivers[].performance`, `teams[].car.performance`, or `race_dynamics` in a
snapshot. A complete fictional example is `examples/race-optional.json`.

```sh
f1-forecast --database artifacts/optional-demo.sqlite3 predict examples/race-optional.json \
  --simulations 5000 --output artifacts/optional-demo.json
```

## Driver evidence

| Field | Meaning |
| --- | --- |
| `qualifying_teammate_delta_pct` | Percentage lap-time difference to the teammate on comparable qualifying runs; negative means faster |
| `race_teammate_delta_pct` | Percentage lap-time difference on comparable clean race stints; negative means faster |
| `clean_lap_variability_pct` | Standard deviation of comparable clean-lap residuals, as percent of reference lap time; lower means more consistent |
| `tyre_management` | Optional 0..1 driver tyre-management estimate; higher is better |
| `retirement_probability` | Total per-race nonfinish probability including incidents and mechanical failures, not an additional independent failure probability |

Delta percentages use percentage points: `-0.2` means 0.2 percent faster,
not 20 percent. These inputs require comparable conditions. Teammate differences
are evidence, not a uniquely identified measure of driver talent.

## Car evidence

| Field | Meaning |
| --- | --- |
| `qualifying_gap_pct`, `race_gap_pct` | Percentage lap-time deficit to a stated reference under comparable conditions; lower means faster |
| `braking`, `aerodynamics`, `handling`, `power_delivery` | Optional 0..1 capability estimates; higher means better |
| `medium_speed_cornering`, `downforce`, `aerodynamic_efficiency` | Optional 0..1 capability estimates; higher means better |
| `tyre_degradation_s_per_lap` | Nonnegative pace-loss slope in seconds per additional lap of tyre age, after correcting for known conditions |

Record the comparison benchmark, tyres, stint context, adjustments, uncertainty,
and supporting source in the snapshot's `sources` and `notes`. The engine does not
derive these measurements from raw telemetry, identify physical downforce or drag,
or infer degradation from aero ratings. Degradation currently adjusts a session-level
tyre feature; it does not evolve lap by lap or distinguish compounds.

## Bounded feature adaptation

Both performance objects accept `weight` (0..1, default 0.5). This controls evidence
influence, not forecast confidence. Zero disables the supplied measurements.
The adapter clips estimates to 0..1 and blends them with existing ratings, limiting
each rating change to 0.2. Overlapping capability estimates are averaged:

- Power delivery and aerodynamic efficiency inform straight-line performance.
- Aerodynamics, downforce and medium-speed cornering inform high-speed corner fit.
- Braking, handling and medium-speed cornering inform low-speed corner fit.
- Measured degradation and driver tyre management inform the tyre feature.

Explicit heuristic scales, pending fitting with real measurements:

| Measurement | Rating estimate before clipping and blending |
| --- | --- |
| Car pace deficit `g` in percent | `1 - g / 5` |
| Driver teammate delta `d` in percent | `0.5 - d / 2` |
| Clean-lap variability `v` in percent | `1 - v / 2` |
| Tyre degradation slope `s` in seconds/lap | `1 - s / 0.2` |

The total DNF probability is blended directly with the existing per-race risk using
`weight`, without the 0.2 rating cap. Weight 1 replaces the existing risk, including
when it comes from a Transformer, so risk is not counted twice. Qualifying ignores
the retirement input.

These adapters preserve the existing eight base features and Transformer feature
schema. Supplied evidence therefore reaches numerical scoring, neural training and
inference, and numerical pace feedback. The separate Transformer is still retrained
explicitly. Old checkpoints load, but compatibility is not proof of accuracy on the
new evidence distribution. No new coefficients or probability calibration are fitted
just by supplying an optional field.

## Race events

`race_dynamics` is allowed only for race sessions. It accepts:

| Field | Default | Meaning |
| --- | --- | --- |
| `safety_car_probability` | 0 | Probability of a safety-car period |
| `virtual_safety_car_probability` | 0 | Probability of a virtual-safety-car period |
| `red_flag_probability` | 0 | Probability of a red-flag interruption |
| `event_lap_fraction` | null | Approximate interruption timing from 0 to 1; null samples uniformly from 10% to 90% of race distance |
| `neutralized_fraction` | 0.1 | Approximate fraction of race distance interrupted, capped by remaining distance |
| `pit_opportunity_probability` | 0.5 | Per-driver probability of benefiting from an interruption pit opportunity |
| `restart_variability` | 0.25 | Restart noise scale in latent model-score units, bounded 0..1 |

With any positive event probability, explicit event sampling replaces the legacy
generic disruption noise. All-zero probabilities leave the original simulation
unchanged. Types are sampled independently and can co-occur; their probabilities
do not need to sum to one. They share one approximate timing window. Safety cars
and red flags compress latent performance differences, and restarts add variation
when racing resumes. Virtual safety cars do not bunch the field or add restart noise.
Pit opportunities depend on timing, strategy and pit-crew ratings. A red flag uses
a shared reset rather than a random cheap-stop benefit.

The same event draw affects the entire field. Returned scenario `race_event_rates`
show the sampled frequency of each event type. Retirement sampling retains its own
random sequence and total-risk input. Event types are not yet causally coupled to
collisions, and multi-car accidents, repeated interruptions, detailed pit timing,
tyre changes and official red-flag classification rules are not simulated.
The supplied event probabilities apply within every weather scenario; the adapter
does not automatically infer circuit or wet-weather event probabilities.

## Evidence and evaluation

Each optional object accepts `available_at`. A date after the snapshot cutoff is
rejected. Sources must also pass existing time checks. If no timestamp is supplied,
the inputs remain user assumptions: timestamp validation cannot prove source truth.
Target-session actual incidents, weather and results must never be inserted into
its prediction inputs.

Evaluation now includes `position_interval_coverage` (fraction of drivers whose
actual position falls inside their p10..p90 interval) and `position_interval_width`.
Use these alongside winner Brier score and position log loss. Discrete position
intervals can exceed nominal 80% coverage; higher coverage alone is not necessarily
better, particularly when intervals become much wider. Higher displayed confidence
alone is not a quality improvement.

`scripts/recheck_optional_inputs.py` repeats all four saved Transformer experiments
with their original data hashes, chronological folds, training settings and seeds:

```sh
# Run against a preserved copy of the original package first via PYTHONPATH.
python scripts/recheck_optional_inputs.py run artifacts/optional-inputs/before
# Then run against the updated package.
python scripts/recheck_optional_inputs.py run artifacts/optional-inputs/after
python scripts/recheck_optional_inputs.py compare \
  artifacts/optional-inputs/before artifacts/optional-inputs/after
```

Choose fresh output directories for another experiment. The comparison writes
matched metrics to `comparison.json` and generates a new HTML/CSV backtest report.
The current archived datasets contain no new optional measurements, so this is a
missing-input compatibility experiment. It cannot establish the benefit of braking,
aero, teammate, tyre, or event measurements. That requires genuine pre-session
measurements and a matched comparison with and without those inputs on held-out
events, without selecting favorable coefficients after inspecting their results.

# Tyre Degradation And Strategy

Optional race snapshots now accept `tyre_strategy`. Predictions include a `tyre_strategy_analysis`, and backtest reports display best supported alternatives by stop count. Missing inputs leave the existing forecast unchanged. Strategy costs are separate diagnostics: no unvalidated race-time-to-ranking conversion is applied.

## Inputs

Supply evidence availability, a source description, circuit ID, car specification, target track temperature, race length, green-flag pit loss and optional neutralized pit loss. Supply available tyre sets with unique IDs, compound (SOFT/MEDIUM/HARD), initial age and an explicit fresh-compound pace offset in seconds. The dry two-compound constraint defaults on and must be configured for the applicable session; this module does not infer sporting regulations.

Each historical lap needs a globally unique stint ID, lap number, tyre age, lap duration, observation and availability timestamps, circuit ID, car specification, compound, track temperature, clean/dry flags, and fuel correction in seconds. Fuel correction is added to elapsed time to normalize the fuel-burn pace benefit; zero must be an intentional assumption, not missing data. The caller must screen pit, traffic and neutralized laps. All evidence must be available by the forecast cutoff.

Raw collection's observed stint slopes are deliberately not auto-converted: they cannot isolate fuel burn or traffic from tyre wear. A trusted fuel correction and clean-lap assessment are required before those observations become these optional inputs. Fresh-compound offsets are supplied separately because stint intercepts cannot identify them reliably.

## Learning

Match exact circuit, car specification and compound, dry running and track temperature within five degrees. Require at least three stints, each with six clean laps and five laps of tyre-age span. Fit a nonnegative linear-plus-quadratic age curve after removing each stint's intercept, with equal stint weight. This controls differing baseline pace but does not establish causal tyre wear. Report sample counts and within-stint residual error, not a confidence interval.

## Strategy Comparison

Compare available tyre-set orders with zero, one or two stops. Stop timing uses a five-lap grid; this is not a continuous optimum. Set ages advance every lap, sets cannot be reused, stint lengths sum to the race distance, and no stint may go outside the age range common to its supporting historical stints. Report the ten quickest supported alternatives and the best for each stop count.

Costs sum compound offsets, degradation and pit losses. They are relative seconds, not total race times. An optional one-neutralized-stop sensitivity subtracts the difference between green and neutralized pit losses; it does not predict a safety car or its timing. Wet strategies, traffic, undercuts, tyre warm-up and stochastic uncertainty are not simulated.

## Running

Use normal prediction with a snapshot containing the optional input, or run:

```sh
.venv/bin/python scripts/analyze_tyre_strategy.py path/to/snapshot.json --output artifacts/new-strategy-analysis
```

The output directory must be new. It contains `strategy.html` and `strategy.json`. Without adequate comparable data or tyre inventory, the output explicitly reports insufficient evidence rather than inventing curves or extrapolating.

Tests use known synthetic curves to verify parameter recovery, fuel/intercept controls, sparse-data fallback, inventory/age constraints, pit-loss sensitivity and cutoff guards. No real-world accuracy improvement is claimed; a chronological strategy-specific evaluation is still required before changing finishing probabilities.

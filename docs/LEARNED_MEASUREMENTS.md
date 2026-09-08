# Learning from optional measurements

The optional [joint driver/car estimator](JOINT_PERFORMANCE.md) adds season-aware
effects from comparable runs, teammate contrasts and dated upgrade evidence.

The `learned` mode keeps the original Transformer inputs and ratings intact. A
separate regularized linear model learns a small correction to driver pace from
the optional measurements and their quality. Corrections are capped at 0.5 in
model score units. This preserves the existing checkpoint feature schema and
allows old checkpoints to retain their original behavior.

The [final matched backtest](LEARNED_MEASUREMENTS_BACKTEST.md) records all seven
experiments and the remaining collection gap.

## Evidence quality

Quality-aware enrichment requires at least six matched pace laps. Teammate
comparisons count the smaller of the two drivers' supporting samples. Car pace
requires at least two team drivers, each meeting the sample threshold. Matching
uses compound, tyre-age groups, session time, qualifying phase, rainfall and track
temperature groups of 5 C. Only prior weather observations within three minutes
of a lap start may provide its conditions. Pit, deleted, interrupted, generated,
unverified and anomalously slow laps remain excluded from pace evidence.
Stint variability also separates wet/dry conditions and track-temperature groups,
requiring at least six samples and a four-lap tyre-age span within each group.

Every populated measurement includes sample count, cohort count, spread where
available, usable-lap fraction, observation time, source session and event. These
are evidence descriptors; they are not probabilities of correctness. Exact fuel
load, traffic effects and physical braking/aero capability cannot be established
from this timing feed, so they are not fabricated. Stint slopes remain descriptive
and are not automatically used as physical tyre degradation.

## Model behavior

- `legacy`: the existing fixed-weight mappings, retained for compatibility and comparisons.
- `ignore`: original base ratings, without driver/car measurement blending.
- `learned`: the original base ratings plus the separately learned measurement correction.

The separate branch receives values, presence indicators, quality availability,
sample size, spread, usable fraction, age, same-event/practice context and simple
weather/circuit interactions. Missing inputs produce zero correction. Each
feature needs support from at least six earlier training weekends; coefficients
for unsupported features remain zero. Values are scaled by fixed physical units,
not converted into skill ratings. The old `weight=0.25` is not used to blend
ratings in learned mode; setting weight to zero still disables that evidence.

Correction coefficients are fitted to incident-free driver comparisons using
only the training portion of each chronological fold. L2 penalties of 0.1, 1 and
10 shrink weak coefficients toward zero. Earlier validation forecasts choose
whether any candidate correction is used. A candidate must improve both position
log loss and winner Brier by more than 0.001 over the validation baseline, with
at least two measured validation weekends. The unchanged baseline is always a
candidate. Selection may still fail to generalize to future races.

`--calibrate` tests conditional pace temperatures 1, 0.8 and 1.25 on earlier
validation forecasts. The default temperature is retained unless both probability
scores improve. This step precedes correction selection, so optional evidence must
add value beyond any calibration benefit. Retirement risk and weather scenario
probabilities are not recalibrated by this step. Probability calibration on future
events remains an empirical question, and metadata does not claim it established.

The validation window also selects the base model's epoch. It is development data,
not an independent test. Entire target weekends are excluded from fitting and
selection. The stored checkpoint includes selected coefficients, temperature,
training/validation identities and selection scores. Loading checks its feature
schema and coefficient dimensions. No new public-data collection can prove that
a historical archive was available in precisely the same form at the time.

## Usage

```sh
uv run f1-forecast enrich-performance data/processed/real-history-2024-2025.json \
  --collection data/processed/performance-full-2024-2025 --quality-aware \
  --output data/processed/real-history-quality-measurements-new.json \
  --report artifacts/quality-measurements-enrichment-new.json

uv run f1-forecast backtest data/processed/real-history-quality-measurements-v2.json \
  --model transformer --optional-mode learned --calibrate \
  --min-train-events 8 --validation-events 2 --test-from-season 2025 \
  --simulations 1000 --output artifacts/learned-measurements-new.json

uv run python scripts/backtest_measurement_models.py \
  --enriched data/processed/real-history-quality-measurements-v2.json \
  --output artifacts/measurement-model-backtest-new
```

The last command compares original inputs, fixed weights, prediction-only
enrichment, training-only enrichment, learned correction, baseline calibration,
and learned correction with calibration. Each arm uses the same test forecasts,
outcome labels, seed, simulation count and base training settings. Each has a
readable year/circuit/session/scenario report, exact JSON forecasts and paired
uncertainty intervals. The final experimental checkpoint is saved in the new
experiment directory; it does not replace any configured production model.

## Collection status

The full 2024-2025 calendar was requested. The provider rate limit paused new
downloads after 65 sessions. Reusing 10 already collected 2025 sessions produced
75 unique sessions, 40,552 retained laps and 23,882 laps passing the original pace
gates. Quality-aware enrichment additionally requires matched conditions and
sample support and populates 36 of 86 source snapshots. There are 165 pending
sessions, recorded explicitly in the manifest. The collection is not complete.

After the provider limit resets, resume the saved collection with:

```sh
uv run python scripts/collect_full_performance.py \
  --output data/processed/performance-full-2024-2025
```

The collector stops again on provider limits or no progress. No background task
was scheduled. Regenerate enrichment into new files and use a new experiment
directory when collection coverage changes.

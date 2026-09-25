# Verified inputs and separate session mixtures

This experiment tests better measured inputs, separate qualifying/race mixtures,
and small probability corrections. The existing primary checkpoints are retained
unless a candidate satisfies the complete declared policy. Historical outcomes
were already inspected; the experiment is retrospective, not a prospective test.

## Revised interval policy

New primary-bundle selection, architecture comparisons and this experiment use
`interval_quality_policy()`. The original strict `AcceptancePolicy()` remains
available to reproduce the preceding guarded experiments.

The revised policy requires central-80% position intervals to cover at least 80%
of outcomes in each checked group. Their aggregate calendar-weekend bootstrap
95% lower coverage bound must also reach 80%. Quantile bounds are inclusive
integer positions; excess coverage due to discreteness is allowed but is not
required to remain at the incumbent's level. Interval score and width must not
worsen. This avoids rejecting every decrease in excess coverage without giving
unreliable narrow intervals a free pass.

The preregistered tolerances are 0.002 absolute pairwise accuracy (0.2 percentage
points) and 0.02 winner accuracy (2 percentage points). These are non-inferiority
margins, not claimed accuracy improvements. MAE and Brier must improve by at least
0.001 and 0.0001 respectively; position log loss cannot worsen. Three training
seeds and three chronological validation windows remain required. Every relevant
session needs at least six weekends in each selection window. Protected aggregate
uncertainty bounds must pass as well as the seed/window checks. The bootstrap
intervals are marginal, not simultaneous or prospective guarantees.

## Input preparation

```sh
.venv/bin/python scripts/prepare_session_inputs.py \
  --output artifacts/NEW-SESSION-INPUTS
```

The plan is saved before collecting up to twelve missing, recent 2025 FP2 archives.
Targets are determined by date and archive availability, never forecast errors.
Existing normalized archives are checksum-verified and copied into a new
collection. The completed preparation is `artifacts/session-inputs-20260909`.

The importer previously skipped an entire performance object if it already
existed, even when its relevant pace measurement was empty. The opt-in
`fill_missing=True` path fills only absent fields using strict quality cohorts.
Existing measured values, retirement risks, weights, joint effects, and supplied
quality descriptions are preserved. Explicit zero-weight objects stay disabled.
The combined object's availability is no earlier than either original or new
evidence. Existing object weights still apply to the added fields; they are not
newly calibrated reliability estimates.

Preparation uses practice sessions only, with matching driver/team identities,
the same season, at most 30 days of age and availability before prediction.
Historical availability retains the explicit assumption of six hours after the
reconstructed session end. No target qualifying/race laps or outcomes are used as
input, and weather observations are not relabelled as forecasts. Feedback, grids,
entrants and non-measurement inputs are checked for exact preservation.

Twelve additional practice archives were collected. In recent Grand Prix driver
rows, missing checked driver pace fields decreased from 73.4% to 54.1%; missing
car pace fields decreased from 62.7% to 48.0%. Seventy-seven Grand Prix and fifteen
sprint snapshots gained measurements. Remaining missing values stay missing;
fuel-corrected tyre curves, verified grid penalties and detailed upgrade history
are not invented by this preparation.

## Frozen rolling experiment

```sh
.venv/bin/python scripts/backtest_session_mixtures.py \
  --enriched artifacts/NEW-SESSION-INPUTS \
  --output artifacts/NEW-SESSION-EXPERIMENT
```

Existing and enriched inputs receive otherwise identical Transformer recipes,
with seeds 42, 43 and 44. The extra quality-feature, recency and listwise options
are disabled to isolate the effect of measured inputs. For each of three rolling
folds, development, calibration fitting, candidate selection and six later test
weekends are disjoint and chronologically ordered. Feedback availability is
checked at each boundary. An earlier outer test may enter later development only
after its outcomes would have become available.

Qualifying and race candidates are selected separately. Fixed heuristic shares
are 0%, 5%, 10% and 20% for each input family. A separate earlier calibration
window with at least twelve weekends per session selects among temperatures
1.0, 0.8 and 1.25. Only 10% of the distribution receives a non-identity correction.
Fitting minimizes mean position log loss plus winner Brier and a small temperature
penalty. It must improve the fitting objective by 0.001 to change temperature.
This is fitting, not acceptance: the later selection gate must still pass.

Mixtures and temperature corrections preserve complete position distributions
and their row/column sums. Identity returns the original distribution exactly.
Each session selects only passing candidates; a combined selection gate checks
the resulting policy. Failure returns the incumbent. Choices and their feedback
cutoff are saved before generating later predictions. Later diagnostic scores do
not get another opportunity to choose a candidate.

The final run is `artifacts/session-mixtures-20260912-final`. Its `report.md`
provides a readable summary; `comparison.json` contains paired weekend intervals
and breakdowns by session, quality tier, season, retirement involvement and seed.
Each fold stores checkpoints, calibration fitting scores and every rejection
reason in `frozen-selection.json`. The root `recipe.json` records source and code
hashes, thresholds, candidate grids, seeds and calendar partitions. No installed
checkpoint is replaced and no background monitoring is scheduled.

## Completed evaluation

The final run evaluated 34 sessions across 18 distinct later Grand Prix weekends,
with three training seeds (102 paired forecasts). All three folds retained the
incumbent for both qualifying and race. Candidates failed metric protections in
individual windows/seeds or lacked sufficient paired-weekend evidence to exclude
regression. Sprint evaluation was skipped because later-weekend support was
insufficient.

The enriched-input diagnostic improved winner Brier from 0.802881 to 0.794550 and
position log loss from 2.666917 to 2.665340, but worsened MAE from 3.299947 to
3.308489 and interval score from 14.228659 to 14.252094. It was not promoted.
These numbers describe this rolling evaluation, not the older 668-session
comparison. The selected procedure exactly preserves incumbent predictions.

Remaining priorities are reducing the still-large measured-input gaps and
collecting enough later sprint weekends. Any further candidate procedure must be
fixed before its confirmation outcomes arrive; these inspected historical
results cannot become an untouched confirmation set.

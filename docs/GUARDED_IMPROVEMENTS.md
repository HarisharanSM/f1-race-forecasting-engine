# Guarded accuracy and probability improvements

The shared acceptance gate protects the incumbent before any candidate can be selected.
This page describes the original strict experiments. New primary bundles and architecture comparisons now use the [revised interval-quality policy](SESSION_MIXTURES.md); the strict policy remains available for reproducing these experiments.
Experiments preserve existing checkpoints, input snapshots and reports.

## Acceptance contract

Candidates must reduce validation expected-position MAE by at least 0.001 and
winner Brier by at least 0.0001. Position log loss, pairwise accuracy, winner
accuracy, central-80% interval coverage, interval width and interval score must
not worsen. Interval score is width plus ten times the distance of an outcome
outside the interval. Coverage is protected conservatively here, as requested;
an increase in coverage alone is not an improvement in calibration.

Selection requires three chronological windows, each containing at least six
weekends for every relevant format/session, and three separately trained seeds
on identical sessions. Window reservation expands when classifications leave
partial weekends. Only dates and session coverage determine this expansion,
never scores or outcomes. Each seed and each session/window must protect every
metric. Aggregate paired-calendar-weekend 95% intervals must also satisfy the
improvement thresholds and non-regression constraints. Whole weekends retain
all session, format and seed rows together when resampled; sessions are equally
weighted. Seeds are not counted as additional independent weekends.

These marginal bootstrap intervals are not a simultaneous statistical guarantee
and do not capture all temporal dependence or repeated model-development bias.
A zero change in all scores does not qualify as an improvement. It is the exact
fallback when no candidate passes. Sparse evidence is rejected rather than
loosening the contract. All rejection reasons are saved.

Version-1 primary bundles remain loadable, but their old optional mixtures return
the exact primary because their two-weekend selection did not satisfy this gate.
Version-2 bundles persist the policy, seed evidence and all candidate audits.
The default primary checkpoint is not replaced by this change.

## Mixture replay

```sh
.venv/bin/python scripts/backtest_guarded_mixture.py \
  --output artifacts/NEW-MIXTURE-EXPERIMENT
```

The runner verifies source checksums and reproduces both sets of saved metrics
before evaluating 5%, 10%, 20% and 25% heuristic shares. It blends complete
position-probability matrices, including matching scenarios; win, podium,
expected positions and intervals are derived coherently from the mixture.
Zero heuristic weight uses the original forecast exactly.

The recipe is written before replay. Seasons before 2025 select weights; the
selection is frozen before reporting later comparison metrics. Saved forecasts
contain only one training trajectory, so this screen alone cannot pass the
three-seed gate. Later candidate scores are diagnostics, not a second selection.

Reports contain paired error breakdowns by qualifying/race, Grand Prix/sprint,
input quality tier, season, retirement involvement and training seed. The
`richer_existing` tier means a saved snapshot without the explicit reduced-input
marker; it does not assert that every optional measurement is present.

## Recent input and training experiments

```sh
.venv/bin/python scripts/backtest_guarded_training.py \
  --output artifacts/NEW-TRAINING-EXPERIMENT
```

This runner uses existing validated snapshots from 2023 onward, with a separate
six-weekend later comparison. It reserves enough preceding selection weekends
for the complete gate and fits components before the first selection window.
The same frozen components predict all selection and later weekends. This is a
matched experiment, not a replay of the earlier expanding-history model scores.
Sprint is skipped if it lacks the required training, selection and later support.

Fixed recipes, each repeated with seeds 42, 43 and 44:

1. The unchanged 27-feature incumbent recipe.
2. Eight added quality indicators: reduced inputs, imputed weather, missing race
   qualifying order, missing driver/car pace evidence, known measurement quality,
   usable fraction and observation age. Missing evidence is distinct from zero.
   Imputed feedback weather cannot act as an observed training condition.
3. Those indicators plus a 365-day training-weight half-life, measured relative
   to the newest eligible training session. Epoch validation stays unweighted.
4. The preceding recipe plus clean finishing-order log likelihood with weight
   0.1, alongside the existing pairwise and retirement objectives. It models
   the joint clean order under the Gumbel pace model, not the complete disrupted
   race outcome; final forecast scores still determine acceptance.

The four small heuristic mixtures are also evaluated against each seed's matched
incumbent. No additional source downloads or fabricated values are introduced.
The input-quality audit shows what is actually present in recent snapshots.
The new training options default off, preserve legacy checkpoint dimensions,
and require retraining for the expanded feature schema. Quality features are
currently restricted to legacy adapters without the separate pace calibrator.

## Artifacts and interpretation

The completed runs are `artifacts/guarded-mixture-20260909-final` and
`artifacts/guarded-training-20260909-final`. Each has a readable `report.md`, exact
comparison JSON, paired rows, source/code hashes and frozen selection records.
The training run also saves its checkpoints and input-quality audit.

Historical results had already been inspected before these experiments. Freezing
a recipe now makes its execution auditable; it does not turn known historical
outcomes into an untouched test. No successful candidate is automatically
promoted, and no future monitoring is scheduled. Before a prospective claim,
freeze the selected model, data cutoff, feature recipe, gate and evaluation
procedure before those future weekend outcomes arrive.

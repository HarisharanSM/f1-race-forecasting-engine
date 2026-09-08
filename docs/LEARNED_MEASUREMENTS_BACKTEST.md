# Learned measurement backtest, 5 September 2026

For the subsequent continuous 2020-2025 evaluation, including 2024 test forecasts,
see [the full historical backtest](BACKTEST_2020_2025.md). Its training history and
sprint validation window differ, so compare models within each experiment.

The revised model modestly improved average position error and probability scores
over the original baseline. Winner/pole accuracy was unchanged. Most of the gain
came from calibration; measured-input corrections contributed a smaller increment.
The result is exploratory, and does not establish a reliable position-error gain.

## Final matched results

Seven fresh runs evaluated the same 45 held-out 2025 forecasts across 23 weekends,
using seed 42, 1,000 simulations per scenario and the existing Transformer training
configuration. This is 45 test forecasts repeated seven ways, not 315 independent
sessions. Whole target weekends were excluded from fitting and validation.

| Experiment | Position error | Pairwise accuracy | Winner/pole accuracy | Winner Brier | Position log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original inputs | 3.320441 | 75.9805% | 33.3333% | 0.796308 | 2.704769 |
| Fixed weights, expanded quality-aware data | 3.315649 | 75.6725% | 33.3333% | 0.802128 | 2.706800 |
| Enriched at prediction only | 3.327534 | 76.0273% | 35.5556% | 0.794840 | 2.704752 |
| Enriched during training only | 3.310852 | 75.6959% | 31.1111% | 0.802876 | 2.706685 |
| Separate learned correction | 3.312350 | 76.0741% | 33.3333% | 0.795836 | 2.704214 |
| Original inputs plus calibration | 3.305902 | 75.9454% | 33.3333% | 0.794460 | 2.698268 |
| Learned correction plus calibration | 3.302956 | 76.0039% | 33.3333% | 0.794131 | 2.696996 |

Position error, Brier and log loss are better when lower. Relative to the original
baseline, the final approach reduced position error by 0.5266%, Brier by 0.2734%,
and log loss by 0.2874%. Winner/pole picks remained 15/45. Displayed exact-position
confidence rose from 10.8610% to 11.1176%, an increase of 0.2566 percentage points.
Displayed confidence alone is not a calibration metric.

Nominal 80% position-interval coverage changed from 87.2807% to 87.1696%, with mean
width shrinking from 10.8921 to 10.7692 positions. Position outcomes are discrete,
and these wide intervals do not establish useful or fully calibrated confidence.

The final learned correction was selected in 2 of 23 test folds. Temperature
changed in 5 of 23 folds. Only earlier validation forecasts made these decisions.
The original model with calibration alone achieved a position error of 3.305902,
so the additional optional-measurement benefit was 0.002946 positions on average.
No settings were tuned using the reported held-out outcomes.

## Uncertainty and coverage

The 95% paired-weekend bootstrap interval for the final change in position error
was -0.042392 to +0.001145 positions, spanning no improvement. The Brier interval
was -0.005058 to -0.000303; the log-loss interval was -0.017496 to +0.000208. These
5,000-resample intervals describe this experiment only: training-seed variation,
dependence across expanding training windows, prior inspection of 2025 results
and imperfect historical input vintages limit broader claims.

The collection contains 75 sessions and 40,552 laps, including 23,882 passing the
original pace gates. New quality-aware matching populates 36 of 86 snapshots,
including 10 held-out forecasts across five weekends. The other enriched records
expand earlier training evidence. The provider rate limit paused downloads with
165 sessions still pending. Thus the full-season collection recommendation is not
yet fulfilled; the saved manifest supports resumption after the limit resets.

During verification, stint variability was found to pool wet/dry observations.
The final implementation separates conditions, adds a regression test and reruns
all seven arms. Use the `v2` data/report below; the initial run is retained as an
intermediate development artifact. The correction was made for input consistency,
not to optimize test outcomes.

## Artifacts and verification

- Dataset: `data/processed/real-history-quality-measurements-v2.json`.
- Dataset SHA-256: `2f79b14cd1dce9976c6ae7b87d26b1eee8c36f68141b24c40313e92b1b21693a`.
- Enrichment audit: `artifacts/quality-measurements-enrichment-v2.json`.
- Final comparison: `artifacts/measurement-model-backtest-v2/comparison.html`.
- Exact paired metrics: `artifacts/measurement-model-backtest-v2/comparison.json`.
- Session report: `artifacts/measurement-model-backtest-v2/learned_calibrated.html`.
- Experimental checkpoint: `artifacts/measurement-model-backtest-v2/model`.

All 161 tests passed. Checks include original-rating preservation, distinguishing
missing from zero measurements, regularization, support thresholds, checkpoint
round trips, future-observation rejection, held-out-label/weather invariance,
ablation input identity, weather matching and wet/dry stint separation. Lint,
formatting and diff checks passed. Three dependency deprecation warnings remain.
The final saved checkpoint was loaded successfully. Existing source datasets,
reports and configured models were preserved.

See [implementation and reproduction](LEARNED_MEASUREMENTS.md) for model controls,
collection resumption and complete limitations.

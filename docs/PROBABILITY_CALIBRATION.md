# Probability Calibration

## Conservative Revision

The runner now defaults to conservative calibration. Use `--method legacy` to reproduce the original experiment described below. Existing saved artifacts keep their original behavior; default forecasts still have no calibration applied.

The revised split fits temperature on 76 sessions from 2020-2021 and checks blends separately in 2022, 2023 and 2024 (153 selection sessions). Candidate blends retain 90%, 75% or 50% of the original distribution. The smallest successful correction is chosen, or exact identity if none passes. Both original and balanced matrices are coherent, so their blend preserves driver and position probability sums. DNF and weather probabilities remain unchanged.

Each validation season must have at least six weekends for that format/session, with at least twelve fitting weekends. Every season must improve position log loss by more than 0.001 and winner Brier by more than 0.0001. Winner accuracy must not decrease; position MAE may increase by at most 0.01, interval width by at most 0.25 positions, and coverage may decline by at most one percentage point. These fixed practical thresholds are conservative screening rules, not statistical guarantees. Full per-window scores and rejection reasons are saved in the audit.

All four groups fell back to identity. On the 82 later sessions, original and revised scores are exactly equal: winner Brier 0.7810, position log loss 2.6564 and position MAE 3.1410. This avoids the original calibration downgrade; it does not improve the original model.

The revised method was developed after inspecting the previous later-period failure. Consequently, 2025-2026 is a retrospective regression comparison, not an untouched test, even though its labels are excluded from fitting and selection. A genuinely unseen future evaluation remains necessary. The optional artifact must be explicitly selected; no defaults were replaced and no future monitoring was scheduled.

The optional probability calibrator checks predicted chances against observed frequencies. It is separate from the existing pace-temperature training option. Default forecasts remain unchanged.

## Chronological Evaluation

- Fit on 176 earlier out-of-fold forecasts from 2020-2023.
- Select the fitted map or identity on 53 forecasts from 2024.
- Freeze the artifact before evaluating 82 later forecasts from 2025-2026.
- Verify source dataset hashes, backtest folds, feedback availability, driver rosters, and disjoint calendar weekends. Reject applying calibration to development weekends or earlier forecast cutoffs.

These are retrospective holdouts for calibration, not genuinely untouched prospective weekends: the project previously inspected these historical results. Base models continue their existing chronological updates. No holdout outcomes are used to refit calibration.

## Method

Separate temperatures are fitted for Grand Prix and sprint qualifying/race. The fixed grid is 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2. Fitting minimizes mean position log loss plus 0.002 times squared log temperature. Selection requires a log-loss improvement exceeding 0.001. At least 12 fitting and 6 selection weekends are required per group; otherwise the identity map is retained.

Each scenario's position matrix is power-transformed and row/column balanced. Each driver's distribution still sums to one, and each position has one expected occupant. Scenario weights are unchanged. Win, podium, expected position and intervals are recomputed from the resulting distributions. Retirement and weather probabilities are not calibrated.

Reliability diagrams compare mean predicted and observed frequencies for win, podium, displayed position and retirement. Sessions have equal weight. Bins with fewer than 30 driver observations or 10 weekends are labelled sparse. Driver observations are dependent; score-difference intervals use 5,000 paired whole-weekend bootstrap draws, seed 42, and do not account for all temporal dependence or development bias.

## Observed Results

Grand Prix qualifying selected temperature 1.5; its 2024 selection log loss improved from 2.7503 to 2.7149. Grand Prix race retained identity. Sprint groups had insufficient fitting support and also retained identity.

On the later 82 sessions:

| Metric | Original | Calibrated |
| --- | ---: | ---: |
| Position log loss, lower is better | 2.6564 | 2.6580 |
| Winner Brier, lower is better | 0.7810 | 0.7860 |
| Expected position error, lower is better | 3.1410 | 3.1848 |
| Position interval coverage | 87.01% | 88.48% |
| Mean interval width | 10.28 | 10.88 |

The map did not generalize. Increased interval coverage accompanied wider intervals and is not evidence of improved confidence quality. The experiment remains opt-in and is not promoted to production defaults. Do not retune against these holdout results and then describe them as untouched.

## Reproduction And Use

Run `.venv/bin/python scripts/calibrate_probabilities.py --output artifacts/NEW-DIRECTORY` from the repository. Output must be a new directory. It contains the frozen calibrator, a deployment review, reliability plots and exact metrics, and detailed before/after reports preserving the year/circuit/session workflow and development-role labels.

Prediction supports `--probability-calibrator PATH` as an explicit experimental option. The artifact rejects earlier cutoffs, development weekends, mismatched model family or data mode, and repeated application. It is not interchangeable with a newly configured model: freeze the model recipe together with the calibration artifact for a prospective evaluation. A model change requires a fresh earlier-data calibration study.

For an honest untouched evaluation, register and freeze the full procedure before future weekend outcomes become available. No future evaluation is automatically scheduled by this experiment.

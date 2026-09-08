# Optional-input backtest comparison, 5 September 2026

The original source was preserved before implementation and both versions were
run afresh with the same datasets, training configuration, chronological folds,
simulation counts and random seeds. Each version trained expanding-window
Transformer models for all 233 previously held-out sessions. Neither used an LLM.

| Experiment | Held-out sessions |
| --- | ---: |
| Grand Prix, 2025 | 45 |
| Sprint, 2025 | 10 |
| Grand Prix, 2020-2023 | 162 |
| Sprint, 2021-2023 | 16 |
| Total | 233 |

There are 331 source session records across the four datasets, including development
sessions. The split contains 117 qualifying and 116 race forecasts. Shared 2021-2022
Friday qualifying appears in both separately trained formats, as in the existing
experiment; these are not 233 independent physical sessions.

## Results

Metrics below average over held-out forecasts; driver metrics first average within
each session. All 233 complete position-probability distributions were unchanged.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Expected-position MAE, lower is better | 3.316587 | 3.316587 | 0 |
| Pairwise ranking accuracy | 76.4666% | 76.4666% | 0 percentage points |
| Winner/pole accuracy | 44.6352% | 44.6352% | 0 percentage points |
| Winner Brier score, lower is better | 0.776096 | 0.776096 | 0 |
| Position log loss, lower is better | 2.707028 | 2.707028 | 0 |
| Mean displayed exact-position confidence | 11.6601% | 11.6601% | 0 percentage points |
| Nominal 80% position-interval coverage | 86.9392% | 86.9392% | 0 percentage points |
| Mean interval width, positions | 10.950836 | 10.950836 | 0 |

Race-only winner accuracy was 56.0345%; qualifying pole accuracy was 33.3333%, both
unchanged. Broad intervals and discrete positions affect coverage: coverage above
80% alone is not evidence of well-calibrated or useful confidence.

## Interpretation

No source records contain the newly supported optional measurements or race-event
priors. Missing inputs deliberately retain original behavior. Thus the measured
accuracy improvement is **0%**, and confidence/probability quality is **unchanged**.
This is evidence of compatibility, not evidence that the added inputs help or hurt.

The mappings from measurements to ratings and the event effects remain heuristics.
Functional tests establish that supplied inputs are validated and influence the
appropriate computations. They do not establish real-world predictive improvement.
Evaluating that requires genuine pre-session measurements, held-out events and a
comparison with the same model/settings using and omitting those measurements.
Actual target-session incidents must remain feedback rather than forecast inputs.

Historical inputs retain their existing reconstruction and publication-time
limitations. Training selects epochs using earlier validation weekends; these
results do not evaluate a final checkpoint on its own training data.

## Artifacts

- `artifacts/optional-inputs/before/`: four fresh original-code backtests.
- `artifacts/optional-inputs/after/`: four fresh updated-code backtests.
- `artifacts/optional-inputs/after/comparison.json`: exact matched results, deltas,
  session-type breakdowns and optional-data coverage.
- `artifacts/optional-inputs/after/backtest-report.html`: regenerated combined report.
- `artifacts/optional-inputs/after/backtest-report.csv`: scenario table export.

See `scripts/recheck_optional_inputs.py` for reproduction and
`docs/OPTIONAL_INPUTS.md` for input units, mappings and model limits.

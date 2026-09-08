# Expanded collection backtest, 5 September 2026

## Outcome

The enriched data did not improve the model overall in this experiment. Keep it
optional and do not treat this run as evidence to replace the existing model.

Both arms were freshly trained on expanding historical windows, evaluating the
same 45 held-out 2025 sessions (23 qualifying, 22 races, across 23 weekends).
Entire target weekends were excluded from model training and inner validation.
Original outcome labels and non-optional forecast inputs were verified unchanged.

| Metric | Original inputs | Enriched inputs | Change |
| --- | ---: | ---: | ---: |
| Average expected-position error, lower better | 3.320441 | 3.328451 | +0.008010 (0.24% worse) |
| Pairwise ranking accuracy | 75.9805% | 75.8025% | -0.1780 percentage points |
| Winner/pole accuracy | 33.3333% (15/45) | 31.1111% (14/45) | -2.2222 percentage points |
| Winner Brier score, lower better | 0.796308 | 0.798420 | +0.002112 |
| Position log loss, lower better | 2.704769 | 2.705974 | +0.001205 |
| Mean displayed exact-position confidence | 10.8610% | 10.8229% | -0.0381 percentage points |
| Nominal 80% interval coverage | 87.2807% | 87.5029% | +0.2222 percentage points |
| Mean position-interval width | 10.892105 | 10.958187 | +0.066082 positions |

Higher coverage here comes with wider intervals. It is not evidence of better
calibration. Neither probability score improved overall.

## Coverage and uncertainty

Only the first two weekends of each of 2024 and 2025 were collected. Their
measurements populate 18 of the 86 input snapshots, including 10 of the 45 held-out
forecasts. The 30-day prior-session window allows measurements to appear in later
weekends, not just the four collected weekends. Training on enriched earlier
records also changes later forecasts that have no direct optional measurements.

For the 10 directly enriched forecasts (five weekends), pairwise accuracy rose
from 78.6842% to 78.9474%, but average position error worsened from 3.113682 to
3.138629. Winner/pole picks fell from 3/10 to 2/10; Brier score improved slightly
while position log loss worsened. This is a small, mixed sample, not a reliable win.

The overall 95% paired-weekend bootstrap interval for the change in position error
was -0.006773 to +0.023265 positions, spanning no change. Intervals use 5,000 paired
weekend resamples. They do not capture training-seed uncertainty or remove the
dependence between expanding training folds. Historical input availability is
reconstructed, not a verified point-in-time vintage archive. The 2025 period has
been evaluated previously, so this is exploratory rather than a fresh untouched
prospective test. No model tuning was performed using these results.

## Reproduction and artifacts

Both runs used the existing saved configuration: up to 60 epochs, patience 8,
eight minimum training weekends, two validation weekends, seed 42, and 1,000
simulations per weather scenario. Execution used deterministic CPU operations
with one Torch thread. No final deployed checkpoint was overwritten.

```sh
uv run python scripts/backtest_expanded_collection.py \
  --output artifacts/expanded-collection-backtest-new
```

The script refuses to overwrite an existing experiment directory.

- `artifacts/expanded-collection-backtest/comparison.html`: paired summary.
- `artifacts/expanded-collection-backtest/comparison.json`: exact metrics, intervals and per-session results.
- `artifacts/expanded-collection-backtest/after-report.html`: enriched year/circuit/session/scenario report.
- `artifacts/expanded-collection-backtest/before-report.html`: matching original-input report.
- `artifacts/expanded-collection-backtest/before.json` and `after.json`: full fresh backtests.

The existing combined 2019-2025 report and original datasets remain unchanged.

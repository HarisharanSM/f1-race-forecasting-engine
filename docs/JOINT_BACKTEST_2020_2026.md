# Joint-effects backtest: 2020-2026 year to date

Completed on 6 September 2026. The report contains 311 held-out forecasts across
138 calendar weekends: 271 Grand Prix and 40 sprint forecasts. The 358 source
records also contain 47 development-only records. The latest evaluated session
is Italian Grand Prix qualifying on 5 September 2026; the race is not included.

The matched comparison is the previous learned/calibrated measurement model
versus the same model with optional joint driver/car effects. It is not a
comparison with an uncalibrated or frozen heuristic model.

| Year | Forecasts | Previous model error | Joint model error | Joint input coverage |
| --- | ---: | ---: | ---: | ---: |
| 2020 | 33 | 3.270804 | 3.270804 | 0 |
| 2021 | 43 | 2.916041 | 2.916041 | 0 |
| 2022 | 50 | 3.351136 | 3.351136 | 0 |
| 2023 | 50 | 3.438240 | 3.438240 | 0 |
| 2024 | 53 | 2.997128 | 2.996071 | 53 |
| 2025 | 55 | 3.258908 | 3.259138 | 55 |
| 2026 YTD | 27 | 2.901199 | 2.900473 | 27 |

Error is average absolute error in expected finishing position; lower is better.
Coverage means an estimate exists, not that it has fresh observations or was
selected for use by the validation gate.

## Overall result

| Metric | Previous model | Joint model |
| --- | ---: | ---: |
| Average position error | 3.180756 | 3.180554 |
| Pairwise accuracy | 77.6332% | 77.6110% |
| Correct winner/pole picks | 138 / 311 | 138 / 311 |
| Winner Brier, lower better | 0.762292 | 0.762408 |
| Position log loss, lower better | 2.669226 | 2.670309 |
| Mean displayed position confidence | 12.4674% | 12.4748% |

The position-error reduction is only 0.0064%. The paired-weekend bootstrap 95%
interval for joint-minus-previous error is -0.000840 to +0.000406 positions,
including zero. Probability quality worsens slightly; higher displayed confidence
is not a calibration improvement. This does not establish a useful overall gain.
The full-position log-loss change interval is +0.000282 to +0.002091. These
retrospective bootstrap intervals do not account for all temporal dependence or
repeated inspection of historical results.

For 2026 alone, position error changes from 2.901199 to 2.900473, while position
log loss worsens from 2.570325 to 2.578165. Winner/pole picks remain 7 of 27.

## Data and exclusions

The 2026 source cutoffs are approximately 08:58 UTC on 6 September. There are
23 usable Grand Prix records (12 qualifying, 11 races) and four usable sprint
records (British and Dutch sprint qualifying and races).

- Australian GP qualifying lists only 19 of the 22 race entrants. Both sessions
  are excluded rather than inventing the remaining qualifying positions.
- Chinese and Canadian sprint qualifying have incomplete rosters; Miami sprint
  qualifying is incomplete or differs from the race roster. These weekends are
  excluded from sprint evaluation.
- Italian GP qualifying has a verified 22-driver roster. Its race and subsequent
  Grands Prix have no eligible completed result at the collection cutoff.
- The collection added 55 usable session archives from 2026 before a provider
  rate limit paused downloads. The combined timing archive has 130 collected
  sessions, 72,800 retained laps and 41,644 laps passing the original pace gates.
  Joint matching applies additional conditions. Sixty timing sessions remain
  pending, including future sessions; no background retry was scheduled.
- No verified upgrade chronology was supplied. Missing physical inputs are not
  invented. Evidence age and conditional effect uncertainty remain visible.

## Verification and continuity

The 2020-2025 joint comparison was run first. All its 284 forecasts per arm were
then preserved exactly. Before reusing them, the extension checked the complete
record prefix, dataset hash, training configuration, simulation count and seed.
The newly trained 2026 folds use all eligible preceding history and retain the
same event-index seed convention as a fresh full-period run. Both formats keep
separate models. Each target weekend is excluded from its own fitting and
validation, including when it contains only qualifying.

The fresh previous-model GP arm reproduced all 248 forecasts from the earlier
2020-2025 revised experiment exactly. All 176 forecasts from 2020-2023 remain
unchanged between arms, as no joint timing data exists for those years. Source
manifests, publication cutoffs, fold separation and exact report datasets passed
validation. All 183 tests passed; code lint, formatting and diff checks passed.
Six dependency deprecation warnings remain.

## Files and reproduction

- `artifacts/joint-effects-backtest-2020-2026/revised-report.html`: combined report.
- `artifacts/joint-effects-backtest-2020-2026/comparison.html`: yearly comparison.
- `artifacts/joint-effects-backtest-2020-2026/comparison.json`: exact scores and intervals.
- The same directory contains original/previous/enriched records, source
  manifests, enrichment audits and both formats' backtests.

```sh
uv run python scripts/backtest_2020_2025.py --joint-effects --end-season 2026 \
  --resume-from artifacts/joint-effects-backtest-2020-2025 \
  --collection data/processed/performance-with-2026 \
  --output artifacts/joint-effects-backtest-2020-2026-new
```

Omit `--resume-from` for a fresh full-period rerun. This uses the saved verified
2026 YTD datasets, not a refreshed later vintage. Current-year collection requires
`fetch-history --include-current-season`; outputs must use new paths. The importer
rejects future years, waits for the six-hour result-publication assumption, and
records incomplete/unavailable sessions. Weather remains an archived day-ahead
forecast reconstruction, not target-session observed weather supplied as an input.

No existing reports or deployed model checkpoints were replaced.

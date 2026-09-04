# Backtests for 2020–2023

The combined readable report now accepts earlier-year backtests alongside the existing 2024–2025 history. Historical Grand Prix runs use 2019 for initial training so eligible 2020 sessions can be evaluated without training on their own results. Earlier 2020–2023 weekends progressively enter later training windows. The latest two eligible Grand Prix weekends are held out for epoch selection, and the target weekend is excluded from both fitting and validation.

## Weather data is different in these years

The fixed-lead weather product used in the 2024–2025 dataset does not cover these years. The earlier importer uses **actual F1 practice observations carried forward as a persistence estimate**. It does not call observed target-session weather a forecast.

- Weather observations, session timing and race-control messages come from the F1 timing archive, with FastF1's documented public mirror used for unavailable original files.
- The latest usable practice must have ended at least one hour before the prediction cutoff. Practice observations older than four days are rejected.
- Its mean temperature and wind are carried forward. Its observed rain fraction, clipped to 5–95%, is an explicitly heuristic rain prior.
- The cutoff is one hour before the session's reconstructed timing-record start. This accounts for rescheduled sessions while retaining the broader historical-reconstruction limitation.
- Target-session observed weather is feedback only. Practice and target observations are separately sourced and timestamped.

The report labels this method on each affected session and in the year summary. Scores across years do not isolate model changes because weather input methods and racing conditions differ.

## Results, session formats and availability

Grand Prix and sprint classifications come from Jolpica. For 2023 sprint shootouts, the importer uses OpenF1's exact session result. Complete classifications and matching fields are required; invalid rows are excluded rather than renumbered or invented.

In 2021–2022, Friday qualifying set the Sprint grid, and the Sprint then set the Grand Prix grid. The same Friday qualifying classification appears in both separately trained report formats and is clearly labelled. These are two model evaluations of a shared session, not two distinct qualifying sessions. The Grand Prix race uses the actual archived start grid.

A race's qualifying-order input is assumed available 30 minutes after qualifying ends, to support the same-day 2023 shootout/race schedule. Feedback for ML remains delayed six hours after session end. Neither assumption is independently verified publication timing. The existing 2024–2025 reports retain their original assumptions and scores.

Sprints did not exist in 2020. The earliest 2021 sprint weekends form the initial development window: one weekend for fitting and one for validation. Forecasts begin at the next usable sprint weekend. This small early sample makes initial sprint forecasts particularly uncertain. Sprint and Grand Prix models remain separate.

Repeated visits to a circuit are selected by year and round, so Austria/Styria and the two Silverstone events are individually accessible. Missing sessions and early training-only sessions remain explicit.

## Recorded run

The completed run adds **178 held-out forecasts** and saves separate final Transformer checkpoints. All forecasts were fixed before their target feedback could enter later training. The combined report retains the previous 55 forecasts, for 233 in total.

| Year | Grand Prix qualifying/race forecasts | Sprint qualifying/race forecasts | GP average position error: Transformer / baseline |
| --- | ---: | ---: | ---: |
| 2020 | 33 | 0 | 3.30 / 3.12 |
| 2021 | 43 | 2 | 2.94 / 2.77 |
| 2022 | 44 | 6 | 3.28 / 3.21 |
| 2023 | 42 | 8 | 3.48 / 3.46 |

Average position error compares expected positions with actual classifications; lower is better. The Transformer did **not** outperform the baseline on this metric in any year, for either format. These backtests establish measured performance rather than demonstrate improvement. Confidence remains uncalibrated.

Coverage exclusions: 2020 Eifel qualifying lacks usable prior practice weather; 2021 Monaco race has a roster/classification mismatch; both 2023 British GP sessions fail the qualifying classification check. Belgian and Qatar 2023 sprint weekends fail completeness or roster checks. One 2019 warm-up race is also excluded. The first two 2021 sprint weekends are training and validation history, leaving São Paulo as that year's held-out sprint weekend.

Verification passed for dataset and raw-record checksums, chronological folds, target-weekend exclusion, prior practice timing, forecast schemas and both saved model checkpoints. The code suite passed 84 tests. Detailed evidence is saved in `artifacts/legacy-data/verification.json`; yearly scores are in each backtest's `by_season` field.

## Reproduce collection and training

```sh
f1-forecast fetch-history --seasons 2019 2020 2021 2022 2023 \
  --weather-mode practice_persistence \
  --output data/processed/history-2019-2023-complete.json --cache data/raw/historical
f1-forecast backtest data/processed/history-2019-2023-complete.json \
  --model transformer --test-from-season 2020 --min-train-events 8 --validation-events 2 \
  --epochs 60 --patience 8 --simulations 1000 --seed 42 \
  --output artifacts/legacy-data/backtest.json --save-model artifacts/legacy-data/model

f1-forecast fetch-history --seasons 2021 2022 2023 --race-format sprint \
  --weather-mode practice_persistence \
  --output data/processed/sprint-history-2021-2023-final.json --cache data/raw/historical
f1-forecast backtest data/processed/sprint-history-2021-2023-final.json \
  --model transformer --test-from-season 2021 --min-train-events 1 --validation-events 1 \
  --epochs 60 --patience 8 --simulations 1000 --seed 42 \
  --output artifacts/legacy-sprint/backtest.json --save-model artifacts/legacy-sprint/model
```

Use new output and checkpoint paths when repeating these commands. The cache retains raw responses and actual retrieval times; collection can be reconstructed from it without replacing the already saved results.

## Update the combined report

```sh
f1-forecast report artifacts/real-data/backtest.json \
  --records data/processed/real-history-2024-2025.json \
  --manifest data/processed/real-history-2024-2025.manifest.json \
  --sprint-backtest artifacts/sprint-data/backtest.json \
  --sprint-records data/processed/sprint-history-2024-2025.json \
  --sprint-manifest data/processed/sprint-history-2024-2025.manifest.json \
  --add-backtest artifacts/legacy-data/backtest.json data/processed/history-2019-2023-complete.json data/processed/history-2019-2023-complete.manifest.json \
  --add-backtest artifacts/legacy-sprint/backtest.json data/processed/sprint-history-2021-2023-final.json data/processed/sprint-history-2021-2023-final.manifest.json \
  --output artifacts/real-data/backtest-report.html
```

The renderer verifies each backtest against its exact dataset checksum. It adds the old results without rerunning or changing the existing 2025 predictions. The HTML and CSV contain every saved scenario and its actual result, plus the feedback audit. A local preview may need refreshing after the report is rebuilt.

Sources: [FastF1 timing archive implementation and public mirror](https://github.com/theOehrly/Fast-F1/blob/master/fastf1/_api.py), [FastF1 historical session-format handling](https://github.com/theOehrly/Fast-F1/blob/master/fastf1/events.py), [Jolpica](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md), [OpenF1](https://openf1.org/docs/), [fixed-lead weather archive coverage](https://open-meteo.com/en/docs/previous-runs-api).

# Readable backtest results

Open `artifacts/real-data/backtest-report.html` in a browser. It is a self-contained report that works offline, with a companion CSV for spreadsheet analysis. The local preview is available at `http://127.0.0.1:8765/backtest-report.html` while its preview server is running.

Choose **year → circuit → session → scenario**. Each session contains:

- The actual session date and the forecast cutoff.
- Separate Grand Prix qualifying/race and sprint qualifying/race views.
- A combined forecast plus up to five scenario tables (three weather scenarios in the current runs).
- Common-language descriptions of dry, changing, and mostly wet conditions, with scenario probability, temperature, wind, expected wet running and disruption assumptions.
- Forecasted position, full driver name, team, confidence at that position, an 80% position range, actual position and position difference.
- The complete actual classification in its own table.
- Feedback details: observed weather, recorded incidents, pace-learning exclusions, retirement targets, and each later model that used those results for fitting or validation.

**Confidence is the estimated probability of the exact displayed position**, conditional on the selected scenario. It is not a subjective high/medium/low label, a driver's win probability at every row, or the probability of the scenario. These estimates have not been calibrated. Drivers are ordered by average simulated finish; the most likely winner can differ from the top driver in that ordering. The report explains both.

## Generate the report without retraining

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

The sprint arguments are optional. Dataset checksums must match the exact records used for the saved backtest. The report never reruns or changes its predictions. The output CSV includes actual positions, scenario confidence and feedback-use counts. The report's Download tables button exports all displayed formats and years, including actual-only training history. Print view prints the currently selected session and scenario.

## Real sprint data and separate training

```sh
f1-forecast fetch-history --seasons 2024 2025 --race-format sprint \
  --output data/processed/sprint-history-2024-2025.json --cache data/raw/historical
f1-forecast backtest data/processed/sprint-history-2024-2025.json \
  --model transformer --test-from-season 2025 \
  --min-train-events 3 --validation-events 1 --epochs 60 --patience 8 \
  --simulations 1000 --seed 42 \
  --output artifacts/sprint-data/backtest.json --save-model artifacts/sprint-data/model
```

Use new output/checkpoint paths when repeating collection or training. Sprint race results come from Jolpica's sprint classification; sprint qualifying positions come from OpenF1's exact sprint qualifying session, never from the subsequent race grid. The same archived day-ahead weather and observed track weather workflow applies. Sources: [Jolpica documentation](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md), [OpenF1 session results](https://openf1.org/docs/#session-result), [Open-Meteo previous runs](https://open-meteo.com/en/docs/previous-runs-api).

Sprint records use `race_format: sprint` and distinct `-sprint` event IDs. Session types remain `qualifying` and `race`. Grand Prix is the default format for existing files. The two formats use separate training datasets and checkpoints; mixed-format neural training and using a Grand Prix checkpoint for sprint inputs are rejected. This fits sprint performance separately with the existing architecture rather than pretending a Grand Prix forecast is a sprint forecast. Use separate forecast databases as well when collecting future feedback; keep distinct event IDs.

## Recorded coverage

The combined report contains **331 actual session records and 233 held-out predictions** across 2019–2025. The earlier-season runs add 203 Grand Prix records (including 2019 warm-up history) and 20 sprint records, with 162 and 16 held-out predictions respectively. Shared Friday qualifying in 2021–2022 is represented in both formats. See [2020–2023 coverage, weather assumptions and results](EARLIER_SEASONS.md).

The original 2024–2025 Grand Prix run contributes 86 actual sessions and 45 held-out 2025 predictions. Its sprint run contributes 22 actual sessions and 10 held-out 2025 predictions. The six 2024 sprint weekends are development history. Five eligible 2025 sprint weekends are evaluated in order, then a final sprint checkpoint is saved.

The 2025 United States sprint weekend is excluded because OpenF1's sprint qualifying classification is incomplete or cannot be matched to the sprint roster. The earlier Grand Prix exclusions are preserved. All missing sessions appear in the coverage section; missing forecasts are not replaced with actual results disguised as predictions. Small samples, excluded disrupted weekends, reconstructed availability and uncalibrated uncertainty limit any accuracy claim.

## What feedback actually means

The report derives feedback use from the saved chronological folds and source availability times. A result cannot train its own forecast. Later eligible windows may use it to fit weights; validation windows use it to select the epoch without directly fitting on it. Final-model use is shown separately, after the held-out predictions have been fixed.

Actual order is the target for incident-free driver-pair comparisons. Known collisions, mechanical issues, penalties and non-finishes mask pace comparisons for those drivers. Race non-finish labels still train the retirement head. Actual weather conditions completed training examples. Event messages are displayed as messages; multiple messages do not imply multiple distinct safety-car deployments.

“Eligible” in the actual table describes the pace mask, not proof that the row has already fitted a particular model. The explicit audit below shows the actual later fitting and validation uses. There is no automatic claim that every feedback item improved predictions.

# Actual F1 data and reproducible training

The primary dataset is `data/processed/real-history-2024-2025.json`. Every row pairs a real qualifying or Grand Prix snapshot with its actual classification, recorded incidents and observed weather. This original dataset contains Grand Prix sessions. A separate sprint dataset and model now feed the [readable combined report](REPORTING.md). Synthetic demo data remains available only for pipeline tests and is rejected when mixed with real training data.

## Download, backtest and save a model

```sh
uv sync --all-extras
source .venv/bin/activate
f1-forecast fetch-history --seasons 2024 2025 \
  --output data/processed/real-history-2024-2025.json \
  --cache data/raw/historical
f1-forecast backtest data/processed/real-history-2024-2025.json \
  --model transformer --test-from-season 2025 \
  --min-train-events 8 --validation-events 2 --epochs 60 --patience 8 \
  --simulations 1000 --seed 42 \
  --output artifacts/real-data/backtest.json \
  --save-model artifacts/real-data/model
```

Use new dataset and checkpoint paths on a repeat run. Raw responses are cached with their original URL and retrieval timestamp; rerunning collection with the same cache resumes successful downloads. A new cache directory fetches a fresh archive vintage. Missing data produces a documented exclusion, never synthetic replacement data.

The backtest trains separately before each eligible 2025 weekend, using only earlier weekends whose feedback is already available. The latest two eligible weekends select the training epoch; the target weekend remains outside weight fitting and validation. Its actual qualifying result may enter the subsequent race prediction, as requested. Earlier 2025 results become eligible for later 2025 forecasts. The final checkpoint is fitted only after all test forecasts have been fixed; the reported scores belong to those earlier fold models, not to re-predictions by the final checkpoint.

## Data sources

| Information | Source and treatment |
| --- | --- |
| Drivers, teams, qualifying and race classifications, archived grid, circuit coordinates | [Jolpica](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md); full-season pagination is merged and validated. Duplicate or missing positions are excluded, not renumbered. |
| Session timing, actual track weather, race-control messages | [OpenF1](https://openf1.org/docs/); matched to the exact session and filtered by its time window. At least five valid weather samples are required. |
| Pre-session forecast temperature, wind and precipitation amount | [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api), GFS day-1 variables at circuit coordinates. These use forecast lead time, not the observed race weather. |
| Driver and team pace | Explicit empirical proxies from available earlier classifications: up to ten driver and twenty team appearances, with known incidents excluded. |
| Unsupported car/circuit properties | Neutral priors, explicitly identified as unknown. No invented chassis specifications. Planned lap count is null; completed target-race laps never enter inputs. |

The archive does not provide a usable precipitation probability for this query. The scenario rain prior is explicitly heuristic: `clip(1 - exp(-precipitation_mm / 0.5), 0.05, 0.95)`. It is not a probability supplied by the weather provider. Actual rain exposure is the fraction of available track samples reporting rain, which is not a direct measurement of track wetness.

The importer preserves raw race status text. Both `Lapped` and `+N Lap(s)` are finishers. Generic `Retired` does not establish a mechanical cause; only supported status/race-control evidence adds an incident label. DNS, DSQ and retirement ordering retain the simulator's documented approximations.

## Historical reconstruction and limits

Every imported snapshot and feedback record is marked `data_mode: historical_reconstruction`, `synthetic: false`. Each source preserves its real `retrieved_at`; `available_at` and `availability_basis` separately record the reconstructed historical assumption. Ordinary live snapshots still reject sources retrieved after their prediction cutoff.

Prediction cutoffs are one hour before the archived session start. Result availability is conservatively assumed six hours after the archived session end. The day-ahead weather issue time is approximated as valid time minus 24 hours. These are explicit assumptions, not proof of original publication times.

Final classification archives supply driver/team identities and the grid, so this is a retrospective roster/grid reconstruction. Post-event corrections, disqualifications and withdrawals can differ from what was known at the cutoff. Circuit schedules can also use local calendar dates; a unique race within 30 hours is matched to OpenF1 timing. The manifest records session keys for audit. Save original pre-event snapshots for stronger prospective evaluation.

Exclusions reduce coverage and can bias measured accuracy, particularly on disrupted weekends. A Transformer trained on two seasons is not established as a reliable future predictor; probabilities need calibration and performance can shift with car regulations. No LLM guesses are used to fill historical results and no remote LLM fine-tuning is required.

## Artifacts and future use

- `data/processed/real-history-2024-2025.json`: actual training records with per-source provenance.
- `data/processed/real-history-2024-2025.manifest.json`: file checksum, session coverage, exclusions and weather sample counts.
- `artifacts/real-data/backtest.json`: chronological splits, per-session forecasts and ML/baseline metrics.
- `artifacts/real-data/model/`: PyTorch weights, metadata, development-data checksum and earlier-result history.

Raw downloads, processed records and model artifacts stay local and are ignored by Git. The committed importer and commands reproduce the workflow against the selected archive vintage.

```sh
# Once next-session.json has been populated with confirmed upcoming inputs:
f1-forecast predict next-session.json --ml-model artifacts/real-data/model \
  --output artifacts/next-forecast.json
# Or configure the HTTP API / CLI:
export F1_ML_MODEL=artifacts/real-data/model
```

Continue collecting verified results after each event. Retrain with the accumulated real records into a new model directory, assess its chronological backtest, then load that checkpoint. Training does not silently replace an active model. Synthetic checkpoints and real snapshots cannot be mixed.

## Recorded run: 4 September 2026

Downloaded 86 actual sessions from 44 weekends: 44 qualifying and 42 Grand Prix sessions. The dataset contains 41 sessions from 2024 and 45 from 2025. The model used 82 sessions for final weight fitting and four from the latest two weekends for validation. The model was successfully reloaded and its checksums verified.

The expanding-window test covers 45 sessions from 23 weekends in 2025, with 1,000 simulations per forecast. No hyperparameter search was performed on these test scores.

| Metric | Transformer | Frozen heuristic |
| --- | ---: | ---: |
| Overall expected-position MAE (lower is better) | 3.3204 | 3.3393 |
| Qualifying expected-position MAE | 3.1920 | 3.2701 |
| Grand Prix expected-position MAE | 3.4548 | 3.4116 |
| Correct pole selection | 6 / 23 | 5 / 23 |
| Correct race winner | 9 / 22 | 5 / 22 |

The overall MAE difference is small; Grand Prix MAE worsened. These results do not demonstrate a statistically established improvement. The full JSON report includes other metrics, every held-out forecast and the exact fold splits.

Ten of the 96 scheduled qualifying/race sessions were excluded: both sessions of 2024 Monaco, Netherlands and Azerbaijan and 2025 Emilia Romagna because the qualifying archive had duplicate or missing positions; the 2024 São Paulo race because the six-hour qualifying-availability assumption crossed the race cutoff; and the 2025 São Paulo race because the qualifying and race driver fields did not match. Both-session exclusions are conservative because the current importer requires a usable qualifying classification before constructing the weekend. All rate-limited downloads were recovered.

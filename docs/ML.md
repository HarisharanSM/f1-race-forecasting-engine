# Transformer forecasts and training backtests

The forecasting engine now has a locally trained PyTorch Transformer. It replaces heuristic pace scores and retirement risk when a checkpoint is loaded. LLM research and evidence extraction remain available, but neural training does not need an LLM or API credentials.

## Train on actual race data

Use the [real-data workflow](REAL_DATA.md) to download 2024–2025 records, evaluate 2025 chronologically, and save `artifacts/real-data/model`. `--test-from-season 2025` keeps 2024 out of reported test metrics while retaining it as development history. Later 2025 folds may train on earlier 2025 weekends.

## Optional synthetic pipeline test

```sh
uv sync --all-extras
source .venv/bin/activate
f1-forecast ml-demo --output artifacts/ml-demo
```

This creates eight fictional weekends, trains expanding-window models, evaluates the final four weekends, and saves a final model plus forecasts for a ninth weekend. Files include:

- `history.json`: the historical snapshot/feedback records.
- `backtest.json`: fold splits, full out-of-sample forecasts and evaluation metrics.
- `model/metadata.json`: dates, selected epoch, learning curves and configuration.
- `model/weights.pt` and `model/history.json`: reloadable weights and form history.
- `next-qualifying-input.json`, `next-race-input.json` and their forecast files.

The original two-session example is too short for neural training/validation. This larger demonstration proves the pipeline runs, not real F1 accuracy. Its inputs and results are explicitly fictional. A Transformer may perform worse than the heuristic baseline; the report displays that comparison without assuming improvement.

## Backtest that trains the model

Supply a JSON list of `{"snapshot": {...}, "feedback": {...}}` records using the existing schemas. Use genuine pre-session snapshots for a strict live-vintage test. The archival importer also supports explicitly marked `historical_reconstruction` records: retrieval timestamps remain authentic, while assumed historical availability is recorded separately. Feedback contains actual results, observed weather and recorded incidents.

```sh
f1-forecast backtest historical-records.json --model transformer \
  --epochs 60 --min-train-events 3 --validation-events 1 \
  --output artifacts/ml-backtest.json --save-model artifacts/model-v1
```

For each eligible target weekend:

1. Gather earlier-event feedback available before its first prediction cutoff.
2. Reserve the latest eligible weekend(s) for validation. Purge delayed training labels unavailable before the validation window.
3. Fit the neural weights on the remaining events. Select the epoch with lowest validation loss and stop early when improvement stalls.
4. Forecast the target weekend with those fixed weights. Its qualifying results may legitimately enter the race's grid input, but no target-weekend labels train that fold.
5. Compare against the frozen heuristic baseline, then expand history for the following weekend.

Whole event weekends stay together; drivers are never randomly split between training and testing. Warm-up weekends are listed separately and excluded from accuracy metrics. Defaults require at least four eligible development weekends and a fifth weekend for the first out-of-sample forecast.

Reports provide MAE, pairwise ranking accuracy, winner accuracy, winner Brier score and position log loss, overall and separately for qualifying/races. Positive `mae_improvement` means the Transformer beat the baseline. Validation loss is used for early stopping; the outer forecasts are the actual backtest. Keep a further untouched future period if these results guide model/hyperparameter selection.

`--save-model` fits a final checkpoint after all test forecasts are fixed. Its latest development weekend remains reserved for validation/early stopping. This checkpoint is for later events; it has not been evaluated on the events it now uses for development. The reported backtest scores belong to the earlier fold models. Loading the final checkpoint to predict its development events is rejected.

Without `--model transformer`, the existing numerical online-learning backtest remains available.

## Train directly and use the result

```sh
f1-forecast train-ml historical-records.json --output artifacts/model-v1

# Omit the file to use accumulated verified feedback from the selected database.
f1-forecast --database data/forecast.sqlite3 train-ml --output artifacts/model-v2

# Restrict fitting to labels available by a historical timestamp.
f1-forecast train-ml historical-records.json --as-of 2026-08-01T00:00:00Z \
  --output artifacts/model-as-of-august

f1-forecast predict next-session.json --ml-model artifacts/model-v2 \
  --output artifacts/next-forecast.json
```

Training options: `--epochs`, `--patience`, `--min-train-events`, `--validation-events`, `--seed`. Four eligible weekends are the default workflow minimum, not sufficient data for reliable deep learning. Output directories must be new/empty to preserve existing models.

Submitting feedback stores it for later fitting; it does not silently alter an active Transformer. Rerun training or the training backtest against the growing history, evaluate the candidate, then explicitly load it. This workflow is separate from remote LLM fine-tuning. Existing lightweight feedback updates still support the heuristic baseline.

In Python:

```python
from f1_forecast.neural import TrainingConfig, train_model
from f1_forecast.ml_backtest import backtest_transformer
from f1_forecast.service import ForecastService

report = backtest_transformer(records, config=TrainingConfig(epochs=60),
                             save_model="artifacts/model-v1")
service = ForecastService("data/forecast.sqlite3", ml_model="artifacts/model-v1")
forecast = service.predict(next_snapshot)
```

For the HTTP server, set `F1_ML_MODEL=artifacts/model-v1`. Requests use the configured model by default; `"use_ml": false` selects the baseline. Without a checkpoint, the original heuristic forecast remains available. CLI `--ml-model` overrides the environment setting.

## Model architecture

The model uses [PyTorch's TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html), with two attention layers, four heads and 32 hidden units by default. Each driver is a token; attention compares competitors within the entered field. There are no positional embeddings, so roster list order cannot act as a hidden finishing-position feature. Padding masks handle different field sizes.

This is a field-attention Transformer, not a temporal language model. Recent form enters as chronological summaries of up to ten driver appearances and twenty team appearances. An LSTM would need meaningful ordered lap/stint sequences, which the current snapshot-based collectors do not provide.

Current features include driver/car/circuit ratings, weather, session type, actual race grid and earlier driver/team form. History is filtered by label availability at each prediction cutoff and excludes the target weekend. New drivers and teams receive neutral history defaults; no fixed driver-ID embedding is required.

Separate qualifying/race pace heads learn pairwise rankings with logistic loss. A retirement head learns race nonfinish risk with binary cross-entropy. Known retirements, collisions and penalties are masked out of pace comparisons while retirement labels still train the risk head. Training uses AdamW, weight decay, dropout, gradient clipping and validation-based early stopping.

At inference, learned pace plus Gumbel noise generates full-field rankings. The Gumbel pairwise probability matches the logistic ranking objective before disruption effects. Learned retirement risk is applied once, independently of the heuristic retirement model. Existing scenario mixing produces win/podium/position probabilities; probabilities still require real-data calibration.

## Conditional weather and leakage limits

Training uses each completed session's observed weather as a condition to learn performance given conditions. These observations are accessed only for eligible training/validation sessions after feedback availability. Held-out inference uses forecast scenario wet exposure and forecast temperature/wind. Actual target-session weather and finishing order never enter its forecast.

Validation loss measures this conditional supervised objective; backtest metrics measure the full forecast under uncertain weather. Dates, source cutoffs and historical ratings must be authentic—timestamps alone do not establish that. LLM calls are excluded from the training backtest to avoid historical-result recall.

Saved models include weights, feature schema, train/validation dates, learning curves and historical summaries. File hashes are checked on load and PyTorch uses `weights_only=True`. Synthetic and real histories/checkpoints cannot be mixed. Forecasts record `forecast_model`, `ml_model_id` and `ml_training_cutoff` alongside the numerical model version.

Weather-regime probabilities and disruption effects retain heuristic assumptions. The model still does not simulate every lap, tyre strategy or FIA classification edge case. More historical weekends, telemetry-derived ratings, calibrated probabilities and an untouched test set are required before claiming real predictive accuracy.

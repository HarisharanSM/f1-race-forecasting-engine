# Matched Ranking Model Comparison

## Completed 2020-2026 Comparison

307 matched sessions across 138 weekends: 271 Grand Prix and 36 sprint. The extra blend-selection window excludes four sprint forecasts covered by previous experiments.

| Model | Position MAE | Winner Brier | Position log loss |
| --- | ---: | ---: | ---: |
| Transformer | 3.2422 | 0.7715 | 2.6698 |
| Linear ranker | 3.6240 | 0.8384 | 2.7442 |
| Small MLP | 3.2796 | 0.7798 | 2.7013 |
| Selected ensemble | 3.2778 | 0.7803 | 2.6662 |

Lower is better for all three metrics. The Transformer has the strongest position error and winner-probability score here. Ensemble-minus-Transformer log loss has a 95% paired-weekend bootstrap interval of -0.0174 to +0.0085, including no improvement. Its position-error difference interval is +0.0114 to +0.0605, and its winner-Brier difference interval is +0.0036 to +0.0135. The ensemble is not an established overall upgrade. No default model was replaced.

Results: `artifacts/ranking-model-comparison-2020-2026/comparison.html`. Historical selection bias and nonidentical coverage mean these scores must not be directly compared with earlier full-pipeline backtest numbers.

The experiment compares the field Transformer, a linear pairwise-logistic ranker, a small independent per-driver MLP and an ensemble of their finishing-position probability matrices. Linear and MLP models use separate qualifying/race pace heads and the same DNF objective as the Transformer. Model labels and checkpoint loading identify the architecture explicitly; existing Transformer checkpoints remain supported.

## Identical Inputs And Evaluation

All models receive the same 27 features, history summaries, optional legacy-input adapters, clean-driver pair masks and actual-weather training conditions. Forecast inference uses forecast weather scenarios only. All share the optimizer settings, maximum epoch budget, early-stopping patience, chronological windows, simulation count (1,000) and prediction seed (42). The number of parameters differs deliberately. Features are not standardized or selected differently between architectures.

No architecture gets the specialized learned measurement correction or pace calibration. Consequently, this is not directly comparable with previously optimized joint-model scores, which also cover more target sessions. Same inputs and training budget establish a controlled comparison, not optimal tuning for each architecture.

## Chronology

For each target weekend, retain only earlier weekends whose feedback is available before the earliest target forecast cutoff. Reserve the latest two eligible weekends for ensemble selection. Split the remaining development history into gradient training and an earlier epoch-selection window using the existing configuration. No ensemble-selection feedback is supplied to component training or component history features. All architectures must cover the same target sessions or the entire fold is skipped. Grand Prix and sprint models remain separate.

Select the smallest-index minimum validation position-log-loss candidate among seven predeclared weights: each individual model, each equal two-model mixture, or an equal three-model mixture. Blend probability matrices, not ranks, to preserve position and driver probability sums. The fixed tie order favors Transformer, linear, MLP, then mixtures. The two-weekend blend selection sample is small and can be noisy; no claim of stable superiority follows from selection alone. Weights and all candidate losses are recorded for every fold. The ensemble cutoff includes its later selection outcomes.

## Reports And Reproduction

```sh
.venv/bin/python scripts/compare_ranking_models.py --output artifacts/new-model-comparison
```

The default source is the verified incident-era 2020-2026 enriched dataset. Source hashes are checked against its saved reports. Output must be new. `comparison.html` contains overall/yearly scores; `comparison.json` includes provenance and 1,000 paired calendar-weekend bootstrap intervals against the Transformer. Format-specific JSON files retain forecasts, selected weights, model metadata, exact fold memberships and skipped weekends. No existing reports or deployed model are replaced.

The historical periods have been repeatedly inspected. Treat this as retrospective development evidence, not a pristine untouched test. 2026 is a saved year-to-date collection through Italian GP qualifying, not a complete season or a newly refreshed dataset. Whole-weekend bootstrap intervals do not model all temporal dependence or prior model-selection bias. The comparison must be frozen and evaluated prospectively before deployment changes.

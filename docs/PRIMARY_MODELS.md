# Primary Transformer And Optional Ensemble

Real-data CLI and HTTP predictions now default to the `field_transformer` checkpoint in `artifacts/primary-models/<race_format>`. Grand Prix and sprint use separate models. Explicit `--ml-model` or `F1_ML_MODEL` overrides remain supported. Synthetic examples retain the heuristic unless a bundle/checkpoint is explicitly provided. Missing real-data primary models fail with a clear message rather than silently switching to a heuristic.

## Using The Models

```sh
# Default: primary field Transformer
f1-forecast predict next-session.json --output artifacts/next-primary.json

# Optional, validation-gated probability mixture
f1-forecast predict next-session.json --model-policy ensemble --output artifacts/next-ensemble.json

# Explicit heuristic or custom bundle root
f1-forecast predict next-session.json --model-policy heuristic
f1-forecast predict next-session.json --primary-bundle artifacts/another-primary-bundle
```

Python uses `ForecastService(model_policy="primary")` by default; choose `model_policy="ensemble"` to combine models. HTTP requests accept `model_policy: "primary"` or `"ensemble"`; `use_ml: false` still explicitly selects the heuristic. Selecting an explicit checkpoint and ensemble mode together is rejected to avoid silently combining unrelated models.

## Training And Selection

`scripts/train_primary_models.py` trains from the verified enriched historical datasets and refuses to overwrite existing bundles. Use `--output` for a new installation. All component models share the 27-feature representation and training objective from the controlled architecture comparison, without specialized learned measurement correction or pace calibration.

The last two earlier weekends are reserved for ensemble selection; component fitting and epoch selection use only earlier development data. This deliberately keeps those newest selection outcomes out of component weights and history features. Bundle cutoffs include the selection results, and every development weekend is excluded from prediction even under a changed event ID. These bundles are for later weekends, not historical backtest replay. Checkpoint checksums, format and synthetic/real checks remain enforced.

Candidate Transformer/linear/MLP weights are 100/0/0, 90/10/0, 90/0/10, 80/10/10, 75/25/0 and 75/0/25 percent. To pass, validation position log loss must improve by more than 0.001 and winner Brier by more than 0.0001, without increased position MAE or reduced winner accuracy. Otherwise ensemble mode returns the Transformer distribution unchanged. Positive-weight models are loaded lazily; zero-weight alternatives are not loaded. Blending position matrices preserves coherent probabilities and derives win/podium chances from the mixture.

## Installed Bundles

- Grand Prix optional ensemble: 75% Transformer, 0% linear, 25% MLP. Selection feedback ends 2026-09-05 21:00 UTC.
- Sprint optional ensemble: 100% Transformer; alternatives did not pass. Selection feedback ends 2026-08-22 17:00 UTC.
- Default primary mode is 100% Transformer for both formats, irrespective of ensemble selection.

The two-weekend validation sample is small and was drawn from previously inspected historical data. These weights are experimental, not a proven accuracy improvement. The earlier unrestricted ensemble did not establish an overall advantage. No future backtest was run or scheduled by deployment; prospective evaluation remains necessary. Older model checkpoints and reports are preserved.

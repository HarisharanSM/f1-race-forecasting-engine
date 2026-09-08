"""Matched architecture comparison with separate epoch-selection and blend-selection windows."""

from dataclasses import replace
from uuid import uuid4

import numpy as np

from .engine import evaluate, predict, summarize
from .ml_data import chronological_split, event_groups, records_from_json
from .models import Forecast, utcnow
from .neural import fit_records

KINDS = ("field_transformer", "linear_ranker", "mlp_ranker")
WEIGHTS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    (1 / 3, 1 / 3, 1 / 3),
)


def blend_forecasts(forecasts, weights):
    weights = np.asarray(weights)
    if (
        len(weights) != len(forecasts)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Invalid ensemble weights")
    first = forecasts[0]
    ids = [r.driver_id for r in first.standings]
    for forecast in forecasts:
        if (
            (forecast.event_id, forecast.session, forecast.race_format)
            != (first.event_id, first.session, first.race_format)
            or {r.driver_id for r in forecast.standings} != set(ids)
            or [s.scenario for s in forecast.scenarios] != [s.scenario for s in first.scenarios]
        ):
            raise ValueError("Ensemble forecasts must have identical targets and scenarios")

    def combine(groups):
        matrices, risks = [], []
        for group in groups:
            by_id = {r.driver_id: r for r in group.standings}
            matrices.append([by_id[d].position_probabilities for d in ids])
            risks.append([by_id[d].dnf_probability for d in ids])
        return summarize(
            ids,
            np.clip(np.einsum("m,mij->ij", weights, matrices), 0, 1),
            np.clip(weights @ np.asarray(risks), 0, 1),
        )

    result = first.model_copy(deep=True)
    result.id = str(uuid4())
    result.forecast_model = "probability_ensemble"
    result.ml_model_id = None
    result.standings = combine(forecasts)
    result.winner = max(result.standings, key=lambda r: r.win_probability).driver_id
    for i, scenario in enumerate(result.scenarios):
        scenario.standings = combine([f.scenarios[i] for f in forecasts])
        scenario.winner = max(scenario.standings, key=lambda r: r.win_probability).driver_id
    result.warnings.append(
        f"Ensemble weights {weights.tolist()} selected on earlier reserved weekends."
    )
    return Forecast.model_validate(result.model_dump())


def compare(rows, config, simulations=1000, progress=None):
    records = records_from_json(rows)
    if any(r.feedback.available_at > utcnow() for r in records):
        raise ValueError("Future feedback cannot be evaluated")
    groups = event_groups(records)
    results, folds, skipped = [], [], []
    # Identical legacy feature adapters, no architecture-specific measurement correction/calibration.
    config = replace(config, optional_mode="legacy", calibrate=False)
    for index, target in enumerate(groups):
        if target[0].snapshot.season < 2020:
            continue
        cutoff = min(r.snapshot.as_of for r in target)
        earlier = [r for group in groups[:index] for r in group if r.feedback.available_at < cutoff]
        try:
            development, selection = chronological_split(
                earlier, config.min_train_events + config.validation_events, 2
            )
            train, validation = chronological_split(
                development, config.min_train_events, config.validation_events
            )
            if {r.snapshot.session for r in target + selection} - {
                r.snapshot.session for r in train
            }:
                raise ValueError("Training lacks target session type")
        except ValueError as exc:
            skipped.append({"event_id": target[0].snapshot.event_id, "reason": str(exc)})
            continue
        models = [
            fit_records(development, replace(config, seed=config.seed + index), network_kind=k)
            for k in KINDS
        ]
        selection_forecasts = [
            [predict(r.snapshot, ml_model=m, simulations=simulations, seed=42) for m in models]
            for r in selection
        ]
        scores = [
            float(
                np.mean(
                    [
                        evaluate(blend_forecasts(fs, w), r.feedback)["position_log_loss"]
                        for fs, r in zip(selection_forecasts, selection, strict=True)
                    ]
                )
            )
            for w in WEIGHTS
        ]
        weights = WEIGHTS[int(np.argmin(scores))]
        fold = {
            "event_id": target[0].snapshot.event_id,
            "prediction_cutoff": cutoff.isoformat(),
            "training_events": sorted({r.snapshot.event_id for r in train}),
            "validation_events": sorted({r.snapshot.event_id for r in validation}),
            "ensemble_validation_events": sorted({r.snapshot.event_id for r in selection}),
            "ensemble_selected_through": max(
                r.feedback.available_at for r in selection
            ).isoformat(),
            "weights": list(weights),
            "candidate_log_losses": scores,
            "models": {k: m.metadata for k, m in zip(KINDS, models, strict=True)},
        }
        for record in target:
            forecasts = [
                predict(record.snapshot, ml_model=m, simulations=simulations, seed=42)
                for m in models
            ]
            ensemble = blend_forecasts(forecasts, weights)
            ensemble.ml_training_cutoff = max(r.feedback.available_at for r in selection)
            forecasts.append(ensemble)
            results.append(
                {
                    "event_id": record.snapshot.event_id,
                    "session": record.snapshot.session.value,
                    "season": record.snapshot.season,
                    "round": record.snapshot.round,
                    "forecasts": {
                        k: f.model_dump(mode="json")
                        for k, f in zip((*KINDS, "ensemble"), forecasts, strict=True)
                    },
                    "metrics": {
                        k: evaluate(f, record.feedback)
                        for k, f in zip((*KINDS, "ensemble"), forecasts, strict=True)
                    },
                }
            )
        folds.append(fold)
        if progress:
            progress(f"{fold['event_id']}: {len(results)} matched sessions")
    if not results:
        raise ValueError("Insufficient history for three disjoint chronological windows")
    return {"results": results, "folds": folds, "skipped": skipped}

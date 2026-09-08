"""Regularized measured-pace correction and validation-only probability selection."""

from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from .engine import evaluate, predict
from .measurement_features import MEASUREMENT_NAMES, measurement_matrix, without_performance
from .ml_data import excluded_drivers, feature_matrix, history_summary

MIN_MEASURED_EVENTS = 6
PENALTIES = (0.1, 1.0, 10.0)
TEMPERATURES = (1.0, 0.8, 1.25)


def training_pairs(model, records, history, stage):
    pairs, offsets, targets, weights, matrices = [], [], [], [], []
    support = [set() for _ in MEASUREMENT_NAMES]
    for record in records:
        s, actual = record.snapshot, record.feedback
        if s.session.value != stage:
            continue
        x = measurement_matrix(s, actual.actual_weather.wet_fraction)
        matrices.append(x)
        for column in np.where(np.any(x != 0, axis=0))[0]:
            support[column].add(s.event_id)
        base = feature_matrix(
            without_performance(s),
            history,
            actual.actual_weather.wet_fraction,
            actual.actual_weather.air_temperature_c,
            actual.actual_weather.wind_speed_ms,
        )
        with torch.inference_mode():
            pace, _ = model.network(
                torch.from_numpy(base).unsqueeze(0), torch.zeros((1, len(base)), dtype=torch.bool)
            )
        pace = pace[0].numpy()
        excluded = excluded_drivers(actual)
        ids = [i for i, driver in enumerate(s.drivers) if driver.id not in excluded]
        indices = [(a, b) for index, a in enumerate(ids) for b in ids[index + 1 :]]
        if not indices:
            continue
        a, b = np.asarray(indices).T
        ranks = [actual.finishing_order.index(d.id) for d in s.drivers]
        pairs.append(x[a] - x[b])
        offsets.append(pace[a] - pace[b])
        targets.append(np.array(ranks)[a] < np.array(ranks)[b])
        weights.extend([1 / len(indices)] * len(indices))
    if not pairs:
        return None
    support_counts = np.array([len(events) for events in support])
    active = support_counts >= MIN_MEASURED_EVENTS
    values = np.concatenate(matrices)
    scale = np.maximum(0.1, values.std(axis=0))
    active &= values.std(axis=0) > 1e-6
    return {
        "x": torch.tensor(np.concatenate(pairs) / scale * active, dtype=torch.float32),
        "offset": torch.tensor(np.concatenate(offsets), dtype=torch.float32),
        "target": torch.tensor(np.concatenate(targets), dtype=torch.float32),
        "weights": torch.tensor(weights, dtype=torch.float32),
        "active": active,
        "scale": scale,
        "event_support": support_counts.tolist(),
    }


def fit_weights(batch, penalty):
    if batch is None or not batch["active"].any():
        return np.zeros(len(MEASUREMENT_NAMES)).tolist()
    w = torch.zeros(len(MEASUREMENT_NAMES), requires_grad=True)
    optimizer = torch.optim.LBFGS([w], lr=1, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = batch["offset"] + batch["x"] @ w
        losses = F.binary_cross_entropy_with_logits(logits, batch["target"], reduction="none")
        loss = (losses * batch["weights"]).sum() / batch[
            "weights"
        ].sum() + penalty * w.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    result = w.detach().numpy() / batch["scale"] * batch["active"]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite optional measurement coefficients")
    return result.tolist()


def validation_scores(model, records):
    rows = [
        evaluate(predict(r.snapshot, ml_model=model, simulations=1000, seed=42), r.feedback)
        for r in records
    ]
    return {
        key: float(np.mean([r[key] for r in rows])) for key in ("position_log_loss", "winner_brier")
    }


def improves(candidate, reference):
    return (
        candidate["position_log_loss"] < reference["position_log_loss"] - 0.001
        and candidate["winner_brier"] < reference["winner_brier"] - 0.001
    )


def fit_adjustment(model, train, validation, config):
    from .neural import NeuralForecaster

    history = [history_summary(r) for r in train]
    metadata = deepcopy(model.metadata)
    metadata.update(
        trained_through=max(r.feedback.available_at for r in train).isoformat(),
        development_events=sorted({r.snapshot.event_id for r in train}),
    )
    candidate_model = NeuralForecaster(model.network, metadata, history)
    baseline = validation_scores(candidate_model, validation)
    temperature, selected_scores = 1.0, baseline
    trials = [{"kind": "baseline", "temperature": 1.0, **baseline}]
    if config.calibrate:
        for value in TEMPERATURES[1:]:
            candidate_model.metadata["pace_temperature"] = value
            scores = validation_scores(candidate_model, validation)
            trials.append({"kind": "temperature", "temperature": value, **scores})
            if improves(scores, baseline) and sum(scores.values()) < sum(selected_scores.values()):
                temperature, selected_scores = value, scores
    candidate_model.metadata["pace_temperature"] = temperature
    calibrated_scores = selected_scores
    selected, penalty_selected = None, None
    support = {}
    measured_validation = {
        r.snapshot.event_id
        for r in validation
        if np.any(measurement_matrix(r.snapshot, r.snapshot.weather.rain_probability))
    }
    if config.optional_mode == "learned" and len(measured_validation) >= 2:
        batches = {
            s: training_pairs(candidate_model, train, history, s) for s in ("qualifying", "race")
        }
        support = {
            s: b["event_support"] if b else [0] * len(MEASUREMENT_NAMES) for s, b in batches.items()
        }
        for penalty in PENALTIES:
            adjustment = {
                "feature_names": list(MEASUREMENT_NAMES),
                "weights": {s: fit_weights(b, penalty) for s, b in batches.items()},
            }
            candidate_model.metadata["measurement_adjustment"] = adjustment
            scores = validation_scores(candidate_model, validation)
            trials.append(
                {"kind": "measurements", "penalty": penalty, "temperature": temperature, **scores}
            )
            if improves(scores, calibrated_scores) and sum(scores.values()) < sum(
                selected_scores.values()
            ):
                selected, penalty_selected, selected_scores = adjustment, penalty, scores
    if selected:
        model.metadata["measurement_adjustment"] = selected
    model.metadata["pace_temperature"] = temperature
    model.metadata["validation_selection"] = {
        "training_events": sorted({r.snapshot.event_id for r in train}),
        "validation_events": sorted({r.snapshot.event_id for r in validation}),
        "feedback_through": max(r.feedback.available_at for r in validation).isoformat(),
        "optional_correction_selected": selected is not None,
        "penalty": penalty_selected,
        "pace_temperature": temperature,
        "baseline": baseline,
        "selected": selected_scores,
        "trials": trials,
        "support_by_feature": support,
        "rule": "At least six training events per feature, two measured validation events; "
        "both validation position log loss and winner Brier must improve by >0.001. "
        "Temperature selection precedes optional correction selection. Missing evidence "
        "has zero correction; coefficients shrink toward zero. Validation reuse for "
        "epoch and option selection is development, not independent evaluation.",
    }

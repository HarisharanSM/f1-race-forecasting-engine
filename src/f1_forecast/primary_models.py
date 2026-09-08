"""Primary Transformer deployment with a conservatively gated probability ensemble."""

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from .engine import evaluate, predict
from .ml_data import chronological_split, records_from_json
from .model_comparison import KINDS, blend_forecasts
from .models import utcnow
from .neural import NeuralForecaster, fit_records

DEFAULT_BUNDLE = Path("artifacts/primary-models")
CANDIDATES = (
    (1.0, 0.0, 0.0),
    (0.9, 0.1, 0.0),
    (0.9, 0.0, 0.1),
    (0.8, 0.1, 0.1),
    (0.75, 0.25, 0.0),
    (0.75, 0.0, 0.25),
)


def select_weights(forecasts, actuals):
    if not forecasts or not actuals:
        raise ValueError("Ensemble selection requires earlier forecasts and outcomes")
    trials = []
    for weights in CANDIDATES:
        scores = [
            evaluate(blend_forecasts(fs, weights), a)
            for fs, a in zip(forecasts, actuals, strict=True)
        ]
        trials.append(
            {
                "weights": list(weights),
                "metrics": {k: float(np.mean([s[k] for s in scores])) for k in scores[0]},
            }
        )
    base = trials[0]["metrics"]
    accepted = [
        r
        for r in trials[1:]
        if r["metrics"]["position_log_loss"] < base["position_log_loss"] - 0.001
        and r["metrics"]["winner_brier"] < base["winner_brier"] - 0.0001
        and r["metrics"]["expected_position_mae"] <= base["expected_position_mae"]
        and r["metrics"]["winner_correct"] >= base["winner_correct"]
    ]
    chosen = (
        min(accepted, key=lambda r: r["metrics"]["position_log_loss"]) if accepted else trials[0]
    )
    return chosen["weights"], trials


def train_bundle(rows, config, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Choose a new bundle directory")
    records = records_from_json(rows)
    if any(r.feedback.available_at > utcnow() for r in records):
        raise ValueError("Cannot train with future results")
    config = replace(config, optional_mode="legacy", calibrate=False)
    development, selection = chronological_split(
        records, config.min_train_events + config.validation_events, 2
    )
    models = [fit_records(development, config, network_kind=k) for k in KINDS]
    predictions = [
        [predict(r.snapshot, ml_model=m, simulations=1000, seed=42) for m in models]
        for r in selection
    ]
    weights, trials = select_weights(predictions, [r.feedback for r in selection])
    destination.mkdir(parents=True)
    for kind, model in zip(KINDS, models, strict=True):
        model.save(destination / kind)
    manifest = {
        "version": 1,
        "primary": "field_transformer",
        "model_order": list(KINDS),
        "components": {k: m.model_id for k, m in zip(KINDS, models, strict=True)},
        "dataset_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "weights": weights,
        "trials": trials,
        "selected_through": max(r.feedback.available_at for r in selection).isoformat(),
        "selection_events": sorted({r.snapshot.event_id for r in selection}),
        "development_events": sorted({r.snapshot.event_id for r in development + selection}),
        "development_weekends": sorted(
            {f"{r.snapshot.season}-{r.snapshot.round}" for r in development + selection}
        ),
        "race_format": records[0].snapshot.race_format,
        "synthetic": records[0].snapshot.synthetic,
        "rule": "At least 75% Transformer. Improve validation log loss >0.001 and winner Brier >0.0001, "
        "without worse MAE or winner accuracy; otherwise exact Transformer. Two reserved weekends, "
        "not used for component fitting or epoch selection. Limited evidence; no future gain guaranteed.",
    }
    (destination / "bundle.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class PrimaryBundle:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "bundle.json").read_text())
        meta = self.metadata
        weights = np.asarray(meta["weights"], dtype=float)
        if (
            meta.get("version") != 1
            or meta.get("primary") != KINDS[0]
            or meta.get("model_order") != list(KINDS)
            or weights.shape != (3,)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1)
            or weights[0] < 0.75
        ):
            raise ValueError("Invalid primary ensemble manifest")
        self.directory = directory
        self.models = {}

    def model(self, kind):
        if kind not in self.models:
            model = NeuralForecaster.load(self.directory / kind)
            if model.metadata["architecture"] != kind:
                raise ValueError("Bundle component architecture mismatch")
            if model.model_id != self.metadata["components"][kind]:
                raise ValueError("Bundle component checkpoint mismatch")
            self.models[kind] = model
        return self.models[kind]

    def predict(self, snapshot, *, ensemble=False, **kwargs):
        meta = self.metadata
        if (
            datetime.fromisoformat(meta["selected_through"]) >= snapshot.as_of
            or snapshot.event_id in meta["development_events"]
            or f"{snapshot.season}-{snapshot.round}" in meta["development_weekends"]
        ):
            raise ValueError("Primary bundle includes target or future development feedback")
        if snapshot.race_format != meta["race_format"] or snapshot.synthetic != meta["synthetic"]:
            raise ValueError("Primary bundle format or data mode mismatch")
        primary = predict(snapshot, ml_model=self.model(KINDS[0]), **kwargs)
        primary.ml_training_cutoff = datetime.fromisoformat(meta["selected_through"])
        if not ensemble or meta["weights"] == [1, 0, 0]:
            if ensemble:
                primary.warnings.append(
                    "Ensemble validation gate retained the primary Transformer unchanged."
                )
            return primary
        active = [(k, w) for k, w in zip(KINDS[1:], meta["weights"][1:], strict=True) if w > 0]
        forecasts = [primary] + [
            predict(snapshot, ml_model=self.model(k), **kwargs) for k, _ in active
        ]
        result = blend_forecasts(forecasts, [meta["weights"][0]] + [w for _, w in active])
        result.warnings.append(meta["rule"])
        return result

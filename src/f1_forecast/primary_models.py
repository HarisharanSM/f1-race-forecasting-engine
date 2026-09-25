"""Primary Transformer deployment with a conservatively gated probability ensemble."""

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from .acceptance import acceptance_gate, interval_quality_policy, pair_record, selection_split
from .engine import predict
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


def select_weights(forecasts, actuals, *, records=None, seeded_forecasts=None, policy=None):
    """No candidate is selected without matched records and repeated training seeds."""
    if not forecasts or not actuals or len(forecasts) != len(actuals):
        raise ValueError("Ensemble selection requires matched earlier forecasts and outcomes")
    policy = policy or interval_quality_policy()
    if records is None or seeded_forecasts is None:
        return list(CANDIDATES[0]), [
            {
                "weights": list(CANDIDATES[0]),
                "gate": {
                    "accepted": False,
                    "reasons": ["Missing chronological multi-seed validation evidence"],
                },
            }
        ]
    if len(records) != len(actuals) or any(
        r.feedback != a for r, a in zip(records, actuals, strict=True)
    ):
        raise ValueError("Selection records and feedback differ")
    trials = []
    for weights in CANDIDATES:
        pairs = [
            pair_record(
                r, fs[0], fs[0] if weights == CANDIDATES[0] else blend_forecasts(fs, weights), seed
            )
            for seed, predictions in seeded_forecasts.items()
            for r, fs in zip(records, predictions, strict=True)
        ]
        gate = acceptance_gate(pairs, policy)
        trials.append(
            {"weights": list(weights), "metrics": gate["overall"]["candidate"], "gate": gate}
        )
    accepted = [r for r in trials[1:] if r["gate"]["accepted"]]
    chosen = min(accepted, key=lambda r: r["metrics"]["winner_brier"]) if accepted else trials[0]
    return chosen["weights"], trials


def train_bundle(rows, config, destination, *, policy=None):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Choose a new bundle directory")
    records = records_from_json(rows)
    if any(r.feedback.available_at > utcnow() for r in records):
        raise ValueError("Cannot train with future results")
    config = replace(config, optional_mode="legacy", calibrate=False)
    policy = policy or interval_quality_policy()
    try:
        development, selection = selection_split(
            records, config.min_train_events + config.validation_events, policy
        )
        seeds = [config.seed + i for i in range(policy.training_seeds)]
    except ValueError:
        # Small histories still produce a usable primary, never a weakly selected blend.
        development, selection = chronological_split(
            records, config.min_train_events + config.validation_events, 2
        )
        seeds = [config.seed]
    seeded_forecasts, models = {}, None
    for seed in seeds:
        fitted = [
            fit_records(development, replace(config, seed=seed), network_kind=k) for k in KINDS
        ]
        if models is None:
            models = fitted
        seeded_forecasts[seed] = [
            [predict(r.snapshot, ml_model=m, simulations=1000, seed=42) for m in fitted]
            for r in selection
        ]
    weights, trials = select_weights(
        seeded_forecasts[seeds[0]],
        [r.feedback for r in selection],
        records=selection,
        seeded_forecasts=seeded_forecasts,
        policy=policy,
    )
    destination.mkdir(parents=True)
    for kind, model in zip(KINDS, models, strict=True):
        model.save(destination / kind)
    manifest = {
        "version": 2,
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
        "acceptance_policy": trials[0]["gate"]["policy"],
        "training_seeds": seeds,
        "rule": "At least 75% Transformer. Three chronological windows of at least six weekends "
        "per session and three training seeds. Improve MAE and Brier; protect log loss, pairwise "
        "and winner accuracy within declared tolerances, nominal 80% coverage, width and interval score. Require paired weekend "
        "uncertainty checks. Insufficient evidence retains the exact primary. Future gains are not guaranteed.",
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
            meta.get("version") not in {1, 2}
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
        approved = meta.get("version") == 2 and any(
            trial.get("weights") == meta["weights"] and trial.get("gate", {}).get("accepted")
            for trial in meta.get("trials", [])
        )
        if not ensemble or meta["weights"] == [1, 0, 0] or not approved:
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

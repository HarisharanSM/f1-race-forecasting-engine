"""Expanding-window Transformer fitting with entire future weekends held out."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .engine import evaluate, predict
from .ml_data import chronological_split, event_groups, records_from_json
from .models import utcnow
from .neural import TrainingConfig, fit_records


def mean_metrics(rows: list[dict], key: str) -> dict:
    return (
        {metric: float(np.mean([row[key][metric] for row in rows])) for metric in rows[0][key]}
        if rows
        else {}
    )


def backtest_transformer(
    rows: list[dict],
    *,
    config: TrainingConfig | None = None,
    simulations: int = 1000,
    seed: int = 42,
    save_model: str | Path | None = None,
    test_from_season: int | None = None,
    progress=None,
) -> dict:
    config = config or TrainingConfig()
    if not 100 <= simulations <= 100000 or seed < 0:
        raise ValueError("Require 100..100000 simulations and a nonnegative seed")
    if save_model and Path(save_model).exists() and any(Path(save_model).iterdir()):
        raise ValueError("Model directory is not empty; choose a new checkpoint directory")
    records = records_from_json(rows)
    if any(r.feedback.available_at > utcnow() for r in records):
        raise ValueError("Backtest requires already available historical results")
    groups = event_groups(records)
    scored, skipped, folds = [], [], []
    for index, target in enumerate(groups):
        if test_from_season is not None and target[0].snapshot.season < test_from_season:
            skipped.append(
                {"event_id": target[0].snapshot.event_id, "reason": "Before requested test season"}
            )
            continue
        cutoff = min(r.snapshot.as_of for r in target)
        # Never use any label from the held-out weekend, even its qualifying result.
        eligible = [
            r for group in groups[:index] for r in group if r.feedback.available_at < cutoff
        ]
        try:
            train, _ = chronological_split(
                eligible, config.min_train_events, config.validation_events
            )
        except ValueError as exc:
            skipped.append({"event_id": target[0].snapshot.event_id, "reason": str(exc)})
            continue
        if {r.snapshot.session for r in target} - {r.snapshot.session for r in train}:
            skipped.append(
                {
                    "event_id": target[0].snapshot.event_id,
                    "reason": "Warm-up history lacks the target session type",
                }
            )
            continue
        model = fit_records(eligible, replace(config, seed=config.seed + index))
        fold = {
            "event_id": target[0].snapshot.event_id,
            "prediction_cutoff": cutoff.isoformat(),
            "model_id": model.model_id,
            "trained_through": model.metadata["trained_through"],
            "training_events": model.metadata["training_events"],
            "validation_events": model.metadata["validation_events"],
            "best_epoch": model.metadata["best_epoch"],
            "best_validation_loss": model.metadata["best_validation_loss"],
        }
        for record in target:
            neural = predict(record.snapshot, simulations=simulations, seed=seed, ml_model=model)
            baseline = predict(record.snapshot, simulations=simulations, seed=seed)
            scored.append(
                {
                    "event_id": record.snapshot.event_id,
                    "session": record.snapshot.session.value,
                    "season": record.snapshot.season,
                    "model_id": model.model_id,
                    "ml": evaluate(neural, record.feedback),
                    "frozen_baseline": evaluate(baseline, record.feedback),
                    "forecast": neural.model_dump(mode="json"),
                }
            )
        folds.append(fold)
        if progress:
            progress(f"Evaluated {fold['event_id']}: {len(scored)} held-out sessions")
    if not scored:
        raise ValueError(
            "No out-of-sample folds: provide more completed weekends or reduce min_train_events"
        )
    overall_ml, overall_baseline = (
        mean_metrics(scored, "ml"),
        mean_metrics(scored, "frozen_baseline"),
    )
    report = {
        "method": "expanding-window, event-grouped Transformer training with inner validation",
        "synthetic": records[0].snapshot.synthetic,
        "sessions_evaluated": len(scored),
        "data_modes": sorted({r.snapshot.data_mode for r in records}),
        "test_from_season": test_from_season,
        "dataset_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "folds": folds,
        "warmup_events": skipped,
        "results": scored,
        "summary": {
            "ml": overall_ml,
            "frozen_baseline": overall_baseline,
            "mae_improvement": overall_baseline["expected_position_mae"]
            - overall_ml["expected_position_mae"],
        },
        "by_session": {
            session: {
                "ml": mean_metrics([r for r in scored if r["session"] == session], "ml"),
                "frozen_baseline": mean_metrics(
                    [r for r in scored if r["session"] == session], "frozen_baseline"
                ),
            }
            for session in sorted({r["session"] for r in scored})
        },
        "by_season": {
            str(year): {
                "sessions_evaluated": sum(r["season"] == year for r in scored),
                "ml": mean_metrics([r for r in scored if r["season"] == year], "ml"),
                "frozen_baseline": mean_metrics(
                    [r for r in scored if r["season"] == year], "frozen_baseline"
                ),
            }
            for year in sorted({r["season"] for r in scored})
        },
        "limitations": (
            "Synthetic data demonstrates execution only. "
            if records[0].snapshot.synthetic
            else "Real archived outcomes; historical input reconstruction is not a point-in-time vintage backtest. "
        )
        + "Actual weather conditions training "
        "examples after results are available; held-out forecasts use forecast scenarios only. "
        "Probabilities are not calibrated. The final checkpoint is not evaluated on its own development events.",
    }
    if save_model:
        # Refit only after all test forecasts are fixed; this artifact is for subsequent events.
        final = fit_records(records, config)
        final.save(save_model)
        report["final_model"] = {
            "directory": str(Path(save_model).resolve()),
            "model_id": final.model_id,
            "trained_through": final.metadata["trained_through"],
            "training_events": final.metadata["training_events"],
            "validation_events": final.metadata["validation_events"],
        }
    return report

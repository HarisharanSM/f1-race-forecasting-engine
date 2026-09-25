"""Freeze and test quality features, recency, listwise likelihood and small mixtures."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from f1_forecast.acceptance import (
    AcceptancePolicy,
    acceptance_gate,
    breakdown,
    pair_record,
    selection_split,
)
from f1_forecast.engine import predict
from f1_forecast.ml_data import (
    QUALITY_FEATURE_NAMES,
    chronological_split,
    quality_matrix,
    records_from_json,
)
from f1_forecast.model_comparison import blend_forecasts
from f1_forecast.models import utcnow
from f1_forecast.neural import TrainingConfig, fit_records
from f1_forecast.performance_collection import atomic_json, digest


def split_windows(records, config, policy, holdout_weekends=6):
    earlier, later = chronological_split(
        records,
        config.min_train_events
        + config.validation_events
        + policy.windows * policy.weekends_per_window,
        holdout_weekends,
    )
    development, selection = selection_split(
        earlier, config.min_train_events + config.validation_events, policy
    )
    return development, selection, later


def run(source, output, start_season=2023):
    if output.exists():
        raise ValueError("Choose a new output directory")
    policy = AcceptancePolicy()
    variants = {
        "incumbent": {},
        "quality": {"quality_features": True},
        "quality_recency": {"quality_features": True, "recency_half_life_days": 365.0},
        "quality_recency_listwise": {
            "quality_features": True,
            "recency_half_life_days": 365.0,
            "listwise_weight": 0.1,
        },
    }
    seeds, mixtures = (42, 43, 44), (0.05, 0.10, 0.20, 0.25)
    loaded, skipped = {}, {}
    for kind in ("grand_prix", "sprint"):
        raw = json.loads((source / f"{kind}-records.json").read_text())
        reference = json.loads((source / f"{kind}-backtest.json").read_text())
        if digest(raw) != reference["dataset_sha256"]:
            raise ValueError("Source checksum mismatch")
        records = [r for r in records_from_json(raw) if r.snapshot.season >= start_season]
        if any(r.feedback.available_at > utcnow() for r in records):
            raise ValueError("Future outcomes in source records")
        config = replace(
            TrainingConfig(**reference["reproduction_config"]),
            optional_mode="legacy",
            calibrate=False,
        )
        try:
            partitions = split_windows(records, config, policy)
        except ValueError as exc:
            skipped[kind] = str(exc)
            continue
        loaded[kind] = (records, config, partitions)
    recipe = {
        "created_at": utcnow().isoformat(),
        "status": "frozen_retrospective_experiment",
        "policy": asdict(policy),
        "variants": variants,
        "training_seeds": seeds,
        "heuristic_shares": mixtures,
        "start_season": start_season,
        "simulations_per_scenario": 1000,
        "simulation_seed": 42,
        "holdout_weekends": 6,
        "skipped": skipped,
        "sources": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.glob("*-records.json"))
        },
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted(Path("src/f1_forecast").glob("*.py"))]
        },
        "formats": {
            kind: {
                "config": asdict(config),
                **{
                    role: sorted({f"{r.snapshot.season}-{r.snapshot.round:02d}" for r in partition})
                    for role, partition in zip(
                        ("development", "selection", "later"), partitions, strict=True
                    )
                },
            }
            for kind, (_, config, partitions) in loaded.items()
        },
        "limitations": "Recent verified existing snapshots only; no fabricated or newly downloaded inputs. "
        "All historical results have previously been inspected. Later periods are retrospective comparisons, "
        "not untouched prospective evidence. Listwise loss models clean-order likelihood before race disruptions. "
        "No variant or model is deployed by this experiment.",
    }
    output.mkdir(parents=True)
    atomic_json(output / "recipe.json", recipe)
    quality = {
        kind: {
            "sessions": len(records),
            "mean_input_indicators": dict(
                zip(
                    QUALITY_FEATURE_NAMES,
                    np.concatenate([quality_matrix(r.snapshot) for r in records]).mean(0).tolist(),
                    strict=True,
                )
            ),
        }
        for kind, (records, _, _) in loaded.items()
    }
    atomic_json(output / "input-quality.json", quality)
    summary = {}
    for kind, (_, config, (development, selection, later)) in loaded.items():
        models = {}
        pairs = {name: [] for name in variants if name != "incumbent"}
        pairs.update({f"mixture_{w:.0%}": [] for w in mixtures})
        for seed in seeds:
            models[seed] = {}
            for name, settings in variants.items():
                print(f"{kind}: training {name}, seed {seed}", flush=True)
                model = fit_records(development, replace(config, seed=seed, **settings))
                models[seed][name] = model
                model.save(output / "checkpoints" / kind / str(seed) / name)
            for record in selection:
                forecasts = {
                    name: predict(record.snapshot, ml_model=model, seed=42)
                    for name, model in models[seed].items()
                }
                incumbent = forecasts["incumbent"]
                for name in variants:
                    if name != "incumbent":
                        pairs[name].append(pair_record(record, incumbent, forecasts[name], seed))
                heuristic = predict(record.snapshot, seed=42)
                for w in mixtures:
                    pairs[f"mixture_{w:.0%}"].append(
                        pair_record(
                            record,
                            incumbent,
                            blend_forecasts([incumbent, heuristic], [1 - w, w]),
                            seed,
                        )
                    )
        gates = {name: acceptance_gate(rows, policy) for name, rows in pairs.items()}
        accepted = [name for name in pairs if gates[name]["accepted"]]
        chosen = (
            min(accepted, key=lambda n: gates[n]["overall"]["candidate"]["winner_brier"])
            if accepted
            else "incumbent"
        )
        # Selection is durably frozen before any later forecasts or evaluation are produced.
        atomic_json(
            output / f"{kind}-frozen-selection.json",
            {
                "chosen": chosen,
                "gates": gates,
                "recipe_sha256": digest(recipe),
                "promoted": False,
            },
        )
        later_pairs = {name: [] for name in pairs}
        selected_pairs = []
        for seed in seeds:
            for record in later:
                forecasts = {
                    name: predict(record.snapshot, ml_model=model, seed=42)
                    for name, model in models[seed].items()
                }
                incumbent = forecasts["incumbent"]
                heuristic = predict(record.snapshot, seed=42)
                forecasts.update(
                    {
                        f"mixture_{w:.0%}": blend_forecasts([incumbent, heuristic], [1 - w, w])
                        for w in mixtures
                    }
                )
                for name, paired_rows in later_pairs.items():
                    paired_rows.append(pair_record(record, incumbent, forecasts[name], seed))
                selected_pairs.append(pair_record(record, incumbent, forecasts[chosen], seed))
        result = {
            "chosen": chosen,
            "selection": {n: breakdown(rows, policy) for n, rows in pairs.items()},
            "later_retrospective": {n: breakdown(rows, policy) for n, rows in later_pairs.items()},
            "selected_later": breakdown(selected_pairs, policy),
        }
        atomic_json(output / f"{kind}-comparison.json", result)
        atomic_json(output / f"{kind}-pairs.json", {"selection": pairs, "later": later_pairs})
        summary[kind] = {"chosen": chosen, "later": result["selected_later"]["overall"]}
    atomic_json(
        output / "summary.json", {"formats": summary, "skipped": skipped, "promoted": False}
    )
    lines = [
        "# Guarded training experiments",
        "",
        "Recipe frozen before fitting; selection frozen before later evaluation. No model promoted.",
        "Recent matched experiment, not directly comparable with the 668-session expanding-history report.",
        "",
    ]
    for kind, result in summary.items():
        gates = json.loads((output / f"{kind}-frozen-selection.json").read_text())["gates"]
        lines += [
            f"## {kind}",
            "",
            f"Selected: {result['chosen']}.",
            "",
            "Validation deltas versus the matched incumbent, averaged over three training seeds.",
            "Negative MAE, Brier and log-loss deltas are better; positive accuracy deltas are better.",
            "",
            "| Candidate | MAE delta | Brier delta | Log-loss delta | Pairwise delta | Coverage delta | Passed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for name, gate in gates.items():
            d = gate["overall"]["delta"]
            lines.append(
                f"| {name} | {d['expected_position_mae']:+.6f} | {d['winner_brier']:+.6f} | {d['position_log_loss']:+.6f} | {d['pairwise_accuracy']:+.2%} | {d['position_interval_coverage']:+.2%} | {gate['accepted']} |"
            )
        lines += [
            "",
            "Rejection reasons and per-window/seed scores are in the frozen-selection file.",
            "Later results and error breakdowns by session, quality, season and retirement are in the comparison file.",
            "",
        ]
    for kind, reason in skipped.items():
        lines += [f"{kind}: skipped — {reason}.", ""]
    lines += [
        "These are retrospective comparisons. The gate protects coverage conservatively as requested,",
        "alongside interval width and interval score. No future gain is guaranteed.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"formats": summary, "skipped": skipped}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/mixed-input-backtest-2010-2026")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-season", type=int, default=2023)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args.source, args.output, args.start_season)

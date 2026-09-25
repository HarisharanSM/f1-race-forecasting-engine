"""Retrospective mixture screen with frozen validation selection and a later comparison."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from f1_forecast.acceptance import AcceptancePolicy, acceptance_gate, breakdown, pair_record
from f1_forecast.engine import evaluate, predict
from f1_forecast.ml_data import records_from_json
from f1_forecast.model_comparison import blend_forecasts
from f1_forecast.models import Forecast, utcnow
from f1_forecast.performance_collection import atomic_json, digest

WEIGHTS = (0.0, 0.05, 0.10, 0.20, 0.25)


def replay(source, kind, progress=print):
    rows = json.loads((source / f"{kind}-records.json").read_text())
    report = json.loads((source / f"{kind}-backtest.json").read_text())
    if digest(rows) != report["dataset_sha256"]:
        raise ValueError("Saved backtest dataset checksum mismatch")
    records = {(r.snapshot.event_id, r.snapshot.session.value): r for r in records_from_json(rows)}
    folds = {f["event_id"]: f for f in report["folds"]}
    seen = set()
    for index, result in enumerate(report["results"]):
        key = (result["event_id"], result["session"])
        if key in seen:
            raise ValueError("Duplicate saved forecast")
        seen.add(key)
        record = records[key]
        fold = folds[result["event_id"]]
        forecast = Forecast.model_validate(result["forecast"])
        if (
            forecast.event_id != record.snapshot.event_id
            or forecast.session != record.snapshot.session
            or forecast.ml_model_id != fold["model_id"]
            or forecast.ml_training_cutoff is None
            or forecast.ml_training_cutoff >= record.snapshot.as_of
            or result["event_id"] in fold["training_events"] + fold["validation_events"]
        ):
            raise ValueError("Invalid held-out forecast provenance")
        heuristic = predict(
            record.snapshot, simulations=forecast.simulations_per_scenario, seed=forecast.seed
        )
        for actual, saved in (
            (evaluate(heuristic, record.feedback), result["frozen_baseline"]),
            (evaluate(forecast, record.feedback), result["ml"]),
        ):
            if any(not np.isclose(actual[m], saved[m], atol=1e-10, rtol=0) for m in saved):
                raise ValueError(
                    "Saved metrics no longer reproduce; do not compare changed recipes"
                )
        yield record, forecast, heuristic, report["reproduction_config"]["seed"]
        if (index + 1) % 50 == 0:
            progress(f"{kind}: replayed {index + 1} saved sessions")


def run(source, output, test_from=2025):
    if output.exists():
        raise ValueError("Choose a new output directory")
    policy = AcceptancePolicy()
    files = [
        source / f"{kind}-{suffix}.json"
        for kind in ("grand_prix", "sprint")
        for suffix in ("records", "backtest")
    ]
    recipe = {
        "created_at": utcnow().isoformat(),
        "purpose": "Retrospective screen, not prospective confirmation or deployment",
        "heuristic_weights": WEIGHTS,
        "policy": asdict(policy),
        "test_from_season": test_from,
        "sources": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted(Path("src/f1_forecast").glob("*.py"))]
        },
        "training_seed_evidence": "Saved single training trajectory only; cannot pass three-seed gate",
    }
    output.mkdir(parents=True)
    atomic_json(output / "recipe.json", recipe)
    groups = {kind: {w: [] for w in WEIGHTS} for kind in ("grand_prix", "sprint")}
    for kind in groups:
        for record, transformer, heuristic, seed in replay(source, kind):
            for weight in WEIGHTS:
                candidate = (
                    transformer
                    if weight == 0
                    else blend_forecasts([transformer, heuristic], [1 - weight, weight])
                )
                groups[kind][weight].append(pair_record(record, transformer, candidate, seed))
    # No later-period metrics participate in candidate selection.
    selection, choices = {}, {}
    for kind, candidates in groups.items():
        selection[kind] = {
            str(w): acceptance_gate([r for r in rows if r["season"] < test_from], policy)
            for w, rows in candidates.items()
        }
        accepted = [w for w in WEIGHTS[1:] if selection[kind][str(w)]["accepted"]]
        choices[kind] = (
            min(
                accepted,
                key=lambda w: selection[kind][str(w)]["overall"]["candidate"]["winner_brier"],
            )
            if accepted
            else 0.0
        )
    atomic_json(
        output / "frozen-selection.json",
        {
            "selected_heuristic_weights": choices,
            "validation": selection,
            "recipe_sha256": digest(recipe),
            "status": "not_promoted",
        },
    )
    # Candidate holdout breakdowns are diagnostic only, not another selection opportunity.
    audit = {}
    for kind, candidates in groups.items():
        audit[kind] = {
            str(w): {
                "all": breakdown(rows, policy),
                "validation": breakdown([r for r in rows if r["season"] < test_from], policy),
                "later_retrospective": breakdown(
                    [r for r in rows if r["season"] >= test_from], policy
                ),
            }
            for w, rows in candidates.items()
        }
    atomic_json(output / "comparison.json", audit)
    atomic_json(
        output / "pairs.json",
        {k: {str(w): rows for w, rows in v.items()} for k, v in groups.items()},
    )
    lines = [
        "# Guarded Transformer–heuristic mixture",
        "",
        "Retrospective replay. No installed model changed. Selection uses earlier seasons only.",
        "A single saved training trajectory cannot satisfy the three-seed acceptance gate.",
        "",
        "| Format | Heuristic share | MAE | Winner Brier | Position log loss | Pairwise accuracy | Selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for kind in groups:
        for w in WEIGHTS:
            m = audit[kind][str(w)]["all"]["overall"]["candidate"]
            lines.append(
                f"| {kind} | {w:.0%} | {m['expected_position_mae']:.6f} | {m['winner_brier']:.6f} | {m['position_log_loss']:.6f} | {m['pairwise_accuracy']:.2%} | {'yes' if choices[kind] == w else 'no'} |"
            )
    lines += [
        "",
        "Full error breakdowns by session, input tier, season, retirement involvement and seed,",
        "with paired calendar-weekend uncertainty intervals, are in comparison.json.",
        "Interval coverage is protected conservatively; interval score also penalizes width and misses.",
        "Any later inspection is exploratory. Freeze a new recipe before genuinely future outcomes.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "selected_heuristic_weights": choices}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/mixed-input-backtest-2010-2026")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-from", type=int, default=2025)
    args = parser.parse_args()
    run(args.source, args.output, args.test_from)

"""Frozen rolling evaluation of verified inputs and separate qualifying/race mixtures."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from f1_forecast.acceptance import acceptance_gate, breakdown, pair_record
from f1_forecast.engine import predict
from f1_forecast.ml_data import records_from_json
from f1_forecast.models import utcnow
from f1_forecast.neural import TrainingConfig, fit_records
from f1_forecast.performance_collection import atomic_json, digest
from f1_forecast.session_selection import (
    CALIBRATION_BLEND,
    SHARES,
    TEMPERATURES,
    fit_temperature,
    interval_quality_policy,
    match_records,
    revised_forecast,
    rolling_splits,
    select_session_candidates,
)

SEEDS = (42, 43, 44)
FAMILIES = ("existing", "enriched")


def key(record):
    return record.snapshot.event_id, record.snapshot.session


def describe_partition(rows):
    return {
        "weekends": sorted({f"{r.snapshot.season}-{r.snapshot.round:02d}" for r in rows}),
        "first_cutoff": min(r.snapshot.as_of for r in rows).isoformat(),
        "last_feedback": max(r.feedback.available_at for r in rows).isoformat(),
        "sessions": len(rows),
    }


def run(source, enriched_source, output):
    if output.exists():
        raise ValueError("Choose a new experiment directory")
    policy, loaded, skipped = interval_quality_policy(), {}, {}
    source_files = []
    input_audit = json.loads((enriched_source / "input-audit.json").read_text())
    for kind in ("grand_prix", "sprint"):
        original_path, added_path = (
            source / f"{kind}-records.json",
            enriched_source / f"{kind}-records.json",
        )
        reference_path = source / f"{kind}-backtest.json"
        source_files += [original_path, added_path, reference_path]
        raw, added = json.loads(original_path.read_text()), json.loads(added_path.read_text())
        reference = json.loads(reference_path.read_text())
        if (
            digest(raw) != reference["dataset_sha256"]
            or digest(added) != input_audit[kind]["output_sha256"]
        ):
            raise ValueError("Experiment input checksum mismatch")
        all_records = records_from_json(raw)
        added_map = match_records(all_records, records_from_json(added))
        records = [r for r in all_records if r.snapshot.season >= 2023]
        if any(r.feedback.available_at > utcnow() for r in records):
            raise ValueError("Source contains future feedback")
        config = replace(
            TrainingConfig(**reference["reproduction_config"]),
            optional_mode="legacy",
            calibrate=False,
            quality_features=False,
            recency_half_life_days=None,
            listwise_weight=0.0,
        )
        try:
            folds = rolling_splits(records, config, policy=policy)
        except ValueError as exc:
            skipped[kind] = str(exc)
            continue
        loaded[kind] = (config, folds, added_map)
    recipe = {
        "created_at": utcnow().isoformat(),
        "status": "frozen_before_fitting",
        "policy": asdict(policy),
        "seeds": SEEDS,
        "families": FAMILIES,
        "heuristic_shares": SHARES,
        "temperature_grid": TEMPERATURES,
        "calibration_blend": CALIBRATION_BLEND,
        "simulation_seed": 42,
        "simulations_per_scenario": 1000,
        "source_hashes": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
        },
        "code_hashes": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted(Path("src/f1_forecast").glob("*.py"))]
        },
        "formats": {
            kind: {
                "config": asdict(config),
                "folds": [
                    {role: describe_partition(rows) for role, rows in fold.items()}
                    for fold in folds
                ],
            }
            for kind, (config, folds, _) in loaded.items()
        },
        "skipped": skipped,
        "limitations": "Retrospective rolling comparison, not untouched future evidence. "
        "Whole weekends form the evaluation units; training seeds do not multiply independent evidence. "
        "Calibration fitting, candidate selection and later evaluation use disjoint chronological weekends. "
        "Earlier outer outcomes may enter later folds only after they are available. "
        "Historical practice availability remains reconstructed; no model is promoted automatically.",
    }
    output.mkdir(parents=True)
    atomic_json(output / "recipe.json", recipe)
    all_selected, all_enriched, summaries = [], [], {}
    for kind, (config, folds, added_map) in loaded.items():
        selected_rows, enriched_rows, fold_reports = [], [], []
        for fold_index, fold in enumerate(folds, 1):
            folder = output / kind / f"fold-{fold_index}"
            models = {}
            for seed in SEEDS:
                models[seed] = {}
                for family in FAMILIES:
                    print(f"{kind}, fold {fold_index}: train {family}, seed {seed}", flush=True)
                    rows = (
                        fold["development"]
                        if family == "existing"
                        else [added_map[key(r)] for r in fold["development"]]
                    )
                    model = fit_records(rows, replace(config, seed=seed))
                    model.save(folder / "models" / str(seed) / family)
                    models[seed][family] = model

            heuristics = {}

            def forecasts(rows, models=models, added_map=added_map, heuristics=heuristics):
                result = {}
                for seed in SEEDS:
                    result[seed] = {}
                    for family in FAMILIES:
                        family_rows = (
                            rows if family == "existing" else [added_map[key(r)] for r in rows]
                        )
                        result[seed][family] = {
                            key(r): predict(r.snapshot, ml_model=models[seed][family], seed=42)
                            for r in family_rows
                        }
                        for r in family_rows:
                            if (family, key(r)) not in heuristics:
                                heuristics[family, key(r)] = predict(r.snapshot, seed=42)
                return result

            calibration_forecasts = forecasts(fold["calibration"])
            selection_forecasts = forecasts(fold["selection"])
            stage_options, stage_choices, gate_audits, calibration_audits = {}, {}, {}, {}
            for stage in ("qualifying", "race"):
                options, candidate_rows = {}, {}
                calibration_audits[stage] = {}
                for family in FAMILIES:
                    for share in SHARES:
                        examples = [
                            (
                                r,
                                seed,
                                revised_forecast(
                                    calibration_forecasts[seed][family][key(r)],
                                    heuristics[family, key(r)],
                                    share,
                                ),
                            )
                            for seed in SEEDS
                            for r in fold["calibration"]
                            if r.snapshot.session.value == stage
                        ]
                        temperature, audit = fit_temperature(examples)
                        calibration_audits[stage][f"{family}/{share}"] = audit
                        for t in dict.fromkeys((1.0, temperature)):
                            if family == "existing" and share == 0 and t == 1:
                                continue
                            name = f"{family}/heuristic={share:g}/temperature={t:g}"
                            options[name] = (family, share, t)
                            candidate_rows[name] = [
                                pair_record(
                                    r,
                                    selection_forecasts[seed]["existing"][key(r)],
                                    revised_forecast(
                                        selection_forecasts[seed][family][key(r)],
                                        heuristics[family, key(r)],
                                        share,
                                        t,
                                    ),
                                    seed,
                                )
                                for seed in SEEDS
                                for r in fold["selection"]
                                if r.snapshot.session.value == stage
                            ]
                chosen, gates = select_session_candidates(candidate_rows, policy)
                stage_options[stage], stage_choices[stage], gate_audits[stage] = (
                    options,
                    chosen,
                    gates,
                )

            def selected_forecast(
                record,
                predictions,
                seed,
                choices,
                stage_options=stage_options,
                heuristics=heuristics,
            ):
                choice = choices[record.snapshot.session.value]
                if choice == "incumbent":
                    return predictions[seed]["existing"][key(record)]
                family, share, temperature = stage_options[record.snapshot.session.value][choice]
                return revised_forecast(
                    predictions[seed][family][key(record)],
                    heuristics[family, key(record)],
                    share,
                    temperature,
                )

            combined_pairs = [
                pair_record(
                    r,
                    selection_forecasts[seed]["existing"][key(r)],
                    selected_forecast(r, selection_forecasts, seed, stage_choices),
                    seed,
                )
                for seed in SEEDS
                for r in fold["selection"]
            ]
            combined_gate = acceptance_gate(combined_pairs, policy)
            final_choices = (
                stage_choices
                if combined_gate["accepted"]
                else {s: "incumbent" for s in stage_choices}
            )
            frozen = {
                "stage_choices": stage_choices,
                "final_choices": final_choices,
                "stage_gates": gate_audits,
                "combined_gate": combined_gate,
                "calibration": calibration_audits,
                "recipe_sha256": digest(recipe),
                "selected_through": max(
                    r.feedback.available_at for r in fold["selection"]
                ).isoformat(),
            }
            # Persist choices before generating or evaluating any later-period forecast.
            atomic_json(folder / "frozen-selection.json", frozen)
            later_forecasts = forecasts(fold["later"])
            current, enriched = [], []
            for seed in SEEDS:
                for r in fold["later"]:
                    base = later_forecasts[seed]["existing"][key(r)]
                    current.append(
                        pair_record(
                            r,
                            base,
                            selected_forecast(r, later_forecasts, seed, final_choices),
                            seed,
                        )
                    )
                    enriched.append(
                        pair_record(r, base, later_forecasts[seed]["enriched"][key(r)], seed)
                    )
            selected_rows.extend(current)
            enriched_rows.extend(enriched)
            result = {
                "fold": fold_index,
                "choices": final_choices,
                "selected": breakdown(current, policy),
                "enriched_only_diagnostic": breakdown(enriched, policy),
            }
            atomic_json(folder / "later-results.json", result)
            atomic_json(
                folder / "later-pairs.json", {"selected": current, "enriched_only": enriched}
            )
            fold_reports.append(result)
            print(f"{kind}, fold {fold_index}: frozen choices {final_choices}", flush=True)
        summaries[kind] = {
            "folds": fold_reports,
            "selected": breakdown(selected_rows, policy),
            "enriched_only_diagnostic": breakdown(enriched_rows, policy),
        }
        all_selected.extend(selected_rows)
        all_enriched.extend(enriched_rows)
    report = {
        "formats": summaries,
        "skipped": skipped,
        "promoted": False,
        "selected": breakdown(all_selected, policy),
        "enriched_only_diagnostic": breakdown(all_enriched, policy),
    }
    atomic_json(output / "comparison.json", report)
    lines = [
        "# Separate qualifying and race mixtures",
        "",
        "Frozen retrospective experiment. Existing primary models are unchanged.",
        "",
        "The revised gate checks nominal 80% coverage plus interval score and width; it does not preserve excess coverage.",
        "Accuracy tolerances were fixed before fitting: 0.2 percentage points for pairwise accuracy and 2 points for winner accuracy.",
        "MAE, winner Brier and position log loss retain their protections.",
        "",
        "| Format | Fold | Qualifying choice | Race choice | Later weekends |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for kind, value in summaries.items():
        for fold in value["folds"]:
            lines.append(
                f"| {kind} | {fold['fold']} | {fold['choices']['qualifying']} | {fold['choices']['race']} | {fold['selected']['overall']['weekends']} |"
            )
    lines += [
        "",
        "## Later evaluation",
        "",
        "Seeds are repeated fits, not extra independent weekends.",
        "",
        "| Metric | Incumbent | Selected procedure | Enriched-only diagnostic |",
        "| --- | ---: | ---: | ---: |",
    ]
    if all_selected:
        for metric, value in report["selected"]["overall"]["baseline"].items():
            lines.append(
                f"| {metric} | {value:.6f} | {report['selected']['overall']['candidate'][metric]:.6f} | {report['enriched_only_diagnostic']['overall']['candidate'][metric]:.6f} |"
            )
    lines += [
        "",
        "The enriched-only column is diagnostic, not a separately promoted model.",
        "Exact per-session, season, retirement and seed breakdowns are in comparison.json.",
        "Every fold's frozen-selection.json records fitted temperatures, candidate scores and rejection reasons.",
        "",
    ]
    lines.extend(f"{kind}: skipped — {reason}." for kind, reason in skipped.items())
    lines += [
        "",
        "Historical outcomes were previously inspected. A new frozen future evaluation is needed for a prospective claim.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "skipped": skipped, "promoted": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/mixed-input-backtest-2010-2026")
    )
    parser.add_argument("--enriched", type=Path, default=Path("artifacts/session-inputs-20260909"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args.source, args.enriched, args.output)

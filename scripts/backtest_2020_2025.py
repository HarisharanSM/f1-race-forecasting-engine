"""Continuous 2020-2025 matched backtests with separate GP and sprint histories."""

import argparse
import hashlib
import html
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from backtest_expanded_collection import aggregate, validate_pair
from recheck_optional_inputs import metrics, read

from f1_forecast.ml_backtest import backtest_transformer, mean_metrics
from f1_forecast.ml_data import records_from_json
from f1_forecast.models import Feedback, Forecast
from f1_forecast.neural import TrainingConfig
from f1_forecast.performance_collection import atomic_json, digest
from f1_forecast.performance_enrichment import enrich_file
from f1_forecast.reporting import render_backtest_report

SOURCES = {
    "grand_prix": [
        ("legacy-data", "history-2019-2023-complete"),
        ("real-data", "real-history-2024-2025"),
    ],
    "sprint": [
        ("legacy-sprint", "sprint-history-2021-2023-final"),
        ("sprint-data", "sprint-history-2024-2025"),
    ],
}


def merge_sources(kind, end_season=2025):
    rows, exclusions, sources = [], [], []
    for suite, name in SOURCES[kind]:
        path = Path("data/processed") / f"{name}.json"
        data = read(path)
        manifest = read(path.with_suffix(".manifest.json"))
        reference = read(Path("artifacts") / suite / "backtest.json")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError(f"Source manifest checksum mismatch: {path}")
        if digest(data) != reference["dataset_sha256"]:
            raise ValueError(f"Source differs from original backtest: {path}")
        rows.extend(data)
        exclusions.extend(manifest.get("exclusions", []))
        sources.append(
            {
                "path": str(path.resolve()),
                "dataset_sha256": digest(data),
                "file_sha256": manifest["sha256"],
                "sessions": len(data),
            }
        )
    if end_season == 2026:
        name = (
            "real-history-2026-ytd-verified"
            if kind == "grand_prix"
            else "sprint-history-2026-ytd-verified"
        )
        path = Path("data/processed") / f"{name}.json"
        data, manifest = read(path), read(path.with_suffix(".manifest.json"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("2026 source manifest checksum mismatch")
        if manifest["race_format"] != kind or any(r["snapshot"]["season"] != 2026 for r in data):
            raise ValueError("2026 source format or season mismatch")
        from datetime import datetime

        if any(
            datetime.fromisoformat(r["feedback"]["available_at"])
            > datetime.fromisoformat(manifest["collection_cutoff"])
            for r in data
        ):
            raise ValueError("2026 source contains results beyond its collection cutoff")
        rows.extend(data)
        exclusions.extend(manifest["exclusions"])
        sources.append(
            {
                "path": str(path.resolve()),
                "file_sha256": manifest["sha256"],
                "dataset_sha256": digest(data),
                "sessions": len(data),
                "collection_cutoff": manifest["collection_cutoff"],
            }
        )
    rows.sort(key=lambda r: r["snapshot"]["as_of"])
    records_from_json(rows)
    return rows, {
        "sources": sources,
        "exclusions": exclusions,
        "race_format": kind,
        "dataset_sha256": digest(rows),
        "sessions": len(rows),
    }


def extend_report(previous, previous_rows, extension, rows, config):
    """Reuse exact prefix forecasts; new folds retain their full-history group-index seeds."""
    if (
        rows[: len(previous_rows)] != previous_rows
        or digest(previous_rows) != previous["dataset_sha256"]
    ):
        raise ValueError("Earlier records changed; run a fresh backtest instead of resuming")
    if previous.get("reproduction_config") != asdict(config):
        raise ValueError("Earlier training configuration differs")
    if any(
        r["forecast"]["simulations_per_scenario"] != 1000 or r["forecast"]["seed"] != 42
        for r in previous["results"]
    ):
        raise ValueError("Earlier simulation configuration differs")
    old_events = {r["snapshot"]["event_id"] for r in previous_rows}
    if any(r["event_id"] in old_events for r in extension["results"]):
        raise ValueError("Extension overlaps previously evaluated events")
    results = previous["results"] + extension["results"]
    combined = {
        **extension,
        "test_from_season": 2020,
        "sessions_evaluated": len(results),
        "results": results,
        "folds": previous["folds"] + extension["folds"],
        "warmup_events": previous["warmup_events"]
        + [r for r in extension["warmup_events"] if r["event_id"] not in old_events],
        "reused_prefix_sha256": digest(previous_rows),
        "reused_forecasts": len(previous["results"]),
    }
    ml, baseline = mean_metrics(results, "ml"), mean_metrics(results, "frozen_baseline")
    combined["summary"] = {
        "ml": ml,
        "frozen_baseline": baseline,
        "mae_improvement": baseline["expected_position_mae"] - ml["expected_position_mae"],
    }
    combined["by_session"] = {
        s: {
            arm: mean_metrics([r for r in results if r["session"] == s], arm)
            for arm in ("ml", "frozen_baseline")
        }
        for s in ("race", "qualifying")
    }
    return combined


def verify_folds(report, rows):
    records = records_from_json(rows)
    by_event = {}
    for record in records:
        by_event.setdefault(record.snapshot.event_id, []).append(record)
    for fold in report["folds"]:
        train, validation = set(fold["training_events"]), set(fold["validation_events"])
        if train & validation or fold["event_id"] in train | validation:
            raise ValueError("Target/validation/training overlap")
        cutoff = min(r.snapshot.as_of for r in by_event[fold["event_id"]])
        from datetime import datetime

        if datetime.fromisoformat(fold["trained_through"]) >= cutoff:
            raise ValueError("Model fitted with future feedback")
        selection = fold.get("validation_selection")
        if selection and (
            set(selection["validation_events"]) != validation
            or datetime.fromisoformat(selection["feedback_through"]) >= cutoff
        ):
            raise ValueError("Optional model selection uses future feedback")
    scored = {(r["event_id"], r["session"]) for r in report["results"]}
    warmup = {r["event_id"] for r in report["warmup_events"]}
    if any(
        (r.snapshot.event_id, r.snapshot.session.value) not in scored
        and r.snapshot.event_id not in warmup
        for r in records
    ):
        raise ValueError("Unaccounted source session")


def run(args):
    if args.resume_from and (not args.joint_effects or args.end_season != 2026):
        raise ValueError("Resume is supported for the joint 2026 extension")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    configs = {
        "grand_prix": TrainingConfig(**read("artifacts/real-data/model/metadata.json")["config"]),
        "sprint": replace(
            TrainingConfig(**read("artifacts/legacy-sprint/model/metadata.json")["config"]),
            validation_events=2,
        ),
    }
    bundles, all_paired, coverage, inputs, selections = (
        {"baseline": [], "revised": []},
        [],
        {},
        {},
        {},
    )
    for kind in SOURCES:
        original, manifest = merge_sources(kind, args.end_season)
        original_path = args.output / f"{kind}-original-records.json"
        atomic_json(original_path, original)
        atomic_json(args.output / f"{kind}-source-manifest.json", manifest)
        enriched_path = args.output / f"{kind}-enriched-records.json"
        enrichment = enrich_file(
            original_path,
            args.collection,
            enriched_path,
            args.output / f"{kind}-enrichment-audit.json",
            quality_aware=True,
            joint_effects=args.joint_effects,
        )
        enriched = read(enriched_path)
        present = validate_pair(original, enriched)
        baseline_rows = original
        if args.joint_effects:
            previous_path = args.output / f"{kind}-previous-records.json"
            enrich_file(
                original_path,
                args.collection,
                previous_path,
                args.output / f"{kind}-previous-enrichment-audit.json",
                quality_aware=True,
            )
            baseline_rows = read(previous_path)
            validate_pair(baseline_rows, enriched)
            present = {
                (r.snapshot.event_id, r.snapshot.session.value): any(
                    e and r.snapshot.session.value in e.joint_effects
                    for e in [d.performance for d in r.snapshot.drivers]
                    + [t.car.performance for t in r.snapshot.teams]
                )
                for r in records_from_json(enriched)
            }
        coverage[kind] = enrichment
        inputs[kind] = manifest
        reports = {}
        for arm, rows in (("baseline", baseline_rows), ("revised", enriched)):
            config = replace(
                configs[kind],
                optional_mode="learned" if args.joint_effects or arm == "revised" else "ignore",
                calibrate=args.joint_effects or arm == "revised",
            )
            report = backtest_transformer(
                rows,
                config=config,
                test_from_season=2026 if args.resume_from else 2020,
                simulations=1000,
                seed=42,
                progress=lambda message, label=f"{kind}/{arm}": print(
                    f"{label}: {message}", flush=True
                ),
            )
            if args.resume_from:
                old_path = args.resume_from / f"{kind}-{arm}.json"
                old_rows = read(
                    args.resume_from
                    / f"{kind}-{'previous' if arm == 'baseline' else 'enriched'}-records.json"
                )
                report = extend_report(read(old_path), old_rows, report, rows, config)
                report["reused_report"] = str(old_path.resolve())
                report["reused_report_sha256"] = hashlib.sha256(old_path.read_bytes()).hexdigest()
            verify_folds(report, rows)
            report["reproduction_config"] = asdict(config)
            atomic_json(args.output / f"{kind}-{arm}.json", report)
            reports[arm] = report
            bundles[arm].append({"report": report, "records": rows, "manifest": manifest})
            render_backtest_report(bundles[arm], args.output / f"{arm}-report.html")
        selections[kind] = {
            "folds": len(reports["revised"]["folds"]),
            "optional_correction_selected": sum(
                f.get("validation_selection", {}).get("optional_correction_selected", False)
                for f in reports["revised"]["folds"]
            ),
            "temperature_changed": sum(
                f.get("pace_temperature", 1) != 1 for f in reports["revised"]["folds"]
            ),
        }
        by_key = {(r["snapshot"]["event_id"], r["snapshot"]["session"]): r for r in original}
        for a, b in zip(reports["baseline"]["results"], reports["revised"]["results"], strict=True):
            key = a["event_id"], a["session"]
            if key != (b["event_id"], b["session"]):
                raise ValueError("Mismatched held-out sessions")
            row = by_key[key]
            fa, fb = Forecast.model_validate(a["forecast"]), Forecast.model_validate(b["forecast"])
            actual = Feedback.model_validate(row["feedback"])
            s = row["snapshot"]
            all_paired.append(
                {
                    "event_id": f"{s['season']}-{s['round']:02d}",
                    "source_event_id": key[0],
                    "format": kind,
                    "session": key[1],
                    "year": s["season"],
                    "before": metrics(fa, actual),
                    "after": metrics(fb, actual),
                    "optional_inputs_present": present[key],
                    "distribution_unchanged": fa.standings == fb.standings,
                    "frozen_heuristic": b["frozen_baseline"],
                }
            )
        print(f"Completed {kind}: {reports['revised']['sessions_evaluated']} forecasts", flush=True)
    result = {
        "end_season": args.end_season,
        "joint_effects": args.joint_effects,
        "overall": aggregate(all_paired),
        "by_year": {
            str(y): aggregate([r for r in all_paired if r["year"] == y])
            for y in range(2020, args.end_season + 1)
        },
        "by_format": {
            kind: aggregate([r for r in all_paired if r["format"] == kind]) for kind in SOURCES
        },
        "by_year_format": {
            str(y): {
                kind: aggregate([r for r in all_paired if r["year"] == y and r["format"] == kind])
                for kind in SOURCES
            }
            for y in range(2020, args.end_season + 1)
        },
        "by_session": {
            s: aggregate([r for r in all_paired if r["session"] == s])
            for s in ("race", "qualifying")
        },
        "optional_coverage_by_year": {
            str(y): sum(r["optional_inputs_present"] for r in all_paired if r["year"] == y)
            for y in range(2020, args.end_season + 1)
        },
        "directly_enriched": aggregate([r for r in all_paired if r["optional_inputs_present"]]),
        "enrichment": coverage,
        "sources": inputs,
        "selection": selections,
        "configs": {kind: asdict(config) for kind, config in configs.items()},
        "method": "Fresh continuous expanding-window training. 2019 initializes Grand Prix training. "
        "GP: 8 minimum training and 2 validation weekends. Sprint: 1 training and 2 validation "
        "weekends; 2021 sprints are development only. Separate models, identical paired folds, "
        "seed 42, 1000 simulations per scenario. Revised uses learned optional correction and "
        "validation-selected pace temperature; baseline uses original inputs without calibration.",
        "limitations": "This experiment carries history across 2023/2024, so its 2025 results are not "
        "the same experiment as training only on 2024/2025. Optional timing measurements "
        "are available only for parts of 2024/2025. Earlier improvements reflect calibration. "
        "Earlier weather is practice persistence, not an archived forecast. Historical "
        "availability is reconstructed. Missing classifications stay excluded. Shared 2021/2022 "
        "Friday qualifying in GP and sprint is not independent physical data. Paired intervals "
        "resample whole calendar weekends across formats; overlapping training windows and "
        "training-seed uncertainty remain. The historical period has been inspected previously.",
        "results": all_paired,
    }
    if args.joint_effects:
        result["method"] = (
            "Fresh continuous expanding-window evaluation from 2020 through 2025, with 2019 GP warm-up. "
            "Separate GP and sprint models; GP uses 8 minimum training and 2 validation weekends, "
            "sprint uses 1 training and 2 validation weekends. Identical folds, seed 42 and 1000 "
            "simulations per scenario. Both arms use learned measurements and validation-selected "
            "calibration. The previous model uses quality-aware timing proxies; the revised model "
            "also receives joint driver-season and evolving team effects."
        )
        result["limitations"] = (
            "No joint timing evidence is available for 2020-2023, so those forecasts should remain "
            "unchanged. Only parts of 2024-2025 have collected timing; later estimates can carry "
            "stale observations and have wider uncertainty. No verified upgrade chronology was supplied. "
            "2019 GP and 2021 sprint sessions are development history, not held-out results. Missing "
            "classifications remain excluded. Earlier weather uses reconstructed practice persistence, "
            "not archived forecasts. Historical publication times are assumptions. Shared Friday "
            "qualifying in 2021/2022 GP and sprint is not independent physical evidence. Bootstrap "
            "intervals resample calendar weekends across formats but do not fully model temporal "
            "dependence or repeated inspection of this historical test period."
        )
        if any(not r["distribution_unchanged"] for r in all_paired if r["year"] < 2024):
            raise ValueError("Joint evidence changed forecasts before its coverage period")
    if args.end_season == 2026:
        result["method"] = result["method"].replace(
            "through 2025", "through completed 2026 sessions"
        )
        if args.resume_from:
            result["method"] = result["method"].replace(
                "Fresh continuous", "Continued chronological"
            )
            result["method"] += (
                " Exact 2020-2025 records and forecasts were verified and reused; 2026 folds were newly trained on the full earlier history with unchanged event-index seeds."
            )
        result["limitations"] = result["limitations"].replace(
            "parts of 2024-2025", "parts of 2024-2026"
        )
        result["limitations"] += (
            " 2026 is year-to-date, not a completed season. Its collection cutoffs and exclusions are saved in source manifests. Incomplete qualifying rosters exclude affected weekends."
        )
    atomic_json(args.output / "comparison.json", result)
    render_summary(result, args.output)
    print(
        json.dumps({k: result[k] for k in ("overall", "by_year", "selection")}, indent=2),
        flush=True,
    )


def render_summary(result, output):
    joint = result.get("joint_effects", False)
    baseline_label = "Previous learned model" if joint else "Baseline"
    coverage_label = "With joint estimates" if joint else "With new measurements"
    previous_link = (
        "<a href='../joint-effects-backtest/joint.html'>2025-only joint experiment</a>"
        if joint
        else "<a href='../measurement-model-backtest-v2/comparison.html'>Previous 2025 experiment</a>"
    )
    rows = []
    for year, summary in result["by_year"].items():
        a, b = summary["before"], summary["after"]
        rows.append(
            f"<tr><td>{year}</td><td>{summary['sessions']}</td><td>{a['expected_position_mae']:.3f}</td>"
            f"<td>{b['expected_position_mae']:.3f}</td><td>{100 * b['pairwise_accuracy']:.2f}%</td>"
            f"<td>{100 * b['winner_correct']:.2f}%</td><td>{b['winner_brier']:.4f}</td>"
            f"<td>{result['optional_coverage_by_year'][year]}</td></tr>"
        )
    all_scores = result["overall"]
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>2020-2025 backtest</title>
<style>body{font:17px Georgia,serif;background:#f4f0e6;color:#16343c;max-width:1200px;margin:auto;padding:24px}
h1,h2{font-weight:500}a{color:#00695f}p{line-height:1.6}.scroll{overflow:auto}table{border-collapse:collapse;
background:#fffdf8;width:100%;font:14px monospace}td,th{padding:12px;text-align:right;border-bottom:1px solid #ddd}
th:first-child,td:first-child{text-align:left}</style><h1>2020-2025: revised forecast backtest</h1>"""
    page = page.replace("2020-2025", f"2020-{result.get('end_season', 2025)}")
    page += f"<p>{all_scores['sessions']} held-out model evaluations across {all_scores['weekends']} calendar weekends. "
    page += "Grand Prix and sprint results use separately trained models.</p>"
    page += (
        "<p><a href='revised-report.html'>Explore revised forecasts by year, circuit and session</a> | "
        f"<a href='baseline-report.html'>{baseline_label} report</a> | "
        "<a href='comparison.json'>Exact results and uncertainty intervals</a> | "
        f"{previous_link}</p>"
    )
    page += (
        f"<div class='scroll'><table><thead><tr><th>Year</th><th>Forecasts</th><th>{baseline_label} error</th>"
        "<th>Revised error</th><th>Ranking accuracy</th><th>Winner / pole</th><th>Brier</th>"
        f"<th>{coverage_label}</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    a, b = all_scores["before"], all_scores["after"]
    page += (
        f"<p>Average position error: {a['expected_position_mae']:.4f} to {b['expected_position_mae']:.4f}. "
        f"Winner Brier: {a['winner_brier']:.4f} to {b['winner_brier']:.4f}. "
        f"Position log loss: {a['position_log_loss']:.4f} to {b['position_log_loss']:.4f}. "
        "Lower is better for these three metrics.</p>"
    )
    page += (
        f"<p>The comparison baseline above is {'the previous learned and calibrated model' if joint else 'the original-input Transformer'}. The detailed "
        "session reports also show a separate frozen heuristic baseline. An improvement in "
        "winner Brier does not imply improvement in the full position distribution: read "
        "position log loss alongside it.</p>"
    )
    page += f"<h2>Method</h2><p>{html.escape(result['method'])}</p><h2>Coverage and limits</h2><p>{html.escape(result['limitations'])}</p></html>"
    (output / "comparison.html").write_text(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--end-season", type=int, choices=[2025, 2026], default=2025)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Verified completed joint 2020-2025 experiment to extend without changing its forecasts",
    )
    parser.add_argument(
        "--joint-effects",
        action="store_true",
        help="Compare previous learned/calibrated inputs with added joint driver/car effects",
    )
    parser.add_argument(
        "--collection", type=Path, default=Path("data/processed/performance-full-2024-2025")
    )
    run(parser.parse_args())

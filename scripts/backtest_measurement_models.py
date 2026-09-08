"""Compare fixed, separate learned and calibrated optional-input models on matched folds."""

import argparse
import html
import json
from dataclasses import replace
from pathlib import Path

import torch
from backtest_expanded_collection import aggregate, validate_pair
from recheck_optional_inputs import metrics, read

from f1_forecast.ml_backtest import backtest_transformer
from f1_forecast.models import Feedback, Forecast
from f1_forecast.neural import TrainingConfig
from f1_forecast.performance_collection import atomic_json, digest
from f1_forecast.reporting import render_backtest_report


def run(args):
    original, enriched = read(args.original), read(args.enriched)
    coverage = validate_pair(original, enriched)
    reference = read("artifacts/real-data/backtest.json")
    if digest(original) != reference["dataset_sha256"]:
        raise ValueError("Original dataset differs from saved reference experiment")
    config = TrainingConfig(**read("artifacts/real-data/model/metadata.json")["config"])
    settings = reference["results"][0]["forecast"]
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    arms = [
        ("baseline", original, original, "ignore", False),
        ("fixed_weights", enriched, enriched, "legacy", False),
        ("prediction_only", original, enriched, "legacy", False),
        ("training_only", enriched, original, "legacy", False),
        ("learned", enriched, enriched, "learned", False),
        ("calibrated_baseline", original, original, "ignore", True),
        ("learned_calibrated", enriched, enriched, "learned", True),
    ]
    actuals = {
        (r["snapshot"]["event_id"], r["snapshot"]["session"]): Feedback.model_validate(
            r["feedback"]
        )
        for r in original
    }
    baseline_rows, summaries = None, {}
    for name, training, inference, mode, calibrate in arms:
        report = backtest_transformer(
            training,
            prediction_rows=inference,
            config=replace(config, optional_mode=mode, calibrate=calibrate),
            simulations=settings["simulations_per_scenario"],
            seed=settings["seed"],
            test_from_season=reference["test_from_season"],
            save_model=args.output / "model" if name == "learned_calibrated" else None,
            progress=lambda message, name=name: print(f"{name}: {message}", flush=True),
        )
        report["reproduction_config"] = vars(
            replace(config, optional_mode=mode, calibrate=calibrate)
        )
        keys = [(r["event_id"], r["session"]) for r in report["results"]]
        if keys != [(r["event_id"], r["session"]) for r in reference["results"]]:
            raise ValueError("Held-out forecasts differ from reference experiment")
        atomic_json(args.output / f"{name}.json", report)
        render_backtest_report(
            [{"report": report, "records": inference}], args.output / f"{name}.html"
        )
        if baseline_rows is None:
            baseline_rows = report["results"]
        paired = []
        for before, after in zip(baseline_rows, report["results"], strict=True):
            key = (after["event_id"], after["session"])
            a, b = (
                Forecast.model_validate(before["forecast"]),
                Forecast.model_validate(after["forecast"]),
            )
            paired.append(
                {
                    "event_id": key[0],
                    "session": key[1],
                    "optional_inputs_present": coverage[key],
                    "before": metrics(a, actuals[key]),
                    "after": metrics(b, actuals[key]),
                    "distribution_unchanged": a.standings == b.standings,
                }
            )
        summary = {
            "overall": aggregate(paired),
            "by_session": {
                s: aggregate([r for r in paired if r["session"] == s])
                for s in ("race", "qualifying")
            },
            "directly_enriched": aggregate([r for r in paired if r["optional_inputs_present"]]),
            "correction_selected_folds": sum(
                f.get("validation_selection", {}).get("optional_correction_selected", False)
                for f in report["folds"]
            ),
            "temperature_changed_folds": sum(
                f.get("pace_temperature", 1) != 1 for f in report["folds"]
            ),
            "folds": len(report["folds"]),
            "results": paired,
        }
        summaries[name] = summary
        atomic_json(
            args.output / "comparison.json",
            {
                "original_sha256": digest(original),
                "enriched_sha256": digest(enriched),
                "source_snapshots": len(original),
                "enriched_snapshots": sum(coverage.values()),
                "config": vars(config),
                "arms": summaries,
                "limitations": "Exploratory matched 2025 evaluation, not an untouched prospective test. "
                "Historical source availability is reconstructed. Every option is selected "
                "only on earlier validation weekends, reused for epoch selection. Temperature "
                "scales conditional pace only, not DNF risk or weather probabilities. "
                "Bootstrap intervals resample weekends; training-seed uncertainty and "
                "dependence across expanding training folds remain. Unmeasured fuel, "
                "traffic, physical braking and aero quantities remain uncertain.",
            },
        )
        print(json.dumps({name: {k: v for k, v in summary.items() if k != "results"}}), flush=True)
    render_comparison(read(args.output / "comparison.json"), args.output)


def render_comparison(result, output):
    titles = {
        "baseline": "Original inputs",
        "fixed_weights": "Fixed weights, expanded data",
        "prediction_only": "New inputs only at prediction",
        "training_only": "New inputs only in training",
        "learned": "Learned measurement correction",
        "calibrated_baseline": "Original inputs + calibration",
        "learned_calibrated": "Learned correction + calibration",
    }
    rows = []
    for name, arm in result["arms"].items():
        s = arm["overall"]["after"]
        rows.append(
            f"<tr><td><a href='{name}.html'>{titles[name]}</a></td>"
            f"<td>{s['expected_position_mae']:.4f}</td><td>{100 * s['pairwise_accuracy']:.2f}%</td>"
            f"<td>{100 * s['winner_correct']:.2f}%</td><td>{s['winner_brier']:.4f}</td>"
            f"<td>{s['position_log_loss']:.4f}</td><td>{100 * s['displayed_position_confidence']:.2f}%</td>"
            f"<td>{arm['correction_selected_folds']}/{arm['folds']}</td></tr>"
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Optional measurement model comparison</title>
<style>body{font:17px Georgia,serif;color:#16343c;background:#f4f0e6;max-width:1250px;margin:auto;padding:24px}
h1,h2{font-weight:500}a{color:#00695f}.scroll{overflow:auto}table{background:#fffdf8;border-collapse:collapse;
width:100%;font:14px monospace}td,th{padding:12px;text-align:right;border-bottom:1px solid #d6ded9}
td:first-child,th:first-child{text-align:left;min-width:220px}p{line-height:1.6;max-width:950px}</style>
<h1>Learning from optional measurements</h1><p>Seven matched experiments on the same 45 held-out 2025 forecasts.
Training and validation use earlier weekends. Click a model to explore its year, circuit, session and weather scenarios.</p>
<div class="scroll"><table><thead><tr><th>Experiment</th><th>Position error</th><th>Ranking accuracy</th>
<th>Winner / pole</th><th>Brier score</th><th>Log loss</th><th>Displayed confidence</th><th>Correction used</th></tr></thead><tbody>"""
    page += "".join(rows) + "</tbody></table></div>"
    final = result["arms"]["learned_calibrated"]["overall"]
    original, revised = final["before"], final["after"]
    gain = (
        100
        * (original["expected_position_mae"] - revised["expected_position_mae"])
        / original["expected_position_mae"]
    )
    lower, upper = final["paired_weekend_bootstrap_95pct_delta"]["expected_position_mae"]
    page += (
        f"<h2>Measured outcome</h2><p>The revised model changes average position error from "
        f"{original['expected_position_mae']:.4f} to {revised['expected_position_mae']:.4f} "
        f"({gain:+.2f}% improvement). Winner/pole accuracy changes from "
        f"{100 * original['winner_correct']:.2f}% to {100 * revised['winner_correct']:.2f}%. "
        "Compare with the calibrated original-input model to see how much improvement "
        "comes from calibration rather than the optional measurements.</p>"
        f"<p>The 95% paired-weekend interval for the change in position error is "
        f"{lower:+.4f} to {upper:+.4f} positions; an interval crossing zero does not "
        "establish a reliable improvement. The previously inspected 2025 period remains "
        "an exploratory evaluation.</p>"
    )
    page += (
        "<p>Lower position error, Brier score and log loss are better. Higher displayed confidence alone "
        "is not evidence of better probabilities. Correction used counts weekends where earlier validation "
        "supported the measured-input correction.</p>"
    )
    page += f"<p>{result['enriched_snapshots']} of {result['source_snapshots']} source snapshots contain optional evidence.</p>"
    page += (
        "<p><a href='comparison.json'>Exact paired metrics and uncertainty intervals</a> | "
        "<a href='../real-data/backtest-report.html'>Existing combined historical report</a></p>"
    )
    page += (
        "<p>Collection coverage and pending downloads are recorded separately. "
        "<a href='../../data/processed/performance-full-2024-2025/quality-report.html'>"
        "View data coverage and pending sessions</a>.</p>"
    )
    page += f"<h2>Scope and limits</h2><p>{html.escape(result['limitations'])}</p></html>"
    (output / "comparison.html").write_text(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="data/processed/real-history-2024-2025.json")
    parser.add_argument("--enriched", required=True)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())

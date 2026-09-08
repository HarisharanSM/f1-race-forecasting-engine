"""Matched chronological evaluation of original and optionally enriched history."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch
from recheck_optional_inputs import metrics, read, summarize

from f1_forecast.ml_backtest import backtest_transformer
from f1_forecast.models import Feedback, Forecast, Snapshot
from f1_forecast.neural import TrainingConfig
from f1_forecast.performance_collection import atomic_json, digest
from f1_forecast.reporting import render_backtest_report


def optional(snapshot):
    return bool(
        snapshot.race_dynamics
        or any(d.performance for d in snapshot.drivers)
        or any(t.car.performance for t in snapshot.teams)
    )


def validate_pair(original, enriched):
    if len(original) != len(enriched):
        raise ValueError("Dataset lengths differ")
    coverage = {}
    for before, after in zip(original, enriched, strict=True):
        if before["feedback"] != after["feedback"]:
            raise ValueError("Outcome labels differ")
        a, b = [Snapshot.model_validate(r["snapshot"]) for r in (before, after)]
        key = (a.event_id, a.session.value)
        coverage[key] = optional(b)
        # Only optional measurements and appended evidence may differ between arms.
        x, y = a.model_dump(mode="json"), b.model_dump(mode="json")
        if y["sources"][: len(x["sources"])] != x["sources"]:
            raise ValueError("Original source provenance changed")
        if y["notes"][: len(x["notes"])] != x["notes"]:
            raise ValueError("Original notes changed")
        y["sources"], y["notes"] = x["sources"], x["notes"]
        y["race_dynamics"] = x["race_dynamics"]
        for da, db in zip(x["drivers"], y["drivers"], strict=True):
            db["performance"] = da["performance"]
        for ta, tb in zip(x["teams"], y["teams"], strict=True):
            tb["car"]["performance"] = ta["car"]["performance"]
            if tb["car"]["upgrades"][: len(ta["car"]["upgrades"])] != ta["car"]["upgrades"]:
                raise ValueError("Original upgrade provenance changed")
            tb["car"]["upgrades"] = ta["car"]["upgrades"]
        if x != y:
            raise ValueError("Non-optional forecast inputs changed")
    return coverage


def aggregate(rows):
    if not rows:
        return {"sessions": 0}
    result = summarize(rows)
    events = sorted({r["event_id"] for r in rows})
    result["weekends"] = len(events)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(events), size=(5000, len(events)))
    counts = np.array([sum(r["event_id"] == e for r in rows) for e in events])
    intervals = {}
    for metric in result["before"]:
        sums = np.array(
            [
                sum(r["after"][metric] - r["before"][metric] for r in rows if r["event_id"] == e)
                for e in events
            ]
        )
        estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        intervals[metric] = np.quantile(estimates, [0.025, 0.975]).tolist()
    result["paired_weekend_bootstrap_95pct_delta"] = intervals
    return result


def run(args):
    original, enriched = read(args.original), read(args.enriched)
    coverage = validate_pair(original, enriched)
    saved, metadata = read(args.reference), read(args.metadata)
    if digest(original) != saved["dataset_sha256"]:
        raise ValueError("Original dataset does not match the reference experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = TrainingConfig(**metadata["config"])
    settings = saved["results"][0]["forecast"]
    reports = []
    for label, records in (("before", original), ("after", enriched)):
        report = backtest_transformer(
            records,
            config=config,
            simulations=settings["simulations_per_scenario"],
            seed=settings["seed"],
            test_from_season=saved["test_from_season"],
            progress=lambda message, label=label: print(f"{label}: {message}", flush=True),
        )
        if [(r["event_id"], r["session"]) for r in report["results"]] != [
            (r["event_id"], r["session"]) for r in saved["results"]
        ]:
            raise ValueError("Held-out folds differ from the reference experiment")
        report["reproduction_config"] = metadata["config"]
        atomic_json(args.output / f"{label}.json", report)
        render_backtest_report(
            [{"report": report, "records": records}], args.output / f"{label}-report.html"
        )
        reports.append(report)
    actuals = {
        (r["snapshot"]["event_id"], r["snapshot"]["session"]): Feedback.model_validate(
            r["feedback"]
        )
        for r in original
    }
    matched = []
    for a, b in zip(reports[0]["results"], reports[1]["results"], strict=True):
        key = (a["event_id"], a["session"])
        fa, fb = Forecast.model_validate(a["forecast"]), Forecast.model_validate(b["forecast"])
        matched.append(
            {
                "event_id": key[0],
                "session": key[1],
                "optional_inputs_present": coverage[key],
                "before": metrics(fa, actuals[key]),
                "after": metrics(fb, actuals[key]),
                "distribution_unchanged": fa.standings == fb.standings,
            }
        )
    result = {
        "original_sha256": digest(original),
        "enriched_sha256": digest(enriched),
        "config": metadata["config"],
        "seed": settings["seed"],
        "simulations_per_scenario": settings["simulations_per_scenario"],
        "overall": aggregate(matched),
        "directly_enriched": aggregate([r for r in matched if r["optional_inputs_present"]]),
        "without_direct_enrichment": aggregate(
            [r for r in matched if not r["optional_inputs_present"]]
        ),
        "by_session": {
            s: aggregate([r for r in matched if r["session"] == s]) for s in ("qualifying", "race")
        },
        "limitations": "Only the first two weekends of 2024/2025 were collected; older observations "
        "can enrich later snapshots within 30 days. Retraining on enriched earlier records can "
        "also affect forecasts without direct optional inputs. Historical availability is "
        "reconstructed, not proven point-in-time. Confidence intervals resample paired weekends "
        "and do not capture training-seed uncertainty or eliminate dependence across expanding "
        "folds. This is exploratory evaluation, not a new untouched prospective test. Higher "
        "displayed confidence alone is not improved calibration. No tuning used these results.",
        "results": matched,
    }
    atomic_json(args.output / "comparison.json", result)
    render_comparison(result, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2), flush=True)


def render_comparison(result, output):
    labels = {
        "expected_position_mae": "Average position error (lower is better)",
        "pairwise_accuracy": "Pairwise ranking accuracy",
        "winner_correct": "Winner / pole accuracy",
        "winner_brier": "Winner Brier score (lower is better)",
        "position_log_loss": "Position log loss (lower is better)",
        "position_interval_coverage": "80% position-interval coverage",
        "position_interval_width": "Mean interval width (positions)",
        "displayed_position_confidence": "Displayed exact-position confidence",
    }
    sections = []
    for key, title in (
        ("overall", "All held-out forecasts"),
        ("directly_enriched", "Forecasts with optional measurements"),
        ("without_direct_enrichment", "Forecasts without direct optional measurements"),
    ):
        summary = result[key]
        if not summary["sessions"]:
            continue
        rows = "".join(
            f"<tr><td>{html.escape(labels.get(metric, metric))}</td><td>{value:.6f}</td>"
            f"<td>{summary['after'][metric]:.6f}</td>"
            f"<td>{summary['delta_after_minus_before'][metric]:+.6f}</td>"
            f"<td>{summary['paired_weekend_bootstrap_95pct_delta'][metric][0]:+.6f} to "
            f"{summary['paired_weekend_bootstrap_95pct_delta'][metric][1]:+.6f}</td></tr>"
            for metric, value in summary["before"].items()
        )
        sections.append(
            f"<h2>{title}</h2><p>{summary['sessions']} sessions / {summary['weekends']} weekends</p>"
            "<div class='scroll'><table><thead><tr><th>Metric</th><th>Before</th><th>After</th>"
            f"<th>Change</th><th>95% paired interval for change</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Expanded collection backtest</title><style>"
        "body{font:16px Georgia,serif;background:#f4f1e9;color:#182e35;max-width:1200px;margin:auto;padding:24px}"
        "h1,h2{font-weight:500}a{color:#006d68}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;"
        "background:#fffdf7;font:13px monospace}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}"
        "th:first-child,td:first-child{text-align:left}p{line-height:1.6}</style>"
        "<h1>Expanded collection: matched backtest</h1>"
        f"<p>Fresh expanding-window Transformer fits. Same test folds, seed {result['seed']}, "
        f"{result['simulations_per_scenario']:,} simulations per scenario, and unchanged outcome "
        "labels. Original data preserved.</p>"
        "<p><a href='after-report.html'>Open enriched session report</a> | "
        "<a href='before-report.html'>Original-input session report</a> | "
        "<a href='comparison.json'>Full paired results</a> | "
        "<a href='../real-data/backtest-report.html'>Existing combined historical report (unchanged)</a></p>"
        "<p>Accuracy and coverage values are fractions (0.01 = one percentage point). Lower MAE, "
        "Brier score and log loss are better. Coverage should be read alongside interval width; "
        "higher displayed confidence is not necessarily better.</p>"
        + "".join(sections)
        + f"<h2>Limitations</h2><p>{html.escape(result['limitations'])}</p></html>"
    )
    (output / "comparison.html").write_text(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", default="data/processed/real-history-2024-2025.json")
    parser.add_argument(
        "--enriched", default="data/processed/real-history-2024-2025-performance.json"
    )
    parser.add_argument("--reference", default="artifacts/real-data/backtest.json")
    parser.add_argument("--metadata", default="artifacts/real-data/model/metadata.json")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())

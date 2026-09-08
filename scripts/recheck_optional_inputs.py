"""Repeat the saved 233-session experiment and compare matched forecast distributions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from f1_forecast.ml_backtest import backtest_transformer
from f1_forecast.models import Feedback, Forecast
from f1_forecast.neural import TrainingConfig

SUITES = {
    "real-data": "real-history-2024-2025.json",
    "sprint-data": "sprint-history-2024-2025.json",
    "legacy-data": "history-2019-2023-complete.json",
    "legacy-sprint": "sprint-history-2021-2023-final.json",
}


def read(path):
    return json.loads(Path(path).read_text())


def metrics(forecast, actual):
    from f1_forecast.engine import evaluate

    result = evaluate(forecast, actual)
    positions = {d: i + 1 for i, d in enumerate(actual.finishing_order)}
    result.update(
        position_interval_coverage=float(
            np.mean(
                [
                    s.p10_position <= positions[s.driver_id] <= s.p90_position
                    for s in forecast.standings
                ]
            )
        ),
        position_interval_width=float(
            np.mean([s.p90_position - s.p10_position for s in forecast.standings])
        ),
        displayed_position_confidence=float(
            np.mean([s.position_probabilities[s.position - 1] for s in forecast.standings])
        ),
    )
    return result


def run(output):
    import f1_forecast.engine

    output.mkdir(parents=True, exist_ok=False)
    print(f"Engine: {f1_forecast.engine.__file__}", flush=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for suite, dataset in SUITES.items():
        saved = read(f"artifacts/{suite}/backtest.json")
        metadata = read(f"artifacts/{suite}/model/metadata.json")
        rows = read(f"data/processed/{dataset}")
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        if digest != saved["dataset_sha256"]:
            raise ValueError(f"Dataset differs from saved experiment: {dataset}")
        original = saved["results"][0]["forecast"]
        report = backtest_transformer(
            rows,
            config=TrainingConfig(**metadata["config"]),
            simulations=original["simulations_per_scenario"],
            seed=original["seed"],
            test_from_season=saved["test_from_season"],
            progress=lambda message, label=suite: print(f"{label}: {message}", flush=True),
        )
        if [(r["event_id"], r["session"]) for r in report["results"]] != [
            (r["event_id"], r["session"]) for r in saved["results"]
        ]:
            raise ValueError("Held-out sessions do not match saved experiment")
        report["reproduction_config"] = metadata["config"]
        (output / f"{suite}.json").write_text(json.dumps(report, indent=2))


def compare(before, after):
    all_rows, summaries = [], {}
    optional_records = 0
    bundles = []
    for suite, dataset in SUITES.items():
        records = read(f"data/processed/{dataset}")
        optional_records += sum(
            bool(
                r["snapshot"].get("race_dynamics")
                or any(d.get("performance") for d in r["snapshot"]["drivers"])
                or any(t["car"].get("performance") for t in r["snapshot"]["teams"])
            )
            for r in records
        )
        actuals = {
            (r["snapshot"]["event_id"], r["snapshot"]["session"]): Feedback.model_validate(
                r["feedback"]
            )
            for r in records
        }
        old, new = read(before / f"{suite}.json"), read(after / f"{suite}.json")
        if old["dataset_sha256"] != new["dataset_sha256"]:
            raise ValueError("Comparison requires identical datasets")
        matched = []
        for a, b in zip(old["results"], new["results"], strict=True):
            key = (a["event_id"], a["session"])
            if key != (b["event_id"], b["session"]):
                raise ValueError("Comparison requires identical held-out sessions")
            fa, fb = Forecast.model_validate(a["forecast"]), Forecast.model_validate(b["forecast"])
            matched.append(
                {
                    "suite": suite,
                    "event_id": key[0],
                    "session": key[1],
                    "before": metrics(fa, actuals[key]),
                    "after": metrics(fb, actuals[key]),
                    "distribution_unchanged": fa.standings == fb.standings,
                }
            )
        summaries[suite] = summarize(matched)
        all_rows.extend(matched)
        bundles.append({"report": new, "records": records})
    result = {
        "overall": summarize(all_rows),
        "by_suite": summaries,
        "by_session": {
            s: summarize([r for r in all_rows if r["session"] == s]) for s in ("qualifying", "race")
        },
        "records_with_optional_inputs": optional_records,
        "interpretation": "Missing-input compatibility experiment. Archived datasets do not "
        "contain the new optional measurements. This cannot estimate their predictive benefit. "
        "Higher displayed confidence alone is not evidence of better probability quality.",
        "results": all_rows,
    }
    (after / "comparison.json").write_text(json.dumps(result, indent=2))
    from f1_forecast.reporting import render_backtest_report

    render_backtest_report(bundles, after / "backtest-report.html")
    print(json.dumps({"overall": result["overall"], "by_session": result["by_session"]}, indent=2))


def summarize(rows):
    a = {k: float(np.mean([r["before"][k] for r in rows])) for k in rows[0]["before"]}
    b = {k: float(np.mean([r["after"][k] for r in rows])) for k in a}
    return {
        "sessions": len(rows),
        "unchanged_distributions": sum(r["distribution_unchanged"] for r in rows),
        "before": a,
        "after": b,
        "delta_after_minus_before": {k: b[k] - a[k] for k in a},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("output", type=Path)
    comparator = sub.add_parser("compare")
    comparator.add_argument("before", type=Path)
    comparator.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.output)
    else:
        compare(args.before, args.after)

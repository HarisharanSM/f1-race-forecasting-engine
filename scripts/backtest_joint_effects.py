"""Matched 2025 evaluation: existing measured inputs versus added joint effects."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from backtest_expanded_collection import validate_pair

from f1_forecast.ml_backtest import backtest_transformer
from f1_forecast.neural import TrainingConfig
from f1_forecast.performance_collection import atomic_json
from f1_forecast.reporting import render_backtest_report


def comparison_summary(results):
    summary = {
        name: {
            "metrics": r["summary"],
            "folds_with_correction": sum(
                f["validation_selection"]["optional_correction_selected"] for f in r["folds"]
            ),
        }
        for name, r in results.items()
    }
    events = {}
    for a, b in zip(results["previous"]["results"], results["joint"]["results"], strict=True):
        if (a["event_id"], a["session"]) != (b["event_id"], b["session"]):
            raise ValueError("Backtest session sets differ")
        events.setdefault(a["event_id"], []).append(
            b["ml"]["expected_position_mae"] - a["ml"]["expected_position_mae"]
        )
    values, rng = list(events.values()), np.random.default_rng(42)
    samples = [
        np.mean([v for i in rng.integers(len(values), size=len(values)) for v in values[i]])
        for _ in range(5000)
    ]
    summary["mae_change_uncertainty"] = {
        "direction": "joint minus previous; negative is better",
        "method": "Paired calendar-weekend bootstrap, 5000 resamples, seed 42",
        "interval_95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "limitation": "Small retrospective sample; interval does not model all temporal dependence or selection bias.",
    }
    return summary


def run(args):
    previous = json.loads(args.previous.read_text())
    joint = json.loads(args.joint.read_text())
    validate_pair(previous, joint)
    config = replace(
        TrainingConfig(**json.loads(args.config.read_text())["config"]),
        optional_mode="learned",
        calibrate=True,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = {}
    for name, records in (("previous", previous), ("joint", joint)):
        report = backtest_transformer(
            records,
            config=config,
            simulations=1000,
            seed=42,
            test_from_season=2025,
            save_model=args.output / f"{name}-model",
            progress=lambda message, name=name: print(f"{name}: {message}", flush=True),
        )
        report["reproduction_config"] = vars(config)
        atomic_json(args.output / f"{name}.json", report)
        render_backtest_report(
            [{"report": report, "records": records}], args.output / f"{name}.html"
        )
        results[name] = report
    keys = lambda r: [(s["event_id"], s["session"]) for s in r["results"]]
    if keys(results["previous"]) != keys(results["joint"]):
        raise ValueError("Backtest session sets differ")
    summary = comparison_summary(results)
    atomic_json(args.output / "comparison.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("data/processed/real-history-quality-measurements-v2.json"),
    )
    parser.add_argument(
        "--joint", type=Path, default=Path("data/processed/real-history-joint-effects.json")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("artifacts/real-data/model/metadata.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())

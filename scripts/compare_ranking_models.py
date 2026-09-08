"""Run matched 2020-2026 ranking and probability-ensemble comparisons."""

import argparse
import html
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from f1_forecast.model_comparison import KINDS, compare
from f1_forecast.neural import TrainingConfig
from f1_forecast.performance_collection import digest


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results, provenance = [], {}
    for kind in ("grand_prix", "sprint"):
        source = args.source / f"{kind}-enriched-records.json"
        rows = json.loads(source.read_text())
        reference = json.loads((args.source / f"{kind}-revised.json").read_text())
        if digest(rows) != reference["dataset_sha256"]:
            raise ValueError("Source checksum mismatch")
        config = TrainingConfig(**reference["reproduction_config"])
        config = replace(config, optional_mode="legacy", calibrate=False)
        report = compare(
            rows, config, progress=lambda s, label=kind: print(f"{label}: {s}", flush=True)
        )
        report["dataset_sha256"] = digest(rows)
        report["config"] = asdict(config)
        (args.output / f"{kind}.json").write_text(json.dumps(report, separators=(",", ":")))
        results.extend(report["results"])
        provenance[kind] = {
            "source": str(source.resolve()),
            "sha256": digest(rows),
            "config": asdict(config),
        }
    names = (*KINDS, "ensemble")

    def summarize(rows):
        return {
            name: {
                metric: float(np.mean([r["metrics"][name][metric] for r in rows]))
                for metric in rows[0]["metrics"][name]
            }
            for name in names
        }

    weekend_groups = {}
    for row in results:
        weekend_groups.setdefault((row["season"], row["round"]), []).append(row)
    groups = list(weekend_groups.values())
    draws = np.random.default_rng(42).integers(len(groups), size=(1000, len(groups)))
    intervals = {
        name: {
            metric: np.quantile(
                [
                    np.mean(
                        [
                            r["metrics"][name][metric] - r["metrics"]["field_transformer"][metric]
                            for i in draw
                            for r in groups[i]
                        ]
                    )
                    for draw in draws
                ],
                [0.025, 0.975],
            ).tolist()
            for metric in ("expected_position_mae", "winner_brier", "position_log_loss")
        }
        for name in names
        if name != "field_transformer"
    }
    summary = {
        "sessions": len(results),
        "weekends": len(groups),
        "overall": summarize(results),
        "by_year": {
            str(y): summarize([r for r in results if r["season"] == y])
            for y in sorted({r["season"] for r in results})
        },
        "paired_weekend_bootstrap_95pct_delta_vs_transformer": intervals,
        "provenance": provenance,
        "limitations": "Architecture comparison on identical 27 features including the same legacy optional-input adapters. "
        "No architecture receives the specialized learned measurement correction or pace calibration. "
        "Same clean-pair ranking and DNF objective, training budget, simulator and seed. "
        "Two earlier weekends select epochs; a separate later two-weekend window selects among seven fixed probability blends. "
        "No selection labels are supplied to component models or their history features. "
        "All models share the same target weekends; extra validation reduces coverage versus previous reports. "
        "This is not a head-to-head with the previously optimized production pipeline. "
        "Historical retrospective comparison, not a pristine untouched test. 2026 uses saved YTD data through Italian GP qualifying. "
        "Bootstrap resamples calendar weekends jointly across formats; it does not capture all temporal dependence or repeated model-development bias. "
        "No deployed model is replaced automatically.",
    }
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2))

    def table(scores):
        return (
            "<table><tr><th>Model</th><th>Position error</th><th>Pairwise accuracy</th><th>Winner/pole</th><th>Winner Brier</th><th>Log loss</th></tr>"
            + "".join(
                f"<tr><td>{name}</td><td>{r['expected_position_mae']:.4f}</td><td>{r['pairwise_accuracy']:.2%}</td>"
                f"<td>{r['winner_correct']:.2%}</td><td>{r['winner_brier']:.4f}</td><td>{r['position_log_loss']:.4f}</td></tr>"
                for name, r in scores.items()
            )
            + "</table>"
        )

    page = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Ranking model comparison</title><style>body{font:17px/1.6 Georgia;background:#f4f0e6;color:#16343c;max-width:1100px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow:auto}</style>"
    page += f"<h1>Ranking model comparison</h1><p>{summary['sessions']} matched sessions across {summary['weekends']} weekends.</p><p>{html.escape(summary['limitations'])}</p>"
    page += "<div class='scroll'>" + table(summary["overall"]) + "</div>"
    for year, scores in summary["by_year"].items():
        page += f"<h2>{year}</h2><div class='scroll'>{table(scores)}</div>"
    page += "<p><a href='comparison.json'>Exact metrics, uncertainty intervals and provenance</a> | <a href='grand_prix.json'>Grand Prix forecasts and fold audit</a> | <a href='sprint.json'>Sprint forecasts and fold audit</a></p></html>"
    (args.output / "comparison.html").write_text(page)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/incident-era-backtest-2020-2026")
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())

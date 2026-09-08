"""Conservative multi-period calibration with a labelled retrospective later comparison."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from backtest_2020_2025 import verify_folds

from f1_forecast.engine import evaluate
from f1_forecast.ml_backtest import mean_metrics
from f1_forecast.models import Feedback, Forecast, Snapshot
from f1_forecast.performance_collection import atomic_json, digest
from f1_forecast.probability_calibration import (
    fit_calibrator,
    fit_conservative_calibrator,
    reliability,
    stratum,
)
from f1_forecast.reporting import render_backtest_report


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    examples, sources = [], {}
    for kind in ("grand_prix", "sprint"):
        records = json.loads((args.experiment / f"{kind}-enriched-records.json").read_text())
        report = json.loads((args.experiment / f"{kind}-revised.json").read_text())
        if digest(records) != report["dataset_sha256"]:
            raise ValueError("Source report and records checksum mismatch")
        verify_folds(report, records)
        by_key = {(r["snapshot"]["event_id"], r["snapshot"]["session"]): r for r in records}
        for row in report["results"]:
            record = by_key[row["event_id"], row["session"]]
            examples.append(
                (
                    Snapshot.model_validate(record["snapshot"]),
                    Forecast.model_validate(row["forecast"]),
                    Feedback.model_validate(record["feedback"]),
                )
            )
        sources[kind] = (records, report)
    conservative = args.method == "conservative"
    fit_end = 2021 if conservative else 2023
    fit = [r for r in examples if r[0].season <= fit_end]
    selection = [r for r in examples if fit_end < r[0].season <= 2024]
    test = [r for r in examples if r[0].season >= 2025]
    if not test:
        raise ValueError("No later holdout forecasts")
    calibrator = (fit_conservative_calibrator if conservative else fit_calibrator)(fit, selection)
    if calibrator.fitted_through >= min(s.as_of for s, _, _ in test):
        raise ValueError("Selection feedback overlaps holdout cutoffs")
    calibrator.audit["source_datasets"] = {k: r[1]["dataset_sha256"] for k, r in sources.items()}
    calibrator.save(args.output / "calibrator.json")
    # The fitted artifact is frozen before any holdout scores are calculated.
    after = [(s, calibrator.apply(s, f), a) for s, f, a in test]
    strata = sorted({stratum(s) for s, _, _ in test})
    diagnostics = {
        key: {
            "before": reliability([r for r in test if stratum(r[0]) == key]),
            "after": reliability([r for r in after if stratum(r[0]) == key]),
        }
        for key in strata
    }
    differences = {}
    for (s, f, a), (_, revised, _) in zip(test, after, strict=True):
        changes = {
            k: evaluate(revised, a)[k] - evaluate(f, a)[k]
            for k in ("position_log_loss", "winner_brier", "expected_position_mae")
        }
        differences.setdefault(f"{s.season}-{s.round:02d}", []).append(changes)
    events, rng = list(differences.values()), np.random.default_rng(42)
    draws = rng.integers(len(events), size=(5000, len(events)))
    intervals = {
        k: np.quantile(
            [np.mean([v[k] for i in draw for v in events[i]]) for draw in draws], [0.025, 0.975]
        ).tolist()
        for k in ("position_log_loss", "winner_brier", "expected_position_mae")
    }
    paired = {
        "before": [evaluate(f, a) for _, f, a in test],
        "after": [evaluate(f, a) for _, f, a in after],
    }
    summary = {
        arm: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        for arm, rows in paired.items()
    }
    result = {
        "fit_sessions": len(fit),
        "selection_sessions": len(selection),
        "test_sessions": len(test),
        "test_weekends": len(events),
        "split": f"Fit <={fit_end}; select {fit_end + 1}-2024; retrospective evaluation >=2025",
        "limitations": "Retrospective temporal holdout: these historical weekends were inspected in prior project work, "
        "so this is not a genuinely untouched prospective test. Base models update chronologically; "
        "calibration is frozen before 2025. DNF and weather probabilities are not calibrated. "
        "Bins use equal session weights; driver trials within a weekend are dependent. Sparse bins "
        "have fewer than 30 observations or 10 weekends. Bootstrap uses 5000 calendar-weekend resamples, "
        "seed 42, and does not capture all temporal dependence or development selection bias.",
        "calibrator": calibrator.model_dump(mode="json"),
        "summary": summary,
        "paired_weekend_bootstrap_95pct_delta": intervals,
        "strata": diagnostics,
    }
    atomic_json(args.output / "reliability.json", result)
    atomic_json(
        args.output / "deployment-review.json",
        {
            "status": "experimental_not_promoted",
            "reason": "Review later holdout scores before deployment; defaults are unchanged.",
            "test_outcomes_used_to_refit": False,
            "prospective_test_required": True,
        },
    )
    bundles = {"before": [], "after": []}
    mapped = {(s.event_id, s.session.value): f for s, f, _ in after}
    actuals = {(s.event_id, s.session.value): a for s, _, a in test}
    for kind, (records, original) in sources.items():
        for arm, bundle in bundles.items():
            rows = []
            for row in original["results"]:
                key = row["event_id"], row["session"]
                if key not in mapped:
                    continue
                updated = dict(row)
                if arm == "after":
                    updated.update(
                        forecast=mapped[key].model_dump(mode="json"),
                        ml=evaluate(mapped[key], actuals[key]),
                    )
                rows.append(updated)
            report = {
                **original,
                "results": rows,
                "sessions_evaluated": len(rows),
                "test_from_season": 2025,
                "probability_calibration": {
                    "split": result["split"],
                    "limitations": result["limitations"],
                    "applied": arm == "after",
                    "temperatures": calibrator.temperatures,
                    "blends": calibrator.blends,
                    "roles": {
                        f"{s.event_id}:{s.session.value}": role
                        for group, role in (
                            (fit, "Calibration fitting"),
                            (selection, "Calibration selection"),
                            (test, "Later calibration holdout"),
                        )
                        for s, _, _ in group
                        if s.race_format == kind
                    },
                },
                "summary": {
                    "ml": mean_metrics(rows, "ml"),
                    "frozen_baseline": mean_metrics(rows, "frozen_baseline"),
                },
            }
            atomic_json(args.output / f"{kind}-{arm}.json", report)
            bundle.append({"report": report, "records": records})
    for arm, bundle in bundles.items():
        render_backtest_report(bundle, args.output / f"{arm}-report.html")
    render(result, args.output / "reliability.html")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "fit_sessions",
                    "selection_sessions",
                    "test_sessions",
                    "summary",
                    "paired_weekend_bootstrap_95pct_delta",
                )
            },
            indent=2,
        )
    )


def render(result, path):
    sections = []
    for key, group in result["strata"].items():
        stats = result["calibrator"]["audit"]["strata"][key]
        sections.append(
            f"<h2>{html.escape(key)}</h2><p>Temperature: {result['calibrator']['temperatures'][key]:g}. "
            f"Blend: {result['calibrator'].get('blends', {}).get(key, 1):.0%}. "
            f"{html.escape(stats.get('fallback_reason') or '')} "
            f"Fit: {stats['fit_weekends']} weekends; selection: {stats['selection_weekends']}. "
            f"Holdout: {group['after']['sessions']} sessions.</p>"
        )
        for event in ("winner", "podium", "displayed_position", "dnf"):
            before, after = [group[arm]["events"][event] for arm in ("before", "after")]
            if after["ece"] is None:
                continue
            points, rows = [], []
            for arm, color in (("before", "#9a6c35"), ("after", "#14634d")):
                for b in group[arm]["events"][event]["bins"]:
                    if b["count"]:
                        points.append(
                            f"<circle cx='{40 + 300 * b['predicted']:.2f}' cy='{340 - 300 * b['observed']:.2f}' r='4' fill='{color}'><title>{arm}: {b['count']} observations</title></circle>"
                        )
            for b in after["bins"]:
                if b["count"]:
                    rows.append(
                        f"<tr><td>{b['low']:.0%}-{b['high']:.0%}</td><td>{b['predicted']:.1%}</td><td>{b['observed']:.1%}</td><td>{b['count']}</td><td>{b['weekends']}</td><td>{'Sparse' if b['sparse'] else 'Supported'}</td></tr>"
                    )
            sections.append(
                f"<section><h3>{event.replace('_', ' ').title()}</h3><p>ECE: {before['ece']:.4f} to {after['ece']:.4f}; Brier per driver: {before['brier']:.4f} to {after['brier']:.4f}. Lower is better, but ECE depends on binning.</p>"
                "<svg viewBox='0 0 380 380' role='img' aria-label='Predicted chance versus observed frequency'><path d='M40 40 V340 H340 M40 340 L340 40' fill='none' stroke='#acb5ad'/>"
                + "".join(points)
                + "<text x='120' y='375'>Predicted probability (0 to 1)</text><text x='5' y='20'>Observed frequency (0 to 1)</text></svg>"
                "<p>Brown: before. Green: after. Diagonal: perfect agreement. DNF is diagnostic only.</p><div class='scroll'><table><tr><th>Bin</th><th>Mean predicted</th><th>Observed</th><th>Driver observations</th><th>Weekends</th><th>Support</th></tr>"
                + "".join(rows)
                + "</table></div></section>"
            )
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{result['summary']['before'][k]:.6f}</td><td>{result['summary']['after'][k]:.6f}</td></tr>"
        for k in (
            "position_log_loss",
            "winner_brier",
            "expected_position_mae",
            "position_interval_coverage",
        )
    )
    page = """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Probability calibration audit</title><style>body{font:16px/1.6 Georgia,serif;background:#f5f5f0;color:#192725;max-width:1100px;margin:auto;padding:24px}h1,h2{color:#14634d}section{background:white;border:1px solid #dce2dc;padding:20px;margin:20px 0}svg{width:360px;max-width:100%}svg text{font:11px sans-serif}table{border-collapse:collapse;width:100%;font:13px/1.5 monospace}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow:auto}a{color:#14634d}.notice{background:#f1ebcf;padding:16px}</style>
<h1>Do predicted chances match observed frequencies?</h1>"""
    page += f"<p>{result['fit_sessions']} earlier fit sessions; {result['selection_sessions']} selection sessions; {result['test_sessions']} later holdout sessions across {result['test_weekends']} weekends.</p>"
    page += "<p class='notice'><strong>Experimental only: defaults unchanged.</strong> Lower log loss and Brier scores are better. Higher interval coverage alone does not establish better calibration, especially if intervals become wider.</p>"
    page += f"<p class='notice'>{html.escape(result['split'])}. {html.escape(result['limitations'])}</p>"
    page += f"<p>{html.escape(result['calibrator']['audit']['rule'])}</p>"
    page += "<p><a href='after-report.html'>Calibrated holdout forecasts</a> | <a href='before-report.html'>Same holdout before calibration</a> | <a href='reliability.json'>Exact audit</a> | <a href='calibrator.json'>Frozen calibrator</a></p>"
    page += (
        "<div class='scroll'><table><tr><th>Metric</th><th>Before</th><th>After</th></tr>"
        + rows
        + "</table></div>"
        + "".join(sections)
        + "</html>"
    )
    path.write_text(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", type=Path, default=Path("artifacts/joint-effects-backtest-2020-2026")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("conservative", "legacy"), default="conservative")
    run(parser.parse_args())

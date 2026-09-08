"""Chronological 2010-2026 evaluation, preferring existing richer snapshots."""

import argparse
import html
import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

import torch

from f1_forecast.historical import classification, projected, ratings
from f1_forecast.ml_backtest import backtest_transformer, mean_metrics
from f1_forecast.ml_data import records_from_json
from f1_forecast.models import (
    ActualWeather,
    Circuit,
    Feedback,
    Session,
    Snapshot,
    Source,
    Weather,
    utcnow,
)
from f1_forecast.neural import TrainingConfig
from f1_forecast.performance_collection import atomic_json, digest
from f1_forecast.reporting import render_backtest_report


def read(path):
    return json.loads(Path(path).read_text())


def assemble(collection, richer):
    manifest = read(collection / "manifest.json")
    previous, reduced, exclusions = [], [], []
    existing = {
        (r["snapshot"]["season"], r["snapshot"]["round"], r["snapshot"]["session"]): r
        for r in richer
    }

    def source(year, endpoint):
        job = manifest["jobs"][f"{year}:{endpoint}"]
        payload = read(collection / "sources" / f"{year}-{endpoint}.json")
        if digest(payload) != job["sha256"]:
            raise ValueError("Historical source checksum mismatch")
        return payload

    for year in range(2010, 2027):
        results, quals = source(year, "results"), source(year, "qualifying")
        schedule = source(year, "schedule")["races"]
        for rnd, event in sorted(schedule.items(), key=lambda p: int(p[0])):
            for stage, payload, field in (
                (Session.QUALIFYING, quals, "QualifyingResults"),
                (Session.RACE, results, "Results"),
            ):
                key = (year, int(rnd), stage.value)
                label = f"{year}-{int(rnd):02d}"
                try:
                    race = payload["races"][rnd]
                    ordered = classification(race[field])
                    timing = event if stage == Session.RACE else event.get("Qualifying", {})
                    if not timing.get("date"):
                        raise ValueError("Missing session date")
                    start = datetime.fromisoformat(
                        timing["date"] + "T" + timing.get("time", "12:00:00Z")
                    )
                    available = start + timedelta(hours=12)
                    if available > utcnow():
                        raise ValueError("Session results not yet available at collection cutoff")
                    cutoff = start - timedelta(hours=1)
                    sources = [Source.model_validate(s) for s in payload["sources"]]
                    drivers, teams, prior_sources = ratings(ordered, previous, cutoff)
                    qorder = None
                    if stage == Session.RACE:
                        qrows = classification(quals["races"][rnd]["QualifyingResults"])
                        qorder = [r["Driver"]["driverId"] for r in qrows]
                        if set(qorder) != {d.id for d in drivers}:
                            raise ValueError(
                                "Race/qualifying roster mismatch; no invented entrants"
                            )
                        qtime = event.get("Qualifying", {})
                        qa = datetime.fromisoformat(
                            qtime["date"] + "T" + qtime.get("time", "12:00:00Z")
                        ) + timedelta(hours=12)
                        if qa >= cutoff:
                            raise ValueError(
                                "Qualifying availability cannot be established before race"
                            )
                        prior_sources += [
                            projected(
                                Source.model_validate(s),
                                qa,
                                "Qualifying classification assumed available 12 hours after scheduled or imputed midday start",
                            )
                            for s in quals["sources"]
                        ]
                    notes = [
                        "REDUCED INPUT: archived classifications and prior-result form only; optional measurements unavailable.",
                        "Missing weather is imputed: rain/wet fraction 0.2, temperature 25 C, wind 0. These are assumptions, including training weather, NOT observations or archived forecasts.",
                        "Unsupported car/circuit traits use model defaults. No target laps, fastest laps, finishing position or incidents enter forecast inputs.",
                        "Final roster reconstructed retrospectively; missing session clock times assumed 12:00 UTC. Classification availability assumed 12 hours later. Grid penalties unavailable; qualifying order only.",
                    ]
                    c = event["Circuit"]
                    snapshot = Snapshot(
                        event_id=label,
                        season=year,
                        round=int(rnd),
                        session=stage,
                        session_start=start,
                        as_of=cutoff,
                        drivers=drivers,
                        teams=teams,
                        circuit=Circuit(
                            id=c["circuitId"],
                            name=c["circuitName"],
                            latitude=float(c["Location"]["lat"]),
                            longitude=float(c["Location"]["long"]),
                        ),
                        weather=Weather(
                            rain_probability=0.2,
                            air_temperature_c=25,
                            wind_speed_ms=0,
                            issued_at=cutoff,
                            valid_at=start,
                            source="Missing weather: fixed imputation, not observed",
                        ),
                        qualifying_order=qorder,
                        sources=prior_sources,
                        notes=notes,
                        data_mode="historical_reconstruction",
                    )
                    retired = [
                        r["Driver"]["driverId"]
                        for r in ordered
                        if stage == Session.RACE
                        and r.get("status") != "Finished"
                        and not r.get("status", "").startswith("+")
                    ]
                    feedback = Feedback(
                        event_id=label,
                        session=stage,
                        available_at=available,
                        finishing_order=[r["Driver"]["driverId"] for r in ordered],
                        retired_drivers=retired,
                        actual_weather=ActualWeather(
                            wet_fraction=0.2, air_temperature_c=25, wind_speed_ms=0
                        ),
                        sources=[
                            projected(
                                s,
                                available,
                                "Classification verified; weather fields are missing-data imputations, NOT verified observations; availability assumed 12 hours after start",
                            )
                            for s in sources
                        ],
                        verified=True,
                        data_mode="historical_reconstruction",
                    )
                    if key not in existing:
                        reduced.append(
                            {
                                "snapshot": snapshot.model_dump(mode="json"),
                                "feedback": feedback.model_dump(mode="json"),
                            }
                        )
                    previous.append(
                        {
                            "key": f"{label}-{stage}",
                            "stage": stage,
                            "available": available,
                            "clean": [r for r in ordered if r["Driver"]["driverId"] not in retired],
                            "sources": sources,
                        }
                    )
                except (ValueError, KeyError) as exc:
                    if key not in existing:
                        exclusions.append(
                            {
                                "event_id": label,
                                "session": stage.value,
                                "reason": f"Missing archived session data: {exc}"
                                if isinstance(exc, KeyError)
                                else str(exc),
                            }
                        )
    rows = sorted([*richer, *reduced], key=lambda r: r["snapshot"]["as_of"])
    records_from_json(rows)
    return rows, exclusions


def tier(row):
    return (
        "reduced"
        if any(n.startswith("REDUCED INPUT:") for n in row["snapshot"]["notes"])
        else "richer_existing"
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    bundles, results = [], []
    for kind in ("grand_prix", "sprint"):
        original = read(args.source / f"{kind}-enriched-records.json")
        reference = read(args.source / f"{kind}-revised.json")
        if digest(original) != reference["dataset_sha256"]:
            raise ValueError("Richer source checksum mismatch")
        rows, exclusions = (
            assemble(args.collection, original) if kind == "grand_prix" else (original, [])
        )
        config = replace(
            TrainingConfig(**reference["reproduction_config"]),
            optional_mode="legacy",
            calibrate=False,
        )
        path = args.output / f"{kind}-backtest.json"
        if path.exists():
            saved = read(path)
            if saved["dataset_sha256"] != digest(rows) or saved["reproduction_config"] != asdict(
                config
            ):
                raise ValueError("Resume source/config changed")
        atomic_json(args.output / f"{kind}-records.json", rows)
        atomic_json(
            args.output / f"{kind}-coverage.json",
            {
                "exclusions": exclusions,
                "tiers": {
                    t: sum(tier(r) == t for r in rows) for t in ("reduced", "richer_existing")
                },
            },
        )
        path = args.output / f"{kind}-backtest.json"
        if path.exists():
            report = read(path)
            if report["dataset_sha256"] != digest(rows) or report["reproduction_config"] != asdict(
                config
            ):
                raise ValueError("Resume source/config changed")
        else:
            print(f"{kind}: {len(rows)} input sessions, {len(exclusions)} exclusions", flush=True)
            report = backtest_transformer(
                rows, config=config, test_from_season=2010, progress=lambda s: print(s, flush=True)
            )
            report["reproduction_config"] = asdict(config)
            report["limitations"] += (
                " Reduced-input sessions use imputed training weather, not observed weather; see snapshot notes."
            )
            atomic_json(path, report)
        tiers = {(r["snapshot"]["event_id"], r["snapshot"]["session"]): tier(r) for r in rows}
        results.extend(
            {**r, "tier": tiers[r["event_id"], r["session"]], "format": kind}
            for r in report["results"]
        )
        bundles.append({"report": report, "records": rows, "manifest": {"exclusions": exclusions}})
    render_backtest_report(bundles, args.output / "report.html")

    def summarize(items):
        return {
            "sessions": len(items),
            "transformer": mean_metrics(items, "ml"),
            "baseline": mean_metrics(items, "frozen_baseline"),
        }

    summary = {
        "overall": summarize(results),
        "by_year": {
            str(y): summarize([r for r in results if r["season"] == y]) for y in range(2010, 2027)
        },
        "by_input_tier": {
            t: summarize([r for r in results if r["tier"] == t])
            for t in ("reduced", "richer_existing")
        },
    }
    atomic_json(args.output / "summary.json", summary)
    page = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>2010-2026 mixed-input backtest</title><style>body{font:17px/1.6 Georgia;background:#f4f0e6;color:#173d38;max-width:1100px;margin:auto;padding:24px}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #bbb;text-align:left}</style><h1>2010-2026 mixed-input backtest</h1><p>Fresh chronological field Transformer training. Richer existing snapshots take priority; missing GP sessions use explicitly imputed reduced inputs. Richer does not mean complete telemetry. 2026 is saved year-to-date data, not a full season. Early weekends are training warm-up, not scored forecasts. Sprint coverage is the existing verified dataset only.</p><p>Reduced weather fields, including training weather, are fixed assumptions, NOT measured conditions. Missing clock times are midday assumptions. Retrospective final rosters are used. Optional tyres, incidents and upgrades remain absent where unavailable. These results are not directly comparable to a different era or coverage mix.</p><p><a href='report.html'>Interactive year/circuit/session forecast report</a> | <a href='summary.json'>Exact metrics</a></p><table><tr><th>Period / inputs</th><th>Sessions</th><th>Position error</th><th>Pair accuracy</th><th>Winner/pole</th><th>Brier</th><th>80% interval coverage</th></tr>"
    for label, score in [
        ("Overall", summary["overall"]),
        *summary["by_input_tier"].items(),
        *summary["by_year"].items(),
    ]:
        m = score["transformer"]
        if m:
            page += f"<tr><td>{html.escape(label)}</td><td>{score['sessions']}</td><td>{m['expected_position_mae']:.3f}</td><td>{m['pairwise_accuracy']:.1%}</td><td>{m['winner_correct']:.1%}</td><td>{m['winner_brier']:.3f}</td><td>{m['position_interval_coverage']:.1%}</td></tr>"
    (args.output / "index.html").write_text(page + "</table></html>")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/incident-era-backtest-2020-2026")
    )
    parser.add_argument(
        "--collection", type=Path, default=Path("data/processed/full-history-2010-2026")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/mixed-input-backtest-2010-2026")
    )
    run(parser.parse_args())

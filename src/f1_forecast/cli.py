"""Command-line entry point; JSON contracts are shared with the Python and HTTP APIs."""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from .backtest import backtest
from .data import Fetcher, fetch_actual, fetch_event
from .demo import run_demo
from .llm import LLM, Document
from .models import Feedback, ScenarioPlan, Session, Snapshot
from .service import ForecastService
from .store import Store
from .training import export_training, start_finetune


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    content = (
        value.model_dump_json(indent=2)
        if hasattr(value, "model_dump_json")
        else json.dumps(value, indent=2)
    )
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content + "\n", encoding="utf-8")
    else:
        print(content)


def parser():
    root = argparse.ArgumentParser(description="F1 qualifying and Grand Prix scenario forecasts")
    root.add_argument("--database", default=os.getenv("F1_DATABASE", "data/forecast.sqlite3"))
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run an offline fictional weekend and feedback loop")
    demo.add_argument(
        "--output", default="artifacts/demo-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    )
    demo.add_argument("--simulations", type=int, default=3000)
    prediction = commands.add_parser("predict")
    prediction.add_argument("input")
    prediction.add_argument("--output")
    prediction.add_argument("--simulations", type=int, default=5000)
    prediction.add_argument("--seed", type=int, default=42)
    prediction.add_argument("--max-scenarios", type=int, choices=range(1, 6), default=5)
    prediction.add_argument("--scenarios", help="Optional JSON ScenarioPlan")
    prediction.add_argument("--llm", action="store_true")
    prediction.add_argument(
        "--probability-calibrator",
        help="Frozen earlier-validation calibration JSON; optional, never fitted during prediction",
    )
    prediction.add_argument(
        "--ml-model",
        default=os.getenv("F1_ML_MODEL"),
        help="Explicit trained checkpoint; overrides primary bundle",
    )
    prediction.add_argument(
        "--model-policy", choices=["primary", "ensemble", "heuristic"], default="primary"
    )
    prediction.add_argument(
        "--primary-bundle", help="Root containing grand_prix and sprint primary bundles"
    )
    feedback = commands.add_parser("feedback")
    feedback.add_argument("forecast_id")
    feedback.add_argument("actual")
    event = commands.add_parser(
        "fetch-event", help="Fetch schedule, provisional roster and weather"
    )
    event.add_argument("--season", type=int, required=True)
    event.add_argument("--round", type=int, required=True)
    event.add_argument("--session", choices=[s.value for s in Session], required=True)
    event.add_argument("--laps", type=int, required=True)
    event.add_argument("--output", required=True)
    actual = commands.add_parser("fetch-feedback")
    actual.add_argument("forecast_id")
    actual.add_argument("--session-key", type=int, required=True)
    actual.add_argument("--output", required=True)
    extraction = commands.add_parser(
        "extract", help="Enrich car traits from sourced documents via LLM"
    )
    extraction.add_argument("input")
    extraction.add_argument("documents", help="JSON list of source/text documents")
    extraction.add_argument("--output", required=True)
    extraction.add_argument("--report", required=True)
    research = commands.add_parser("research", help="LLM web research for a future session")
    research.add_argument("input")
    research.add_argument("--output", required=True)
    research.add_argument("--report", required=True)
    extraction_actual = commands.add_parser(
        "extract-feedback", help="LLM fallback; returns unverified results"
    )
    extraction_actual.add_argument("documents")
    extraction_actual.add_argument("--event-id", required=True)
    extraction_actual.add_argument("--session", choices=[s.value for s in Session], required=True)
    extraction_actual.add_argument("--output", required=True)
    replay = commands.add_parser("backtest")
    replay.add_argument("records", help="JSON list of snapshot/feedback pairs")
    replay.add_argument("--simulations", type=int, default=1000)
    replay.add_argument("--output")
    replay.add_argument("--model", choices=["heuristic", "transformer"], default="heuristic")
    replay.add_argument(
        "--test-from-season", type=int, help="Start Transformer evaluation in this season"
    )
    replay.add_argument("--save-model", help="Save a Transformer refitted after the backtest")
    train_ml = commands.add_parser(
        "train-ml", help="Train a Transformer from a file or database feedback"
    )
    train_ml.add_argument(
        "records", nargs="?", help="JSON history; omit to use the selected database"
    )
    train_ml.add_argument("--output", required=True, help="New model directory")
    train_ml.add_argument("--as-of", help="Optional timezone-aware historical training cutoff")
    ml_demo = commands.add_parser(
        "ml-demo", help="Train and backtest on fictional multi-weekend data"
    )
    ml_demo.add_argument(
        "--output", default="artifacts/ml-demo-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    )
    ml_demo.add_argument("--simulations", type=int, default=1000)
    for command in (replay, train_ml, ml_demo):
        command.add_argument(
            "--optional-mode", choices=["legacy", "ignore", "learned"], default="legacy"
        )
        command.add_argument(
            "--calibrate",
            action="store_true",
            help="Select pace temperature on earlier validation forecasts",
        )
        command.add_argument("--epochs", type=int, default=60)
        command.add_argument("--patience", type=int, default=8)
        command.add_argument("--min-train-events", type=int, default=3)
        command.add_argument("--validation-events", type=int, default=1)
        command.add_argument("--seed", type=int, default=42)
    history = commands.add_parser("fetch-history", help="Build actual historical F1 training data")
    history.add_argument(
        "--weather-mode",
        choices=["archived_forecast", "practice_persistence"],
        default="archived_forecast",
    )
    history.add_argument("--race-format", choices=["grand_prix", "sprint"], default="grand_prix")
    history.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025])
    history.add_argument("--output", default="data/processed/real-history-2024-2025.json")
    history.add_argument("--cache", default="data/raw/historical")
    history.add_argument(
        "--include-current-season",
        action="store_true",
        help="Allow completed sessions from the current year; excludes unavailable and future results",
    )
    performance = commands.add_parser(
        "fetch-performance",
        help="Collect practice, qualifying and race laps/stints with quality reports",
    )
    performance.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025])
    performance.add_argument("--rounds", nargs="+", type=int)
    performance.add_argument(
        "--sessions",
        nargs="+",
        choices=["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"],
        default=["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"],
    )
    performance.add_argument("--provider", choices=["auto", "fastf1", "openf1"], default="auto")
    performance.add_argument(
        "--telemetry",
        action="store_true",
        help="Also cache FastF1 car/position traces (larger downloads)",
    )
    performance.add_argument("--resume", action="store_true")
    performance.add_argument(
        "--max-sessions", type=int, default=20, help="New session attempts per batch (1..40)"
    )
    performance.add_argument(
        "--settle-hours",
        type=float,
        default=6,
        help="Minimum hours after scheduled start; 0 checks just-completed sessions",
    )
    performance.add_argument("--slow-lap-factor", type=float, default=1.07)
    performance.add_argument("--cache", default="data/raw/performance")
    performance.add_argument("--output", required=True, help="Collection directory")
    enrichment = commands.add_parser(
        "enrich-performance", help="Fill optional inputs using eligible collected lap evidence"
    )
    enrichment.add_argument("input", help="Snapshot or historical snapshot/feedback list")
    enrichment.add_argument("--collection", required=True)
    enrichment.add_argument("--output", required=True)
    enrichment.add_argument("--report", required=True)
    enrichment.add_argument(
        "--joint-effects",
        action="store_true",
        help="Fit driver-season and evolving team effects jointly",
    )
    enrichment.add_argument(
        "--upgrades",
        help="Optional JSON list of sourced, dated team upgrades; requires --joint-effects",
    )
    enrichment.add_argument(
        "--quality-aware",
        action="store_true",
        help="Require stronger matched samples and attach quality statistics",
    )
    readable = commands.add_parser(
        "report", help="Create readable backtest tables with actual results and ML feedback"
    )
    readable.add_argument("backtest", help="Saved Transformer backtest JSON")
    readable.add_argument(
        "--records", required=True, help="Exact historical dataset used in the backtest"
    )
    readable.add_argument("--manifest", help="Optional collection manifest with excluded sessions")
    readable.add_argument("--sprint-backtest", help="Optional separately trained sprint backtest")
    readable.add_argument("--sprint-records", help="Exact sprint dataset")
    readable.add_argument("--sprint-manifest", help="Optional sprint collection manifest")
    readable.add_argument(
        "--add-backtest",
        nargs=3,
        action="append",
        default=[],
        metavar=("BACKTEST", "RECORDS", "MANIFEST"),
        help="Add another completed backtest and its exact records/manifest",
    )
    readable.add_argument("--output", default="artifacts/real-data/backtest-report.html")
    export = commands.add_parser("export-training")
    export.add_argument("--output", required=True)
    training = commands.add_parser("fine-tune", help="Submit a billable LLM fine-tuning job")
    training.add_argument("dataset")
    training.add_argument("--base-model", required=True)
    status = commands.add_parser("training-status")
    status.add_argument("job_id")
    return root


def main():
    args = parser().parse_args()
    try:
        if args.command == "demo":
            write(None, run_demo(args.output, args.simulations))
        elif args.command == "predict":
            snapshot = Snapshot.model_validate(read(args.input))
            service = ForecastService(
                args.database,
                LLM() if args.llm else None,
                ml_model=args.ml_model,
                probability_calibrator=args.probability_calibrator,
                model_policy=args.model_policy,
                primary_bundle=args.primary_bundle,
            )
            plan = ScenarioPlan.model_validate(read(args.scenarios)) if args.scenarios else None
            write(
                args.output,
                service.predict(
                    snapshot,
                    simulations=args.simulations,
                    seed=args.seed,
                    max_scenarios=args.max_scenarios,
                    plan=plan,
                ),
            )
        elif args.command == "feedback":
            write(
                None,
                ForecastService(args.database).feedback(
                    args.forecast_id, Feedback.model_validate(read(args.actual))
                ),
            )
        elif args.command in ("fetch-event", "fetch-feedback"):
            fetcher = Fetcher()
            try:
                if args.command == "fetch-event":
                    result = fetch_event(
                        fetcher, args.season, args.round, Session(args.session), laps=args.laps
                    )
                else:
                    snapshot, _ = Store(args.database).get(args.forecast_id)
                    result = fetch_actual(fetcher, snapshot, args.session_key)
                write(args.output, result)
            finally:
                fetcher.close()
        elif args.command == "research":
            snapshot, report = LLM().research(Snapshot.model_validate(read(args.input)))
            write(args.output, snapshot)
            write(args.report, report)
        elif args.command == "extract":
            snapshot, report = LLM().extract(
                Snapshot.model_validate(read(args.input)),
                [Document.model_validate(d) for d in read(args.documents)],
            )
            write(args.output, snapshot)
            write(args.report, report)
        elif args.command == "extract-feedback":
            write(
                args.output,
                LLM().extract_actual(
                    [Document.model_validate(d) for d in read(args.documents)],
                    args.event_id,
                    args.session,
                ),
            )
        elif args.command == "backtest":
            if args.model == "transformer":
                from .ml_backtest import backtest_transformer

                write(
                    args.output,
                    backtest_transformer(
                        read(args.records),
                        config=ml_config(args),
                        simulations=args.simulations,
                        seed=args.seed,
                        save_model=args.save_model,
                        test_from_season=args.test_from_season,
                        progress=lambda message: print(message, file=sys.stderr),
                    ),
                )
            else:
                if args.save_model or args.test_from_season is not None:
                    raise ValueError(
                        "--save-model and --test-from-season require --model transformer"
                    )
                write(
                    args.output,
                    backtest(read(args.records), simulations=args.simulations, seed=args.seed),
                )
        elif args.command == "train-ml":
            from .neural import train_model

            rows = read(args.records) if args.records else Store(args.database).training_records()
            write(
                None,
                train_model(
                    rows,
                    args.output,
                    config=ml_config(args),
                    as_of=datetime.fromisoformat(args.as_of) if args.as_of else None,
                ),
            )
        elif args.command == "ml-demo":
            from .ml_demo import run_ml_demo

            write(
                None, run_ml_demo(args.output, config=ml_config(args), simulations=args.simulations)
            )
        elif args.command == "report":
            from .reporting import render_backtest_report

            if bool(args.sprint_backtest) != bool(args.sprint_records):
                raise ValueError("Supply both --sprint-backtest and --sprint-records")
            if args.sprint_manifest and not args.sprint_records:
                raise ValueError("--sprint-manifest requires sprint records and backtest")
            bundles = [
                {
                    "report": read(args.backtest),
                    "records": read(args.records),
                    "manifest": read(args.manifest) if args.manifest else None,
                }
            ]
            if args.sprint_backtest:
                bundles.append(
                    {
                        "report": read(args.sprint_backtest),
                        "records": read(args.sprint_records),
                        "manifest": read(args.sprint_manifest) if args.sprint_manifest else None,
                    }
                )
            for extra_report, extra_records, extra_manifest in args.add_backtest:
                bundles.append(
                    {
                        "report": read(extra_report),
                        "records": read(extra_records),
                        "manifest": read(extra_manifest),
                    }
                )
            write(None, render_backtest_report(bundles, args.output))
        elif args.command == "fetch-history":
            from .historical import collect_history
            from .legacy_history import collect_legacy_history

            collector = (
                collect_legacy_history
                if args.weather_mode == "practice_persistence"
                else collect_history
            )
            if args.include_current_season and args.weather_mode != "archived_forecast":
                raise ValueError(
                    "Current-season collection requires archived_forecast weather mode"
                )
            write(
                None,
                collector(
                    args.seasons,
                    args.output,
                    args.cache,
                    race_format=args.race_format,
                    progress=lambda message: print(message, file=sys.stderr),
                    **({"include_current_season": True} if args.include_current_season else {}),
                ),
            )
        elif args.command == "fetch-performance":
            from .performance_collection import collect_performance

            result = collect_performance(
                args.seasons,
                args.output,
                args.cache,
                rounds=args.rounds,
                sessions=args.sessions,
                provider=args.provider,
                telemetry=args.telemetry,
                resume=args.resume,
                slow_lap_factor=args.slow_lap_factor,
                max_sessions=args.max_sessions,
                settle_hours=args.settle_hours,
                progress=lambda message: print(message, file=sys.stderr),
            )
            write(None, result)
            if not result["sessions_collected"] and (
                result["sessions_failed"] or result["catalog_errors"]
            ):
                raise ValueError(f"No sessions collected; inspect {result['report']}")
        elif args.command == "enrich-performance":
            from .performance_enrichment import enrich_file

            write(
                None,
                enrich_file(
                    args.input,
                    args.collection,
                    args.output,
                    args.report,
                    quality_aware=args.quality_aware,
                    joint_effects=args.joint_effects,
                    upgrades=args.upgrades,
                ),
            )
        elif args.command == "export-training":
            write(None, export_training(Store(args.database), args.output))
        elif args.command == "fine-tune":
            write(None, start_finetune(args.dataset, args.base_model))
        elif args.command == "training-status":
            from openai import OpenAI

            job = OpenAI().fine_tuning.jobs.retrieve(args.job_id)
            write(
                None,
                {"job_id": job.id, "status": job.status, "fine_tuned_model": job.fine_tuned_model},
            )
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def ml_config(args):
    from .neural import TrainingConfig

    return TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        min_train_events=args.min_train_events,
        validation_events=args.validation_events,
        seed=args.seed,
        optional_mode=args.optional_mode,
        calibrate=args.calibrate,
    )


if __name__ == "__main__":
    main()

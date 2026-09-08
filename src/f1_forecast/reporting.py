"""Readable, offline backtest reports from saved forecasts and their exact source records."""

import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from .ml_data import excluded_drivers, records_from_json
from .models import Forecast


def stamp(value):
    return datetime.fromisoformat(value)


def render_backtest_report(bundles, output):
    """Each bundle contains report, records, and an optional collection manifest."""
    sessions, exclusions, summaries = [], [], []
    seen = set()
    for bundle in bundles:
        rows, report = bundle["records"], bundle["report"]
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        if report.get("dataset_sha256") != digest:
            raise ValueError(
                "Backtest and records checksum mismatch; use the exact evaluated dataset"
            )
        records = records_from_json(rows)
        record_map = {(r.snapshot.event_id, r.snapshot.session.value): r for r in records}
        results = {}
        for result in report["results"]:
            key = (result["event_id"], result["session"])
            if key not in record_map or key in results:
                raise ValueError("Missing or duplicated source session for a forecast")
            forecast = Forecast.model_validate(result["forecast"])
            if (forecast.event_id, forecast.session.value) != key:
                raise ValueError("Forecast identity does not match its backtest row")
            if forecast.race_format != record_map[key].snapshot.race_format:
                raise ValueError("Forecast race format does not match its source record")
            expected = {d.id for d in record_map[key].snapshot.drivers}
            for standings in [forecast.standings, *[s.standings for s in forecast.scenarios]]:
                if {s.driver_id for s in standings} != expected or len(standings) != len(expected):
                    raise ValueError("Forecast roster does not match its source record")
                if sorted(s.position for s in standings) != list(range(1, len(expected) + 1)):
                    raise ValueError("Forecast positions must form a complete ordering")
                if any(len(s.position_probabilities) != len(expected) for s in standings):
                    raise ValueError("Incomplete forecast probability distribution")
            results[key] = (forecast, result)
        kind = records[0].snapshot.race_format
        summaries.append({"format": kind, "evaluated": len(results), "metrics": report["summary"]})
        for record in records:
            snapshot, actual = record.snapshot, record.feedback
            key = (snapshot.event_id, snapshot.session.value)
            if key in seen:
                raise ValueError("Duplicate session across report bundles")
            seen.add(key)
            names = {d.id: d.name for d in snapshot.drivers}
            teams = {t.id: t.name for t in snapshot.teams}
            team_names = {d.id: teams[d.team_id] for d in snapshot.drivers}
            actual_positions = {d: p for p, d in enumerate(actual.finishing_order, 1)}
            forecast, scored = results.get(key, (None, None))
            usage = feedback_usage(record, report, records)
            masked = excluded_drivers(actual)
            n_clean = len(names) - len(masked)
            actual_rows = [
                {
                    "position": p,
                    "driver": names[d],
                    "driver_id": d,
                    "team": team_names[d],
                    "status": "Retired / non-finish"
                    if d in actual.retired_drivers
                    else "Classified",
                    "pace_used": d not in masked,
                }
                for d, p in actual_positions.items()
            ]
            scenario_rows = []
            if forecast:
                for scenario in forecast.scenarios:
                    s = scenario.scenario
                    scenario_rows.append(
                        {
                            "name": s.name,
                            "probability": s.probability,
                            "brief": scenario_brief(s.wet_fraction),
                            "wet_fraction": s.wet_fraction,
                            "disruption_probability": s.disruption_probability,
                            "race_event_rates": scenario.race_event_rates,
                            "temperature": snapshot.weather.air_temperature_c,
                            "wind_kmh": snapshot.weather.wind_speed_ms * 3.6,
                            "winner": names[scenario.winner],
                            "rows": readable_rows(
                                scenario.standings, names, team_names, actual_positions
                            ),
                        }
                    )
            events = [
                {
                    "kind": e.kind.replace("_", " "),
                    "driver": names.get(e.driver_id, "Session"),
                    "description": e.description,
                }
                for e in actual.events
            ]
            sessions.append(
                {
                    "id": snapshot.event_id + ":" + snapshot.session.value,
                    "event_id": snapshot.event_id,
                    "year": snapshot.season,
                    "round": snapshot.round,
                    "event_key": f"{snapshot.season}-{snapshot.round:02d}",
                    "input_notes": snapshot.notes,
                    "weather_method": "imputed"
                    if any(note.startswith("REDUCED INPUT:") for note in snapshot.notes)
                    else "practice_persistence"
                    if any(
                        "WEATHER METHOD: practice persistence" in note for note in snapshot.notes
                    )
                    else "archived_forecast",
                    "format_note": next(
                        (note for note in snapshot.notes if note.startswith("Shared Friday")), None
                    ),
                    "circuit": snapshot.circuit.name,
                    "circuit_id": snapshot.circuit.id,
                    "format": snapshot.race_format,
                    "session": snapshot.session.value,
                    "date": snapshot.session_start.isoformat(),
                    "cutoff": snapshot.as_of.isoformat(),
                    "joint_effects": joint_effect_rows(snapshot),
                    "evaluated": forecast is not None,
                    "tyre_strategy": forecast.tyre_strategy_analysis if forecast else None,
                    "interruptions": forecast.interruption_analysis if forecast else None,
                    "calibration_role": report.get("probability_calibration", {})
                    .get("roles", {})
                    .get(snapshot.event_id + ":" + snapshot.session.value),
                    "calibration_notice": report.get("probability_calibration", {}).get(
                        "limitations"
                    ),
                    "forecast": readable_rows(
                        forecast.standings, names, team_names, actual_positions
                    )
                    if forecast
                    else [],
                    "winner": names[forecast.winner] if forecast else None,
                    "scenarios": scenario_rows,
                    "actual": actual_rows,
                    "actual_winner": names[actual.finishing_order[0]],
                    "weather": actual.actual_weather.model_dump(),
                    "forecast_weather": snapshot.weather.model_dump(mode="json"),
                    "metrics": scored["ml"] if scored else None,
                    "baseline": scored["frozen_baseline"] if scored else None,
                    "feedback": {
                        "available_at": actual.available_at.isoformat(),
                        "verified": actual.verified,
                        "clean_drivers": n_clean,
                        "clean_pairs": n_clean * (n_clean - 1) // 2,
                        "masked": [names[d] for d in sorted(masked)],
                        "retired": [names[d] for d in actual.retired_drivers],
                        "events": events,
                        "event_counts": dict(Counter(e["kind"] for e in events)),
                        **usage,
                    },
                    "sources": sorted({s.url for s in actual.sources}),
                    "reconstructed": snapshot.data_mode == "historical_reconstruction",
                    "synthetic": snapshot.synthetic,
                }
            )
        manifest = bundle.get("manifest") or {}
        for entry in manifest.get("exclusions", []):
            exclusions.append({**entry, "format": kind})
    if not sessions:
        raise ValueError("No sessions to report")
    data = {
        "sessions": sorted(sessions, key=lambda s: s["date"]),
        "exclusions": exclusions,
        "summaries": summaries,
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    content = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    # Escape script terminators and JS line separators. All UI strings are assigned as textContent.
    content = content.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    content = content.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = Path(__file__).with_name("report_template.html").read_text()
    output = Path(output)
    if output.suffix.lower() != ".html":
        raise ValueError("Readable report output must end in .html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__REPORT_DATA__", content), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    csv_path.write_text(report_csv(data), encoding="utf-8-sig")
    return {
        "html": str(output.resolve()),
        "csv": str(csv_path.resolve()),
        "sessions": len(sessions),
        "evaluated_sessions": sum(s["evaluated"] for s in sessions),
    }


def joint_effect_rows(snapshot):
    teams = {t.id: t for t in snapshot.teams}
    result = []
    for driver in snapshot.drivers:
        team = teams[driver.team_id]
        estimates = [
            e.joint_effects.get(snapshot.session.value) if e else None
            for e in (driver.performance, team.car.performance)
        ]
        if not any(estimates):
            continue
        result.append(
            {
                "driver": driver.name,
                "team": team.name,
                "driver_effect": estimates[0].model_dump(mode="json") if estimates[0] else None,
                "car_effect": estimates[1].model_dump(mode="json") if estimates[1] else None,
            }
        )
    return result


def scenario_brief(wet):
    if wet == 0:
        return "Dry running throughout. Rain is not assumed in this scenario."
    if wet < 0.6:
        return "Changing conditions: dry running mixed with rain or a wet track."
    return "Mostly wet running: rain or wet conditions for most of the session."


def readable_rows(standings, names, teams, actual):
    return [
        {
            "position": s.position,
            "driver": names[s.driver_id],
            "driver_id": s.driver_id,
            "team": teams[s.driver_id],
            "confidence": s.position_probabilities[s.position - 1],
            "expected_position": s.expected_position,
            "low": s.p10_position,
            "high": s.p90_position,
            "win_probability": s.win_probability,
            "actual": actual[s.driver_id],
            "error": abs(s.position - actual[s.driver_id]),
        }
        for s in sorted(standings, key=lambda s: s.position)
    ]


def feedback_usage(record, report, records):
    """Distinguish gradient fitting, epoch selection, and never-used feedback."""

    def role(fold):
        sid = record.snapshot.event_id
        cutoff = stamp(fold["prediction_cutoff"]) if "prediction_cutoff" in fold else None
        if cutoff and record.feedback.available_at >= cutoff:
            return None
        validation_ids = fold.get("validation_events", [])
        validation = [
            r
            for r in records
            if r.snapshot.event_id in validation_ids
            and (cutoff is None or r.feedback.available_at < cutoff)
        ]
        if sid in validation_ids:
            return "validation"
        if (
            sid in fold.get("training_events", [])
            and validation
            and record.feedback.available_at < min(r.snapshot.as_of for r in validation)
        ):
            return "training"
        return None

    fit, validation = [], []
    for fold in report.get("folds", []):
        use = role(fold)
        if use:
            target = next(r for r in records if r.snapshot.event_id == fold["event_id"])
            label = f"{target.snapshot.circuit.name} ({target.snapshot.session_start.date()})"
            (fit if use == "training" else validation).append(label)
    final = report.get("final_model")
    return {
        "training_runs": fit,
        "validation_runs": validation,
        "final_model_role": role(final) if final else None,
        "first_training_run": fit[0] if fit else None,
    }


def report_csv(data):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "Year",
            "Circuit",
            "Format",
            "Session",
            "Date UTC",
            "Scenario",
            "Scenario probability",
            "Forecast position",
            "Driver",
            "Team",
            "Confidence at forecast position",
            "80% range low",
            "80% range high",
            "Actual position",
            "Used for clean pace feedback",
            "Later weight-fitting runs",
            "Later validation runs",
            "Final model role",
        ]
    )
    for session in data["sessions"]:
        groups = (
            (
                [{"name": "Combined forecast", "probability": 1, "rows": session["forecast"]}]
                + session["scenarios"]
            )
            if session["evaluated"]
            else [
                {
                    "name": "Training history: no held-out prediction",
                    "probability": "",
                    "rows": [
                        {"driver": r["driver"], "team": r["team"], "actual": r["position"]}
                        for r in session["actual"]
                    ],
                }
            ]
        )
        for group in groups:
            for row in group["rows"]:
                # Quoting alone does not prevent spreadsheet formula injection.
                cells = [
                    session["year"],
                    session["circuit"],
                    session["format"],
                    session["session"],
                    session["date"],
                    group["name"],
                    group["probability"],
                    row.get("position", ""),
                    row["driver"],
                    row["team"],
                    row.get("confidence", ""),
                    row.get("low", ""),
                    row.get("high", ""),
                    row["actual"],
                    row["driver"] not in session["feedback"]["masked"],
                    len(session["feedback"]["training_runs"]),
                    len(session["feedback"]["validation_runs"]),
                    session["feedback"]["final_model_role"] or "Not used",
                ]
                writer.writerow(
                    [
                        ("'" + v)
                        if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@"))
                        else v
                        for v in cells
                    ]
                )
    return stream.getvalue()

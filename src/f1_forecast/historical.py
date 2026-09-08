"""Real F1 archives with explicit, auditable retrospective feature reconstruction."""

import hashlib
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .data import DataUnavailable, Fetcher
from .models import (
    ActualWeather,
    Car,
    Circuit,
    Driver,
    Feedback,
    RaceEvent,
    Session,
    Snapshot,
    Source,
    Team,
    Weather,
    utcnow,
    validate_order,
)

MODE = "historical_reconstruction"
JOLPICA = "https://api.jolpi.ca/ergast/f1"
OPENF1 = "https://api.openf1.org/v1"


def stamp(value):
    return datetime.fromisoformat(value)


def projected(source, available, basis):
    return source.model_copy(update={"available_at": available, "availability_basis": basis})


class Archive(Fetcher):
    """Resumable, immutable raw-response cache with the original retrieval timestamp."""

    def __init__(self, directory, client=None):
        super().__init__(client=client, archive=None)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last_request = 0.0

    def get(self, url, params=None):
        key = hashlib.sha256(json.dumps([url, params or {}], sort_keys=True).encode()).hexdigest()
        path = self.directory / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            return record["payload"], Source.model_validate(record["source"])
        interval = (
            0.3
            if url.startswith(
                ("https://livetiming.formula1.com/", "https://livetiming-mirror.fastf1.dev/")
            )
            else 1.5
        )
        delay = interval - (time.monotonic() - self.last_request)
        if delay > 0:
            time.sleep(delay)
        self.last_request = time.monotonic()
        payload, source = super().get(url, params)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.directory, delete=False, suffix=".tmp"
        ) as handle:
            json.dump({"source": source.model_dump(mode="json"), "payload": payload}, handle)
            temporary = Path(handle.name)
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
        record = json.loads(path.read_text())
        payload, source = record["payload"], Source.model_validate(record["source"])
        return payload, source


def season_table(archive, season, endpoint=""):
    """Jolpica pages count result rows, so a race can span multiple pages."""
    races, sources, offset = {}, [], 0
    while True:
        suffix = f"{endpoint}/" if endpoint else ""
        payload, source = archive.get(
            f"{JOLPICA}/{season}/{suffix}", {"limit": 100, "offset": offset}
        )
        mr = payload["MRData"]
        page = mr["RaceTable"]["Races"]
        sources.append(source)
        for race in page:
            key = int(race["round"])
            if int(race["season"]) != season:
                raise DataUnavailable("Jolpica returned a different season")
            if key not in races:
                races[key] = race.copy()
                races[key]["_sources"] = [source]
                for field in ("QualifyingResults", "Results", "SprintResults"):
                    if field in race:
                        races[key][field] = list(race[field])
            else:
                races[key]["_sources"].append(source)
                for field in ("QualifyingResults", "Results", "SprintResults"):
                    races[key].setdefault(field, []).extend(race.get(field, []))
        total, limit = int(mr["total"]), int(mr["limit"])
        offset += limit
        if offset >= total:
            break
        if not page or limit <= 0:
            raise DataUnavailable("Jolpica pagination stopped before all results were downloaded")
    return races, sources


def classification(rows):
    if not rows or any(not str(r.get("position", "")).isdigit() for r in rows):
        raise DataUnavailable("Classification has missing positions")
    ordered = sorted(rows, key=lambda r: int(r["position"]))
    if [int(r["position"]) for r in ordered] != list(range(1, len(ordered) + 1)):
        raise DataUnavailable("Classification contains duplicated/missing positions")
    ids = [r["Driver"]["driverId"] for r in ordered]
    if len(set(ids)) != len(ids):
        raise DataUnavailable("Classification contains duplicated drivers")
    return ordered


def session_for(event, sessions, race_format="grand_prix"):
    race_start = stamp(event["date"] + "T" + event["time"])
    candidates = [
        s
        for s in sessions
        if s["session_name"] == "Race"
        and abs((stamp(s["date_start"]) - race_start).total_seconds()) <= 30 * 3600
    ]
    if len(candidates) != 1:
        raise DataUnavailable("No unique OpenF1 race matching the event date")
    meeting = candidates[0]["meeting_key"]
    names = (
        {"Qualifying": "Qualifying", "Race": "Race"}
        if race_format == "grand_prix"
        else {"Sprint Qualifying": "Qualifying", "Sprint Shootout": "Qualifying", "Sprint": "Race"}
    )
    matched = [s for s in sessions if s["meeting_key"] == meeting and s["session_name"] in names]
    result = {names[s["session_name"]]: s for s in matched}
    if set(result) != {"Qualifying", "Race"} or len(matched) != 2:
        raise DataUnavailable("Qualifying or race session metadata is missing or ambiguous")
    return result


def archived_weather(archive, event, starts):
    loc = event["Circuit"]["Location"]
    lo = min(starts).date().isoformat()
    hi = max(starts).date().isoformat()
    variables = ("temperature_2m", "precipitation", "wind_speed_10m")
    payload, source = archive.get(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        {
            "latitude": float(loc["lat"]),
            "longitude": float(loc["long"]),
            "start_date": lo,
            "end_date": hi,
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            "models": "gfs_seamless",
            "hourly": ",".join(v + "_previous_day1" for v in variables),
        },
    )
    return payload["hourly"], source


def forecast_from_archive(hourly, source, start):
    dates = [stamp(t + "+00:00") for t in hourly["time"]]
    idx = min(range(len(dates)), key=lambda i: abs((dates[i] - start).total_seconds()))
    if abs((dates[idx] - start).total_seconds()) > 3600:
        raise DataUnavailable("No day-ahead forecast near the session start")
    values = [
        hourly[k + "_previous_day1"][idx]
        for k in ("temperature_2m", "precipitation", "wind_speed_10m")
    ]
    if any(v is None or not math.isfinite(v) for v in values):
        raise DataUnavailable("Day-ahead weather archive has missing values")
    # Precipitation probabilities are absent in this archive; do not pretend amounts are probabilities.
    # This explicitly labelled heuristic is only a scenario prior, not an observed or provider probability.
    rain_prior = float(np.clip(1 - math.exp(-max(0, values[1]) / 0.5), 0.05, 0.95))
    issued = dates[idx] - timedelta(days=1)
    return (
        Weather(
            rain_probability=rain_prior,
            air_temperature_c=values[0],
            wind_speed_ms=values[2],
            issued_at=issued,
            valid_at=dates[idx],
            source=source.url,
        ),
        projected(
            source,
            issued,
            "Archived GFS day-1 lead time; issue time approximated as valid time minus 24h",
        ),
        values[1],
    )


def observed_weather(samples, session):
    start, end = stamp(session["date_start"]), stamp(session["date_end"])
    rows = [
        r
        for r in samples
        if r.get("session_key") == session["session_key"]
        and start <= stamp(r["date"]) <= end
        and all(r.get(k) is not None for k in ("rainfall", "air_temperature", "wind_speed"))
    ]
    if len(rows) < 5:
        raise DataUnavailable("Fewer than five valid observed weather samples")
    return ActualWeather(
        wet_fraction=float(np.mean([r["rainfall"] for r in rows])),
        air_temperature_c=float(np.mean([r["air_temperature"] for r in rows])),
        wind_speed_ms=float(np.mean([r["wind_speed"] for r in rows])),
    ), len(rows)


def result_events(rows, controls, session, stage):
    events, retired = [], []
    mechanical = (
        "engine",
        "gearbox",
        "transmission",
        "hydraulic",
        "electrical",
        "brake",
        "suspension",
        "power unit",
        "water",
        "oil",
        "fuel",
        "overheat",
        "driveshaft",
    )
    if stage == Session.RACE:
        for row in rows:
            status = row.get("status", "Unknown")
            lower = status.lower()
            if lower in {"finished", "lapped"} or re.fullmatch(r"\+\d+ laps?", lower):
                continue
            driver = row["Driver"]["driverId"]
            retired.append(driver)
            kind = "other"
            if any(w in lower for w in mechanical):
                kind = "mechanical"
            elif "collision" in lower or "accident" in lower:
                kind = "collision"
            elif "disqualified" in lower:
                kind = "penalty"
            events.append(RaceEvent(kind=kind, driver_id=driver, description=status))
    numbers = {int(r["number"]): r["Driver"]["driverId"] for r in rows if r.get("number")}
    for row in controls:
        if row.get("session_key") != session["session_key"]:
            continue
        message = row.get("message", "")
        kind = (
            "red_flag"
            if row.get("flag") == "RED"
            else "safety_car"
            if "SAFETY CAR" in message.upper()
            else "penalty"
            if "PENALTY" in message.upper()
            else None
        )
        if kind:
            events.append(
                RaceEvent(
                    kind=kind, driver_id=numbers.get(row.get("driver_number")), description=message
                )
            )
    return events, retired


def ratings(rows, previous, cutoff):
    """Explicit empirical form proxies; no fabricated chassis specifications."""
    teams, drivers = {}, []
    history = [p for p in previous if p["available"] < cutoff]
    used = []

    def form(identifier, team_id, stage):
        values = []
        for record in reversed(history):
            if record["stage"] != stage:
                continue
            clean = record["clean"]
            for i, old in enumerate(clean):
                if (identifier and old["Driver"]["driverId"] == identifier) or (
                    team_id and old["Constructor"]["constructorId"] == team_id
                ):
                    values.append(1 - i / max(1, len(clean) - 1))
                    used.append(record)
                    if len(values) >= (20 if team_id else 10):
                        return float(np.mean(values))
        return float(np.mean(values)) if values else 0.5

    for row in sorted(rows, key=lambda r: r["Driver"]["driverId"]):
        d, t = row["Driver"], row["Constructor"]
        tid = t["constructorId"]
        if tid not in teams:
            teams[tid] = Team(
                id=tid,
                name=t["name"],
                car=Car(
                    name="Unspecified chassis; historical result-based pace proxy",
                    qualifying_pace=form(None, tid, Session.QUALIFYING),
                    race_pace=form(None, tid, Session.RACE),
                ),
            )
        drivers.append(
            Driver(
                id=d["driverId"],
                name=d["givenName"] + " " + d["familyName"],
                team_id=tid,
                number=int(row["number"]) if row.get("number") else None,
                qualifying_skill=form(d["driverId"], None, Session.QUALIFYING),
                race_skill=form(d["driverId"], None, Session.RACE),
            )
        )
    source_records = {r["key"]: r for r in used}.values()
    sources = [
        projected(
            source,
            r["available"],
            "Prior classification; conservative assumed availability six hours after session end",
        )
        for r in source_records
        for source in r["sources"]
    ]
    return drivers, list(teams.values()), sources


def collect_history(
    seasons,
    output,
    cache="data/raw/historical",
    progress=None,
    race_format="grand_prix",
    include_current_season=False,
):
    if race_format not in {"grand_prix", "sprint"}:
        raise ValueError("Unknown race format")
    output = Path(output)
    if output.exists():
        raise ValueError("History output already exists; choose a new file")
    archive = Archive(cache)
    records, exclusions, previous, lineage = [], [], [], []
    collection_cutoff = utcnow()
    try:
        for year in sorted(set(seasons)):
            if (
                year < 2024
                or year > collection_cutoff.year
                or (year == collection_cutoff.year and not include_current_season)
            ):
                raise ValueError(
                    "Use completed seasons from 2024 onward, or explicitly include the current season"
                )
            schedule, schedule_sources = season_table(archive, year)
            qualifying = (
                season_table(archive, year, "qualifying")[0] if race_format == "grand_prix" else {}
            )
            endpoint = "results" if race_format == "grand_prix" else "sprint"
            races, _r_sources = season_table(archive, year, endpoint)
            sessions, session_source = archive.get(f"{OPENF1}/sessions", {"year": year})
            for round_number, event in sorted(schedule.items()):
                if race_format == "sprint" and round_number not in races:
                    continue
                key = f"{year}-{round_number:02d}" + ("-sprint" if race_format == "sprint" else "")
                try:
                    if round_number not in races and round_number not in qualifying:
                        raise DataUnavailable(
                            "No completed classification available at collection time"
                        )
                    metadata = session_for(event, sessions, race_format)
                    rrows = (
                        classification(
                            races[round_number][
                                "Results" if race_format == "grand_prix" else "SprintResults"
                            ]
                        )
                        if round_number in races
                        else []
                    )
                    if race_format == "sprint":
                        sq, sq_source = archive.get(
                            f"{OPENF1}/session_result",
                            {"session_key": metadata["Qualifying"]["session_key"]},
                        )
                        qrows = sprint_qualifying_rows(
                            sq, rrows, metadata["Qualifying"]["session_key"]
                        )
                        qualifying[round_number] = {
                            "QualifyingResults": qrows,
                            "_sources": [sq_source, *races[round_number]["_sources"]],
                        }
                    else:
                        qrows = classification(qualifying[round_number]["QualifyingResults"])
                    if (
                        include_current_season
                        and rrows
                        and {r["Driver"]["driverId"] for r in qrows}
                        != {r["Driver"]["driverId"] for r in rrows}
                    ):
                        raise DataUnavailable(
                            "Incomplete qualifying classification: entered race roster differs"
                        )
                    meeting = metadata["Race"]["meeting_key"]
                    weather_rows, actual_source = archive.get(
                        f"{OPENF1}/weather", {"meeting_key": meeting}
                    )
                    controls, control_source = archive.get(
                        f"{OPENF1}/race_control", {"meeting_key": meeting}
                    )
                    starts = [stamp(s["date_start"]) for s in metadata.values()]
                    hourly, weather_source = archived_weather(archive, event, starts)
                except (ValueError, KeyError, TypeError) as exc:
                    exclusions.append({"event_id": key, "session": "both", "reason": str(exc)})
                    if progress:
                        progress(f"{key} excluded: {exc}")
                    continue
                for stage, name, rows, raw_sources in (
                    (Session.QUALIFYING, "Qualifying", qrows, qualifying[round_number]["_sources"]),
                    (Session.RACE, "Race", rrows, races.get(round_number, {}).get("_sources", [])),
                ):
                    session = metadata[name]
                    start, end = stamp(session["date_start"]), stamp(session["date_end"])
                    cutoff, available = start - timedelta(hours=1), end + timedelta(hours=6)
                    if not rows or available > collection_cutoff:
                        exclusions.append(
                            {
                                "event_id": key,
                                "session": stage.value,
                                "reason": "Session classification unavailable or six-hour publication delay not elapsed",
                            }
                        )
                        continue
                    raw_source = raw_sources[0]
                    events, retired = result_events(rows, controls, session, stage)
                    try:
                        forecast, fsource, precipitation = forecast_from_archive(
                            hourly, weather_source, start
                        )
                        observed, sample_count = observed_weather(weather_rows, session)
                        drivers, teams, prior_sources = ratings(rows, previous, cutoff)
                        c = event["Circuit"]
                        circuit = Circuit(
                            id=c["circuitId"],
                            name=c["circuitName"],
                            latitude=float(c["Location"]["lat"]),
                            longitude=float(c["Location"]["long"]),
                            laps=None,
                        )
                        sources = [
                            projected(
                                raw_source,
                                cutoff,
                                "Identity-only projection: roster reconstructed from final classification; assumes entrants known by cutoff",
                            ),
                            projected(
                                schedule_sources[0],
                                cutoff,
                                "Circuit identity and schedule reconstructed from archive",
                            ),
                            projected(
                                session_source,
                                cutoff,
                                "Session start reconstructed from archive; actual end is feedback-only",
                            ),
                            fsource,
                            *prior_sources,
                        ]
                        sources.extend(
                            projected(
                                s, cutoff, "Identity-only projection of archived classification"
                            )
                            for s in raw_sources[1:]
                        )
                        qualifying_order = grid = None
                        if stage == Session.RACE:
                            ids = [d.id for d in drivers]
                            qualifying_order = [
                                r["Driver"]["driverId"]
                                for r in qrows
                                if r["Driver"]["driverId"] in ids
                            ]
                            validate_order(qualifying_order, ids)
                            q_available = stamp(metadata["Qualifying"]["date_end"]) + timedelta(
                                hours=6
                            )
                            if q_available > cutoff:
                                raise DataUnavailable(
                                    "Qualifying result availability assumption is after race cutoff"
                                )
                            sources.extend(
                                projected(
                                    s,
                                    q_available,
                                    "Actual qualifying order; assumed available six hours after session end",
                                )
                                for s in qualifying[round_number]["_sources"]
                            )
                            grid = [
                                r["Driver"]["driverId"]
                                for r in sorted(
                                    rows,
                                    key=lambda r: (
                                        int(r.get("grid", 0)) if int(r.get("grid", 0)) > 0 else 999,
                                        qualifying_order.index(r["Driver"]["driverId"]),
                                    ),
                                )
                            ]
                            sources.append(
                                projected(
                                    raw_source,
                                    cutoff,
                                    "Grid-only projection from final race archive; assumes published grid known by cutoff",
                                )
                            )
                        snapshot = Snapshot(
                            event_id=key,
                            season=year,
                            round=round_number,
                            session=stage,
                            race_format=race_format,
                            session_start=start,
                            as_of=cutoff,
                            drivers=drivers,
                            teams=teams,
                            circuit=circuit,
                            weather=forecast,
                            qualifying_order=qualifying_order,
                            starting_grid=grid,
                            sources=sources,
                            data_mode=MODE,
                            notes=[
                                "Real archived classifications and track observations; reconstructed historical inputs.",
                                "Roster/grid are identity-only/start-grid projections; final revisions may differ from event-time records.",
                                "Pace ratings are empirical earlier-result proxies; unsupported car/circuit traits retain neutral priors.",
                                "Planned lap count unavailable: null, encoded as missing zero; actual completed laps are never used as input.",
                                f"GFS day-ahead precipitation {precipitation} mm; rain probability is an explicit heuristic conversion, not a provider probability.",
                            ],
                        )
                        feedback_sources = [
                            projected(
                                s,
                                available,
                                "Observed session data/final classification; conservative assumed availability six hours after session end, retrieval time preserved",
                            )
                            for s in (*raw_sources, actual_source, control_source)
                        ]
                        actual = Feedback(
                            event_id=key,
                            session=stage,
                            available_at=available,
                            finishing_order=[r["Driver"]["driverId"] for r in rows],
                            retired_drivers=retired,
                            events=events,
                            actual_weather=observed,
                            sources=feedback_sources,
                            verified=True,
                            data_mode=MODE,
                        )
                        records.append(
                            {
                                "snapshot": snapshot.model_dump(mode="json"),
                                "feedback": actual.model_dump(mode="json"),
                            }
                        )
                        lineage.append(
                            {
                                "event_id": key,
                                "session": stage.value,
                                "weather_samples": sample_count,
                                "forecast_precipitation_mm": precipitation,
                                "session_key": session["session_key"],
                            }
                        )
                    except (ValueError, KeyError, TypeError) as exc:
                        exclusions.append(
                            {"event_id": key, "session": stage.value, "reason": str(exc)}
                        )
                    excluded = set(retired) | {
                        e.driver_id
                        for e in events
                        if e.kind in ("collision", "penalty", "mechanical")
                    }
                    previous.append(
                        {
                            "key": key + ":" + stage.value,
                            "stage": stage,
                            "available": available,
                            "sources": raw_sources,
                            "clean": [r for r in rows if r["Driver"]["driverId"] not in excluded],
                        }
                    )
                if progress:
                    progress(
                        f"{key} {event['raceName']}: {len(records)} sessions collected, {len(exclusions)} exclusions"
                    )
    finally:
        archive.close()
    if not records:
        raise DataUnavailable(
            "No complete real-data sessions collected: " + json.dumps(exclusions[:3])
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(records, indent=2)
    output.write_text(content, encoding="utf-8")
    manifest = {
        "synthetic": False,
        "data_mode": MODE,
        "race_format": race_format,
        "seasons": sorted(set(seasons)),
        "sessions": len(records),
        "weekends": len({r["snapshot"]["event_id"] for r in records}),
        "qualifying_sessions": sum(r["snapshot"]["session"] == "qualifying" for r in records),
        "race_sessions": sum(r["snapshot"]["session"] == "race" for r in records),
        "created_at": utcnow().isoformat(),
        "collection_cutoff": collection_cutoff.isoformat(),
        "include_current_season": include_current_season,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "exclusions": exclusions,
        "lineage": lineage,
        "cache": str(Path(cache).resolve()),
        "limitations": [
            "Retrospective reconstruction, not an immutable event-time vintage archive.",
            "Final classification revisions and roster/grid publication times are not independently verified.",
            "Rain probability is a heuristic from archived day-ahead precipitation; car/circuit traits include unknown neutral priors.",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def sprint_qualifying_rows(results, roster, session_key):
    """Use actual sprint qualifying positions; never substitute the later race grid."""
    by_number = {int(r["number"]): r for r in roster}
    rows = []
    for result in results:
        if result.get("session_key") != session_key:
            raise DataUnavailable("Sprint qualifying response contains a different session")
        number = result.get("driver_number")
        if number not in by_number or not isinstance(result.get("position"), int):
            raise DataUnavailable(
                "Sprint qualifying classification is incomplete or roster differs"
            )
        original = by_number[number]
        rows.append(
            {
                "Driver": original["Driver"],
                "Constructor": original["Constructor"],
                "number": str(number),
                "position": str(result["position"]),
            }
        )
    return classification(rows)

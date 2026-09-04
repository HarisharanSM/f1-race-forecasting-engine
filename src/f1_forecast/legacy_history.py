"""2019–2023 results with real timing observations and a pre-session persistence prior."""

import hashlib
import json
import re
from datetime import UTC, timedelta, timezone
from pathlib import Path

import numpy as np

from .data import DataUnavailable
from .historical import (
    OPENF1,
    Archive,
    classification,
    projected,
    ratings,
    result_events,
    season_table,
    sprint_qualifying_rows,
    stamp,
)
from .models import (
    ActualWeather,
    Circuit,
    Feedback,
    Session,
    Snapshot,
    Weather,
    utcnow,
    validate_order,
)

BASE = "https://livetiming.formula1.com/static/"
MODE = "historical_reconstruction"


class TimingArchive(Archive):
    """Same public mirror fallback used by FastF1, retaining the actual source URL."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mirrored = set()

    def get(self, url, params=None):
        if not url.startswith(BASE):
            return super().get(url, params)
        mirror = url.replace(
            "https://livetiming.formula1.com", "https://livetiming-mirror.fastf1.dev", 1
        )
        mirror_key = hashlib.sha256(
            json.dumps([mirror, params or {}], sort_keys=True).encode()
        ).hexdigest()
        if url in self.mirrored or (self.directory / f"{mirror_key}.json").exists():
            return super().get(mirror, params)
        try:
            return super().get(url, params)
        except DataUnavailable:
            result = super().get(mirror, params)
            self.mirrored.add(url)
            return result


def stream(text):
    rows = []
    for line in text.lstrip("\ufeff").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"(\d+):(\d{2}):(\d{2}(?:\.\d+)?)(\{.*\})", line)
        if not match:
            raise DataUnavailable("Malformed archived timing stream")
        h, m, s, payload = match.groups()
        rows.append(
            (timedelta(hours=int(h), minutes=int(m), seconds=float(s)), json.loads(payload))
        )
    if not rows:
        raise DataUnavailable("Empty archived timing stream")
    return rows


def utc(value):
    parsed = stamp(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def scheduled(session, field="StartDate"):
    offset = session.get("GmtOffset", "00:00:00")
    sign = -1 if offset.startswith("-") else 1
    h, m, s = map(int, offset.lstrip("+-").split(":"))
    zone = timezone(sign * timedelta(hours=h, minutes=m, seconds=s))
    return stamp(session[field]).replace(tzinfo=zone).astimezone(UTC)


def meeting_for(event, index):
    start = stamp(event["date"] + "T" + event["time"])
    candidates = [
        m
        for m in index["Meetings"]
        if any(
            s.get("Name") == "Race"
            and s.get("Path")
            and s.get("StartDate")
            and abs((scheduled(s) - start).total_seconds()) <= 30 * 3600
            for s in m.get("Sessions", [])
        )
    ]
    if len(candidates) != 1:
        raise DataUnavailable("No unique timing meeting matching the race date")
    return candidates[0]


def session_pair(meeting, year, race_format):
    sessions = [s for s in meeting["Sessions"] if s.get("Name") and s.get("Type")]
    if race_format == "grand_prix":
        q = [s for s in sessions if s["Name"] == "Qualifying"]
        r = [s for s in sessions if s["Name"] == "Race"]
    else:
        r = [s for s in sessions if s["Type"] == "Race" and s["Name"] != "Race"]
        q = (
            [s for s in sessions if s["Name"] == "Qualifying"]
            if year <= 2022
            else [
                s
                for s in sessions
                if s["Type"] == "Qualifying"
                and s["Name"] in ("Sprint Shootout", "Sprint Qualifying")
            ]
        )
    if len(q) != 1 or len(r) != 1:
        raise DataUnavailable("Missing or ambiguous qualifying/race timing metadata")
    return q[0], r[0]


def timing_window(archive, session):
    if session["Path"].startswith("2019/"):
        status_text, source = archive.get(BASE + session["Path"] + "SessionStatus.jsonStream")
        heartbeat, _ = archive.get(BASE + session["Path"] + "Heartbeat.jsonStream")
        offset, anchor = next((t, utc(r["Utc"])) for t, r in stream(heartbeat) if r.get("Utc"))
        statuses = [
            {"Utc": (anchor + (t - offset)).isoformat(), "SessionStatus": r.get("Status")}
            for t, r in stream(status_text)
        ]
    else:
        data, source = archive.get(BASE + session["Path"] + "SessionData.json")
        statuses = data.get("StatusSeries", [])
    starts = [utc(s["Utc"]) for s in statuses if s.get("SessionStatus") == "Started"]
    ends = [
        utc(s["Utc"])
        for s in statuses
        if s.get("SessionStatus") in ("Finished", "Finalised", "Ends")
    ]
    if not starts or not ends or max(ends) <= min(starts):
        raise DataUnavailable("No completed timing session window")
    # Session finish/finalisation is feedback only; predictions use the published schedule.
    return min(starts), max(ends), source


def weather_observations(archive, session):
    start, end, status_source = timing_window(archive, session)
    heartbeat, heartbeat_source = archive.get(BASE + session["Path"] + "Heartbeat.jsonStream")
    anchors = [(offset, utc(row["Utc"])) for offset, row in stream(heartbeat) if row.get("Utc")]
    if not anchors:
        raise DataUnavailable("Timing stream lacks a UTC clock anchor")
    anchor_offset, anchor_time = anchors[0]
    text, weather_source = archive.get(BASE + session["Path"] + "WeatherData.jsonStream")
    samples = []
    for offset, row in stream(text):
        date = anchor_time + (offset - anchor_offset)
        if not start <= date <= end:
            continue
        try:
            if row.get("Rainfall") not in ("0", "1", 0, 1):
                continue
            values = [float(row[k]) for k in ("AirTemp", "WindSpeed", "Rainfall")]
            if not all(np.isfinite(values)):
                continue
            samples.append(values)
        except (KeyError, ValueError, TypeError):
            continue
    if len(samples) < 5:
        raise DataUnavailable("Fewer than five valid archived weather samples")
    temp, wind, wet = np.mean(samples, axis=0)
    return {
        "weather": ActualWeather(
            wet_fraction=float(wet), air_temperature_c=float(temp), wind_speed_ms=float(wind)
        ),
        "start": start,
        "end": end,
        "samples": len(samples),
        "sources": [status_source, heartbeat_source, weather_source],
        "session": session,
    }


def persistence_prior(observations, cutoff, start):
    available = observations["end"] + timedelta(hours=1)
    if available >= cutoff:
        raise DataUnavailable("Practice observations were not available before prediction cutoff")
    age = (cutoff - observations["end"]).total_seconds() / 3600
    if age > 96:
        raise DataUnavailable("Practice observations are over four days old")
    actual = observations["weather"]
    # Explicit smoothing prevents certainty; it is not a meteorological forecast product.
    chance = float(np.clip(actual.wet_fraction, 0.05, 0.95))
    weather = Weather(
        rain_probability=chance,
        air_temperature_c=actual.air_temperature_c,
        wind_speed_ms=actual.wind_speed_ms,
        issued_at=available,
        valid_at=start,
        source=observations["sources"][-1].url,
    )
    sources = [
        projected(
            s,
            available,
            "Pre-session practice weather; persistence estimate, assumed available one hour after practice end",
        )
        for s in observations["sources"]
    ]
    return weather, sources, age


def controls(archive, session):
    text, source = archive.get(BASE + session["Path"] + "RaceControlMessages.jsonStream")
    messages = {}
    for _, payload in stream(text):
        values = payload.get("Messages", {})
        values = values.values() if isinstance(values, dict) else values
        for m in values:
            if not isinstance(m, dict) or not m.get("Message"):
                continue
            key = (m.get("Utc", ""), m["Message"], m.get("RacingNumber"))
            messages[key] = {
                "session_key": session["Key"],
                "message": m["Message"],
                "flag": m.get("Flag"),
                "driver_number": int(m["RacingNumber"])
                if str(m.get("RacingNumber", "")).isdigit()
                else None,
            }
    return list(messages.values()), source


def shootout_rows(archive, session, roster):
    """Final sprint-shootout timing classification, with complete positions required."""
    payload, source = archive.get(OPENF1 + "/session_result", {"session_key": session["Key"]})
    return sprint_qualifying_rows(payload, roster, session["Key"]), source


def collect_legacy_history(
    seasons, output, cache="data/raw/historical", race_format="grand_prix", progress=None
):
    years = sorted(set(seasons))
    if not years or min(years) < 2019 or max(years) > 2023:
        raise ValueError("Practice-persistence importer supports completed 2019–2023 seasons")
    output = Path(output)
    if output.exists():
        raise ValueError("History output already exists; choose a new file")
    ar = TimingArchive(cache)
    records, exclusions, lineage, previous = [], [], [], []
    try:
        for year in years:
            if race_format == "sprint" and year < 2021:
                continue
            schedule, _ = season_table(ar, year)
            races, _ = season_table(
                ar, year, "results" if race_format == "grand_prix" else "sprint"
            )
            qualifying, _ = season_table(ar, year, "qualifying")
            index, index_source = ar.get(BASE + str(year) + "/Index.json")
            for rnd, event in sorted(schedule.items()):
                if race_format == "sprint" and rnd not in races:
                    continue
                key = f"{year}-{rnd:02d}" + ("-sprint" if race_format == "sprint" else "")
                try:
                    meeting = meeting_for(event, index)
                    qs, rs = session_pair(meeting, year, race_format)
                    rr = classification(
                        races[rnd]["Results" if race_format == "grand_prix" else "SprintResults"]
                    )
                    q_sources = qualifying[rnd]["_sources"]
                    if race_format == "sprint" and year >= 2023:
                        qr, qs_source = shootout_rows(ar, qs, rr)
                        q_sources = [qs_source, *races[rnd]["_sources"]]
                    else:
                        qr = classification(qualifying[rnd]["QualifyingResults"])
                    q_end = timing_window(ar, qs)[1]
                except (ValueError, KeyError, TypeError) as exc:
                    exclusions.append({"event_id": key, "session": "both", "reason": str(exc)})
                    if progress:
                        progress(f"{key} excluded: {exc}")
                    continue
                for stage, session, rows, sources in (
                    (Session.QUALIFYING, qs, qr, q_sources),
                    (Session.RACE, rs, rr, races[rnd]["_sources"]),
                ):
                    try:
                        observed = weather_observations(ar, session)
                        start = observed["start"]
                        cutoff = start - timedelta(hours=1)
                        available = observed["end"] + timedelta(hours=6)
                        control, csource = controls(ar, session)
                        events, retired = result_events(
                            rows, control, {"session_key": session["Key"]}, stage
                        )
                        # Only practice sessions; never borrow target actual weather or later qualifying weather.
                        candidates = sorted(
                            [
                                s
                                for s in meeting["Sessions"]
                                if s.get("Type") == "Practice"
                                and scheduled(s, "EndDate") + timedelta(hours=1) < cutoff
                            ],
                            key=lambda s: scheduled(s),
                            reverse=True,
                        )
                        prior = None
                        for practice in candidates:
                            try:
                                candidate = weather_observations(ar, practice)
                                weather, wsources, age = persistence_prior(candidate, cutoff, start)
                                prior = candidate
                                break
                            except (ValueError, KeyError, TypeError):
                                continue
                        if prior is None:
                            raise DataUnavailable(
                                "No complete prior practice weather available before cutoff"
                            )
                        drivers, teams, psources = ratings(rows, previous, cutoff)
                        c = event["Circuit"]
                        ids = [d.id for d in drivers]
                        qorder = grid = None
                        input_sources = (
                            [
                                projected(
                                    s,
                                    cutoff,
                                    "Identity/circuit-only reconstruction from final result archive; publication time assumed",
                                )
                                for s in [*sources, *event["_sources"], index_source]
                            ]
                            + wsources
                            + psources
                        )
                        if stage == Session.RACE:
                            qorder = [
                                r["Driver"]["driverId"]
                                for r in qr
                                if r["Driver"]["driverId"] in ids
                            ]
                            validate_order(qorder, ids)
                            q_available = q_end + timedelta(minutes=30)
                            if q_available > cutoff:
                                raise DataUnavailable(
                                    "Qualifying availability assumption is after race cutoff"
                                )
                            grid = [
                                r["Driver"]["driverId"]
                                for r in sorted(
                                    rows,
                                    key=lambda r: (
                                        int(r.get("grid", 0)) or 999,
                                        qorder.index(r["Driver"]["driverId"]),
                                    ),
                                )
                            ]
                            input_sources.extend(
                                projected(
                                    s,
                                    q_available,
                                    "Actual qualifying order for race grid; availability assumed 30 minutes after qualifying end, distinct from the six-hour training-feedback lag",
                                )
                                for s in q_sources
                            )
                        notes = [
                            "Real archived results and F1 timing weather; historical input availability reconstructed.",
                            "Prediction cutoff is one hour before the timing-record start. Race qualifying-order availability is assumed 30 minutes after qualifying end; training feedback remains delayed six hours.",
                            "WEATHER METHOD: practice persistence; not an archived meteorological forecast. Latest available practice averages carried forward; rain prior clipped to 5–95%.",
                            f"Prior practice: {prior['session']['Name']}; observed end {prior['end'].isoformat()}; age {age:.1f} hours at cutoff.",
                            "Final roster/grid reconstructed; car pace is earlier-result form, unsupported traits neutral. Actual completed laps excluded.",
                        ]
                        if year in (2021, 2022) and any(
                            s.get("Type") == "Race" and s.get("Name") != "Race"
                            for s in meeting["Sessions"]
                        ):
                            notes.append(
                                "Shared Friday qualifying set the Sprint grid in this format; the Grand Prix grid came from the Sprint. Same qualifying session may be shown in both separately trained formats."
                            )
                        snapshot = Snapshot(
                            event_id=key,
                            season=year,
                            round=rnd,
                            session=stage,
                            race_format=race_format,
                            session_start=start,
                            as_of=cutoff,
                            drivers=drivers,
                            teams=teams,
                            circuit=Circuit(
                                id=c["circuitId"],
                                name=c["circuitName"],
                                latitude=float(c["Location"]["lat"]),
                                longitude=float(c["Location"]["long"]),
                                laps=None,
                            ),
                            weather=weather,
                            qualifying_order=qorder,
                            starting_grid=grid,
                            sources=input_sources,
                            notes=notes,
                            data_mode=MODE,
                        )
                        actual = Feedback(
                            event_id=key,
                            session=stage,
                            available_at=available,
                            finishing_order=[r["Driver"]["driverId"] for r in rows],
                            retired_drivers=retired,
                            events=events,
                            actual_weather=observed["weather"],
                            sources=[
                                projected(
                                    s,
                                    available,
                                    "Observed timing/classification; availability assumed six hours after session end; real retrieval retained",
                                )
                                for s in [*sources, *observed["sources"], csource]
                            ],
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
                                "weather_method": "practice_persistence",
                                "weather_samples": observed["samples"],
                                "prior_session": prior["session"]["Path"],
                                "prior_samples": prior["samples"],
                                "prior_end": prior["end"].isoformat(),
                                "prediction_cutoff": cutoff.isoformat(),
                                "prior_age_hours": age,
                            }
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
                                "sources": sources,
                                "clean": [
                                    r for r in rows if r["Driver"]["driverId"] not in excluded
                                ],
                            }
                        )
                    except (ValueError, KeyError, TypeError) as exc:
                        exclusions.append(
                            {"event_id": key, "session": stage.value, "reason": str(exc)}
                        )
                if progress:
                    progress(
                        f"{key} {event['raceName']}: {len(records)} sessions; {len(exclusions)} exclusions"
                    )
    finally:
        ar.close()
    if not records:
        raise DataUnavailable("No usable legacy records: " + json.dumps(exclusions[:3]))
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(records, indent=2)
    output.write_text(content)
    manifest = {
        "synthetic": False,
        "data_mode": MODE,
        "race_format": race_format,
        "weather_method": "practice_persistence",
        "seasons": years,
        "sessions": len(records),
        "weekends": len({r["snapshot"]["event_id"] for r in records}),
        "created_at": utcnow().isoformat(),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "exclusions": exclusions,
        "lineage": lineage,
        "limitations": [
            "Real results and observed timing weather; forecast inputs are explicit prior-practice persistence estimates, not saved meteorological forecasts.",
            "Reconstructed roster/grid and publication times. Early Sprint weekends require warm-up; 2021–2022 share Friday qualifying between formats.",
        ],
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

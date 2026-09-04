"""Deterministic public API collectors. LLMs interpret evidence; APIs supply facts."""

import hashlib
import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

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


class DataUnavailable(ValueError):
    pass


class Fetcher:
    def __init__(self, client: httpx.Client | None = None, archive="data/raw"):
        self.client = client or httpx.Client(
            timeout=30,
            headers={"User-Agent": "f1-race-forecasting-engine/0.1"},
        )
        self.owns_client = client is None
        self.archive = Path(archive) if archive else None

    def close(self):
        if self.owns_client:
            self.client.close()

    def get(self, url: str, params: dict | None = None) -> tuple[object, Source]:
        for attempt in range(3):
            try:
                response = self.client.get(url, params=params)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    delay = response.headers.get("Retry-After", str(2**attempt))
                    time.sleep(min(10, float(delay)) if delay.isdigit() else 2**attempt)
                    continue
                response.raise_for_status()
                payload = response.text if url.endswith(".jsonStream") else response.json()
                break
            except httpx.TransportError as exc:
                if attempt == 2:
                    raise DataUnavailable(f"Could not reach {url}: {type(exc).__name__}") from exc
                time.sleep(2**attempt)
            except (httpx.HTTPStatusError, json.JSONDecodeError) as exc:
                raise DataUnavailable(f"Provider returned an error for {url}: {exc}") from exc
        source = Source(url=str(response.url), retrieved_at=utcnow())
        if self.archive:
            self.archive.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(
                (source.url + source.retrieved_at.isoformat()).encode()
            ).hexdigest()
            (self.archive / f"{digest}.json").write_text(
                json.dumps(
                    {
                        "source": source.model_dump(mode="json"),
                        "payload": payload,
                    }
                ),
                encoding="utf-8",
            )
        return payload, source


class Jolpica:
    base = "https://api.jolpi.ca/ergast/f1"

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def race(self, season: int, round_number: int, endpoint: str = "") -> tuple[dict, Source]:
        suffix = f"/{endpoint}" if endpoint else ""
        payload, source = self.fetcher.get(
            f"{self.base}/{season}/{round_number}{suffix}/",
            {"limit": 100},
        )
        try:
            races = payload["MRData"]["RaceTable"]["Races"]
            if len(races) != 1:
                raise DataUnavailable("Event/results are not published yet or event is unknown")
            race = races[0]
            if int(race["season"]) != season or int(race["round"]) != round_number:
                raise DataUnavailable("Provider returned a different event")
            return race, source
        except (KeyError, TypeError) as exc:
            raise DataUnavailable("Unexpected Jolpica response schema") from exc

    def results(self, season: int, round_number: int, session: Session):
        endpoint, key = (
            ("qualifying", "QualifyingResults")
            if session == Session.QUALIFYING
            else ("results", "Results")
        )
        race, source = self.race(season, round_number, endpoint)
        rows = race.get(key, [])
        if not rows or any(not r.get("position") for r in rows):
            raise DataUnavailable("A complete classification is not available")
        rows = sorted(rows, key=lambda r: int(r["position"]))
        if [int(r["position"]) for r in rows] != list(range(1, len(rows) + 1)):
            raise DataUnavailable("Provider classification contains duplicate or missing positions")
        return rows, source


class OpenMeteo:
    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def forecast(self, circuit: Circuit, session_start: datetime) -> tuple[Weather, Source]:
        now = utcnow()
        if session_start <= now or session_start > now + timedelta(days=15):
            raise DataUnavailable(
                "Live weather collection requires a session within the next 15 days"
            )
        payload, source = self.fetcher.get(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": circuit.latitude,
                "longitude": circuit.longitude,
                "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
                "wind_speed_unit": "ms",
                "timezone": "UTC",
                "forecast_days": 16,
            },
        )
        try:
            hourly = payload["hourly"]
            dates = [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in hourly["time"]]
            idx = min(
                range(len(dates)), key=lambda i: abs((dates[i] - session_start).total_seconds())
            )
            if abs((dates[idx] - session_start).total_seconds()) > 3600:
                raise DataUnavailable("No forecast near the requested session time")
            weather = Weather(
                rain_probability=hourly["precipitation_probability"][idx] / 100,
                air_temperature_c=hourly["temperature_2m"][idx],
                wind_speed_ms=hourly["wind_speed_10m"][idx],
                issued_at=source.retrieved_at,
                valid_at=dates[idx],
                source=source.url,
            )
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise DataUnavailable("Missing or invalid weather forecast values") from exc
        return weather, source


def fetch_event(
    fetcher: Fetcher, season: int, round_number: int, session: Session, *, laps: int
) -> Snapshot:
    """Bootstrap neutral ratings, explicit lineup assumptions, and a real weather forecast."""
    api = Jolpica(fetcher)
    event, schedule_source = api.race(season, round_number)
    schedule = event.get("Qualifying") if session == Session.QUALIFYING else event
    if not schedule or not schedule.get("date") or not schedule.get("time"):
        raise DataUnavailable("Session date/time has not been published")
    start = datetime.fromisoformat(schedule["date"] + "T" + schedule["time"])
    circuit_data = event["Circuit"]
    circuit = Circuit(
        id=circuit_data["circuitId"],
        name=circuit_data["circuitName"],
        laps=laps,
        latitude=float(circuit_data["Location"]["lat"]),
        longitude=float(circuit_data["Location"]["long"]),
    )
    notes = ["Car/driver/circuit ratings are neutral assumptions; enrich with current evidence."]
    if session == Session.RACE:
        rows, roster_source = api.results(season, round_number, Session.QUALIFYING)
        qualifying_order = [r["Driver"]["driverId"] for r in rows]
    else:
        if round_number <= 1:
            raise DataUnavailable(
                "For the season opener, supply a confirmed roster in a snapshot file"
            )
        rows, roster_source = api.results(season, round_number - 1, Session.RACE)
        qualifying_order = None
        notes.append(
            "PROVISIONAL ROSTER from the previous race; check substitutions before predicting."
        )
    teams = {}
    drivers = []
    for row in rows:
        d, t = row["Driver"], row["Constructor"]
        teams[t["constructorId"]] = Team(id=t["constructorId"], name=t["name"], car=Car())
        drivers.append(
            Driver(
                id=d["driverId"],
                name=f"{d['givenName']} {d['familyName']}",
                team_id=t["constructorId"],
                number=int(row["number"]) if row.get("number") else None,
            )
        )
    weather, weather_source = OpenMeteo(fetcher).forecast(circuit, start)
    return Snapshot(
        event_id=f"{season}-{round_number:02d}",
        season=season,
        round=round_number,
        session=session,
        session_start=start,
        as_of=utcnow(),
        drivers=drivers,
        teams=list(teams.values()),
        circuit=circuit,
        weather=weather,
        qualifying_order=qualifying_order,
        sources=[schedule_source, roster_source, weather_source],
        notes=notes,
    )


def fetch_actual(fetcher: Fetcher, snapshot: Snapshot, session_key: int) -> Feedback:
    """Published classification plus a matched OpenF1 session, never model-generated truth."""
    if snapshot.synthetic:
        raise ValueError("Cannot fetch real feedback for a synthetic event")
    rows, result_source = Jolpica(fetcher).results(
        snapshot.season, snapshot.round, snapshot.session
    )
    order = [r["Driver"]["driverId"] for r in rows]
    validate_order(order, [d.id for d in snapshot.drivers])
    base = "https://api.openf1.org/v1"
    sessions, session_source = fetcher.get(f"{base}/sessions", {"session_key": session_key})
    expected_name = "Qualifying" if snapshot.session == Session.QUALIFYING else "Race"
    if len(sessions) != 1 or sessions[0].get("session_name") != expected_name:
        raise DataUnavailable("OpenF1 session key does not match the requested session type")
    session = sessions[0]
    start, end = (datetime.fromisoformat(session[k]) for k in ("date_start", "date_end"))
    if (
        abs((start - snapshot.session_start).total_seconds()) > 10800
        or end > utcnow()
        or int(session["year"]) != snapshot.season
    ):
        raise DataUnavailable("OpenF1 session is unfinished or belongs to a different event")
    weather, weather_source = fetcher.get(f"{base}/weather", {"session_key": session_key})
    weather = [r for r in weather if start <= datetime.fromisoformat(r["date"]) <= end]
    required = ("rainfall", "air_temperature", "wind_speed")
    weather = [r for r in weather if all(r.get(k) is not None for k in required)]
    if not weather:
        raise DataUnavailable(
            "Actual session weather is unavailable; do not substitute the forecast"
        )
    mean = lambda key: sum(float(r[key]) for r in weather) / len(weather)
    actual_weather = ActualWeather(
        wet_fraction=mean("rainfall"),
        air_temperature_c=mean("air_temperature"),
        wind_speed_ms=mean("wind_speed"),
    )
    events = []
    retired = []
    if snapshot.session == Session.RACE:
        mechanical_words = (
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
        )
        for row in rows:
            status = row.get("status", "Unknown")
            lower = status.lower()
            driver_id = row["Driver"]["driverId"]
            if lower in {"finished", "lapped"} or re.fullmatch(r"\+\d+ laps?", lower):
                continue
            retired.append(driver_id)
            kind = "other"
            if any(w in lower for w in mechanical_words):
                kind = "mechanical"
            elif "collision" in lower or "accident" in lower:
                kind = "collision"
            elif "disqualified" in lower:
                kind = "penalty"
            events.append(RaceEvent(kind=kind, driver_id=driver_id, description=status))
    controls, control_source = fetcher.get(f"{base}/race_control", {"session_key": session_key})
    numbers = {d.number: d.id for d in snapshot.drivers if d.number is not None}
    for row in controls:
        message = row.get("message", "")
        kind = "other"
        if "SAFETY CAR" in message.upper():
            kind = "safety_car"
        elif row.get("flag") == "RED":
            kind = "red_flag"
        elif "PENALTY" in message.upper():
            kind = "penalty"
        if kind != "other":
            events.append(
                RaceEvent(
                    kind=kind, driver_id=numbers.get(row.get("driver_number")), description=message
                )
            )
    return Feedback(
        event_id=snapshot.event_id,
        session=snapshot.session,
        available_at=utcnow(),
        finishing_order=order,
        retired_drivers=retired,
        events=events,
        actual_weather=actual_weather,
        sources=[result_source, session_source, weather_source, control_source],
        verified=True,
    )

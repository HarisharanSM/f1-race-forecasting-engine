"""Chronological session records and features shared by neural training and inference."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .engine import FEATURES, features
from .models import Feedback, Session, Snapshot, validate_order

FEATURE_NAMES = (
    *FEATURES,
    "race_session",
    "rain_forecast",
    "wet_exposure",
    "temperature",
    "wind",
    "overtaking",
    "tyre_stress",
    "disruption",
    "laps",
    "consistency",
    "reliability",
    "driver_starts",
    "driver_clean_form",
    "driver_latest_form",
    "driver_retirement_rate",
    "team_starts",
    "team_clean_form",
    "team_retirement_rate",
    "history_age",
)


@dataclass
class Record:
    snapshot: Snapshot
    feedback: Feedback


def records_from_json(rows: list[dict]) -> list[Record]:
    records, seen = [], set()
    for row in rows:
        snapshot = Snapshot.model_validate(row["snapshot"])
        actual = Feedback.model_validate(row["feedback"])
        key = (snapshot.event_id, snapshot.session)
        if key in seen:
            raise ValueError("Duplicate event/session in ML history")
        seen.add(key)
        if (actual.event_id, actual.session) != key:
            raise ValueError("History feedback event/session mismatch")
        validate_order(actual.finishing_order, [d.id for d in snapshot.drivers])
        if not actual.verified or actual.available_at <= snapshot.session_start:
            raise ValueError("ML history requires verified post-session feedback")
        if actual.synthetic != snapshot.synthetic:
            raise ValueError("Snapshot and feedback synthetic flags differ")
        if actual.data_mode != snapshot.data_mode:
            raise ValueError("Snapshot and feedback data modes differ")
        records.append(Record(snapshot, actual))
    if not records:
        raise ValueError("Supply historical snapshot/feedback records")
    if len({r.snapshot.synthetic for r in records}) != 1:
        raise ValueError("Keep synthetic and real ML history separate")
    if len({r.snapshot.race_format for r in records}) != 1:
        raise ValueError("Train Grand Prix and sprint histories separately")
    return sorted(records, key=lambda r: r.snapshot.as_of)


def excluded_drivers(actual: Feedback) -> set[str]:
    excluded = set(actual.retired_drivers)
    excluded.update(
        e.driver_id
        for e in actual.events
        if e.driver_id and e.kind in {"mechanical", "collision", "penalty"}
    )
    return excluded


def history_summary(record: Record) -> dict:
    snapshot, actual = record.snapshot, record.feedback
    clean = [d for d in actual.finishing_order if d not in excluded_drivers(actual)]
    return {
        "event_id": snapshot.event_id,
        "session": snapshot.session.value,
        "session_start": snapshot.session_start.isoformat(),
        "available_at": actual.available_at.isoformat(),
        "synthetic": snapshot.synthetic,
        "drivers": [
            {
                "id": d.id,
                "team_id": d.team_id,
                "form": (1 - clean.index(d.id) / max(1, len(clean) - 1)) if d.id in clean else None,
                "retired": d.id in actual.retired_drivers,
            }
            for d in snapshot.drivers
        ],
    }


def feature_matrix(
    snapshot: Snapshot,
    history: list[dict],
    wet: float,
    temperature: float | None = None,
    wind: float | None = None,
) -> np.ndarray:
    """No driver-position encoding: reordering input drivers only reorders feature rows."""
    temperature = snapshot.weather.air_temperature_c if temperature is None else temperature
    wind = snapshot.weather.wind_speed_ms if wind is None else wind
    eligible = sorted(
        [
            h
            for h in history
            if h["session"] == snapshot.session.value
            and h["event_id"] != snapshot.event_id
            and h["synthetic"] == snapshot.synthetic
            and datetime.fromisoformat(h["available_at"]) <= snapshot.as_of
            and datetime.fromisoformat(h["session_start"]) < snapshot.as_of
        ],
        key=lambda h: datetime.fromisoformat(h["session_start"]),
        reverse=True,
    )
    base = features(snapshot, wet, temperature)
    teams = {t.id: t for t in snapshot.teams}
    rows = []
    for i, driver in enumerate(snapshot.drivers):
        own, team = [], []
        latest = None
        for event in eligible:
            for entry in event["drivers"]:
                if entry["id"] == driver.id and len(own) < 10:
                    own.append(entry)
                    latest = latest or datetime.fromisoformat(event["session_start"])
                if entry["team_id"] == driver.team_id and len(team) < 20:
                    team.append(entry)
        own_form = [x["form"] for x in own if x["form"] is not None]
        team_form = [x["form"] for x in team if x["form"] is not None]
        context = [
            float(snapshot.session == Session.RACE),
            snapshot.weather.rain_probability,
            wet,
            (temperature - 25) / 20,
            wind / 30,
            snapshot.circuit.overtaking,
            snapshot.circuit.tyre_stress,
            snapshot.circuit.disruption_probability,
            (snapshot.circuit.laps or 0) / 100,
            driver.consistency,
            teams[driver.team_id].car.reliability,
            len(own) / 10,
            np.mean(own_form) if own_form else 0.5,
            own_form[0] if own_form else 0.5,
            np.mean([x["retired"] for x in own]) if own else 0.05,
            len(team) / 20,
            np.mean(team_form) if team_form else 0.5,
            np.mean([x["retired"] for x in team]) if team else 0.05,
            min(1, (snapshot.as_of - latest).total_seconds() / 86400 / 365) if latest else 1,
        ]
        rows.append([*base[i], *context])
    return np.asarray(rows, dtype=np.float32)


def event_groups(records: list[Record]) -> list[list[Record]]:
    groups: dict[str, list[Record]] = {}
    for record in records:
        groups.setdefault(record.snapshot.event_id, []).append(record)
    return sorted(groups.values(), key=lambda g: min(r.snapshot.as_of for r in g))


def chronological_split(
    records: list[Record], min_train_events: int, validation_events: int
) -> tuple[list[Record], list[Record]]:
    if min_train_events < 1 or validation_events < 1:
        raise ValueError("Training and validation must each contain at least one event")
    groups = event_groups(records)
    if len(groups) < min_train_events + validation_events:
        raise ValueError("Not enough earlier events for separate training and validation windows")
    validation = [r for group in groups[-validation_events:] for r in group]
    cutoff = min(r.snapshot.as_of for r in validation)
    train = [
        r
        for group in groups[:-validation_events]
        for r in group
        if r.feedback.available_at < cutoff
    ]
    if len(event_groups(train)) < min_train_events:
        raise ValueError("Delayed results leave too few training events before validation")
    if {r.snapshot.session for r in validation} - {r.snapshot.session for r in train}:
        raise ValueError("Training must cover every session type present in validation")
    return train, validation

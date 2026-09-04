"""SQLite audit trail and transactional, exactly-once feedback updates."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .engine import LearnerState, evaluate, learn
from .models import Feedback, Forecast, Snapshot, utcnow, validate_order


class Store:
    def __init__(self, path: str | Path = "data/forecast.sqlite3"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS models (
                    version INTEGER PRIMARY KEY, state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forecasts (
                    id TEXT PRIMARY KEY, event_id TEXT NOT NULL, session TEXT NOT NULL,
                    snapshot TEXT NOT NULL, forecast TEXT NOT NULL, messages TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    event_id TEXT NOT NULL, session TEXT NOT NULL, forecast_id TEXT NOT NULL,
                    actual TEXT NOT NULL, metrics TEXT NOT NULL, model_version INTEGER NOT NULL,
                    PRIMARY KEY (event_id, session)
                );
            """)
            db.execute(
                "INSERT OR IGNORE INTO models VALUES (0, ?)", (json.dumps(asdict(LearnerState())),)
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _state(db) -> LearnerState:
        row = db.execute("SELECT state FROM models ORDER BY version DESC LIMIT 1").fetchone()
        return LearnerState(**json.loads(row[0]))

    def state(self) -> LearnerState:
        with self.connect() as db:
            return self._state(db)

    def state_for(self, snapshot: Snapshot) -> LearnerState:
        with self.connect() as db:
            state = self._state(db)
            row = db.execute("SELECT actual FROM feedback LIMIT 1").fetchone()
            if row and Feedback.model_validate_json(row[0]).synthetic != snapshot.synthetic:
                raise ValueError("Use separate databases for synthetic and real predictions")
            if (
                state.trained_through
                and datetime.fromisoformat(state.trained_through) > snapshot.as_of
            ):
                raise ValueError(
                    "Model includes future feedback; use a historical model or backtest"
                )
            return state

    def save(self, snapshot: Snapshot, forecast: Forecast, messages: list[dict]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    forecast.id,
                    snapshot.event_id,
                    snapshot.session,
                    snapshot.model_dump_json(),
                    forecast.model_dump_json(),
                    json.dumps(messages),
                ),
            )

    def get(self, forecast_id: str) -> tuple[Snapshot, Forecast]:
        with self.connect() as db:
            row = db.execute(
                "SELECT snapshot, forecast FROM forecasts WHERE id=?", (forecast_id,)
            ).fetchone()
        if row is None:
            raise ValueError("Forecast not found")
        return Snapshot.model_validate_json(row[0]), Forecast.model_validate_json(row[1])

    def feedback(self, forecast_id: str, actual: Feedback, *, replay: bool = False) -> dict:
        if not actual.verified:
            raise ValueError("Verify actual results and weather before applying feedback")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT snapshot, forecast FROM forecasts WHERE id=?", (forecast_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Forecast not found")
            snapshot = Snapshot.model_validate_json(row[0])
            forecast = Forecast.model_validate_json(row[1])
            if (actual.event_id, actual.session) != (snapshot.event_id, snapshot.session):
                raise ValueError("Feedback event/session does not match the prediction")
            validate_order(actual.finishing_order, [d.id for d in snapshot.drivers])
            if snapshot.synthetic != actual.synthetic:
                raise ValueError("Synthetic and real records cannot be mixed")
            # Keep demo/replay state isolated from real observations.
            prior = db.execute("SELECT actual FROM feedback LIMIT 1").fetchone()
            if prior and Feedback.model_validate_json(prior[0]).synthetic != actual.synthetic:
                raise ValueError("Use separate databases for synthetic and real feedback")
            if actual.available_at <= snapshot.session_start:
                raise ValueError("Feedback must be available after the session start")
            if not replay and actual.available_at > utcnow():
                raise ValueError("Feedback availability cannot be in the future")
            existing = db.execute(
                "SELECT actual, metrics, model_version FROM feedback "
                "WHERE event_id=? AND session=?",
                (actual.event_id, actual.session),
            ).fetchone()
            if existing:
                if Feedback.model_validate_json(existing[0]) != actual:
                    raise ValueError(
                        "Conflicting feedback already exists; rebuild from corrected history"
                    )
                return {
                    "already_applied": True,
                    "metrics": json.loads(existing[1]),
                    "model_version": existing[2],
                }
            state = self._state(db)
            if state.trained_through and actual.available_at < datetime.fromisoformat(
                state.trained_through
            ):
                raise ValueError(
                    "Apply feedback in availability order, or use a fresh backtest database"
                )
            metrics = evaluate(forecast, actual)
            updated = learn(snapshot, actual, state)
            db.execute(
                "INSERT INTO models VALUES (?, ?)", (updated.version, json.dumps(asdict(updated)))
            )
            db.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
                (
                    actual.event_id,
                    actual.session,
                    forecast_id,
                    actual.model_dump_json(),
                    json.dumps(metrics),
                    updated.version,
                ),
            )
            return {"already_applied": False, "metrics": metrics, "model_version": updated.version}

    def memories(self, snapshot: Snapshot, limit: int = 3) -> list[dict]:
        eligible = []
        for row in self.training_records():
            actual = Feedback.model_validate(row["feedback"])
            previous = Snapshot.model_validate(row["snapshot"])
            if (
                actual.available_at <= snapshot.as_of
                and actual.session == snapshot.session
                and actual.synthetic == snapshot.synthetic
                and actual.event_id != snapshot.event_id
            ):
                eligible.append(
                    (
                        actual.available_at,
                        {
                            "event_id": actual.event_id,
                            "circuit_id": previous.circuit.id,
                            "finishing_order": actual.finishing_order,
                            "actual_weather": actual.actual_weather.model_dump(),
                            "events": [e.model_dump() for e in actual.events],
                            "metrics": row["metrics"],
                        },
                    )
                )
        return [r[1] for r in sorted(eligible, key=lambda x: x[0], reverse=True)[:limit]]

    def training_records(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("""
                SELECT f.snapshot, f.messages, b.actual, b.metrics
                FROM feedback b JOIN forecasts f ON f.id=b.forecast_id
            """).fetchall()
        return [
            {
                "snapshot": json.loads(s),
                "messages": json.loads(m),
                "feedback": json.loads(a),
                "metrics": json.loads(e),
            }
            for s, m, a, e in rows
        ]

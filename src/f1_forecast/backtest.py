"""Predict first, score second, and learn only when results become available."""

import tempfile
from pathlib import Path

from .engine import evaluate, predict
from .models import Feedback, Snapshot
from .service import ForecastService


def backtest(records: list[dict], simulations: int = 1000, seed: int = 42) -> dict:
    pairs = [
        (Snapshot.model_validate(r["snapshot"]), Feedback.model_validate(r["feedback"]))
        for r in records
    ]
    pairs.sort(key=lambda pair: pair[0].as_of)
    if not pairs:
        raise ValueError("Supply at least one historical snapshot/feedback pair")
    keys = [(s.event_id, s.session) for s, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("Backtest contains duplicate event/session records")
    scored, pending = [], []
    with tempfile.TemporaryDirectory(prefix="f1-backtest-") as directory:
        service = ForecastService(Path(directory) / "replay.sqlite3")
        for snapshot, actual in pairs:
            ready = sorted(
                (p for p in pending if p[0].available_at <= snapshot.as_of),
                key=lambda p: p[0].available_at,
            )
            for feedback, forecast_id in ready:
                service.store.feedback(forecast_id, feedback, replay=True)
                pending.remove((feedback, forecast_id))
            forecast = service.predict(snapshot, simulations=simulations, seed=seed)
            baseline = predict(snapshot, simulations=simulations, seed=seed)
            # Validate actual record before scoring; rollback is automatic on any failure.
            if (actual.event_id, actual.session) != (snapshot.event_id, snapshot.session):
                raise ValueError("Backtest feedback event/session mismatch")
            if not actual.verified or actual.available_at <= snapshot.session_start:
                raise ValueError("Backtest requires verified, post-session feedback")
            scored.append(
                {
                    "event_id": snapshot.event_id,
                    "session": snapshot.session.value,
                    "model_version": forecast.model_version,
                    "adaptive": evaluate(forecast, actual),
                    "frozen_baseline": evaluate(baseline, actual),
                }
            )
            pending.append((actual, forecast.id))
        for actual, forecast_id in sorted(pending, key=lambda p: p[0].available_at):
            service.store.feedback(forecast_id, actual, replay=True)
    summary = {}
    for kind in ("adaptive", "frozen_baseline"):
        summary[kind] = {
            metric: sum(float(r[kind][metric]) for r in scored) / len(scored)
            for metric in scored[0][kind]
        }
    return {
        "sessions": len(scored),
        "summary": summary,
        "results": scored,
        "limitations": "Numerical model only. Frozen snapshots must be genuinely pre-event; "
        "timestamps cannot prove factual provenance. LLM historical recall can leak results.",
    }

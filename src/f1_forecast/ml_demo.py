"""Clearly fictional training data and a runnable neural walk-forward demonstration."""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from .demo import weekend
from .engine import predict
from .models import Feedback, Snapshot


def synthetic_history(weekends: int = 8, seed: int = 19) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for event in range(weekends):
        q, qa, race, ra = weekend()
        qualifying_order = None
        for snapshot, actual in ((q, qa), (race, ra)):
            shift = timedelta(days=event * 7)
            s = snapshot.model_dump(mode="json")
            s.update(
                event_id=f"ml-fictional-{event + 1:02d}",
                round=event + 1,
                as_of=(snapshot.as_of + shift).isoformat(),
                session_start=(snapshot.session_start + shift).isoformat(),
            )
            s["weather"].update(
                issued_at=s["as_of"],
                valid_at=s["session_start"],
                rain_probability=[0.15, 0.5, 0.8][event % 3],
            )
            s["sources"][0]["retrieved_at"] = s["as_of"]
            s["circuit"]["overtaking"] = 0.25 + (event % 4) * 0.15
            s["notes"] = ["Fictional data for exercising ML training, not real F1 evidence."]
            if qualifying_order is not None:
                s["qualifying_order"] = qualifying_order
            snap = Snapshot.model_validate(s)
            ids = [d.id for d in snap.drivers]
            wet = float(rng.random() < snap.weather.rain_probability) * 0.75
            pace = []
            teams = {t.id: t for t in snap.teams}
            for d in snap.drivers:
                car = teams[d.team_id].car
                value = (car.qualifying_pace if snap.session == "qualifying" else car.race_pace) * 2
                value += d.qualifying_skill * 0.6 + wet * d.wet_skill * 0.4 + rng.normal(0, 0.08)
                if qualifying_order:
                    value += (1 - qualifying_order.index(d.id) / (len(ids) - 1)) * 0.25
                pace.append(value)
            order = [ids[i] for i in np.argsort(-np.asarray(pace))]
            if snap.session == "qualifying":
                qualifying_order = order
            retired = (
                [ids[(event * 3 + 2) % len(ids)]] if snap.session == "race" and event % 2 else []
            )
            if retired:
                order.remove(retired[0])
                order += retired
            a = actual.model_dump(mode="json")
            a.update(
                event_id=snap.event_id,
                available_at=(actual.available_at + shift).isoformat(),
                finishing_order=order,
                retired_drivers=retired,
                events=[
                    {"kind": "mechanical", "driver_id": d, "description": "Fictional failure"}
                    for d in retired
                ],
            )
            a["actual_weather"]["wet_fraction"] = wet
            a["sources"][0]["retrieved_at"] = a["available_at"]
            feedback = Feedback.model_validate(a)
            rows.append(
                {
                    "snapshot": snap.model_dump(mode="json"),
                    "feedback": feedback.model_dump(mode="json"),
                }
            )
    return rows


def run_ml_demo(output: str | Path, config=None, simulations=1000) -> dict:
    from .ml_backtest import backtest_transformer
    from .neural import NeuralForecaster

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("ML demo output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    history = synthetic_history()
    (output / "history.json").write_text(json.dumps(history, indent=2))
    report = backtest_transformer(
        history, config=config, simulations=simulations, save_model=output / "model"
    )
    (output / "backtest.json").write_text(json.dumps(report, indent=2))
    model = NeuralForecaster.load(output / "model")
    for row in synthetic_history(9)[-2:]:
        snapshot = Snapshot.model_validate(row["snapshot"])
        forecast = predict(snapshot, ml_model=model, simulations=simulations)
        (output / f"next-{snapshot.session}-input.json").write_text(
            snapshot.model_dump_json(indent=2)
        )
        (output / f"next-{snapshot.session}-forecast.json").write_text(
            forecast.model_dump_json(indent=2)
        )
    return {
        "synthetic": True,
        "output": str(output.resolve()),
        "sessions_evaluated": report["sessions_evaluated"],
        "summary": report["summary"],
        "model": report["final_model"],
        "note": "Fictional training demonstration; these scores are not real F1 accuracy.",
    }

"""Credential-free, explicitly fictional weekend for exercising the whole loop."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
)
from .service import ForecastService


def weekend() -> tuple[Snapshot, Feedback, Snapshot, Feedback]:
    start = datetime(2025, 6, 14, 14, tzinfo=UTC)
    cutoff = start - timedelta(hours=3)
    teams, drivers = [], []
    names = [
        "Aurora",
        "Velocity",
        "Apex",
        "Solstice",
        "Orion",
        "Vertex",
        "Cobalt",
        "Zenith",
        "Meridian",
        "Nova",
        "Eclipse",
    ]
    for i, name in enumerate(names):
        pace = 0.92 - i * 0.045
        teams.append(
            Team(
                id=name.lower(),
                name=name,
                strategy=0.85 - i * 0.025,
                pit_crew=0.8 - i * 0.02,
                car=Car(
                    name=f"{name} X25",
                    qualifying_pace=pace,
                    race_pace=pace - 0.02,
                    straight_speed=0.8 - i * 0.02,
                    high_speed_cornering=pace,
                    low_speed_cornering=0.68 + i * 0.015,
                    tyre_management=pace,
                    wet_performance=0.55 + (i % 4) * 0.1,
                    cooling=0.75,
                    reliability=0.96 - i * 0.002,
                    capabilities=["Fictional high-speed balance"],
                    weaknesses=["Fictional low-speed traction limitation"],
                ),
            )
        )
        for j in range(2):
            number = i * 2 + j + 1
            drivers.append(
                Driver(
                    id=f"driver_{number:02d}",
                    name=f"Driver {number:02d}",
                    team_id=name.lower(),
                    number=number,
                    qualifying_skill=0.88 - i * 0.018 - j * 0.07,
                    race_skill=0.85 - i * 0.015 - j * 0.03,
                    wet_skill=0.5 + ((i + j) % 5) * 0.1,
                    consistency=0.85 - j * 0.15,
                )
            )
    snapshot = Snapshot(
        event_id="fictional-2025-01",
        season=2025,
        round=1,
        session=Session.QUALIFYING,
        session_start=start,
        as_of=cutoff,
        drivers=drivers,
        teams=teams,
        circuit=Circuit(
            id="harbour_test",
            name="Fictional Harbour Circuit",
            latitude=1.29,
            longitude=103.86,
            laps=58,
            high_speed_weight=0.5,
            straight_weight=0.3,
            low_speed_weight=0.2,
            overtaking=0.35,
            tyre_stress=0.7,
            disruption_probability=0.35,
        ),
        weather=Weather(
            rain_probability=0.35,
            air_temperature_c=29,
            wind_speed_ms=4,
            issued_at=cutoff,
            valid_at=start,
            source="synthetic://weather",
        ),
        sources=[Source(url="synthetic://weekend", retrieved_at=cutoff)],
        notes=["All teams, drivers, ratings and outcomes are invented test data."],
        synthetic=True,
    )
    qualifying = [d.id for d in drivers]
    qualifying[1], qualifying[2] = qualifying[2], qualifying[1]
    qualifying[4], qualifying[6] = qualifying[6], qualifying[4]
    q_actual = Feedback(
        event_id=snapshot.event_id,
        session=Session.QUALIFYING,
        available_at=start + timedelta(hours=3),
        finishing_order=qualifying,
        actual_weather=ActualWeather(wet_fraction=0.25, air_temperature_c=27, wind_speed_ms=5),
        events=[RaceEvent(kind="red_flag", description="Fictional late qualifying interruption")],
        sources=[Source(url="synthetic://qualifying", retrieved_at=start + timedelta(hours=3))],
        verified=True,
        synthetic=True,
    )
    race_data = snapshot.model_dump(mode="json")
    race_start = start + timedelta(days=1)
    race_cutoff = race_start - timedelta(hours=3)
    race_data.update(
        session="race",
        as_of=race_cutoff.isoformat(),
        session_start=race_start.isoformat(),
        qualifying_order=qualifying,
    )
    race_data["weather"].update(
        issued_at=race_cutoff.isoformat(), valid_at=race_start.isoformat(), rain_probability=0.6
    )
    race = Snapshot.model_validate(race_data)
    order = list(qualifying)
    order[0], order[2] = order[2], order[0]
    order.remove("driver_03")
    order.append("driver_03")
    r_actual = Feedback(
        event_id=snapshot.event_id,
        session=Session.RACE,
        available_at=race_start + timedelta(hours=3),
        finishing_order=order,
        retired_drivers=["driver_03"],
        actual_weather=ActualWeather(wet_fraction=0.6, air_temperature_c=26, wind_speed_ms=5),
        events=[
            RaceEvent(
                kind="mechanical", driver_id="driver_03", description="Fictional gearbox failure"
            ),
            RaceEvent(kind="safety_car", description="Fictional mid-race safety car"),
        ],
        sources=[Source(url="synthetic://race", retrieved_at=race_start + timedelta(hours=3))],
        verified=True,
        synthetic=True,
    )
    return snapshot, q_actual, race, r_actual


def run_demo(output: str | Path, simulations: int = 3000) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    database = output / "demo.sqlite3"
    if database.exists():
        raise ValueError("Demo database already exists; choose a new output directory")
    q, qa, r, ra = weekend()
    service = ForecastService(database)
    report = {"synthetic": True, "output_directory": str(output.resolve())}
    for label, snapshot, actual in (("qualifying", q, qa), ("race", r, ra)):
        forecast = service.predict(snapshot, simulations=simulations)
        feedback_result = service.feedback(forecast.id, actual)
        for suffix, model in (("input", snapshot), ("forecast", forecast), ("actual", actual)):
            (output / f"{label}-{suffix}.json").write_text(model.model_dump_json(indent=2))
        report[label] = {
            "forecast_id": forecast.id,
            "winner": forecast.winner,
            "scenarios": len(forecast.scenarios),
            **feedback_result,
        }
    report["model_version_after_feedback"] = service.store.state().version
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report

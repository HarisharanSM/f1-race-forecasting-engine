from datetime import timedelta

import pytest

from f1_forecast.engine import predict
from f1_forecast.ml_demo import synthetic_history
from f1_forecast.models import Snapshot, TyreStrategyInput
from f1_forecast.tyre_strategy import analyze_strategy, learn_curves


@pytest.fixture
def snapshot():
    return Snapshot.model_validate(synthetic_history(1)[1]["snapshot"])


@pytest.fixture
def evidence(snapshot):
    time = snapshot.as_of - timedelta(days=1)
    laps = [
        {
            "stint_id": f"{compound}-{stint}",
            "lap": age,
            "age": age,
            "duration_s": 90 + stint * 2 + slope * age + 0.002 * age**2 - 0.05 * age,
            "fuel_correction_s": 0.05 * age,
            "observed_at": time,
            "available_at": time,
            "circuit_id": snapshot.circuit.id,
            "car_spec": "spec-a",
            "compound": compound,
            "track_temperature_c": 30,
            "clean": True,
            "dry": True,
        }
        for compound, slope in (("SOFT", 0.08), ("HARD", 0.04))
        for stint in range(3)
        for age in range(1, 41)
    ]
    return TyreStrategyInput(
        available_at=time,
        source="Synthetic test, not measured telemetry",
        circuit_id=snapshot.circuit.id,
        car_spec="spec-a",
        track_temperature_c=30,
        race_laps=50,
        green_pit_loss_s=22,
        neutralized_pit_loss_s=12,
        sets=[
            {"id": "s1", "compound": "SOFT", "fresh_pace_offset_s": 0},
            {"id": "h1", "compound": "HARD", "fresh_pace_offset_s": 0.5},
            {"id": "h2", "compound": "HARD", "fresh_pace_offset_s": 0.5},
        ],
        laps=laps,
    )


def test_recovers_curves_with_stint_and_fuel_controls(evidence):
    curves = learn_curves(evidence)
    assert curves["SOFT"]["linear_s_per_lap"] == pytest.approx(0.08)
    assert curves["HARD"]["quadratic_s_per_lap2"] == pytest.approx(0.002)
    assert curves["SOFT"]["within_stint_rmse_s"] < 1e-9


def test_filters_incomparable_and_sparse_data(evidence):
    evidence.laps = [r for r in evidence.laps if not r.stint_id.endswith("2")]
    assert learn_curves(evidence) == {}
    assert analyze_strategy(evidence)["alternatives"] == []
    for r in evidence.laps:
        r.clean = False
    assert learn_curves(evidence) == {}


def test_strategy_inventory_age_and_pit_loss(evidence):
    report = analyze_strategy(evidence)
    assert report["strategies_evaluated"] > 0
    assert {r["stops"] for r in report["best_by_stops"]} == {1, 2}
    for r in report["alternatives"]:
        assert len(set(r["sets"])) == len(r["sets"])
        assert sum(r["stint_laps"]) == 50
        assert max(r["stint_laps"]) <= 40
        assert r["relative_time_s"] - r["one_neutralized_stop_time_s"] == pytest.approx(10)
    evidence.sets[0].initial_age = 40
    assert analyze_strategy(evidence)["alternatives"] == []


def test_snapshot_guards_and_forecast_opt_in(snapshot, evidence):
    snapshot.circuit.laps = 50
    plain = predict(snapshot, simulations=100)
    data = snapshot.model_dump()
    data["tyre_strategy"] = evidence.model_dump()
    enhanced = Snapshot.model_validate(data)
    result = predict(enhanced, simulations=100)
    assert result.standings == plain.standings
    assert result.tyre_strategy_analysis["status"] == "conditional_estimates"
    assert plain.tyre_strategy_analysis is None
    data["tyre_strategy"]["laps"][0]["available_at"] = snapshot.as_of + timedelta(seconds=1)
    with pytest.raises(ValueError, match="cutoff"):
        Snapshot.model_validate(data)


def test_duplicate_laps_rejected(evidence):
    data = evidence.model_dump()
    data["laps"].append(data["laps"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        TyreStrategyInput.model_validate(data)


def test_mismatched_conditions_are_excluded(evidence):
    for row in evidence.laps:
        row.track_temperature_c = 45
    assert learn_curves(evidence) == {}


def test_strategy_report_command(snapshot, evidence, tmp_path):
    import subprocess
    import sys

    snapshot.circuit.laps = 50
    snapshot.tyre_strategy = evidence
    path = tmp_path / "snapshot.json"
    path.write_text(snapshot.model_dump_json())
    output = tmp_path / "report"
    subprocess.run(
        [sys.executable, "scripts/analyze_tyre_strategy.py", str(path), "--output", str(output)],
        check=True,
    )
    assert "conditional_estimates" in (output / "strategy.html").read_text()
    assert "linear_s_per_lap" in (output / "strategy.json").read_text()

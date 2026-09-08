from datetime import timedelta
from typing import ClassVar

import numpy as np
import pytest

from f1_forecast.demo import weekend
from f1_forecast.engine import LearnerState, evaluate, features, learn, predict
from f1_forecast.ml_data import feature_matrix
from f1_forecast.models import CarPerformance, DriverPerformance, RaceDynamics, Snapshot
from f1_forecast.optional_inputs import effective_car, effective_driver, race_event_effects


@pytest.mark.parametrize("index", [0, 2])
def test_empty_or_disabled_optional_measurements_preserve_forecast(index):
    snapshot = weekend()[index]
    changed = snapshot.model_copy(deep=True)
    for driver in changed.drivers:
        driver.performance = DriverPerformance()
    for team in changed.teams:
        team.car.performance = CarPerformance()
    if index == 2:
        changed.race_dynamics = RaceDynamics()
    a, b = predict(snapshot, simulations=300), predict(changed, simulations=300)
    assert a.standings == b.standings
    assert a.scenarios == b.scenarios
    assert np.array_equal(feature_matrix(snapshot, [], 0.5), feature_matrix(changed, [], 0.5))
    changed.drivers[0].performance = DriverPerformance(
        weight=0,
        race_teammate_delta_pct=-1,
        retirement_probability=1,
        clean_lap_variability_pct=0,
        tyre_management=0,
    )
    changed.teams[0].car.performance = CarPerformance(weight=0, race_gap_pct=5, braking=0)
    assert predict(changed, simulations=300).standings == a.standings


@pytest.mark.parametrize(
    "field,column",
    [
        ("braking", 2),
        ("aerodynamics", 2),
        ("handling", 2),
        ("power_delivery", 2),
        ("medium_speed_cornering", 2),
        ("downforce", 2),
        ("aerodynamic_efficiency", 2),
    ],
)
def test_car_capabilities_affect_both_forecasters(field, column):
    low, high = weekend()[2], weekend()[2]
    low.teams[0].car.performance = CarPerformance(**{field: 0})
    high.teams[0].car.performance = CarPerformance(**{field: 1})
    assert features(high, 0)[0, column] > features(low, 0)[0, column]
    assert feature_matrix(high, [], 0)[0, column] > feature_matrix(low, [], 0)[0, column]


def test_pace_teammate_and_tyre_measurements_have_correct_direction_and_bounds():
    snapshot = weekend()[2]
    car, driver = snapshot.teams[0].car, snapshot.drivers[0]
    car.performance = CarPerformance(race_gap_pct=5, tyre_degradation_s_per_lap=0.3)
    driver.performance = DriverPerformance(race_teammate_delta_pct=1)
    assert effective_car(car).race_pace < car.race_pace
    assert effective_car(car).tyre_management < car.tyre_management
    assert effective_driver(driver).race_skill < driver.race_skill
    assert abs(effective_car(car).race_pace - car.race_pace) <= 0.20000001
    original = snapshot.model_dump_json()
    a = predict(snapshot, simulations=1000)
    car.performance = CarPerformance(race_gap_pct=0, tyre_degradation_s_per_lap=0)
    driver.performance = DriverPerformance(race_teammate_delta_pct=-1)
    b = predict(snapshot, simulations=1000)
    rank = lambda f: next(s.expected_position for s in f.standings if s.driver_id == driver.id)
    assert rank(b) < rank(a)
    copy = Snapshot.model_validate_json(original)
    predict(copy, simulations=300)
    assert copy.model_dump_json() == original


def test_driver_tyre_management_and_variability_reach_model_features():
    snapshot = weekend()[2]
    before = feature_matrix(snapshot, [], 0)
    snapshot.drivers[0].performance = DriverPerformance(
        tyre_management=0,
        clean_lap_variability_pct=2,
    )
    after = feature_matrix(snapshot, [], 0)
    assert after[0, 3] < before[0, 3]
    assert after[0, 17] < before[0, 17]  # Consistency in the existing checkpoint schema.
    np.testing.assert_array_equal(after[1:], before[1:])


class FixedModel:
    metadata: ClassVar[dict] = {"trained_through": "2020-01-01T00:00:00Z"}
    model_id = "test-model"

    def check_snapshot(self, snapshot):
        pass

    def scenario_parameters(self, snapshot, scenario):
        return np.zeros(len(snapshot.drivers)), np.full(len(snapshot.drivers), 0.7)


@pytest.mark.parametrize("ml", [None, FixedModel()])
@pytest.mark.parametrize("risk", [0, 1])
def test_total_retirement_probability_replaces_existing_risk_once(ml, risk):
    snapshot = weekend()[2]
    snapshot.drivers[0].performance = DriverPerformance(weight=1, retirement_probability=risk)
    forecast = predict(snapshot, simulations=1000, ml_model=ml)
    row = next(s for s in forecast.standings if s.driver_id == snapshot.drivers[0].id)
    assert row.dnf_probability == risk


@pytest.mark.parametrize("kind", ["safety_car", "virtual_safety_car", "red_flag"])
def test_shared_event_rates_seed_reproducibility_and_full_field(kind):
    snapshot = weekend()[2]
    baseline = predict(snapshot, simulations=1000)
    snapshot.race_dynamics = RaceDynamics(**{kind + "_probability": 1})
    a, b = [predict(snapshot, simulations=1000) for _ in range(2)]
    assert a.standings == b.standings
    assert a.scenarios == b.scenarios
    assert a.standings != baseline.standings
    for scenario in a.scenarios:
        assert scenario.race_event_rates[kind] == 1
        matrix = np.array([s.position_probabilities for s in scenario.standings])
        np.testing.assert_allclose(matrix.sum(axis=0), 1)
        np.testing.assert_allclose(matrix.sum(axis=1), 1)
    # New event draws must not shift the retirement RNG stream.
    assert {s.driver_id: s.dnf_probability for s in a.standings} == {
        s.driver_id: s.dnf_probability for s in baseline.standings
    }


def test_vsc_has_no_restart_or_bunching_and_late_sc_compresses_more():
    snapshot = weekend()[2]
    latent = np.tile(np.arange(len(snapshot.drivers)), (100, 1)).astype(float)
    snapshot.race_dynamics = RaceDynamics(
        virtual_safety_car_probability=1,
        neutralized_fraction=0.2,
        pit_opportunity_probability=0,
        restart_variability=1,
    )
    result, _ = race_event_effects(latent, snapshot, np.random.default_rng(4))
    np.testing.assert_array_equal(result, latent)
    snapshot.race_dynamics = RaceDynamics(
        safety_car_probability=1,
        event_lap_fraction=0.2,
        pit_opportunity_probability=0,
        restart_variability=0,
    )
    early, _ = race_event_effects(latent, snapshot, np.random.default_rng(4))
    snapshot.race_dynamics.event_lap_fraction = 0.8
    late, _ = race_event_effects(latent, snapshot, np.random.default_rng(4))
    assert late.std() < early.std() < latent.std()


@pytest.mark.parametrize("location", ["driver", "car", "race_dynamics"])
def test_future_optional_evidence_is_rejected(location):
    data = weekend()[2].model_dump()
    evidence = {"available_at": data["as_of"] + timedelta(seconds=1)}
    if location == "driver":
        data["drivers"][0]["performance"] = evidence
    elif location == "car":
        data["teams"][0]["car"]["performance"] = evidence
    else:
        data["race_dynamics"] = evidence
    with pytest.raises(ValueError, match="newer than the prediction cutoff"):
        Snapshot.model_validate(data)


@pytest.mark.parametrize(
    "cls,field,value",
    [
        (CarPerformance, "braking", 1.1),
        (CarPerformance, "tyre_degradation_s_per_lap", -0.1),
        (CarPerformance, "aerodynamics", float("nan")),
        (DriverPerformance, "clean_lap_variability_pct", float("inf")),
        (DriverPerformance, "retirement_probability", -1),
        (RaceDynamics, "event_lap_fraction", 2),
        (RaceDynamics, "safety_car_probability", 1.01),
    ],
)
def test_optional_input_validation(cls, field, value):
    with pytest.raises(ValueError):
        cls(**{field: value})


def test_race_dynamics_rejected_for_qualifying():
    data = weekend()[0].model_dump()
    data["race_dynamics"] = {"safety_car_probability": 0.3}
    with pytest.raises(ValueError, match="only supported for race"):
        Snapshot.model_validate(data)


def test_feedback_uses_optional_features_and_measures_interval_coverage():
    _, _, snapshot, actual = weekend()
    snapshot.teams[0].car.performance = CarPerformance(braking=0, race_gap_pct=5)
    forecast = predict(snapshot, simulations=1000)
    metrics = evaluate(forecast, actual)
    positions = {d: p for p, d in enumerate(actual.finishing_order, 1)}
    covered = sum(
        s.p10_position <= positions[s.driver_id] <= s.p90_position for s in forecast.standings
    )
    assert metrics["position_interval_coverage"] == covered / len(forecast.standings)
    assert metrics["position_interval_width"] >= 0
    assert learn(snapshot, actual, LearnerState()).version == 1

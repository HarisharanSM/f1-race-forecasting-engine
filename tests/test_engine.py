from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import pytest

from f1_forecast.demo import weekend
from f1_forecast.engine import LearnerState, default_scenarios, learn, predict
from f1_forecast.models import Scenario, ScenarioPlan, Snapshot
from f1_forecast.service import ForecastService


def test_overall_distribution_is_the_weighted_scenario_mixture():
    result = predict(weekend()[2], simulations=300)
    for row in result.standings:
        expected = np.zeros(len(result.standings))
        for scenario in result.scenarios:
            conditional = next(s for s in scenario.standings if s.driver_id == row.driver_id)
            expected += scenario.scenario.probability * np.array(conditional.position_probabilities)
        assert row.position_probabilities == pytest.approx(expected)


def test_concurrent_feedback_is_applied_once(tmp_path):
    q, actual, _, _ = weekend()
    service = ForecastService(tmp_path / "concurrent.sqlite3")
    forecast = service.predict(q, simulations=100)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: service.feedback(forecast.id, actual), range(3)))
    assert sum(not r["already_applied"] for r in results) == 1
    assert service.store.state().version == 1


def test_synthetic_state_cannot_be_used_for_real_predictions(tmp_path):
    q, actual, r, _ = weekend()
    service = ForecastService(tmp_path / "isolated.sqlite3")
    forecast = service.predict(q, simulations=100)
    service.feedback(forecast.id, actual)
    with pytest.raises(ValueError, match="separate databases"):
        service.predict(r.model_copy(update={"synthetic": False}), simulations=100)


@pytest.mark.parametrize("index", [0, 2])
@pytest.mark.parametrize("maximum", [1, 2, 3, 4, 5])
def test_rankings_are_permutations_and_probability_mass_is_conserved(index, maximum):
    snapshot = weekend()[index]
    result = predict(snapshot, simulations=300, max_scenarios=maximum)
    assert 1 <= len(result.scenarios) <= maximum
    assert sum(s.scenario.probability for s in result.scenarios) == pytest.approx(1)
    for standings in [result.standings] + [s.standings for s in result.scenarios]:
        assert {s.driver_id for s in standings} == {d.id for d in snapshot.drivers}
        assert [s.position for s in standings] == list(range(1, len(snapshot.drivers) + 1))
        matrix = np.array([s.position_probabilities for s in standings])
        assert matrix.sum(axis=0) == pytest.approx(np.ones(len(standings)))
        assert matrix.sum(axis=1) == pytest.approx(np.ones(len(standings)))
        assert sum(s.win_probability for s in standings) == pytest.approx(1)
        assert sum(s.podium_probability for s in standings) == pytest.approx(3)
        assert all(s.p10_position <= s.p90_position for s in standings)
        if index == 0:
            assert all(s.dnf_probability == 0 for s in standings)


def test_seed_is_reproducible_and_model_does_not_mutate():
    snapshot = weekend()[0]
    original = snapshot.model_dump_json()
    a, b = [predict(snapshot, seed=83, simulations=500) for _ in range(2)]
    assert a.standings == b.standings
    assert a.scenarios == b.scenarios
    assert snapshot.model_dump_json() == original


@pytest.mark.parametrize("rain", [0, 1])
def test_weather_extremes_do_not_create_zero_probability_scenarios(rain):
    data = weekend()[0].model_dump()
    data["weather"]["rain_probability"] = rain
    result = default_scenarios(Snapshot.model_validate(data), LearnerState(), 5)
    assert sum(s.probability for s in result.scenarios) == pytest.approx(1)
    assert all((s.wet_fraction > 0) == bool(rain) for s in result.scenarios)


@pytest.mark.parametrize(
    "change",
    [
        "missing_grid",
        "duplicate_driver",
        "unknown_team",
        "future_weather",
        "future_source",
        "late_cutoff",
        "invalid_rating",
        "naive_time",
    ],
)
def test_rejects_invalid_or_leaky_inputs(change):
    data = weekend()[2].model_dump()
    if change == "missing_grid":
        data["qualifying_order"] = None
    elif change == "duplicate_driver":
        data["drivers"][1]["id"] = data["drivers"][0]["id"]
    elif change == "unknown_team":
        data["drivers"][0]["team_id"] = "unknown"
    elif change == "future_weather":
        data["weather"]["issued_at"] = data["as_of"] + timedelta(seconds=1)
    elif change == "future_source":
        data["sources"][0]["retrieved_at"] = data["as_of"] + timedelta(seconds=1)
    elif change == "late_cutoff":
        data["as_of"] = data["session_start"]
    elif change == "invalid_rating":
        data["drivers"][0]["race_skill"] = float("nan")
    else:
        data["as_of"] = data["as_of"].replace(tzinfo=None)
    with pytest.raises(ValueError):
        Snapshot.model_validate(data)


def test_grid_changes_race_outcomes():
    snapshot = weekend()[2]
    reverse = Snapshot.model_validate(
        {**snapshot.model_dump(), "starting_grid": list(reversed(snapshot.qualifying_order))}
    )
    a = predict(snapshot, simulations=2000)
    b = predict(reverse, simulations=2000)
    original = {s.driver_id: s.expected_position for s in a.standings}
    altered = {s.driver_id: s.expected_position for s in b.standings}
    assert altered[snapshot.qualifying_order[0]] > original[snapshot.qualifying_order[0]]
    assert altered[snapshot.qualifying_order[-1]] < original[snapshot.qualifying_order[-1]]


def test_capability_change_and_wet_conditions_affect_results():
    original = weekend()[0]
    data = original.model_dump()
    data["teams"][-1]["car"]["qualifying_pace"] = 1
    improved = predict(Snapshot.model_validate(data), simulations=2000)
    baseline = predict(original, simulations=2000)
    target = original.drivers[-1].id
    position = lambda f: next(s.expected_position for s in f.standings if s.driver_id == target)
    assert position(improved) < position(baseline)
    wet = ScenarioPlan(
        scenarios=[
            Scenario(
                name="Wet",
                probability=1,
                wet_fraction=1,
                disruption_probability=0,
                rationale="Test",
            )
        ]
    )
    dry = ScenarioPlan(
        scenarios=[
            Scenario(
                name="Dry",
                probability=1,
                wet_fraction=0,
                disruption_probability=0,
                rationale="Test",
            )
        ]
    )
    assert (
        predict(original, simulations=500, plan=wet).standings
        != predict(original, simulations=500, plan=dry).standings
    )


def test_feedback_is_persistent_and_exactly_once(tmp_path):
    q, actual, _, _ = weekend()
    service = ForecastService(tmp_path / "test.sqlite3")
    forecast = service.predict(q, simulations=200)
    first = service.feedback(forecast.id, actual)
    second = ForecastService(tmp_path / "test.sqlite3").feedback(forecast.id, actual)
    assert first["model_version"] == second["model_version"] == 1
    assert second["already_applied"]
    assert service.store.state().version == 1
    conflicting = actual.model_copy(
        update={"finishing_order": list(reversed(actual.finishing_order))}
    )
    with pytest.raises(ValueError, match="Conflicting"):
        service.feedback(forecast.id, conflicting)


def test_feedback_errors_do_not_update_state(tmp_path):
    q, actual, _, _ = weekend()
    service = ForecastService(tmp_path / "test.sqlite3")
    forecast = service.predict(q, simulations=100)
    for invalid in [
        actual.model_copy(update={"verified": False}),
        actual.model_copy(update={"event_id": "wrong"}),
        actual.model_copy(update={"finishing_order": actual.finishing_order[:-1]}),
    ]:
        with pytest.raises(ValueError):
            service.feedback(forecast.id, invalid)
    assert service.store.state().version == 0


def test_mechanical_retirement_updates_reliability_without_pace_penalty():
    _, _, snapshot, actual = weekend()
    updated = learn(snapshot, actual, LearnerState())
    assert "race:driver_03" not in updated.driver_effects
    assert updated.reliability["velocity"] == [1, 2]
    assert updated.driver_effects
    assert updated.weights["qualifying"] == LearnerState().weights["qualifying"]


def test_cannot_predict_with_future_trained_state(tmp_path):
    q, actual, r, _ = weekend()
    service = ForecastService(tmp_path / "test.sqlite3")
    forecast = service.predict(q, simulations=100)
    service.feedback(forecast.id, actual)
    with pytest.raises(ValueError, match="future feedback"):
        service.predict(q, simulations=100)
    assert service.predict(r, simulations=100).model_version == 1
    assert service.store.memories(q) == []
    assert service.store.memories(r) == []  # Different session type.


def test_invalid_llm_orders_and_scenario_plans_are_rejected():
    q = weekend()[0]
    with pytest.raises(ValueError, match="every entered driver"):
        predict(q, simulations=100, llm_order=["unknown"] * len(q.drivers))
    with pytest.raises(ValueError, match="sum to one"):
        ScenarioPlan(
            scenarios=[
                Scenario(
                    name="Dry",
                    probability=0.2,
                    wet_fraction=0,
                    disruption_probability=0,
                    rationale="Bad mass",
                )
            ]
        )

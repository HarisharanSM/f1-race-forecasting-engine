from copy import deepcopy
from datetime import timedelta

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from f1_forecast.engine import predict
from f1_forecast.measurement_features import (
    LEGACY_MEASUREMENT_NAMES,
    MEASUREMENT_NAMES,
    measurement_matrix,
    without_performance,
)
from f1_forecast.measurement_learning import fit_weights, training_pairs
from f1_forecast.ml_backtest import backtest_transformer
from f1_forecast.ml_data import feature_matrix, records_from_json
from f1_forecast.ml_demo import synthetic_history
from f1_forecast.models import DriverPerformance, MeasurementQuality, Snapshot
from f1_forecast.neural import NeuralForecaster, TrainingConfig, fit_records
from f1_forecast.performance import summarize_session
from f1_forecast.performance_enrichment import quality_sessions


@pytest.fixture(scope="module")
def history():
    rows = synthetic_history(10)
    for row in rows:
        s = Snapshot.model_validate(row["snapshot"])
        for i, driver in enumerate(s.drivers):
            name = f"{s.session.value}_teammate_delta_pct"
            driver.performance = DriverPerformance(
                **{name: (i % 3 - 1) * 0.2},
                quality={
                    name: MeasurementQuality(
                        samples=12,
                        cohorts=3,
                        usable_fraction=0.6,
                        observed_at=s.as_of - timedelta(hours=10),
                        source_session="FP2",
                        source_event=s.event_id,
                    )
                },
            )
        row["snapshot"] = s.model_dump(mode="json")
    return rows


def test_measurements_keep_missing_distinct_from_zero_and_base_features_unchanged(history):
    snapshot = Snapshot.model_validate(history[0]["snapshot"])
    missing = without_performance(snapshot)
    assert not measurement_matrix(missing, 0.5).any()
    index = 1  # Explicit zero delta still has evidence.
    matrix = measurement_matrix(snapshot, 0.5)
    assert matrix[index, 0] == 0 and matrix[index, 1] == 1
    assert matrix[index, 2] == 1 and matrix[index, 3] > 0
    original_base = feature_matrix(missing, [], 0.5)
    snapshot.drivers[0].performance.qualifying_teammate_delta_pct = 10
    np.testing.assert_array_equal(
        original_base, feature_matrix(without_performance(snapshot), [], 0.5)
    )
    snapshot.drivers.reverse()
    np.testing.assert_array_equal(measurement_matrix(snapshot, 0.5)[0], matrix[-1])


def test_future_quality_is_rejected(history):
    row = deepcopy(history[0]["snapshot"])
    s = Snapshot.model_validate(row)
    row["drivers"][0]["performance"]["quality"]["qualifying_teammate_delta_pct"]["observed_at"] = (
        s.as_of + timedelta(days=1)
    )
    with pytest.raises(ValueError, match="quality is newer"):
        Snapshot.model_validate(row)


def test_sprint_evidence_recognizes_the_same_calendar_weekend(history):
    snapshot = Snapshot.model_validate(history[0]["snapshot"])
    snapshot.event_id = f"{snapshot.season}-{snapshot.round:02d}-sprint"
    for driver in snapshot.drivers:
        for quality in driver.performance.quality.values():
            quality.source_event = f"{snapshot.season}-{snapshot.round:02d}"
    matrix = measurement_matrix(snapshot, 0)
    assert np.all(matrix[:, 7] == 1)


def test_regularization_shrinks_correction_and_leaves_unsupported_columns_zero():
    n = len(MEASUREMENT_NAMES)
    x = torch.zeros((4, n))
    x[:, 0] = torch.tensor([-1, -1, 1, 1])
    batch = {
        "x": x,
        "offset": torch.zeros(4),
        "target": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        "weights": torch.ones(4),
        "active": np.array([True] + [False] * (n - 1)),
        "scale": np.ones(n),
    }
    weak, strong = np.array(fit_weights(batch, 0.1)), np.array(fit_weights(batch, 10))
    assert 0 < strong[0] < weak[0]
    assert not strong[1:].any()


def test_checkpoint_roundtrip_and_missing_fallback_for_learned_branch(history, tmp_path):
    config = TrainingConfig(
        epochs=2,
        hidden_size=16,
        layers=1,
        min_train_events=3,
        validation_events=2,
        optional_mode="learned",
        calibrate=True,
    )
    model = fit_records(records_from_json(history[:12]), config)
    # This short history cannot support a measured correction.
    assert not model.metadata["validation_selection"]["optional_correction_selected"]
    batch = training_pairs(model, records_from_json(history[:8]), [], "race")
    assert not batch["active"].any()
    model.save(tmp_path / "model")
    loaded = NeuralForecaster.load(tmp_path / "model")
    s = Snapshot.model_validate(history[-1]["snapshot"])
    a = predict(s, ml_model=model, simulations=100)
    b = predict(s, ml_model=loaded, simulations=100)
    assert a.standings == b.standings
    # Persisted nonzero corrections also survive serialization and disappear when evidence is omitted.
    weights = {stage: [0.0] * len(MEASUREMENT_NAMES) for stage in ("qualifying", "race")}
    weights["race"][0] = 0.4
    model.metadata["measurement_adjustment"] = {
        "feature_names": list(MEASUREMENT_NAMES),
        "weights": weights,
    }
    model.save(tmp_path / "adjusted")
    loaded = NeuralForecaster.load(tmp_path / "adjusted")
    assert (
        predict(s, ml_model=model, simulations=100).standings
        == predict(s, ml_model=loaded, simulations=100).standings
    )
    assert (
        predict(without_performance(s), ml_model=loaded, simulations=100).standings == b.standings
    )


def test_learned_selection_cannot_use_target_labels_or_weather(history):
    original = history[:18]
    changed = deepcopy(original)
    for row in changed[-2:]:
        row["feedback"]["finishing_order"].reverse()
        row["feedback"]["actual_weather"]["wet_fraction"] = 1
    config = TrainingConfig(
        epochs=2,
        hidden_size=16,
        layers=1,
        min_train_events=6,
        validation_events=2,
        optional_mode="learned",
        calibrate=True,
    )
    before = backtest_transformer(original, config=config, simulations=100)
    after = backtest_transformer(changed, config=config, simulations=100)
    for a, b in zip(before["results"], after["results"], strict=True):
        assert a["forecast"]["standings"] == b["forecast"]["standings"]
    assert before["folds"][0]["validation_selection"] == after["folds"][0]["validation_selection"]


def test_ablation_rejects_nonoptional_changes(history):
    changed = deepcopy(history)
    changed[0]["snapshot"]["drivers"][0]["consistency"] = 0.01
    with pytest.raises(ValueError, match="only change optional"):
        backtest_transformer(history, prediction_rows=changed, simulations=100)


def test_strict_measurements_require_weather_and_sufficient_peer_samples():
    from test_performance import session_data

    data = session_data()
    strict = summarize_session(data["laps"], data["drivers"], strict=True)
    assert all(d["pace_gap_pct"] is None for d in strict["drivers"])
    for lap in data["laps"]:
        lap.update(rainfall=False, track_temperature_c=30)
    strict = summarize_session(data["laps"], data["drivers"], strict=True)
    assert any(d["pace_gap_pct"] is not None for d in strict["drivers"])
    first_team = next(iter(data["drivers"].values()))["team_id"]
    peers = [d for d, row in data["drivers"].items() if row["team_id"] == first_team]
    if len(peers) >= 2:
        reduced = [
            lap
            for lap in data["laps"]
            if lap["driver_number"] != peers[1] or lap["lap_number"] == 1
        ]
        strict = summarize_session(reduced, data["drivers"], strict=True)
        assert (
            next(d for d in strict["drivers"] if d["driver_number"] == peers[0])[
                "teammate_delta_pct"
            ]
            is None
        )


def test_weather_matching_does_not_use_future_observations():
    from test_performance import session_data

    data = session_data()
    data["channels"] = {
        "weather_data": [{"Time": "P0DT1H0M0S", "TrackTemp": 30, "Rainfall": False}]
    }
    result = quality_sessions([data])[0]
    assert all(d["pace_gap_pct"] is None for d in result["quality"]["drivers"])


def test_strict_variability_does_not_mix_wet_and_dry_stint_laps():
    from test_performance import session_data

    data = session_data()
    for lap in data["laps"]:
        lap.update(rainfall=lap["lap_number"] >= 5, track_temperature_c=30)
    result = summarize_session(data["laps"], data["drivers"], strict=True)
    # Neither condition has six observations even though the full stint has nine.
    assert all(d["clean_lap_variability_pct"] is None for d in result["drivers"])


@pytest.mark.parametrize("names", [LEGACY_MEASUREMENT_NAMES, MEASUREMENT_NAMES])
def test_old_and_joint_checkpoint_schemas_roundtrip(history, tmp_path, names):
    from f1_forecast.models import JointEffectEstimate

    model = fit_records(
        records_from_json(history[:12]),
        TrainingConfig(
            epochs=1,
            hidden_size=16,
            layers=1,
            min_train_events=3,
            validation_events=2,
            optional_mode="ignore",
        ),
    )
    s = Snapshot.model_validate(history[-1]["snapshot"])
    s.drivers[0].performance.joint_effects["race"] = JointEffectEstimate(
        mean_pct=-0.5,
        std_pct=0.3,
        available_at=s.as_of - timedelta(days=1),
        observed_at=s.as_of - timedelta(days=2),
        season=s.season,
        samples=30,
        sessions=3,
        current_season_observed=True,
    )
    weights = {stage: [0.0] * len(names) for stage in ("qualifying", "race")}
    weights["race"][len(LEGACY_MEASUREMENT_NAMES) if names == MEASUREMENT_NAMES else 0] = 0.4
    model.metadata["measurement_adjustment"] = {"feature_names": list(names), "weights": weights}
    model.save(tmp_path / "model")
    loaded = NeuralForecaster.load(tmp_path / "model")
    forecast = predict(s, ml_model=model, simulations=100)
    assert forecast.standings == predict(s, ml_model=loaded, simulations=100).standings
    from f1_forecast.models import Scenario

    scenario = Scenario(
        name="dry", probability=1, wet_fraction=0, disruption_probability=0.1, rationale="test"
    )
    with_evidence = loaded.scenario_parameters(s, scenario)[0]
    if names == MEASUREMENT_NAMES:
        s.drivers[0].performance.joint_effects.clear()
        without_joint = loaded.scenario_parameters(s, scenario)[0]
        assert with_evidence[0] - without_joint[0] == pytest.approx(-0.1)

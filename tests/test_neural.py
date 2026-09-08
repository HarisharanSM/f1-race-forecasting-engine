from copy import deepcopy
from datetime import datetime, timedelta

import pytest

torch = pytest.importorskip("torch")

from f1_forecast.api import create_app
from f1_forecast.engine import predict
from f1_forecast.ml_backtest import backtest_transformer
from f1_forecast.ml_data import (
    FEATURE_NAMES,
    chronological_split,
    feature_matrix,
    history_summary,
    records_from_json,
)
from f1_forecast.ml_demo import synthetic_history
from f1_forecast.models import Snapshot
from f1_forecast.neural import FieldTransformer, NeuralForecaster, TrainingConfig, fit_records, pack
from f1_forecast.service import ForecastService


@pytest.fixture(scope="module")
def history():
    return synthetic_history(6)


@pytest.fixture(scope="module")
def trained(history):
    return fit_records(
        records_from_json(history[:8]),
        TrainingConfig(
            epochs=15, patience=5, min_train_events=2, hidden_size=16, layers=1, dropout=0, seed=71
        ),
    )


def test_training_learns_and_keeps_validation_weekend_separate(trained):
    metadata = trained.metadata
    assert metadata["selected_training_loss"] < metadata["initial_training_loss"]
    assert not set(metadata["training_events"]) & set(metadata["validation_events"])
    assert datetime.fromisoformat(metadata["training_feedback_max"]) < datetime.fromisoformat(
        metadata["validation_cutoff"]
    )
    assert metadata["best_epoch"] <= metadata["config"]["epochs"]
    assert set(metadata["trained_sessions"]) == {"qualifying", "race"}


def test_feature_history_drops_future_and_same_weekend_labels(history):
    records = records_from_json(history)
    snapshot = records[4].snapshot
    earlier = [history_summary(r) for r in records[:4]]
    all_history = [history_summary(r) for r in records]
    a = feature_matrix(snapshot, earlier, 0.5)
    b = feature_matrix(snapshot, all_history, 0.5)
    assert a == pytest.approx(b)
    delayed = deepcopy(earlier)
    for h in delayed:
        h["available_at"] = (snapshot.as_of + timedelta(days=1)).isoformat()
    assert feature_matrix(snapshot, delayed, 0.5) == pytest.approx(
        feature_matrix(snapshot, [], 0.5)
    )
    assert a.shape == (len(snapshot.drivers), len(FEATURE_NAMES))


def test_optional_evidence_reaches_loaded_transformer(history, trained, tmp_path):
    import numpy as np

    from f1_forecast.engine import LearnerState, default_scenarios
    from f1_forecast.models import CarPerformance, DriverPerformance

    trained.save(tmp_path / "checkpoint")
    model = NeuralForecaster.load(tmp_path / "checkpoint")
    snapshot = Snapshot.model_validate(history[-1]["snapshot"])
    scenario = default_scenarios(snapshot, LearnerState(), 3).scenarios[0]
    before, _ = model.scenario_parameters(snapshot, scenario)
    snapshot.drivers[0].performance = DriverPerformance()
    snapshot.teams[0].car.performance = CarPerformance()
    missing, _ = model.scenario_parameters(snapshot, scenario)
    np.testing.assert_array_equal(before, missing)
    snapshot.drivers[0].performance = DriverPerformance(
        race_teammate_delta_pct=-1,
        clean_lap_variability_pct=0.1,
    )
    snapshot.teams[0].car.performance = CarPerformance(race_gap_pct=4, braking=0.1)
    changed, risk = model.scenario_parameters(snapshot, scenario)
    assert not np.allclose(before, changed)
    assert np.isfinite(changed).all() and np.isfinite(risk).all()


def test_transformer_is_equivariant_to_driver_input_order_and_ignores_padding(history):
    snapshot = Snapshot.model_validate(history[0]["snapshot"])
    x = torch.from_numpy(feature_matrix(snapshot, [], 0.3)).unsqueeze(0)
    network = FieldTransformer(TrainingConfig(hidden_size=16, layers=1, dropout=0)).eval()
    mask = torch.zeros(x.shape[:2], dtype=torch.bool)
    permutation = torch.arange(x.shape[1] - 1, -1, -1)
    with torch.inference_mode():
        original, _ = network(x, mask)
        reordered, _ = network(x[:, permutation], mask)
        padded = torch.cat([x, torch.randn((1, 4, x.shape[2]))], dim=1)
        padded_mask = torch.cat([mask, torch.ones((1, 4), dtype=torch.bool)], dim=1)
        extra, _ = network(padded, padded_mask)
    assert original[:, permutation].numpy() == pytest.approx(reordered.numpy(), abs=2e-6)
    assert original.numpy() == pytest.approx(extra[:, : x.shape[1]].numpy(), abs=2e-6)


def test_retirements_mask_pace_targets_but_supervise_dnf(history):
    records = records_from_json(history)
    record = next(r for r in records if r.feedback.retired_drivers)
    _, _, clean, _, retired = pack([record], [])
    for i, driver in enumerate(record.snapshot.drivers):
        if driver.id in record.feedback.retired_drivers:
            assert not clean[0, i]
            assert retired[0, i] == 1


def test_delayed_training_results_are_purged(history):
    records = records_from_json(history)
    # The final validation weekend's cutoff is after normal earlier results.
    cutoff = records[-2].snapshot.as_of
    records[0].feedback = records[0].feedback.model_copy(
        update={"available_at": cutoff + timedelta(days=1)}
    )
    train, validation = chronological_split(records, 2, 1)
    assert records[0] not in train
    assert len(validation) == 2
    assert all(r.feedback.available_at < cutoff for r in train)


@pytest.mark.parametrize("index", [8, 9])
def test_saved_checkpoint_predicts_both_sessions_and_conserves_probability(
    trained, history, tmp_path, index
):
    trained.save(tmp_path / "model")
    loaded = NeuralForecaster.load(tmp_path / "model")
    snapshot = Snapshot.model_validate(history[index]["snapshot"])
    a = predict(snapshot, ml_model=trained, simulations=200)
    b = predict(snapshot, ml_model=loaded, simulations=200)
    assert a.standings == b.standings
    assert a.forecast_model == "transformer"
    assert a.ml_model_id == loaded.model_id
    assert sum(s.win_probability for s in a.standings) == pytest.approx(1)
    assert sum(s.podium_probability for s in a.standings) == pytest.approx(3)
    assert all(sum(s.position_probabilities) == pytest.approx(1) for s in a.standings)
    if snapshot.session == "qualifying":
        assert all(s.dnf_probability == 0 for s in a.standings)


def test_checkpoint_rejects_known_events_future_labels_and_real_inputs(trained, history):
    with pytest.raises(ValueError, match="future"):
        trained.check_snapshot(Snapshot.model_validate(history[0]["snapshot"]))
    s = Snapshot.model_validate(history[8]["snapshot"])
    with pytest.raises(ValueError, match="training/validation event"):
        trained.check_snapshot(
            s.model_copy(update={"event_id": history[0]["snapshot"]["event_id"]})
        )
    with pytest.raises(ValueError, match="Synthetic"):
        trained.check_snapshot(s.model_copy(update={"synthetic": False}))


def test_checkpoint_corruption_is_rejected_before_deserialization(trained, tmp_path):
    trained.save(tmp_path / "model")
    (tmp_path / "model" / "weights.pt").write_bytes(b"invalid checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        NeuralForecaster.load(tmp_path / "model")


def test_walk_forward_trains_before_each_fold_and_saves_future_checkpoint(history, tmp_path):
    report = backtest_transformer(
        history,
        config=TrainingConfig(epochs=2, min_train_events=2, layers=1, hidden_size=16),
        simulations=100,
        save_model=tmp_path / "model",
    )
    assert report["sessions_evaluated"] == 6
    assert len(report["warmup_events"]) == 3
    for fold in report["folds"]:
        assert fold["event_id"] not in fold["training_events"] + fold["validation_events"]
        assert datetime.fromisoformat(fold["trained_through"]) < datetime.fromisoformat(
            fold["prediction_cutoff"]
        )
        assert not set(fold["training_events"]) & set(fold["validation_events"])
    assert set(report["by_session"]) == {"qualifying", "race"}
    final = NeuralForecaster.load(tmp_path / "model")
    assert final.model_id not in [fold["model_id"] for fold in report["folds"]]


def test_holdout_actual_weather_and_labels_cannot_change_its_forecast(history):
    config = TrainingConfig(epochs=2, min_train_events=2, layers=1, hidden_size=16)
    # Use a single out-of-sample weekend, so changes cannot legitimately train later folds.
    original = history[:8]
    modified = deepcopy(original)
    for row in modified[-2:]:
        row["feedback"]["actual_weather"]["wet_fraction"] = 1
        row["feedback"]["finishing_order"].reverse()
    a = backtest_transformer(original, config=config, simulations=100)
    b = backtest_transformer(modified, config=config, simulations=100)
    for left, right in zip(a["results"], b["results"], strict=True):
        assert left["forecast"]["standings"] == right["forecast"]["standings"]


def test_too_short_history_fails_instead_of_testing_on_training_data(history):
    with pytest.raises(ValueError, match="No out-of-sample"):
        backtest_transformer(history[:4], simulations=100)


def test_history_validates_provenance_and_unique_sessions(history):
    with pytest.raises(ValueError, match="Duplicate"):
        records_from_json([history[0], history[0]])
    altered = deepcopy(history[:1])
    altered[0]["feedback"]["verified"] = False
    with pytest.raises(ValueError, match="verified"):
        records_from_json(altered)


def test_service_and_http_use_the_trained_model(trained, history, tmp_path):
    from fastapi.testclient import TestClient

    trained.save(tmp_path / "model")
    s = Snapshot.model_validate(history[8]["snapshot"])
    service = ForecastService(tmp_path / "service.sqlite3", ml_model=tmp_path / "model")
    prediction = service.predict(s, simulations=100)
    assert service.store.get(prediction.id)[1].ml_model_id == trained.model_id
    client = TestClient(create_app(str(tmp_path / "api.sqlite3"), ml_model=trained))
    response = client.post(
        "/predictions", json={"snapshot": s.model_dump(mode="json"), "simulations": 100}
    )
    assert response.status_code == 200
    assert response.json()["forecast_model"] == "transformer"
    response = client.post(
        "/predictions",
        json={"snapshot": s.model_dump(mode="json"), "simulations": 100, "use_ml": False},
    )
    assert response.json()["forecast_model"] == "heuristic"


def test_test_season_boundary_keeps_earlier_season_as_history(history):
    rows = deepcopy(history)
    for index, row in enumerate(rows):
        row["snapshot"]["season"] = 2024 if index < 8 else 2025
    report = backtest_transformer(
        rows,
        config=TrainingConfig(epochs=1, min_train_events=2, hidden_size=16, layers=1),
        simulations=100,
        test_from_season=2025,
    )
    assert report["sessions_evaluated"] == 4
    assert len(report["warmup_events"]) == 4
    first_test = report["folds"][0]
    assert rows[0]["snapshot"]["event_id"] in first_test["training_events"]
    assert (
        first_test["event_id"]
        not in first_test["training_events"] + first_test["validation_events"]
    )
    assert report["test_from_season"] == 2025

import numpy as np
import pytest
import torch

from f1_forecast.engine import predict
from f1_forecast.ml_data import records_from_json
from f1_forecast.ml_demo import synthetic_history
from f1_forecast.model_comparison import KINDS, blend_forecasts, compare
from f1_forecast.neural import NeuralForecaster, TrainingConfig, fit_records


def test_identical_folds_and_reserved_ensemble_selection():
    torch.set_num_threads(1)
    report = compare(
        synthetic_history(7),
        TrainingConfig(epochs=1, min_train_events=1, validation_events=1),
        simulations=100,
    )
    for fold in report["folds"]:
        train, valid, ensemble = [
            set(fold[k])
            for k in ("training_events", "validation_events", "ensemble_validation_events")
        ]
        assert not train & valid and not train & ensemble and not valid & ensemble
        assert fold["event_id"] not in train | valid | ensemble
        assert fold["ensemble_selected_through"] < fold["prediction_cutoff"]
        for model in fold["models"].values():
            assert set(model["training_events"]) == train
            assert set(model["validation_events"]) == valid
    for row in report["results"]:
        assert set(row["metrics"]) == {*KINDS, "ensemble"}


def test_target_labels_cannot_change_predictions_or_blend_selection():
    rows = synthetic_history(5)
    config = TrainingConfig(epochs=1, min_train_events=1, validation_events=1)
    before = compare(rows, config, simulations=100)
    rows[-1]["feedback"]["finishing_order"].reverse()
    after = compare(rows, config, simulations=100)
    assert before["folds"][-1]["weights"] == after["folds"][-1]["weights"]
    assert before["folds"][-1]["candidate_log_losses"] == after["folds"][-1]["candidate_log_losses"]
    for kind in (*KINDS, "ensemble"):
        assert (
            before["results"][-1]["forecasts"][kind]["standings"]
            == after["results"][-1]["forecasts"][kind]["standings"]
        )


@pytest.mark.parametrize("kind", ["linear_ranker", "mlp_ranker"])
def test_simple_checkpoint_and_probability_mixture(kind, tmp_path):
    records = records_from_json(synthetic_history(4))
    model = fit_records(
        records[:6], TrainingConfig(epochs=1, min_train_events=1), network_kind=kind
    )
    model.save(tmp_path)
    restored = NeuralForecaster.load(tmp_path)
    s = records[-1].snapshot
    a = predict(s, simulations=100, ml_model=model)
    b = predict(s, simulations=100, ml_model=restored)
    assert a.standings == b.standings
    assert a.forecast_model == kind
    mixed = blend_forecasts([a, b], [0.25, 0.75])
    np.testing.assert_allclose(
        [r.position_probabilities for r in a.standings],
        [r.position_probabilities for r in mixed.standings],
    )
    matrix = np.array([r.position_probabilities for r in mixed.standings])
    np.testing.assert_allclose(matrix.sum(0), 1)
    np.testing.assert_allclose(matrix.sum(1), 1)
    with pytest.raises(ValueError, match="weights"):
        blend_forecasts([a, b], [-1, 2])

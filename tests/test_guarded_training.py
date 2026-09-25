from datetime import timedelta

import numpy as np
import pytest
import torch

from f1_forecast.engine import predict
from f1_forecast.ml_data import (
    FEATURE_NAMES,
    QUALITY_FEATURE_NAMES,
    feature_matrix,
    records_from_json,
)
from f1_forecast.ml_demo import synthetic_history
from f1_forecast.neural import NeuralForecaster, TrainingConfig, fit_records, objective, pack


def test_missingness_is_explicit_and_default_schema_is_unchanged():
    record = records_from_json(synthetic_history(1))[-1]
    s = record.snapshot
    s.notes.append("REDUCED INPUT: missing measurements")
    original = feature_matrix(s, [], 0.2)
    enriched = feature_matrix(s, [], 0.2, quality_features=True)
    assert original.shape[1] == len(FEATURE_NAMES)
    np.testing.assert_array_equal(original, enriched[:, : len(FEATURE_NAMES)])
    flags = dict(zip(QUALITY_FEATURE_NAMES, enriched[0, len(FEATURE_NAMES) :], strict=True))
    assert flags["reduced_inputs"] == flags["weather_imputed"] == 1
    assert flags["driver_pace_missing"] == flags["car_pace_missing"] == 1


def test_imputed_feedback_weather_cannot_masquerade_as_observed_training_weather():
    r = records_from_json(synthetic_history(1))[-1]
    r.snapshot.notes.append("REDUCED INPUT: weather assumed")
    before = pack([r], [], quality_features=True)[0]
    r.feedback.actual_weather.air_temperature_c = 45
    r.feedback.actual_weather.wet_fraction = 1
    after = pack([r], [], quality_features=True)[0]
    torch.testing.assert_close(before, after)


def test_listwise_likelihood_prefers_correct_order_and_masks_retirements():
    class Fixed(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.tensor([values], requires_grad=True)

        def forward(self, x, padding):
            return self.values, torch.zeros_like(self.values)

    x = torch.zeros(1, 3, len(FEATURE_NAMES))
    batch = (
        x,
        torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[True, True, False]]),
        torch.tensor([[0.0, 1.0, 2.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
    )
    good, bad = Fixed([2.0, 0.0, 999.0]), Fixed([0.0, 2.0, -999.0])
    extra_good = objective(good, batch, listwise_weight=1) - objective(good, batch)
    extra_bad = objective(bad, batch, listwise_weight=1) - objective(bad, batch)
    assert extra_good < extra_bad
    extra_good.backward()
    assert good.values.grad[0, 2] == 0


def test_recency_weights_change_training_contribution():
    class Fixed(torch.nn.Module):
        def forward(self, x, padding):
            return torch.tensor([[2.0, 0.0], [2.0, 0.0]]), torch.zeros(2, 2)

    batch = (
        torch.zeros(2, 2, len(FEATURE_NAMES)),
        torch.zeros(2, 2, dtype=torch.bool),
        torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        torch.zeros(2, 2),
    )
    good_weighted = objective(Fixed(), batch, sample_weights=torch.tensor([1.0, 0.1]))
    bad_weighted = objective(Fixed(), batch, sample_weights=torch.tensor([0.1, 1.0]))
    assert good_weighted < bad_weighted


def test_experimental_checkpoint_roundtrip_and_no_future_input(tmp_path):
    torch.set_num_threads(1)
    records = records_from_json(synthetic_history(4))
    config = TrainingConfig(
        epochs=1,
        min_train_events=1,
        quality_features=True,
        recency_half_life_days=365,
        listwise_weight=0.1,
    )
    model = fit_records(records[:6], config)
    model.save(tmp_path)
    loaded = NeuralForecaster.load(tmp_path)
    target = records[-1].snapshot
    a = predict(target, ml_model=model, simulations=100)
    b = predict(target, ml_model=loaded, simulations=100)
    assert a.standings == b.standings
    assert model.metadata["feature_names"] == list(FEATURE_NAMES + QUALITY_FEATURE_NAMES)
    assert model.metadata["training_feedback_max"] < model.metadata["validation_cutoff"]
    target.as_of -= timedelta(days=100)
    with pytest.raises(ValueError, match="future"):
        loaded.check_snapshot(target)


@pytest.mark.parametrize(
    "setting",
    [
        {"recency_half_life_days": 0},
        {"listwise_weight": float("nan")},
        {"quality_features": True, "calibrate": True},
    ],
)
def test_invalid_experimental_settings_rejected(setting):
    with pytest.raises(ValueError):
        TrainingConfig(**setting)

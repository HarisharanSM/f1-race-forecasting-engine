import json

import pytest
import torch

from f1_forecast.engine import predict
from f1_forecast.ml_data import records_from_json
from f1_forecast.ml_demo import synthetic_history
from f1_forecast.neural import TrainingConfig
from f1_forecast.primary_models import CANDIDATES, PrimaryBundle, select_weights, train_bundle
from f1_forecast.service import ForecastService


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("primary")
    rows = synthetic_history(6)
    train_bundle(rows, TrainingConfig(epochs=1, min_train_events=1), root / "grand_prix")
    return root, rows


def test_primary_service_defaults_and_explicit_ensemble(bundle, tmp_path):
    root, _ = bundle
    target = records_from_json(synthetic_history(7))[-1].snapshot
    primary = ForecastService(tmp_path / "primary.sqlite3", primary_bundle=root).predict(
        target, simulations=100
    )
    assert primary.forecast_model == "transformer"
    combined = ForecastService(
        tmp_path / "ensemble.sqlite3", primary_bundle=root, model_policy="ensemble"
    ).predict(target, simulations=100)
    assert combined.forecast_model in {"transformer", "probability_ensemble"}
    assert all(w[0] >= 0.75 for w in CANDIDATES)
    heuristic = ForecastService(tmp_path / "heuristic.sqlite3", model_policy="heuristic").predict(
        target, simulations=100
    )
    assert heuristic.forecast_model == "heuristic"


def test_bundle_rejects_development_and_invalid_weights(bundle):
    root, rows = bundle
    model = PrimaryBundle(root / "grand_prix")
    with pytest.raises(ValueError, match="target or future"):
        model.predict(records_from_json(rows)[-1].snapshot, simulations=100)
    assert model.metadata["weights"][0] >= 0.75


def test_identity_fallback_is_exact_and_lazy(bundle):
    root, _ = bundle
    target = records_from_json(synthetic_history(7))[-1].snapshot
    model = PrimaryBundle(root / "grand_prix")
    model.metadata["weights"] = [1, 0, 0]
    a = model.predict(target, simulations=100)
    b = model.predict(target, ensemble=True, simulations=100)
    assert a.standings == b.standings and a.scenarios == b.scenarios
    assert set(model.models) == {"field_transformer"}


def test_active_blend_loads_only_weighted_models(bundle):
    root, _ = bundle
    model = PrimaryBundle(root / "grand_prix")
    model.metadata["weights"] = [0.75, 0, 0.25]
    target = records_from_json(synthetic_history(7))[-1].snapshot
    result = model.predict(target, ensemble=True, simulations=100)
    assert result.forecast_model == "probability_ensemble"
    assert set(model.models) == {"field_transformer", "mlp_ranker"}


def test_identical_forecasts_fail_strict_blend_gate():
    r = records_from_json(synthetic_history(1))[-1]
    f = predict(r.snapshot, simulations=100)
    weights, _ = select_weights([[f, f, f]], [r.feedback])
    assert weights == [1, 0, 0]


def test_manifest_rejects_transformer_minority(tmp_path):
    meta = {
        "version": 1,
        "primary": "field_transformer",
        "model_order": ["field_transformer", "linear_ranker", "mlp_ranker"],
        "weights": [0.5, 0.5, 0],
    }
    (tmp_path / "bundle.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Invalid primary"):
        PrimaryBundle(tmp_path)

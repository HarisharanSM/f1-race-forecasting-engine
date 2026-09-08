from datetime import timedelta

import numpy as np
import pytest

from f1_forecast.engine import predict, summarize
from f1_forecast.ml_demo import synthetic_history
from f1_forecast.models import Feedback, Forecast, Snapshot
from f1_forecast.probability_calibration import (
    ProbabilityCalibrator,
    fit_calibrator,
    fit_conservative_calibrator,
    reliability,
    temperature_matrix,
    transform,
)


@pytest.fixture(scope="module")
def examples():
    result = []
    for row in synthetic_history(40):
        s, a = Snapshot.model_validate(row["snapshot"]), Feedback.model_validate(row["feedback"])
        forecast = predict(s, simulations=100)
        ids = [d.driver_id for d in forecast.standings]
        n = len(ids)
        matrix = np.full((n, n), 0.45 / n)
        for i, driver in enumerate(ids):
            matrix[i, a.finishing_order.index(driver)] += 0.55
        risks = np.array([d.dnf_probability for d in forecast.standings])
        standings = summarize(ids, matrix, risks)
        forecast.standings = standings
        forecast.winner = max(standings, key=lambda d: d.win_probability).driver_id
        for scenario in forecast.scenarios:
            scenario.standings = standings
            scenario.winner = forecast.winner
        result.append((s, Forecast.model_validate(forecast.model_dump()), a))
    return result


def test_balancing_preserves_probability_constraints():
    matrix = 0.7 * np.eye(22) + 0.3 / 22
    np.testing.assert_array_equal(temperature_matrix(matrix, 1), matrix)
    for t in (0.5, 0.8, 1.25, 2):
        result = temperature_matrix(matrix, t)
        np.testing.assert_allclose(result.sum(0), 1, atol=1e-9)
        np.testing.assert_allclose(result.sum(1), 1, atol=1e-9)
        assert result[:, :3].sum() == pytest.approx(3)
    with pytest.raises(ValueError, match="doubly stochastic"):
        temperature_matrix(np.ones((3, 3)), 0.8)


def test_fitting_uses_disjoint_earlier_windows_and_selects_useful_map(examples):
    calibrator = fit_calibrator(examples[:24], examples[24:36])
    assert all(t < 1 for t in calibrator.temperatures.values())
    assert all(s["selected"] for s in calibrator.audit["strata"].values())
    for s, f, a in examples[36:]:
        after = calibrator.apply(s, f)
        before_loss = reliability([(s, f, a)])["position_log_loss"]
        assert reliability([(s, after, a)])["position_log_loss"] < before_loss
    with pytest.raises(ValueError, match="chronological"):
        fit_calibrator(examples[:24], examples[22:36])
    with pytest.raises(ValueError, match="chronological"):
        fit_calibrator(examples[24:36], examples[:24])


def test_small_strata_keep_identity(examples):
    calibrator = fit_calibrator(examples[:8], examples[8:12])
    assert set(calibrator.temperatures.values()) == {1}
    s, f, _ = examples[-1]
    assert calibrator.apply(s, f).standings == f.standings


def test_holdout_labels_cannot_change_fitted_parameters(examples):
    before = fit_calibrator(examples[:24], examples[24:36])
    changed = [
        (s.model_copy(deep=True), f.model_copy(deep=True), a.model_copy(deep=True))
        for s, f, a in examples
    ]
    for _, _, actual in changed[36:]:
        actual.finishing_order.reverse()
        actual.actual_weather.wet_fraction = 1
    after = fit_calibrator(changed[:24], changed[24:36])
    assert before == after
    for s, _, _ in examples[:36]:
        assert s.as_of < before.fitted_through


def test_artifact_roundtrip_and_temporal_guards(examples, tmp_path):
    calibrator = fit_calibrator(examples[:24], examples[24:36])
    path = tmp_path / "calibrator.json"
    calibrator.save(path)
    loaded = ProbabilityCalibrator.load(path)
    s, f, _ = examples[-1]
    assert loaded.apply(s, f).standings == calibrator.apply(s, f).standings
    with pytest.raises(ValueError, match="target-weekend or future"):
        loaded.apply(*examples[0][:2])
    loaded.fitted_through = s.as_of + timedelta(seconds=1)
    with pytest.raises(ValueError, match="target-weekend or future"):
        loaded.apply(s, f)
    with pytest.raises(ValueError, match="new calibration"):
        calibrator.save(path)
    with pytest.raises(ValueError, match="already been"):
        calibrator.apply(s, calibrator.apply(s, f))


def test_calibration_preserves_scenario_mixture_and_dnf(examples):
    _, f, _ = examples[-1]
    after = transform(f, 0.67)
    ids = [r.driver_id for r in after.standings]
    mixture = np.zeros((len(ids), len(ids)))
    for scenario in after.scenarios:
        rows = {r.driver_id: r for r in scenario.standings}
        mixture += scenario.scenario.probability * np.array(
            [rows[d].position_probabilities for d in ids]
        )
    np.testing.assert_allclose(
        mixture, [r.position_probabilities for r in after.standings], atol=1e-10
    )
    assert {r.driver_id: r.dnf_probability for r in after.standings} == {
        r.driver_id: r.dnf_probability for r in f.standings
    }
    assert after.standings != f.standings


def test_diagnostics_measure_frequencies_and_label_sparse_bins(examples):
    s, f, a = examples[0]
    metrics = reliability([(s, f, a)])
    winner = metrics["events"]["winner"]
    assert winner["ece"] > 0
    assert sum(b["count"] for b in winner["bins"]) == len(f.standings)
    assert all(b["sparse"] for b in winner["bins"])
    assert metrics["events"]["dnf"]["ece"] is None  # Qualifying is not a retirement prediction.
    assert metrics["position_brier"] > 0


def test_engine_opt_in_and_model_family_guard(examples):
    c = fit_calibrator(examples[:24], examples[24:36])
    s, _, _ = examples[-1]
    plain = predict(s, simulations=100)
    revised = predict(s, simulations=100, probability_calibrator=c)
    assert revised.standings == c.apply(s, plain).standings
    wrong = plain.model_copy(update={"forecast_model": "transformer"})
    with pytest.raises(ValueError, match="model family"):
        c.apply(s, wrong)


def multi_period(examples):
    return [
        (s.model_copy(update={"season": 2022 + i // 12}), f, a)
        for i, (s, f, a) in enumerate(examples[24:60])
    ]


def test_conservative_blend_passes_all_periods(examples, tmp_path):
    c = fit_conservative_calibrator(examples[:24], multi_period(examples))
    assert set(c.blends.values()) == {0.1}
    path = tmp_path / "conservative.json"
    c.save(path)
    assert ProbabilityCalibrator.load(path) == c
    f = examples[-1][1]
    assert transform(f, 0.5, 0) == f
    for blend in (0.1, 0.25, 0.5):
        result = transform(f, 0.5, blend)
        matrix = np.array([r.position_probabilities for r in result.standings])
        np.testing.assert_allclose(matrix.sum(0), 1, atol=1e-9)
        np.testing.assert_allclose(matrix.sum(1), 1, atol=1e-9)
    with pytest.raises(ValueError, match="blend"):
        transform(f, 0.5, float("nan"))


def test_conservative_rejects_unstable_or_insufficient_periods(examples):
    selection = multi_period(examples)
    c = fit_conservative_calibrator(examples[:24], selection[:24])
    assert set(c.blends.values()) == {0}
    changed = [(s, f, a.model_copy(deep=True)) for s, f, a in selection]
    for _, _, a in changed[-12:]:
        a.finishing_order.reverse()
    c = fit_conservative_calibrator(examples[:24], changed)
    assert set(c.blends.values()) == {0}
    s, f, _ = examples[-1]
    assert c.apply(s, f).standings == f.standings

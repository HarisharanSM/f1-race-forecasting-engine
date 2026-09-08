import importlib
from copy import deepcopy
from pathlib import Path

import pytest

from f1_forecast.demo import weekend


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("backtest_expanded_collection")


def test_comparison_allows_optional_evidence_only(experiment):
    before = [{"snapshot": weekend()[2].model_dump(mode="json"), "feedback": {"fixed": True}}]
    after = deepcopy(before)
    after[0]["snapshot"]["drivers"][0]["performance"] = {"clean_lap_variability_pct": 0.4}
    after[0]["snapshot"]["notes"].append("Optional evidence")
    assert all(experiment.validate_pair(before, after).values())
    after[0]["snapshot"]["drivers"][0]["name"] = "Unexpected change"
    with pytest.raises(ValueError, match="Non-optional"):
        experiment.validate_pair(before, after)


def test_comparison_rejects_changed_feedback(experiment):
    before = [{"snapshot": weekend()[2].model_dump(mode="json"), "feedback": {"fixed": True}}]
    after = deepcopy(before)
    after[0]["feedback"] = {"fixed": False}
    with pytest.raises(ValueError, match="Outcome labels"):
        experiment.validate_pair(before, after)


def test_paired_intervals_resample_weekends(experiment):
    rows = [
        {
            "event_id": event,
            "before": {"error": 2.0},
            "after": {"error": 1.0},
            "distribution_unchanged": False,
        }
        for event in ("a", "a", "b")
    ]
    result = experiment.aggregate(rows)
    assert result["sessions"] == 3
    assert result["weekends"] == 2
    assert result["delta_after_minus_before"]["error"] == -1
    assert result["paired_weekend_bootstrap_95pct_delta"]["error"] == [-1, -1]
    assert experiment.aggregate([]) == {"sessions": 0}

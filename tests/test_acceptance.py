from copy import deepcopy

import numpy as np
import pytest

from f1_forecast.acceptance import (
    DIRECTIONS,
    AcceptancePolicy,
    acceptance_gate,
    breakdown,
    pair_record,
    protected_metrics,
    selection_split,
)
from f1_forecast.engine import predict
from f1_forecast.ml_data import records_from_json
from f1_forecast.ml_demo import synthetic_history


def evidence():
    rows = []
    base = {m: 0.5 for m in DIRECTIONS}
    for seed in (42, 43, 44):
        for weekend in range(1, 19):
            for session in ("qualifying", "race"):
                rows.append(
                    {
                        "seed": seed,
                        "weekend": f"2024-{weekend:02d}",
                        "season": 2024,
                        "format": "grand_prix",
                        "session": session,
                        "quality": "reduced",
                        "retirement": "without_retirement",
                        "baseline": dict(base),
                        "candidate": {m: v - 0.01 * DIRECTIONS[m] for m, v in base.items()},
                    }
                )
    return rows


@pytest.fixture
def policy():
    return AcceptancePolicy(bootstrap_draws=100)


def test_full_improvement_passes_with_three_disjoint_windows(policy):
    result = acceptance_gate(evidence(), policy)
    assert result["accepted"], result["reasons"]
    windows = [set(r["weekends"]) for r in result["windows"][:3]]
    assert all(len(w) == 6 for w in windows)
    assert not windows[0] & windows[1] and not windows[1] & windows[2]


@pytest.mark.parametrize("metric", list(DIRECTIONS))
def test_any_protected_metric_regression_rejects(metric, policy):
    rows = evidence()
    for row in rows:
        row["candidate"][metric] = row["baseline"][metric] + 0.001 * DIRECTIONS[metric]
    result = acceptance_gate(rows, policy)
    assert not result["accepted"]
    assert any(metric in reason for reason in result["reasons"])


def test_identity_and_insufficient_seed_or_weekend_support_retain_incumbent(policy):
    rows = evidence()
    identity = deepcopy(rows)
    for r in identity:
        r["candidate"] = dict(r["baseline"])
    assert not acceptance_gate(identity, policy)["accepted"]
    result = acceptance_gate([r for r in rows if r["seed"] == 42], policy)
    assert any("training seeds" in r for r in result["reasons"])
    assert not acceptance_gate(rows[:4], policy)["accepted"]


def test_aggregate_gain_cannot_hide_bad_window_or_session(policy):
    rows = evidence()
    for r in rows:
        if r["weekend"] <= "2024-06" and r["session"] == "race":
            r["candidate"]["pairwise_accuracy"] = 0.495
    result = acceptance_gate(rows, policy)
    assert result["overall"]["delta"]["pairwise_accuracy"] > 0
    assert not result["accepted"]
    assert any("window 1" in r and "pairwise_accuracy" in r for r in result["reasons"])


def test_bootstrap_rejects_uncertain_average_gain(policy):
    rows = evidence()
    for r in rows:
        r["candidate"]["winner_brier"] = 0.5 + (-0.06 if int(r["weekend"][-2:]) % 2 else 0.05)
    result = acceptance_gate(rows, policy)
    assert result["overall"]["delta"]["winner_brier"] < 0
    assert any("winner_brier uncertainty" in r for r in result["reasons"])


def test_incomplete_seed_panel_duplicates_and_nan_are_not_accepted(policy):
    rows = evidence()
    assert not acceptance_gate(rows[:-1], policy)["accepted"]
    with pytest.raises(ValueError, match="Duplicate"):
        acceptance_gate(rows + [rows[0]], policy)
    rows[0]["candidate"]["position_interval_score"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        acceptance_gate(rows, policy)


def test_interval_score_and_error_breakdown_use_actual_outcomes(policy):
    record = records_from_json(synthetic_history(1))[-1]
    forecast = predict(record.snapshot, simulations=100)
    metrics = protected_metrics(forecast, record.feedback)
    assert metrics["position_interval_score"] >= metrics["position_interval_width"]
    row = pair_record(record, forecast, forecast, 42)
    report = breakdown([row], policy)
    assert report["overall"]["delta"]["position_interval_score"] == 0
    assert set(report) == {
        "overall",
        "by_format",
        "by_session",
        "by_quality",
        "by_season",
        "by_retirement",
        "by_seed",
    }


def test_partial_weekends_extend_windows_without_using_outcomes(policy):
    records = records_from_json(synthetic_history(28))
    # Remove one race from the newest 18 weekends, as an incomplete classification would.
    records = [
        r for r in records if not (r.snapshot.round == 20 and r.snapshot.session.value == "race")
    ]
    development, selection = selection_split(records, 2, policy)
    assert len({r.snapshot.event_id for r in selection}) > 18
    cutoff = min(r.snapshot.as_of for r in selection)
    assert max(r.feedback.available_at for r in development) < cutoff
    before = [r.snapshot.event_id for r in selection]
    for r in records:
        r.feedback.finishing_order.reverse()
    assert before == [r.snapshot.event_id for r in selection_split(records, 2, policy)[1]]

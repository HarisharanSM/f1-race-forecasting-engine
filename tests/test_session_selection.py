from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pytest
from test_acceptance import evidence
from test_performance import session_data, target

from f1_forecast.acceptance import AcceptancePolicy, acceptance_gate
from f1_forecast.engine import predict
from f1_forecast.ml_data import records_from_json
from f1_forecast.ml_demo import synthetic_history
from f1_forecast.models import DriverPerformance
from f1_forecast.neural import TrainingConfig
from f1_forecast.performance import summarize_session
from f1_forecast.performance_enrichment import enrich_snapshot
from f1_forecast.session_selection import (
    fit_temperature,
    interval_quality_policy,
    match_records,
    rolling_splits,
    select_session_candidates,
)


def quality_data():
    data = session_data()
    laps = [{**lap, "rainfall": False, "track_temperature_c": 30} for lap in data["laps"]]
    data["quality"] = summarize_session(laps, data["drivers"], strict=True)
    return data


def test_fill_missing_preserves_existing_values_weights_and_quality():
    s = target()
    s.drivers[0].performance = DriverPerformance(
        retirement_probability=0.12, weight=0.4, qualifying_teammate_delta_pct=0.123
    )
    before = s.model_dump_json()
    after, audit = enrich_snapshot(s, [quality_data()], quality_aware=True, fill_missing=True)
    assert s.model_dump_json() == before
    p = after.drivers[0].performance
    assert p.race_teammate_delta_pct is not None and audit
    assert p.qualifying_teammate_delta_pct == 0.123
    assert p.retirement_probability == 0.12 and p.weight == 0.4
    repeated, _ = enrich_snapshot(after, [quality_data()], quality_aware=True, fill_missing=True)
    assert repeated.drivers[0].performance == p


def test_disabled_and_future_evidence_not_filled():
    s, data = target(), quality_data()
    s.drivers[0].performance = DriverPerformance(weight=0)
    after, _ = enrich_snapshot(s, [data], quality_aware=True, fill_missing=True)
    assert after.drivers[0].performance == s.drivers[0].performance
    data["ended_at"] = (s.as_of + timedelta(hours=1)).isoformat()
    after, audit = enrich_snapshot(s, [data], quality_aware=True, fill_missing=True)
    assert after == s and not audit
    with pytest.raises(ValueError, match="quality-aware"):
        enrich_snapshot(s, [data], fill_missing=True)


def nominal_evidence():
    rows = evidence()
    for r in rows:
        r["baseline"]["position_interval_coverage"] = 0.9
        r["candidate"]["position_interval_coverage"] = 0.86
    return rows


def test_nominal_rule_allows_less_overcoverage_but_requires_interval_quality():
    rows = nominal_evidence()
    policy = replace(interval_quality_policy(), bootstrap_draws=100)
    assert acceptance_gate(rows, policy)["accepted"]
    assert not acceptance_gate(rows, AcceptancePolicy(bootstrap_draws=100))["accepted"]
    for r in rows:
        r["candidate"]["position_interval_score"] = r["baseline"]["position_interval_score"] + 0.01
    assert not acceptance_gate(rows, policy)["accepted"]


def test_nominal_rule_rejects_undercoverage_and_uncertain_coverage():
    policy = replace(interval_quality_policy(), bootstrap_draws=100)
    rows = nominal_evidence()
    for r in rows:
        r["candidate"]["position_interval_coverage"] = 0.79
    assert any("nominal floor" in s for s in acceptance_gate(rows, policy)["reasons"])
    for r in rows:
        r["candidate"]["position_interval_coverage"] = 0.84 if r["weekend"] != "2024-01" else 0.4
    assert any("coverage uncertainty" in s for s in acceptance_gate(rows, policy)["reasons"])


def test_accuracy_tolerances_are_bounded_and_core_scores_remain_protected():
    rows = nominal_evidence()
    policy = replace(interval_quality_policy(), bootstrap_draws=100)
    for r in rows:
        r["candidate"]["pairwise_accuracy"] = r["baseline"]["pairwise_accuracy"] - 0.001
        r["candidate"]["winner_correct"] = r["baseline"]["winner_correct"] - 0.01
    assert acceptance_gate(rows, policy)["accepted"]
    for r in rows:
        r["candidate"]["position_log_loss"] = r["baseline"]["position_log_loss"] + 0.0001
    assert not acceptance_gate(rows, policy)["accepted"]


def test_rolling_roles_are_disjoint_and_later_labels_do_not_select_windows():
    records = records_from_json(synthetic_history(35))
    following = deepcopy(records)
    for r in following:
        r.snapshot.event_id = "next-" + r.snapshot.event_id
        r.feedback.event_id = r.snapshot.event_id
        r.snapshot.season += 1
        r.snapshot.as_of += timedelta(days=365)
        r.snapshot.session_start += timedelta(days=365)
        r.feedback.available_at += timedelta(days=365)
    records += following
    config = TrainingConfig(min_train_events=8, validation_events=2)
    splits = rolling_splits(records, config)
    seen_tests = set()
    before = []
    for fold in splits:
        sets = [
            {r.snapshot.event_id for r in fold[k]}
            for k in ("development", "calibration", "selection", "later")
        ]
        assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1 :])
        assert not seen_tests & sets[-1]
        seen_tests |= sets[-1]
        for a, b in zip(
            ("development", "calibration", "selection"), ("calibration", "selection", "later")
        ):
            assert max(r.feedback.available_at for r in fold[a]) < min(
                r.snapshot.as_of for r in fold[b]
            )
        before.append(sets)
    for r in records:
        r.feedback.finishing_order.reverse()
    after = [
        [
            {r.snapshot.event_id for r in fold[k]}
            for k in ("development", "calibration", "selection", "later")
        ]
        for fold in rolling_splits(records, config)
    ]
    assert before == after


def test_enrichment_cannot_change_feedback_grid_or_known_measurements():
    records = records_from_json(synthetic_history(1))
    changed = deepcopy(records)
    changed[0].feedback.finishing_order.reverse()
    with pytest.raises(ValueError, match="feedback"):
        match_records(records, changed)
    changed = deepcopy(records)
    changed[-1].snapshot.qualifying_order.reverse()
    with pytest.raises(ValueError, match="non-measurement"):
        match_records(records, changed)


def test_session_selection_rejects_mixed_stages_and_unequal_candidate_panels():
    rows = nominal_evidence()
    with pytest.raises(ValueError, match="separately"):
        select_session_candidates({"one": rows})
    rows = [r for r in rows if r["session"] == "race"]
    with pytest.raises(ValueError, match="identical"):
        select_session_candidates({"one": rows, "two": rows[:-1]})
    changed = deepcopy(rows)
    changed[0]["baseline"]["winner_brier"] += 0.1
    with pytest.raises(ValueError, match="incumbent"):
        select_session_candidates({"one": rows, "two": changed})


def test_calibration_rejects_in_sample_or_mismatched_forecasts():
    records = [
        r for r in records_from_json(synthetic_history(12)) if r.snapshot.session.value == "race"
    ]
    examples = []
    for r in records:
        forecast = predict(r.snapshot, simulations=100)
        forecast.ml_training_cutoff = r.snapshot.as_of
        examples.append((r, 42, forecast))
    with pytest.raises(ValueError, match="future|fitting"):
        fit_temperature(examples)
    for r, _, forecast in examples:
        forecast.ml_training_cutoff = r.snapshot.as_of - timedelta(days=1)
    examples[-1][2].event_id = "wrong-target"
    with pytest.raises(ValueError, match="identities"):
        fit_temperature(examples)


def test_calibration_seeds_do_not_inflate_weekend_support():
    r = records_from_json(synthetic_history(1))[-1]
    forecast = predict(r.snapshot, simulations=100)
    temperature, audit = fit_temperature([(r, seed, forecast) for seed in range(20)])
    assert temperature == 1 and audit["weekends"] == 1

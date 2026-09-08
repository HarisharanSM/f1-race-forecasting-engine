from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pytest

from f1_forecast.demo import weekend
from f1_forecast.joint_performance import apply_joint_effects, comparable_runs, fit_joint_effects
from f1_forecast.measurement_features import (
    LEGACY_MEASUREMENT_NAMES,
    MEASUREMENT_NAMES,
    measurement_matrix,
    without_performance,
)
from f1_forecast.models import Snapshot


def target():
    s = weekend()[2].model_copy(deep=True)
    s.data_mode = "historical_reconstruction"
    s.sources = []
    s.round = 10
    s.drivers = s.drivers[:6]
    s.qualifying_order = [d for d in s.qualifying_order if d in {driver.id for driver in s.drivers}]
    s.teams = [t for t in s.teams if t.id in {d.team_id for d in s.drivers}]
    return Snapshot.model_validate(s.model_dump())


def timing(s, days=7, round_number=9, shifts=None, swaps=False):
    when = s.as_of - timedelta(days=days)
    rows = []
    for i, d in enumerate(s.drivers):
        team = d.team_id
        if swaps and i in (0, 2):
            team = s.drivers[2 if i == 0 else 0].team_id
        team_index = next(j for j, t in enumerate(s.teams) if t.id == team)
        rows.append(
            {
                "driver": d.id,
                "team": team,
                "samples": 6,
                "log_time_pct": 450
                + (-0.3 if i % 2 == 0 else 0.3)
                + (team_index - 1) * 0.8
                + (shifts or {}).get(team, 0),
            }
        )
    return {
        "meta": {
            "season": when.year,
            "round": round_number,
            "code": "FP2",
            "scheduled_start": when.isoformat(),
        },
        "ended_at": (when + timedelta(hours=1)).isoformat(),
        "retrieved_at": (when + timedelta(hours=2)).isoformat(),
        "source_url": "https://example.org/test-timing",
        "source_note": "Synthetic timing fixture",
        "groups": [{"conditions": [i], "rows": deepcopy(rows)} for i in range(12)],
    }


def upgrade(s):
    return {
        "id": "test-upgrade",
        "team_id": s.teams[0].id,
        "introduced_at": (s.as_of - timedelta(days=3)).isoformat(),
        "available_at": (s.as_of - timedelta(days=4)).isoformat(),
        "source_url": "https://example.org/test-upgrade",
    }


def test_joint_model_recovers_teammate_contrast_and_relative_car_pace():
    s = target()
    result = fit_joint_effects(s, [timing(s)])
    d = result["drivers"]
    for a, b in zip(s.drivers[::2], s.drivers[1::2]):
        assert d[b.id]["mean_pct"] - d[a.id]["mean_pct"] == pytest.approx(0.6, abs=0.05)
    t = result["teams"]
    assert t[s.teams[0].id]["mean_pct"] < t[s.teams[-1].id]["mean_pct"]
    assert result["audit"]["data_rank"] < result["audit"]["state_parameters"]
    assert all(v["prior_dependent"] for v in d.values())
    first = s.drivers[0]
    combined = result["combined"][first.id]
    expected_variance = (
        d[first.id]["std_pct"] ** 2
        + t[first.team_id]["std_pct"] ** 2
        + 2 * combined["driver_team_covariance"]
    )
    assert combined["std_pct"] ** 2 == pytest.approx(expected_variance)


def test_upgrade_without_observations_adds_uncertainty_not_speed():
    s = target()
    baseline = fit_joint_effects(s, [timing(s)])
    changed = fit_joint_effects(s, [timing(s)], [upgrade(s)])
    team = s.teams[0].id
    assert changed["teams"][team]["mean_pct"] == pytest.approx(baseline["teams"][team]["mean_pct"])
    assert changed["teams"][team]["std_pct"] > baseline["teams"][team]["std_pct"]
    assert changed["teams"][team]["upgrade_count"] == 1
    assert datetime.fromisoformat(changed["teams"][team]["available_at"]) == datetime.fromisoformat(
        upgrade(s)["available_at"]
    )


def test_post_upgrade_shared_improvement_is_mostly_car_not_driver():
    s = target()
    team = s.teams[0].id
    before = [timing(s, days=14, round_number=8), timing(s)]
    post = timing(s, days=1, round_number=10, shifts={team: -1.0})
    base = fit_joint_effects(s, before, [upgrade(s)])
    result = fit_joint_effects(s, before + [post], [upgrade(s)])
    car_change = result["teams"][team]["mean_pct"] - base["teams"][team]["mean_pct"]
    assert car_change < -0.5
    for d in s.drivers[:2]:
        assert abs(result["drivers"][d.id]["mean_pct"] - base["drivers"][d.id]["mean_pct"]) < abs(
            car_change
        )


def test_season_transition_resets_team_more_than_driver_and_tracks_transfers():
    s = target()
    past = timing(s, days=210, round_number=22)
    result = fit_joint_effects(s, [past])
    links = result["audit"]["transitions"]
    assert {t["retention"] for t in links if t["state"][0] == "driver" and t["previous"]} == {0.8}
    assert {t["retention"] for t in links if t["state"][0] == "team" and t["previous"]} == {0.2}
    assert not any(v["current_season_observed"] for v in result["drivers"].values())
    transferred = fit_joint_effects(s, [past, timing(s, swaps=True)])
    assert transferred["audit"]["transfer_drivers"] == [s.drivers[0].id, s.drivers[2].id]


def test_future_sessions_and_unavailable_upgrades_cannot_change_estimates():
    s = target()
    original = fit_joint_effects(s, [timing(s)])
    future = timing(s, days=-1, round_number=11)
    unseen = upgrade(s)
    unseen["available_at"] = (s.as_of + timedelta(days=1)).isoformat()
    assert fit_joint_effects(s, [timing(s), future], [unseen]) == original
    s.data_mode = "live"
    late = timing(s)
    late["retrieved_at"] = (s.as_of + timedelta(days=1)).isoformat()
    assert not fit_joint_effects(s, [late])["drivers"]


def test_stage_and_missing_support_are_explicit():
    s = target()
    qualifying = timing(s)
    qualifying["meta"]["code"] = "Q"
    assert not fit_joint_effects(s, [qualifying])["drivers"]
    small = timing(s)
    small["groups"] = [{"conditions": [], "rows": small["groups"][0]["rows"][:2]}]
    result = fit_joint_effects(s, [small])
    assert len(result["drivers"]) == 2
    assert not result["teams"]  # Teammate differences cannot observe team pace.
    for row in small["groups"][0]["rows"]:
        row["samples"] = 1
    assert not fit_joint_effects(s, [small])["drivers"]


def test_joint_encoder_is_optional_stage_specific_and_preserves_old_schema():
    s = target()
    result = fit_joint_effects(s, [timing(s)])
    enriched = apply_joint_effects(s, result, [upgrade(s)])
    assert len(enriched.sources) > len(s.sources)
    assert without_performance(enriched).teams[0].car.upgrades == []
    full = measurement_matrix(enriched, 0)
    assert full.shape == (6, len(MEASUREMENT_NAMES))
    old = measurement_matrix(enriched, 0, LEGACY_MEASUREMENT_NAMES)
    np.testing.assert_array_equal(old, full[:, : len(LEGACY_MEASUREMENT_NAMES)])
    assert full[:, len(LEGACY_MEASUREMENT_NAMES) :].any()
    assert not measurement_matrix(without_performance(enriched), 0).any()
    raw = enriched.model_dump()
    raw["session"] = "qualifying"
    raw["qualifying_order"] = None
    raw["starting_grid"] = None
    enriched = Snapshot.model_validate(raw)
    assert not measurement_matrix(enriched, 0).any()


def test_joint_validation_rejects_future_or_wrong_season_estimates():
    s = target()
    enriched = apply_joint_effects(s, fit_joint_effects(s, [timing(s)]))
    data = enriched.model_dump()
    data["drivers"][0]["performance"]["joint_effects"]["race"]["available_at"] = (
        s.as_of + timedelta(days=1)
    )
    with pytest.raises(ValueError, match="newer"):
        Snapshot.model_validate(data)
    data = enriched.model_dump()
    data["drivers"][0]["performance"]["joint_effects"]["race"]["season"] -= 1
    with pytest.raises(ValueError, match="target the snapshot season"):
        Snapshot.model_validate(data)


def test_comparable_run_builder_excludes_unknown_weather_and_interrupted_laps():
    from test_performance import session_data

    data = session_data()
    data["conditioned_laps"] = deepcopy(data["laps"])
    assert comparable_runs([data]) == []
    for lap in data["conditioned_laps"]:
        lap.update(rainfall=False, track_temperature_c=30)
    assert comparable_runs([data])
    for lap in data["conditioned_laps"]:
        lap["pace_usable"] = False
    assert comparable_runs([data]) == []


def test_enrichment_integration_preserves_existing_data_and_feedback(tmp_path, monkeypatch):
    import json

    from test_performance import session_data

    from f1_forecast import performance_enrichment as enrichment
    from f1_forecast.performance_collection import atomic_json

    s = target()
    s.synthetic = False
    data = session_data()
    data["channels"] = {
        "weather_data": [
            {"Time": "PT0S", "TrackTemp": 30, "Rainfall": False},
            {"Time": "PT5M", "TrackTemp": 30, "Rainfall": False},
            {"Time": "PT10M", "TrackTemp": 30, "Rainfall": False},
        ]
    }
    monkeypatch.setattr(enrichment, "load_collection", lambda _: [data])
    source, output, report = [
        tmp_path / name for name in ("input.json", "output.json", "report.json")
    ]
    original = [{"snapshot": s.model_dump(mode="json"), "feedback": {"unchanged": True}}]
    atomic_json(source, original)
    summary = enrichment.enrich_file(source, tmp_path, output, report, joint_effects=True)
    assert summary["snapshots_joint_estimated"] == 1
    changed = json.loads(output.read_text())
    assert changed[0]["feedback"] == original[0]["feedback"]
    loaded = Snapshot.model_validate(changed[0]["snapshot"])
    assert loaded.drivers[0].performance.joint_effects
    assert loaded.drivers[0].race_skill == s.drivers[0].race_skill
    assert json.loads(source.read_text()) == original


def test_joint_evidence_does_not_change_legacy_forecasts():
    from f1_forecast.engine import predict

    s = target()
    enriched = apply_joint_effects(s, fit_joint_effects(s, [timing(s)]), [upgrade(s)])
    assert predict(s, simulations=100).standings == predict(enriched, simulations=100).standings


def test_readable_report_exposes_joint_effects_and_their_evidence_age():
    from f1_forecast.reporting import joint_effect_rows

    s = target()
    assert joint_effect_rows(s) == []
    enriched = apply_joint_effects(s, fit_joint_effects(s, [timing(s)]))
    rows = joint_effect_rows(enriched)
    assert len(rows) == 6
    assert rows[0]["driver"] == s.drivers[0].name
    assert rows[0]["car_effect"]["std_pct"] > 0
    assert datetime.fromisoformat(rows[0]["driver_effect"]["observed_at"]) < s.as_of

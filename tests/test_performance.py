import json
from copy import deepcopy
from datetime import timedelta

import pytest

from f1_forecast.data import DataUnavailable
from f1_forecast.demo import weekend
from f1_forecast.models import CarPerformance, Snapshot
from f1_forecast.performance import assess_laps, has_tyres, summarize_session
from f1_forecast.performance_collection import (
    atomic_json,
    collect_performance,
    digest,
    openf1_session,
    utc,
)
from f1_forecast.performance_enrichment import enrich_file, enrich_snapshot, load_collection


def meta(round_number=1):
    return {
        "season": 2025,
        "round": round_number,
        "event_name": "Test <Circuit>",
        "code": "FP1",
        "session_name": "Practice 1",
        "scheduled_start": f"2025-06-{10 + round_number:02d}T08:00:00+00:00",
    }


def session_data():
    snapshot = weekend()[2]
    drivers = {
        str(d.number): {
            "driver_id": d.id,
            "name": d.name,
            "team_id": d.team_id,
            "team_name": d.team_id,
        }
        for d in snapshot.drivers[:6]
    }
    laps = []
    for index, driver in enumerate(drivers):
        for lap in range(1, 10):
            duration = 90 + index * 0.1 + lap * lap * 0.003
            laps.append(
                {
                    "driver_number": driver,
                    "lap_number": lap,
                    "duration_s": duration,
                    "sector_1_s": duration / 3,
                    "sector_2_s": duration / 3,
                    "sector_3_s": duration / 3,
                    "start_s": 90 * (lap - 1),
                    "end_s": 90 * (lap - 1) + duration,
                    "stint": 1,
                    "compound": "SOFT",
                    "tyre_age_laps": lap,
                    "pit_in": False,
                    "pit_out": False,
                    "deleted": False,
                    "generated": False,
                    "timing_accurate": True,
                    "track_status": "1",
                    "phase": None,
                }
            )
    data = {
        "meta": meta(),
        "provider": "fastf1",
        "source_url": "https://example.org/timing/",
        "source_note": "Test evidence",
        "retrieved_at": "2026-09-01T10:00:00+00:00",
        "ended_at": "2025-06-11T09:00:00+00:00",
        "drivers": drivers,
        "laps": laps,
        "issues": [],
        "channels": {},
        "telemetry": {},
        "collection_version": 1,
    }
    data["laps"] = assess_laps(data["laps"])
    data["quality"] = summarize_session(data["laps"], drivers)
    return data


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("duration_s", None, "missing_lap_time"),
        ("duration_s", float("nan"), "missing_lap_time"),
        ("sector_1_s", 1, "sector_sum_mismatch"),
        ("sector_1_s", None, "missing_sector_time"),
        ("pit_in", True, "pit_lap"),
        ("pit_out", True, "pit_lap"),
        ("deleted", True, "deleted_lap"),
        ("deleted", None, "deletion_status_unknown"),
        ("track_status", "14", "track_interruption"),
        ("track_status", None, "track_status_unknown"),
        ("timing_accurate", None, "timing_unverified"),
        ("generated", True, "generated_lap"),
    ],
)
def test_quality_flags_keep_bad_laps_out_of_pace_without_deleting_them(field, value, reason):
    lap = session_data()["laps"][0]
    lap[field] = value
    rows = assess_laps([lap])
    assert len(rows) == 1
    assert not rows[0]["pace_usable"]
    assert reason in rows[0]["exclusions"]


def test_duplicates_and_slow_laps_are_flagged():
    laps = session_data()["laps"][:9]
    laps.append(deepcopy(laps[0]))
    laps[5]["duration_s"] = 150
    for i in (1, 2, 3):
        laps[5][f"sector_{i}_s"] = 50
    checked = assess_laps(laps)
    assert "duplicate_lap" in checked[0]["exclusions"]
    assert "duplicate_lap" in checked[-1]["exclusions"]
    assert "slow_lap_candidate" in checked[5]["exclusions"]


def test_comparable_pace_and_stint_statistics_preserve_sample_sizes():
    data = session_data()
    first = data["quality"]["drivers"][0]
    assert first["usable_laps"] == 9
    assert first["pace_gap_pct"] == pytest.approx(0)
    assert first["teammate_delta_pct"] < 0
    assert first["teammate_delta_pct_samples"] == 9
    assert first["stints"][0]["observed_pace_slope_s_per_lap"] > 0
    assert "tyre_degradation_s_per_lap" not in first
    # Changing compound removes comparisons rather than inventing a compound correction.
    for lap in data["laps"]:
        if lap["driver_number"] == "1":
            lap["compound"] = "WET"
    first = summarize_session(data["laps"], data["drivers"])["drivers"][0]
    assert first["teammate_delta_pct"] is None
    assert first["pace_gap_pct"] is None


def test_unknown_tyres_and_qualifying_phases_are_not_pooled():
    data = session_data()
    for lap in data["laps"]:
        if lap["driver_number"] == "1":
            lap["phase"] = 2
        if lap["driver_number"] == "2":
            lap["tyre_age_laps"] = None
    quality = summarize_session(data["laps"], data["drivers"])
    assert quality["drivers"][0]["teammate_delta_pct"] is None
    assert quality["drivers"][1]["pace_gap_pct"] is None


@pytest.fixture
def collector(monkeypatch):
    import f1_forecast.performance_collection as module

    calls = []
    monkeypatch.setattr(module, "fastf1_catalog", lambda year, cache: [meta(1), meta(2)])

    def collect(m, cache, telemetry=False):
        calls.append(m)
        return deepcopy(session_data())

    monkeypatch.setattr(module, "fastf1_session", collect)
    return module, calls


def test_collection_is_bounded_resumable_and_checked(collector, tmp_path):
    _, calls = collector
    output, cache = tmp_path / "collection", tmp_path / "cache"
    result = collect_performance([2025], output, cache, provider="fastf1", max_sessions=1)
    assert result["sessions_collected"] == 1 and result["sessions_pending"] == 1
    result = collect_performance(
        [2025], output, cache, provider="fastf1", max_sessions=1, resume=True
    )
    assert result["sessions_collected"] == 2 and len(calls) == 2
    assert "Test &lt;Circuit&gt;" in (output / "quality-report.html").read_text()
    assert len(load_collection(output)) == 2
    with pytest.raises(ValueError, match="exists"):
        collect_performance([2025], output, cache, provider="fastf1")
    with pytest.raises(ValueError, match="settings differ"):
        collect_performance([2025], output, cache, provider="fastf1", telemetry=True, resume=True)
    path = output / "sessions/2025-01-FP1.json"
    data = json.loads(path.read_text())
    data["source_note"] = "altered"
    atomic_json(path, data)
    with pytest.raises(ValueError, match="checksum"):
        collect_performance([2025], output, cache, provider="fastf1", resume=True)
    with pytest.raises(ValueError, match="checksum"):
        load_collection(output)


def test_provider_fallback_and_auth_failure_are_visible(collector, monkeypatch, tmp_path):
    module, _ = collector

    def unavailable(*args, **kwargs):
        raise DataUnavailable("Provider returned 401")

    monkeypatch.setattr(module, "fastf1_session", unavailable)
    monkeypatch.setattr(
        module, "openf1_session", lambda *args: {**session_data(), "provider": "openf1"}
    )
    output = tmp_path / "data"
    result = collect_performance([2025], output, tmp_path / "cache")
    assert result["sessions_collected"] == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(e["provider"] == "openf1" for e in manifest["sessions"].values())
    assert "401" in manifest["sessions"]["2025-01-FP1"]["attempts"][0]["error"]
    assert (
        "unavailable for this batch" in manifest["sessions"]["2025-02-FP1"]["attempts"][0]["error"]
    )


def test_rate_limit_pauses_without_repeated_requests(collector, monkeypatch, tmp_path):
    module, _ = collector
    attempts = []

    def limited(*args, **kwargs):
        attempts.append(1)
        raise DataUnavailable("429 Too Many Requests")

    monkeypatch.setattr(module, "fastf1_session", limited)
    result = collect_performance([2025], tmp_path / "data", tmp_path / "cache", provider="fastf1")
    assert result["sessions_pending"] == 2 and len(attempts) == 1


def target():
    data = weekend()[2].model_dump()
    data.update(synthetic=False, data_mode="historical_reconstruction", sources=[])
    return Snapshot.model_validate(data)


def test_enrichment_is_optional_and_preserves_explicit_inputs():
    snapshot = target()
    original = snapshot.model_dump_json()
    updated, audit = enrich_snapshot(snapshot, [session_data()])
    assert snapshot.model_dump_json() == original
    assert audit and updated.drivers[0].performance.race_teammate_delta_pct < 0
    assert updated.teams[0].car.performance.race_gap_pct is not None
    assert updated.teams[0].car.performance.tyre_degradation_s_per_lap is None
    assert updated.race_dynamics is None
    assert all(s.available_at <= snapshot.as_of for s in updated.sources)
    updated.teams[0].car.performance = CarPerformance(braking=0.8)
    again, _ = enrich_snapshot(updated, [session_data()])
    assert again.teams[0].car.performance == updated.teams[0].car.performance


@pytest.mark.parametrize("change", ["future", "delayed", "old", "other_season", "identity", "team"])
def test_future_stale_or_mismatched_evidence_cannot_enter_snapshot(change):
    snapshot, data = target(), session_data()
    if change == "future":
        data["ended_at"] = (snapshot.as_of + timedelta(hours=1)).isoformat()
    elif change == "delayed":
        data["ended_at"] = (snapshot.as_of - timedelta(hours=1)).isoformat()
    elif change == "old":
        data["ended_at"] = (snapshot.as_of - timedelta(days=31)).isoformat()
    elif change == "other_season":
        data["meta"]["season"] = 2024
    else:
        for row in data["quality"]["drivers"]:
            row["team_id"] = "different_team"
            if change == "identity":
                row["driver_id"] = "different_driver"
    updated, audit = enrich_snapshot(snapshot, [data])
    assert not audit
    assert updated == snapshot


def test_live_snapshots_require_actual_collection_availability():
    snapshot = target().model_copy(update={"data_mode": "live"})
    updated, audit = enrich_snapshot(snapshot, [session_data()])
    assert not audit and updated == snapshot
    data = session_data()
    data["retrieved_at"] = (snapshot.as_of - timedelta(hours=1)).isoformat()
    updated, audit = enrich_snapshot(snapshot, [data])
    assert audit and updated.drivers[0].performance is not None


def test_synthetic_snapshots_rejected():
    with pytest.raises(ValueError, match="synthetic"):
        enrich_snapshot(weekend()[2], [session_data()])


def test_history_enrichment_preserves_feedback_and_emits_audit(collector, tmp_path):
    collection = tmp_path / "collection"
    collect_performance([2025], collection, tmp_path / "cache", provider="fastf1")
    path, output, report = [
        tmp_path / name for name in ("history.json", "enriched.json", "audit.json")
    ]
    original = [{"snapshot": target().model_dump(mode="json"), "feedback": {"unchanged": True}}]
    atomic_json(path, original)
    result = enrich_file(path, collection, output, report)
    assert result["snapshots_enriched"] == 1
    assert json.loads(output.read_text())[0]["feedback"] == original[0]["feedback"]
    assert digest(json.loads(path.read_text())) == result["input_sha256"]
    with pytest.raises(ValueError, match="new output"):
        enrich_file(path, collection, output, report)


def test_openf1_fallback_is_explicitly_provisional():
    from f1_forecast.models import Source

    class Archive:
        def get(self, url, params):
            endpoint = url.rsplit("/", 1)[-1]
            row = {"session_key": 123}
            payloads = {
                "sessions": [
                    {
                        **row,
                        "session_name": "Practice 1",
                        "date_start": meta()["scheduled_start"],
                        "date_end": "2025-06-11T09:00:00Z",
                    }
                ],
                "drivers": [
                    {**row, "driver_number": 1, "full_name": "Driver One", "team_name": "Team"}
                ],
                "laps": [
                    {
                        **row,
                        "driver_number": 1,
                        "lap_number": 2,
                        "lap_duration": 90,
                        "duration_sector_1": 30,
                        "duration_sector_2": 30,
                        "duration_sector_3": 30,
                        "date_start": "2025-06-11T08:05:00Z",
                        "is_pit_out_lap": False,
                    }
                ],
                "stints": [
                    {
                        **row,
                        "driver_number": 1,
                        "lap_start": 1,
                        "lap_end": 10,
                        "compound": "SOFT",
                        "tyre_age_at_start": 0,
                        "stint_number": 1,
                    }
                ],
            }
            return payloads.get(endpoint, []), Source(
                url=url, retrieved_at=utc("2025-06-11T10:00:00Z")
            )

    result = openf1_session(meta(), Archive())
    assert result["laps"][0]["tyre_age_laps"] == 2
    assert result["issues"]
    assert not assess_laps(result["laps"])[0]["pace_usable"]


def test_unknown_compound_and_invalid_age_are_not_tyre_measurements():
    assert has_tyres({"compound": "SOFT", "tyre_age_laps": 1})
    assert not has_tyres({"compound": "UNKNOWN", "tyre_age_laps": 1})
    assert not has_tyres({"compound": "SOFT", "tyre_age_laps": 0})
    assert not has_tyres({"compound": "SOFT", "tyre_age_laps": float("nan")})


def test_resume_refuses_unrelated_directory(tmp_path):
    output = tmp_path / "unrelated"
    output.mkdir()
    atomic_json(output / "unrelated.json", {"keep": True})
    with pytest.raises(ValueError, match="without a collection manifest"):
        collect_performance([2025], output, tmp_path / "cache", resume=True)


def test_cli_reports_total_collection_failure(monkeypatch, capsys, tmp_path):
    from f1_forecast.cli import main

    monkeypatch.setattr("sys.argv", ["f1-forecast", "fetch-performance", "--output", str(tmp_path)])
    monkeypatch.setattr(
        "f1_forecast.performance_collection.collect_performance",
        lambda *args, **kwargs: {
            "sessions_collected": 0,
            "sessions_failed": 1,
            "catalog_errors": 0,
            "report": "quality-report.html",
        },
    )
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2
    assert "No sessions collected" in capsys.readouterr().err

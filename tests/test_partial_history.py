from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from f1_forecast import historical as h
from f1_forecast.models import ActualWeather, Source, Weather


@pytest.fixture
def archive(monkeypatch):
    now = datetime(2026, 9, 6, 8, tzinfo=UTC)
    source = Source(url="https://example.org/archive", retrieved_at=now)
    rows = [
        {
            "Driver": {"driverId": name, "givenName": name, "familyName": "Test"},
            "Constructor": {"constructorId": name, "name": name},
            "number": str(i),
            "position": str(i),
            "status": "Finished",
            "grid": str(i),
        }
        for i, name in enumerate(("a", "b"), 1)
    ]
    schedule, sessions = {}, []
    for rnd, race_start in ((1, now + timedelta(hours=5)), (2, now - timedelta(days=3))):
        schedule[rnd] = {
            "season": "2026",
            "round": str(rnd),
            "raceName": f"Test {rnd}",
            "date": race_start.date().isoformat(),
            "time": race_start.strftime("%H:%M:%SZ"),
            "Circuit": {
                "circuitId": "test",
                "circuitName": "Test",
                "Location": {"lat": "1", "long": "1"},
            },
        }
        for offset, name in ((1, "Qualifying"), (0, "Race")):
            start = race_start - timedelta(days=offset)
            sessions.append(
                {
                    "session_name": name,
                    "meeting_key": rnd,
                    "session_key": rnd * 10 + offset,
                    "date_start": start.isoformat(),
                    "date_end": (start + timedelta(hours=1)).isoformat(),
                }
            )
    tables = {
        "": schedule,
        "results": {n: {"Results": deepcopy(rows), "_sources": [source]} for n in schedule},
        "qualifying": {
            n: {"QualifyingResults": deepcopy(rows), "_sources": [source]} for n in schedule
        },
    }
    monkeypatch.setattr(h, "utcnow", lambda: now)
    monkeypatch.setattr(
        h,
        "Archive",
        lambda _: SimpleNamespace(
            get=lambda url, params: (sessions if url.endswith("/sessions") else [], source),
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(
        h, "season_table", lambda ar, year, endpoint="": (tables[endpoint], [source])
    )
    monkeypatch.setattr(h, "archived_weather", lambda *args: ({}, source))
    monkeypatch.setattr(
        h,
        "observed_weather",
        lambda *args: (ActualWeather(wet_fraction=0, air_temperature_c=25, wind_speed_ms=2), 5),
    )

    def forecast(hourly, src, start):
        issued = start - timedelta(days=1)
        return (
            Weather(
                rain_probability=0.1,
                air_temperature_c=25,
                wind_speed_ms=2,
                issued_at=issued,
                valid_at=start,
                source=src.url,
            ),
            h.projected(src, issued, "Test forecast"),
            0,
        )

    monkeypatch.setattr(h, "forecast_from_archive", forecast)
    return tables


@pytest.mark.parametrize("has_future_results", [False, True])
def test_current_year_keeps_completed_qualifying_but_not_future_race(
    archive, tmp_path, has_future_results
):
    if not has_future_results:
        archive["results"].pop(1)
    report = h.collect_history([2026], tmp_path / "history.json", include_current_season=True)
    assert report["sessions"] == 3
    assert report["qualifying_sessions"] == 2 and report["race_sessions"] == 1
    assert report["exclusions"][0]["event_id"] == "2026-01"
    assert report["exclusions"][0]["session"] == "race"


def test_current_year_rejects_incomplete_qualifying_roster(archive, tmp_path):
    archive["qualifying"][2]["QualifyingResults"].pop()
    report = h.collect_history([2026], tmp_path / "history.json", include_current_season=True)
    assert report["sessions"] == 1
    assert any("Incomplete qualifying" in r["reason"] for r in report["exclusions"])


def test_current_year_requires_opt_in_and_future_year_is_rejected(archive, tmp_path):
    with pytest.raises(ValueError, match="explicitly include"):
        h.collect_history([2026], tmp_path / "current.json")
    with pytest.raises(ValueError, match="explicitly include"):
        h.collect_history([2027], tmp_path / "future.json", include_current_season=True)

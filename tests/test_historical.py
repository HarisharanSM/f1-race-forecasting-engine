from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from f1_forecast.demo import weekend
from f1_forecast.historical import (
    classification,
    forecast_from_archive,
    observed_weather,
    projected,
    ratings,
    result_events,
    season_table,
    session_for,
)
from f1_forecast.models import Session, Snapshot, Source


def row(driver, position, status="Finished"):
    return {
        "Driver": {"driverId": driver, "givenName": driver, "familyName": "Test"},
        "Constructor": {"constructorId": driver, "name": driver},
        "number": str(position),
        "position": str(position),
        "status": status,
    }


def test_archive_preserves_retrieval_time_and_requires_explicit_reconstruction():
    snapshot = weekend()[0].model_dump()
    cutoff = snapshot["as_of"]
    source = Source(url="https://example.org/archive", retrieved_at=cutoff + timedelta(days=30))
    historical = projected(source, cutoff, "Explicit archival availability assumption")
    assert historical.retrieved_at == source.retrieved_at
    snapshot["sources"] = [historical]
    with pytest.raises(ValueError, match="newer"):
        Snapshot.model_validate(snapshot)
    snapshot["data_mode"] = "historical_reconstruction"
    assert Snapshot.model_validate(snapshot).sources[0].retrieved_at > cutoff
    snapshot["sources"] = [source]
    with pytest.raises(ValueError, match="availability"):
        Snapshot.model_validate(snapshot)


def test_paginated_results_keep_all_drivers_and_correct_page_sources():
    source = Source(url="https://example.org/page1", retrieved_at=datetime.now(UTC))
    pages = [
        {
            "MRData": {
                "total": "3",
                "limit": "2",
                "RaceTable": {
                    "Races": [
                        {"season": "2024", "round": "1", "Results": [row("a", 1)]},
                        {"season": "2024", "round": "2", "Results": [row("b", 1)]},
                    ]
                },
            }
        },
        {
            "MRData": {
                "total": "3",
                "limit": "2",
                "RaceTable": {
                    "Races": [{"season": "2024", "round": "2", "Results": [row("c", 2)]}]
                },
            }
        },
    ]

    def get(url, params):
        index = params["offset"] // 2
        return pages[index], source.model_copy(
            update={"url": f"https://example.org/page{index + 1}"}
        )

    races, _ = season_table(SimpleNamespace(get=get), 2024, "results")
    assert len(classification(races[2]["Results"])) == 2
    assert len(races[1]["_sources"]) == 1
    assert [s.url for s in races[2]["_sources"]] == [
        "https://example.org/page1",
        "https://example.org/page2",
    ]


def test_invalid_classification_is_not_silently_renumbered():
    with pytest.raises(ValueError, match="duplicated/missing"):
        classification([row("a", 1), row("b", 1)])


def test_lapped_finishers_are_not_retirements():
    rows = [row("a", 1), row("b", 2, "Lapped"), row("c", 3, "+2 Laps"), row("d", 4, "Retired")]
    _, retired = result_events(rows, [], {"session_key": 1}, Session.RACE)
    assert retired == ["d"]


def test_form_excludes_unavailable_results_and_does_not_read_target_positions():
    now = datetime.now(UTC)
    rows = [row("a", 1), row("b", 2)]
    history = [
        {
            "key": "prior",
            "stage": Session.RACE,
            "available": now + timedelta(days=1),
            "sources": [],
            "clean": rows,
        }
    ]
    drivers, teams, _ = ratings(rows, history, now)
    assert all(d.race_skill == 0.5 for d in drivers)
    assert all(t.car.race_pace == 0.5 for t in teams)
    history[0]["available"] = now - timedelta(days=1)
    before = ratings(rows, history, now)
    changed = [dict(r, position=str(3 - int(r["position"]))) for r in rows]
    assert ratings(changed, history, now) == before


def test_forecast_and_observed_weather_are_separate_and_missing_is_rejected():
    start = datetime(2024, 3, 1, 16, tzinfo=UTC)
    source = Source(url="https://example.org/forecast", retrieved_at=datetime.now(UTC))
    hourly = {
        "time": ["2024-03-01T16:00"],
        "temperature_2m_previous_day1": [20],
        "precipitation_previous_day1": [0],
        "wind_speed_10m_previous_day1": [3],
    }
    weather, provenance, amount = forecast_from_archive(hourly, source, start)
    assert amount == 0 and weather.rain_probability == 0.05
    assert provenance.available_at == start - timedelta(days=1)
    hourly["temperature_2m_previous_day1"] = [None]
    with pytest.raises(ValueError, match="missing"):
        forecast_from_archive(hourly, source, start)
    with pytest.raises(ValueError, match="five"):
        observed_weather(
            [],
            {
                "date_start": start.isoformat(),
                "date_end": (start + timedelta(hours=1)).isoformat(),
                "session_key": 1,
            },
        )


def test_session_match_handles_local_calendar_date_and_requires_unique_race():
    event = {"date": "2024-11-23", "time": "06:00:00Z"}
    sessions = [
        {"session_name": "Race", "meeting_key": 1, "date_start": "2024-11-24T06:00:00Z"},
        {"session_name": "Qualifying", "meeting_key": 1, "date_start": "2024-11-23T06:00:00Z"},
    ]
    assert session_for(event, sessions)["Race"]["meeting_key"] == 1
    with pytest.raises(ValueError, match="unique"):
        session_for(event, sessions + [sessions[0]])

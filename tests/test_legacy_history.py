from datetime import UTC, datetime, timedelta

import pytest

from f1_forecast.data import DataUnavailable
from f1_forecast.legacy_history import (
    persistence_prior,
    scheduled,
    session_pair,
    stream,
    timing_window,
)
from f1_forecast.models import ActualWeather, Source


def test_stream_bom_and_clock_offsets():
    parsed = stream('\ufeff00:00:15.500{"Rainfall":"0"}\r\n01:02:03.250{"Rainfall":"1"}\r\n')
    assert parsed[0][0] == timedelta(seconds=15.5)
    assert parsed[1][0] == timedelta(hours=1, minutes=2, seconds=3.25)
    with pytest.raises(DataUnavailable, match="Malformed"):
        stream("not a timing stream")


def test_local_schedule_dates_convert_to_utc():
    assert scheduled({"StartDate": "2020-07-04T15:00:00", "GmtOffset": "02:00:00"}) == datetime(
        2020, 7, 4, 13, tzinfo=UTC
    )
    assert scheduled({"StartDate": "2021-11-13T16:00:00", "GmtOffset": "-03:00:00"}) == datetime(
        2021, 11, 13, 19, tzinfo=UTC
    )


def test_practice_prior_cannot_use_target_or_unavailable_weather():
    start = datetime(2020, 7, 4, 13, tzinfo=UTC)
    cutoff = start - timedelta(hours=1)
    observations = {
        "end": cutoff - timedelta(hours=3),
        "weather": ActualWeather(wet_fraction=0.4, air_temperature_c=23, wind_speed_ms=2),
        "sources": [Source(url="https://example.org/practice", retrieved_at=datetime.now(UTC))],
    }
    weather, sources, age = persistence_prior(observations, cutoff, start)
    assert weather.rain_probability == 0.4 and weather.air_temperature_c == 23
    assert weather.issued_at < cutoff and age == 3
    assert sources[0].retrieved_at > cutoff
    assert sources[0].available_at < cutoff
    observations["end"] = start
    with pytest.raises(DataUnavailable, match="before prediction cutoff"):
        persistence_prior(observations, cutoff, start)
    observations["end"] = cutoff - timedelta(hours=100)
    with pytest.raises(DataUnavailable, match="four days"):
        persistence_prior(observations, cutoff, start)


def test_2019_clock_uses_actual_heartbeat_not_scheduled_start():
    source = Source(url="https://example.org/status", retrieved_at=datetime.now(UTC))

    class Archive:
        def get(self, url):
            if url.endswith("SessionStatus.jsonStream"):
                return (
                    '00:15:00.000{"Status":"Started"}\r\n01:15:00.000{"Status":"Finished"}',
                    source,
                )
            return '00:00:10.000{"Utc":"2019-03-16T05:45:10Z"}', source

    start, end, _ = timing_window(Archive(), {"Path": "2019/session/"})
    assert start == datetime(2019, 3, 16, 6, tzinfo=UTC)
    assert end == datetime(2019, 3, 16, 7, tzinfo=UTC)


def test_early_sprint_format_uses_friday_qualifying():
    q = {"Name": "Qualifying", "Type": "Qualifying"}
    sprint = {"Name": "Sprint Qualifying", "Type": "Race"}
    gp = {"Name": "Race", "Type": "Race"}
    meeting = {"Sessions": [q, sprint, gp, {"Key": -1}]}
    assert session_pair(meeting, 2021, "sprint") == (q, sprint)
    assert session_pair(meeting, 2021, "grand_prix") == (q, gp)
    with pytest.raises(DataUnavailable, match="Missing"):
        session_pair(meeting, 2023, "sprint")

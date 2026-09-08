import json

import pytest

from f1_forecast.data import DataUnavailable
from f1_forecast.historical_sources import collect_sources, paged_race_data
from f1_forecast.models import Source, utcnow


class FakeArchive:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None):
        self.calls.append(url)
        year = int(url.split("/f1/")[1].split("/")[0])
        race = {"season": str(year), "round": "1", "date": f"{year}-03-10"}
        return {"MRData": {"total": "1", "limit": "100", "RaceTable": {"Races": [race]}}}, Source(
            url=url, retrieved_at=utcnow()
        )


def test_bounded_resume_and_no_false_full_readiness(tmp_path):
    archive = FakeArchive()
    first = collect_sources([2010], tmp_path, max_jobs=1, details=True, archive=archive)
    assert first["attempted_jobs"] == 1 and not first["ready_for_full_backtest"]
    second = collect_sources([2010], tmp_path, max_jobs=10, details=True, archive=archive)
    assert second["collected_jobs"] == 5
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["jobs"]["2010:timing_index"]["status"] == "unsupported_by_provider"
    assert manifest["unresolved_full_inputs"]["archived_pre_session_weather"]
    calls = len(archive.calls)
    collect_sources([2010], tmp_path, max_jobs=1, details=True, archive=archive)
    assert len(archive.calls) == calls
    (tmp_path / "sources/2010-schedule.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        collect_sources([2010], tmp_path, archive=archive)


def test_wrong_race_page_rejected():
    with pytest.raises(DataUnavailable, match="different race"):
        paged_race_data(FakeArchive(), 2010, 2, "laps")

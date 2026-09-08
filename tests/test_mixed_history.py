import importlib
from copy import deepcopy
from pathlib import Path

import pytest

from f1_forecast.performance_collection import atomic_json, digest


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("backtest_mixed_history")
    circuit = {"circuitId": "test", "circuitName": "Test", "Location": {"lat": "0", "long": "0"}}
    event = {
        "date": "2010-03-14",
        "time": "12:00:00Z",
        "Qualifying": {"date": "2010-03-13"},
        "Circuit": circuit,
    }
    rows = [
        {
            "position": str(i + 1),
            "Driver": {"driverId": name, "givenName": name, "familyName": "Driver"},
            "Constructor": {"constructorId": "team", "name": "Team"},
            "status": "Finished",
        }
        for i, name in enumerate(("a", "b"))
    ]
    source = {"url": "https://example.com/results", "retrieved_at": "2026-09-08T00:00:00Z"}
    jobs = {}
    for year in range(2010, 2027):
        for endpoint, field in (
            ("results", "Results"),
            ("qualifying", "QualifyingResults"),
            ("schedule", None),
        ):
            payload = {
                "races": {"1": {**event, **({field: rows} if field else {})}}
                if year == 2010
                else {},
                "sources": [source],
            }
            atomic_json(tmp_path / "sources" / f"{year}-{endpoint}.json", payload)
            jobs[f"{year}:{endpoint}"] = {"sha256": digest(payload)}
    atomic_json(tmp_path / "manifest.json", {"jobs": jobs})
    return runner, tmp_path


def rewrite(runner, root, endpoint, change):
    path = root / "sources" / f"2010-{endpoint}.json"
    payload = runner.read(path)
    change(payload)
    atomic_json(path, payload)
    manifest = runner.read(root / "manifest.json")
    manifest["jobs"][f"2010:{endpoint}"]["sha256"] = digest(payload)
    atomic_json(root / "manifest.json", manifest)


def test_target_result_does_not_enter_own_inputs(archive):
    runner, root = archive
    before, exclusions = runner.assemble(root, [])
    assert not exclusions
    assert len(before) == 2
    assert all(runner.tier(r) == "reduced" for r in before)
    assert "not observed" in before[0]["snapshot"]["weather"]["source"]

    def reverse_result(payload):
        rows = payload["races"]["1"]["Results"]
        rows[0]["position"], rows[1]["position"] = "2", "1"
        rows[0]["status"] = "Collision"

    rewrite(runner, root, "results", reverse_result)
    after, _ = runner.assemble(root, [])
    assert [r["snapshot"] for r in before] == [r["snapshot"] for r in after]
    assert before[1]["feedback"] != after[1]["feedback"]


def test_richer_snapshot_wins_and_invalid_roster_is_excluded(archive):
    runner, root = archive
    rows, _ = runner.assemble(root, [])
    richer = deepcopy(rows[0])
    richer["snapshot"]["notes"] = ["Preserved richer snapshot"]
    merged, _ = runner.assemble(root, [richer])
    assert merged[0] == richer
    assert len(merged) == 2
    rewrite(
        runner,
        root,
        "results",
        lambda p: p["races"]["1"]["Results"][0]["Driver"].update(driverId="c"),
    )
    merged, exclusions = runner.assemble(root, [richer])
    assert len(merged) == 1
    assert "roster mismatch" in exclusions[0]["reason"]


def test_checksum_corruption_is_rejected(archive):
    runner, root = archive
    atomic_json(root / "sources" / "2010-results.json", {})
    with pytest.raises(ValueError, match="checksum"):
        runner.assemble(root, [])

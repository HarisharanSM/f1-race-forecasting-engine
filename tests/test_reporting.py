import csv
import hashlib
import io
import json
from copy import deepcopy

import pytest

pytest.importorskip("torch")

from f1_forecast.historical import sprint_qualifying_rows
from f1_forecast.ml_backtest import backtest_transformer
from f1_forecast.ml_data import records_from_json
from f1_forecast.ml_demo import synthetic_history
from f1_forecast.models import Snapshot
from f1_forecast.neural import TrainingConfig, fit_records
from f1_forecast.reporting import feedback_usage, render_backtest_report


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    rows = synthetic_history(6)
    report = backtest_transformer(
        rows,
        config=TrainingConfig(epochs=1, min_train_events=2),
        simulations=100,
        save_model=tmp_path_factory.mktemp("model"),
    )
    return {"records": rows, "report": report}


def payload(path):
    text = path.read_text()
    return json.loads(
        text.split('<script type="application/json" id="report-data">')[1].split("</script>")[0]
    )


def test_report_has_actuals_conditional_confidence_and_training_history(bundle, tmp_path):
    path = tmp_path / "report.html"
    result = render_backtest_report([bundle], path)
    data = payload(path)
    assert result["sessions"] == 12 and result["evaluated_sessions"] == 6
    assert len({s["event_key"] for s in data["sessions"]}) == 6
    assert sum(s["evaluated"] for s in data["sessions"]) == 6
    training = data["sessions"][0]
    assert not training["forecast"] and training["actual"]
    scored = next(s for s in data["sessions"] if s["evaluated"])
    original = bundle["report"]["results"][0]["forecast"]["scenarios"][0]["standings"][1]
    display = scored["scenarios"][0]["rows"][1]
    assert display["confidence"] == original["position_probabilities"][original["position"] - 1]
    assert display["actual"] == next(
        r["position"] for r in scored["actual"] if r["driver_id"] == display["driver_id"]
    )
    assert data["sessions"][-1]["feedback"]["final_model_role"] == "validation"
    csv_rows = list(
        csv.reader(io.StringIO(path.with_suffix(".csv").read_text(encoding="utf-8-sig")))
    )
    assert csv_rows[0][0] == "Year"
    assert any("Training history" in r[5] for r in csv_rows[1:])


def test_report_rejects_wrong_source_dataset(bundle, tmp_path):
    changed = deepcopy(bundle)
    changed["records"][0]["snapshot"]["notes"].append("different dataset")
    with pytest.raises(ValueError, match="checksum"):
        render_backtest_report([changed], tmp_path / "bad.html")


def test_optional_race_event_frequencies_are_available_in_report(bundle, tmp_path):
    changed = deepcopy(bundle)
    result = next(r for r in changed["report"]["results"] if r["session"] == "race")
    result["forecast"]["scenarios"][0]["race_event_rates"] = {
        "safety_car": 0.3,
        "virtual_safety_car": 0.2,
        "red_flag": 0.05,
    }
    path = tmp_path / "events.html"
    render_backtest_report([changed], path)
    session = next(
        s
        for s in payload(path)["sessions"]
        if s["event_id"] == result["event_id"] and s["session"] == "race"
    )
    assert session["scenarios"][0]["race_event_rates"]["safety_car"] == 0.3
    assert "Optional race events use supplied probabilities" in path.read_text()


def test_feedback_usage_does_not_claim_validation_fits_weights(bundle):
    records = records_from_json(bundle["records"])
    last = feedback_usage(records[-1], bundle["report"], records)
    assert last["training_runs"] == [] and last["validation_runs"] == []
    assert last["final_model_role"] == "validation"
    first = feedback_usage(records[0], bundle["report"], records)
    assert len(first["training_runs"]) == 3
    assert first["final_model_role"] == "training"


def test_names_cannot_break_out_of_embedded_report_data(bundle, tmp_path):
    changed = deepcopy(bundle)
    danger = "</script><img src=x onerror=alert(1)>"
    changed["records"][0]["snapshot"]["drivers"][0]["name"] = danger
    changed["report"]["dataset_sha256"] = hashlib.sha256(
        json.dumps(changed["records"], sort_keys=True).encode()
    ).hexdigest()
    path = tmp_path / "safe.html"
    render_backtest_report([changed], path)
    assert danger not in path.read_text()
    assert danger in [r["driver"] for r in payload(path)["sessions"][0]["actual"]]


def test_sprint_models_and_grand_prix_models_remain_separate(bundle):
    rows = deepcopy(bundle["records"])
    rows[-1]["snapshot"]["race_format"] = "sprint"
    with pytest.raises(ValueError, match="separately"):
        records_from_json(rows)
    model = fit_records(
        records_from_json(bundle["records"][:6]), TrainingConfig(epochs=1, min_train_events=2)
    )
    s = Snapshot.model_validate(bundle["records"][-1]["snapshot"]).model_copy(
        update={"race_format": "sprint"}
    )
    with pytest.raises(ValueError, match="race format"):
        model.check_snapshot(s)


def test_sprint_qualifying_never_uses_sprint_race_positions():
    rows = [
        {
            "number": "1",
            "position": "1",
            "grid": "2",
            "Driver": {"driverId": "a"},
            "Constructor": {"constructorId": "x"},
        },
        {
            "number": "2",
            "position": "2",
            "grid": "1",
            "Driver": {"driverId": "b"},
            "Constructor": {"constructorId": "y"},
        },
    ]
    qualifying = [
        {"session_key": 4, "driver_number": 2, "position": 1},
        {"session_key": 4, "driver_number": 1, "position": 2},
    ]
    result = sprint_qualifying_rows(qualifying, rows, 4)
    assert [r["Driver"]["driverId"] for r in result] == ["b", "a"]
    assert all("grid" not in r for r in result)
    with pytest.raises(ValueError, match="different session"):
        sprint_qualifying_rows(qualifying, rows, 5)

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from f1_forecast.backtest import backtest
from f1_forecast.demo import weekend
from f1_forecast.llm import outcome_messages
from f1_forecast.models import Feedback, Snapshot
from f1_forecast.training import export_training, start_finetune, validate_dataset


def records(count=8):
    result = []
    for i in range(count):
        for snapshot, actual in zip(weekend()[::2], weekend()[1::2]):
            shift = timedelta(days=i * 7)
            data = snapshot.model_dump()
            data.update(event_id=f"test-{i}", synthetic=False)
            for key in ("as_of", "session_start"):
                data[key] += shift
            for key in ("issued_at", "valid_at"):
                data["weather"][key] += shift
            s = Snapshot.model_validate(data)
            data = actual.model_dump()
            data.update(
                event_id=s.event_id, available_at=actual.available_at + shift, synthetic=False
            )
            a = Feedback.model_validate(data)
            result.append(
                {
                    "snapshot": s.model_dump(mode="json"),
                    "feedback": a.model_dump(mode="json"),
                    "messages": outcome_messages(s, []),
                    "metrics": {},
                }
            )
    return result


def test_walk_forward_learns_only_after_results_are_available():
    q, qa, r, ra = weekend()
    result = backtest(
        [
            {"snapshot": r.model_dump(), "feedback": ra.model_dump()},
            {"snapshot": q.model_dump(), "feedback": qa.model_dump()},
        ],
        simulations=100,
    )
    assert [r["model_version"] for r in result["results"]] == [0, 1]
    assert result["results"][0]["adaptive"] == result["results"][0]["frozen_baseline"]


def test_delayed_qualifying_feedback_is_not_used_for_race_forecast():
    q, qa, r, ra = weekend()
    qa = qa.model_copy(update={"available_at": ra.available_at + timedelta(hours=1)})
    result = backtest(
        [
            {"snapshot": q.model_dump(), "feedback": qa.model_dump()},
            {"snapshot": r.model_dump(), "feedback": ra.model_dump()},
        ],
        simulations=100,
    )
    assert [r["model_version"] for r in result["results"]] == [0, 0]


def test_training_split_keeps_weekends_together_and_outcomes_out_of_prompts(tmp_path):
    history = records()
    manifest = export_training(SimpleNamespace(training_records=lambda: history), tmp_path)
    assert manifest["training_examples"] >= 10
    assert not set(manifest["training_events"]) & set(manifest["validation_events"])
    assert validate_dataset(tmp_path) == manifest
    for line in (tmp_path / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        user = json.loads(row["messages"][1]["content"])
        assert "finishing_order" not in user["snapshot"]
        assert "actual_weather" not in user["snapshot"]
        assert list(json.loads(row["messages"][2]["content"])) == ["predicted_order"]


def test_training_rejects_small_and_synthetic_datasets(tmp_path):
    for history in [
        records(2),
        [dict(r, feedback={**r["feedback"], "synthetic": True}) for r in records()],
    ]:
        with pytest.raises(ValueError):
            export_training(SimpleNamespace(training_records=lambda h=history: h), tmp_path)


def test_tampering_is_detected_before_remote_upload(tmp_path):
    history = records()
    export_training(SimpleNamespace(training_records=lambda: history), tmp_path)
    with (tmp_path / "train.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="changed after export"):
        start_finetune(tmp_path, "test-model", client=object())


def test_fine_tuning_submits_both_splits_and_saves_job_without_promoting(tmp_path):
    history = records()
    export_training(SimpleNamespace(training_records=lambda: history), tmp_path)
    uploads, jobs = [], []

    def upload(**kwargs):
        assert kwargs["purpose"] == "fine-tune"
        uploads.append(kwargs["file"].read())
        return SimpleNamespace(id=f"file-{len(uploads)}")

    def create(**kwargs):
        jobs.append(kwargs)
        return SimpleNamespace(id="job-test", status="queued")

    client = SimpleNamespace(
        files=SimpleNamespace(create=upload),
        fine_tuning=SimpleNamespace(jobs=SimpleNamespace(create=create)),
    )
    result = start_finetune(tmp_path, "test-base-model", client)
    assert len(uploads) == 2
    assert jobs[0]["validation_file"] == "file-2"
    assert result["status"] == "queued"
    assert (tmp_path / "job.json").exists()

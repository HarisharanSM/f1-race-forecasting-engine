import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize("joint", [False, True])
def test_historical_summary_labels_the_actual_comparison(monkeypatch, tmp_path, joint):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("backtest_2020_2025")
    metrics = {
        "expected_position_mae": 3.0,
        "pairwise_accuracy": 0.75,
        "winner_correct": 0.4,
        "winner_brier": 0.8,
        "position_log_loss": 2.7,
    }
    summary = {"sessions": 2, "weekends": 1, "before": metrics, "after": metrics}
    runner.render_summary(
        {
            "joint_effects": joint,
            "by_year": {"2020": summary},
            "overall": summary,
            "optional_coverage_by_year": {"2020": 0},
            "method": "Method <test>",
            "limitations": "No early-year joint observations",
        },
        tmp_path,
    )
    page = (tmp_path / "comparison.html").read_text()
    assert "Method &lt;test&gt;" in page
    assert "No early-year joint observations" in page
    if joint:
        assert "Previous learned model error" in page
        assert "With joint estimates" in page
        assert "2025-only joint experiment" in page
    else:
        assert "Baseline error" in page
        assert "With new measurements" in page
        assert "Previous 2025 experiment" in page


def test_resume_rejects_changed_history_or_config(monkeypatch):
    from dataclasses import asdict

    from f1_forecast.neural import TrainingConfig
    from f1_forecast.performance_collection import digest

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("backtest_2020_2025")
    rows, config = [{"snapshot": {"event_id": "old"}}], TrainingConfig()
    previous = {"dataset_sha256": digest(rows), "reproduction_config": asdict(config)}
    with pytest.raises(ValueError, match="Earlier records changed"):
        runner.extend_report(previous, rows, {}, [], config)
    previous["reproduction_config"]["epochs"] += 1
    with pytest.raises(ValueError, match="configuration differs"):
        runner.extend_report(previous, rows, {}, rows, config)

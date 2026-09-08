from datetime import timedelta

import numpy as np
import pytest

from f1_forecast.engine import predict
from f1_forecast.interruptions import estimate, simulate
from f1_forecast.ml_demo import synthetic_history
from f1_forecast.models import Snapshot


def snapshot_with_history():
    s = Snapshot.model_validate(synthetic_history(1)[1]["snapshot"])
    data = s.model_dump()
    time = s.as_of - timedelta(days=400)
    data["race_dynamics"] = {
        "history": [
            {
                "event_id": f"past-{i}",
                "season": s.season - 2,
                "round": i + 1,
                "circuit_id": s.circuit.id,
                "race_format": s.race_format,
                "observed_at": time,
                "available_at": time,
                "source": "Synthetic complete log",
                "complete": True,
                "entrants": [d.id for d in s.drivers],
                "episodes": (
                    [
                        {
                            "kind": "incident",
                            "start_fraction": 0.4,
                            "affected_drivers": [s.drivers[0].id],
                        },
                        {"kind": "safety_car", "start_fraction": 0.4, "duration_fraction": 0.1},
                    ]
                    if i < 4
                    else []
                ),
            }
            for i in range(8)
        ]
    }
    return Snapshot.model_validate(data)


def test_probabilities_count_event_free_races_and_preserve_cooccurrence():
    s = snapshot_with_history()
    assert estimate(s)["probabilities"]["safety_car"] == pytest.approx(0.5)
    latent = np.tile(np.arange(len(s.drivers), dtype=float), (3000, 1))
    a, rates = simulate(latent, s, np.random.default_rng(42))
    assert rates["incident"] == rates["safety_car"]
    assert rates["safety_car"] == pytest.approx(0.5, abs=0.04)
    assert not np.array_equal(a, latent)
    b, _ = simulate(latent, s, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_incomplete_logs_and_future_history_rejected():
    s = snapshot_with_history()
    s.race_dynamics.history[0].complete = False
    assert estimate(s)["status"] == "insufficient_history"
    data = s.model_dump()
    data["race_dynamics"]["history"][0]["available_at"] = s.as_of
    with pytest.raises(ValueError, match="future evidence"):
        Snapshot.model_validate(data)
    data = snapshot_with_history().model_dump()
    data["race_dynamics"]["history"][0].update(season=s.season, round=s.round)
    with pytest.raises(ValueError, match="target weekend"):
        Snapshot.model_validate(data)


def test_opt_in_keeps_dnf_risks_and_sparse_fallback():
    s = snapshot_with_history()
    plain = s.model_copy(update={"race_dynamics": None})
    before = predict(plain, simulations=100)
    after = predict(s, simulations=100)
    assert after.interruption_analysis["status"] == "empirical_episode_resampling"
    assert {r.driver_id: r.dnf_probability for r in before.standings} == {
        r.driver_id: r.dnf_probability for r in after.standings
    }
    s.race_dynamics.history = s.race_dynamics.history[:2]
    assert predict(s, simulations=100).standings == before.standings


def test_vsc_preserves_gaps_without_pitting():
    s = snapshot_with_history()
    s.race_dynamics.pit_opportunity_probability = 0
    for row in s.race_dynamics.history:
        row.episodes = [
            s.race_dynamics.history[0]
            .episodes[-1]
            .model_copy(update={"kind": "virtual_safety_car"})
        ]
    latent = np.tile(np.arange(len(s.drivers), dtype=float), (20, 1))
    after, rates = simulate(latent, s, np.random.default_rng(1))
    np.testing.assert_array_equal(after, latent)
    assert rates["virtual_safety_car"] == 1

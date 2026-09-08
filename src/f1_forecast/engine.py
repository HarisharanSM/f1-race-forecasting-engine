"""Transparent stochastic ranking baseline, not a lap-by-lap physics model."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from uuid import uuid4

import numpy as np

from .models import (
    Feedback,
    Forecast,
    Scenario,
    ScenarioPlan,
    ScenarioResult,
    Session,
    Snapshot,
    Standing,
    utcnow,
    validate_order,
)
from .optional_inputs import (
    effective_car,
    effective_driver,
    explicit_dynamics,
    race_event_effects,
    tyre_rating,
)

FEATURES = ("pace", "driver", "circuit", "tyres", "grid", "wet", "strategy", "cooling")
INITIAL = {
    "qualifying": [2.8, 1.3, 1.2, 0.0, 0.0, 1.6, 0.0, 0.25],
    "race": [2.5, 1.1, 1.0, 0.9, 1.6, 1.5, 0.55, 0.35],
}


class MLForecaster(Protocol):
    metadata: dict
    model_id: str

    def check_snapshot(self, snapshot: Snapshot): ...

    def scenario_parameters(self, snapshot: Snapshot, scenario: Scenario): ...


@dataclass
class LearnerState:
    version: int = 0
    trained_through: str | None = None
    weights: dict = field(default_factory=lambda: {k: list(v) for k, v in INITIAL.items()})
    driver_effects: dict = field(default_factory=dict)
    team_effects: dict = field(default_factory=dict)
    reliability: dict = field(default_factory=dict)
    rain_bias: float = 0.0


def default_scenarios(snapshot: Snapshot, state: LearnerState, maximum: int) -> ScenarioPlan:
    if not 1 <= maximum <= 5:
        raise ValueError("max_scenarios must be between 1 and 5")
    p = float(np.clip(snapshot.weather.rain_probability + state.rain_bias, 0, 1))
    disruption = snapshot.circuit.disruption_probability
    if maximum == 1:
        return ScenarioPlan(
            scenarios=[
                Scenario(
                    name="Weather-weighted outlook",
                    probability=1,
                    wet_fraction=p,
                    disruption_probability=disruption,
                    rationale="Single scenario uses expected wet exposure, an approximation to mixed weather.",
                )
            ]
        )
    # These are mutually exclusive weather regimes; disruptions occur inside each regime.
    regimes = [("Dry", 1 - p, 0.0), ("Wet", p, 0.85)]
    if maximum >= 3:
        regimes = [
            ("Dry", 1 - p, 0.0),
            ("Changing weather", p * 0.55, 0.4),
            ("Sustained rain", p * 0.45, 0.95),
        ]
    return ScenarioPlan(
        scenarios=[
            Scenario(
                name=name,
                probability=prob,
                wet_fraction=wet,
                disruption_probability=min(0.95, disruption + 0.2 * wet),
                rationale="Prior weather regime; probability requires calibration on historical forecasts.",
            )
            for name, prob, wet in regimes
            if prob > 0
        ]
    )


def features(snapshot: Snapshot, wet: float, temperature: float | None = None) -> np.ndarray:
    teams = {t.id: t for t in snapshot.teams}
    circuit = snapshot.circuit
    track_weights = np.array(
        [circuit.straight_weight, circuit.high_speed_weight, circuit.low_speed_weight]
    )
    track_weights /= track_weights.sum()
    grid = snapshot.starting_grid or snapshot.qualifying_order or []
    temperature = snapshot.weather.air_temperature_c if temperature is None else temperature
    heat = max(0, temperature - 25) / 20
    rows = []
    for driver in snapshot.drivers:
        driver = effective_driver(driver)
        team = teams[driver.team_id]
        car = effective_car(team.car)
        race = snapshot.session == Session.RACE
        grid_score = (0.5 - grid.index(driver.id) / (len(grid) - 1)) if race else 0
        rows.append(
            [
                (car.race_pace if race else car.qualifying_pace) - 0.5,
                (driver.race_skill if race else driver.qualifying_skill) - 0.5,
                float(
                    np.dot(
                        track_weights,
                        [car.straight_speed, car.high_speed_cornering, car.low_speed_cornering],
                    )
                )
                - 0.5,
                (tyre_rating(car, driver) - 0.5) * circuit.tyre_stress if race else 0,
                grid_score * (1 - circuit.overtaking * 0.65),
                wet * ((driver.wet_skill + car.wet_performance) / 2 - 0.5),
                ((team.strategy + team.pit_crew) / 2 - 0.5) if race else 0,
                (car.cooling - 0.5) * heat,
            ]
        )
    return np.array(rows)


def scores(
    snapshot: Snapshot, state: LearnerState, wet: float, temperature: float | None = None
) -> np.ndarray:
    base = features(snapshot, wet, temperature) @ np.array(state.weights[snapshot.session])
    for i, driver in enumerate(snapshot.drivers):
        base[i] += state.driver_effects.get(f"{snapshot.session}:{driver.id}", 0)
        base[i] += state.team_effects.get(f"{snapshot.session}:{driver.team_id}", 0)
    return base


def summarize(ids: list[str], distribution: np.ndarray, dnfs: np.ndarray) -> list[Standing]:
    expected = distribution @ np.arange(1, len(ids) + 1)
    order = sorted(range(len(ids)), key=lambda i: (expected[i], -distribution[i, 0], ids[i]))
    rows = []
    for place, i in enumerate(order, 1):
        cumulative = np.cumsum(distribution[i])
        rows.append(
            Standing(
                position=place,
                driver_id=ids[i],
                expected_position=float(expected[i]),
                p10_position=min(len(ids), int(np.searchsorted(cumulative, 0.1)) + 1),
                p90_position=min(len(ids), int(np.searchsorted(cumulative, 0.9)) + 1),
                win_probability=float(distribution[i, 0]),
                podium_probability=float(min(1, distribution[i, :3].sum())),
                dnf_probability=float(dnfs[i]),
                position_probabilities=distribution[i].tolist(),
            )
        )
    return rows


def predict(
    snapshot: Snapshot,
    state: LearnerState | None = None,
    *,
    simulations: int = 5000,
    seed: int = 42,
    max_scenarios: int = 5,
    plan: ScenarioPlan | None = None,
    llm_order: list[str] | None = None,
    llm_model: str | None = None,
    ml_model: MLForecaster | None = None,
    probability_calibrator=None,
) -> Forecast:
    state = state or LearnerState()
    if not 100 <= simulations <= 100000:
        raise ValueError("simulations must be between 100 and 100000 per scenario")
    if not 1 <= max_scenarios <= 5:
        raise ValueError("max_scenarios must be between 1 and 5")
    if state.trained_through and datetime.fromisoformat(state.trained_through) > snapshot.as_of:
        raise ValueError("Model includes future feedback; use a historical model or backtest")
    if ml_model:
        ml_model.check_snapshot(snapshot)
    plan = plan or default_scenarios(snapshot, state, max_scenarios)
    if len(plan.scenarios) > max_scenarios:
        raise ValueError("Scenario plan exceeds max_scenarios")
    ids = [d.id for d in snapshot.drivers]
    if llm_order is not None:
        validate_order(llm_order, ids)
    rng = np.random.default_rng(seed)
    n = len(ids)
    teams = {t.id: t for t in snapshot.teams}
    mixture = np.zeros((n, n))
    mixed_dnfs = np.zeros(n)
    results = []
    for scenario in plan.scenarios:
        if ml_model:
            strength, ml_risks = ml_model.scenario_parameters(snapshot, scenario)
        else:
            strength = scores(snapshot, state, scenario.wet_fraction)
        if llm_order is not None:
            # Bounded prior: cannot replace the numerical model with arbitrary LLM certainty.
            strength += np.array([0.3 * (0.5 - llm_order.index(d) / (n - 1)) for d in ids])
        uncertainty = np.array(
            [0.16 + 0.18 * (1 - effective_driver(d).consistency) for d in snapshot.drivers]
        )
        uncertainty += (
            0.20 * scenario.wet_fraction + min(snapshot.weather.wind_speed_ms, 30) * 0.004
        )
        disrupted = rng.random(simulations) < scenario.disruption_probability
        # Unit Gumbel noise matches the neural pairwise-logistic training objective.
        latent = strength + (
            rng.gumbel(size=(simulations, n))
            if ml_model
            else rng.normal(size=(simulations, n)) * uncertainty
        )
        disruption_noise = rng.normal(size=(simulations, n)) * disrupted[:, None] * 0.22
        from .interruptions import distribution

        learned_interruptions = distribution(snapshot)[1] is not None
        if not explicit_dynamics(snapshot.race_dynamics) and not learned_interruptions:
            latent += disruption_noise
        # A separate RNG isolates new event draws from existing retirement sampling.
        latent, event_rates = race_event_effects(
            latent,
            snapshot,
            np.random.default_rng(np.random.SeedSequence([seed, 9173, len(results)])),
        )
        retired = np.zeros((simulations, n), dtype=bool)
        if snapshot.session == Session.RACE:
            for i, driver in enumerate(snapshot.drivers):
                if ml_model:
                    risk = ml_risks[i]
                else:
                    car = teams[driver.team_id].car
                    counts = state.reliability.get(driver.team_id, [0, 0])
                    mechanical = (20 * (1 - car.reliability) + counts[0]) / (20 + counts[1])
                    incident = (
                        0.015
                        + 0.04 * scenario.wet_fraction
                        + 0.02 * (1 - effective_driver(driver).consistency)
                    )
                    risk = min(0.85, 1 - (1 - mechanical) * (1 - incident))
                evidence = driver.performance
                if (
                    evidence is not None
                    and evidence.retirement_probability is not None
                    and (
                        not ml_model
                        or ml_model.metadata.get("config", {}).get("optional_mode", "legacy")
                        == "legacy"
                    )
                ):
                    risk = (
                        1 - evidence.weight
                    ) * risk + evidence.weight * evidence.retirement_probability
                retired[:, i] = rng.random(simulations) < risk
            # Retirements finish behind finishers, ordered by sampled distance completed.
            latent = np.where(retired, -1000 + rng.random((simulations, n)), latent)
        orders = np.argsort(-latent, axis=1, kind="stable")
        ranks = np.argsort(orders, axis=1)
        matrix = np.stack([np.bincount(ranks[:, i], minlength=n) / simulations for i in range(n)])
        dnfs = retired.mean(axis=0)
        standings = summarize(ids, matrix, dnfs)
        winner = max(standings, key=lambda row: row.win_probability).driver_id
        results.append(
            ScenarioResult(
                scenario=scenario, winner=winner, standings=standings, race_event_rates=event_rates
            )
        )
        mixture += scenario.probability * matrix
        mixed_dnfs += scenario.probability * dnfs
    # Remove floating point drift before schema validation.
    mixture = np.clip(mixture, 0, 1)
    overall = summarize(ids, mixture, np.clip(mixed_dnfs, 0, 1))
    warnings = [
        "Research baseline: probabilities are uncalibrated until evaluated on held-out races."
    ]
    if ml_model and ml_model.metadata.get("config", {}).get("optional_mode") == "learned":
        warnings.append(
            "Optional measurements use a regularized correction selected on earlier "
            "validation weekends. Missing or unsupported evidence has zero correction."
        )
    elif any(d.performance is not None for d in snapshot.drivers) or any(
        t.car.performance is not None for t in snapshot.teams
    ):
        warnings.append(
            "Optional measurements use bounded heuristic mappings, not fitted telemetry models. "
            "Attach pre-cutoff evidence in sources; missing measurements retain existing ratings."
        )
    if ml_model and "validation_selection" in ml_model.metadata:
        warnings.append(
            "Pace temperature and optional influence were selected on earlier validation "
            "forecasts; this does not establish calibrated probabilities on future races."
        )
    if explicit_dynamics(snapshot.race_dynamics):
        warnings.append(
            "Optional race events replace generic disruption noise with approximate timing, "
            "score compression, pit opportunities and restarts; this is not a lap simulation. "
            "Event probabilities are supplied assumptions, not automatically calibrated."
        )
    if snapshot.data_mode == "historical_reconstruction":
        warnings.append(
            "Historical reconstruction: input availability is assumed and source revisions may differ from event-time data."
        )
    if snapshot.synthetic:
        warnings.append("Synthetic demonstration inputs; not a real race forecast.")
    if not snapshot.sources:
        warnings.append("No input source provenance supplied; ratings are user assumptions.")
    if any("PROVISIONAL ROSTER" in note for note in snapshot.notes):
        warnings.append(
            "Roster is provisional; verify entrants and substitutions before relying on it."
        )
    if any("neutral assumptions" in note for note in snapshot.notes):
        warnings.append(
            "Ratings are neutral defaults; add evidence before treating this as an informed forecast."
        )
    if any("unverified synthesis" in note for note in snapshot.notes):
        warnings.append("LLM research is an unverified synthesis; inspect its cited sources.")
    if snapshot.session == Session.RACE and snapshot.starting_grid is None:
        warnings.append(
            "Using actual qualifying order as the grid; apply penalties via starting_grid."
        )
    if llm_order is None:
        warnings.append("LLM disabled; numerical scenario model used.")
    forecast = Forecast(
        id=str(uuid4()),
        event_id=snapshot.event_id,
        session=snapshot.session,
        race_format=snapshot.race_format,
        created_at=utcnow(),
        model_version=state.version,
        forecast_model=(
            "transformer"
            if ml_model.metadata.get("architecture", "field_transformer") == "field_transformer"
            else ml_model.metadata["architecture"]
        )
        if ml_model
        else "heuristic",
        ml_model_id=ml_model.model_id if ml_model else None,
        ml_training_cutoff=ml_model.metadata["trained_through"] if ml_model else None,
        simulations_per_scenario=simulations,
        seed=seed,
        winner=max(overall, key=lambda row: row.win_probability).driver_id,
        standings=overall,
        scenarios=results,
        llm_order=llm_order,
        llm_model=llm_model,
        warnings=warnings,
    )
    if snapshot.tyre_strategy is not None:
        from .tyre_strategy import analyze_strategy

        forecast.tyre_strategy_analysis = analyze_strategy(snapshot.tyre_strategy)
        forecast.warnings.append(
            "Optional tyre strategy analysis is conditional; finishing probabilities are unchanged."
        )
    if snapshot.race_dynamics is not None and snapshot.race_dynamics.history:
        from .interruptions import estimate

        forecast.interruption_analysis = estimate(snapshot)
        forecast.warnings.append(
            "Historical interruption simulation is experimental and not probability-calibrated; see interruption_analysis."
        )
    return (
        probability_calibrator.apply(snapshot, forecast)
        if probability_calibrator is not None
        else forecast
    )


def evaluate(forecast: Forecast, feedback: Feedback) -> dict:
    ids = [s.driver_id for s in forecast.standings]
    validate_order(feedback.finishing_order, ids)
    actual = {driver: i + 1 for i, driver in enumerate(feedback.finishing_order)}
    errors = [abs(s.expected_position - actual[s.driver_id]) for s in forecast.standings]
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]
    winner = feedback.finishing_order[0]
    return {
        "expected_position_mae": float(np.mean(errors)),
        "pairwise_accuracy": sum(actual[a] < actual[b] for a, b in pairs) / len(pairs),
        "winner_correct": forecast.winner == winner,
        "winner_brier": sum(
            (s.win_probability - int(s.driver_id == winner)) ** 2 for s in forecast.standings
        ),
        "position_log_loss": float(
            np.mean(
                [
                    -np.log(max(1e-8, s.position_probabilities[actual[s.driver_id] - 1]))
                    for s in forecast.standings
                ]
            )
        ),
        "position_interval_coverage": float(
            np.mean(
                [
                    s.p10_position <= actual[s.driver_id] <= s.p90_position
                    for s in forecast.standings
                ]
            )
        ),
        "position_interval_width": float(
            np.mean([s.p90_position - s.p10_position for s in forecast.standings])
        ),
    }


def learn(snapshot: Snapshot, feedback: Feedback, state: LearnerState) -> LearnerState:
    """One conservative pairwise-logistic update; incidents are masked from pace learning."""
    from copy import deepcopy

    result = deepcopy(state)
    x = features(
        snapshot, feedback.actual_weather.wet_fraction, feedback.actual_weather.air_temperature_c
    )
    strength = scores(
        snapshot,
        state,
        feedback.actual_weather.wet_fraction,
        feedback.actual_weather.air_temperature_c,
    )
    actual = {driver: i for i, driver in enumerate(feedback.finishing_order)}
    excluded = set(feedback.retired_drivers)
    excluded.update(
        e.driver_id
        for e in feedback.events
        if e.kind in {"mechanical", "collision", "penalty"} and e.driver_id
    )
    gradient = np.zeros(len(FEATURES))
    residual = np.zeros(len(snapshot.drivers))
    counts = np.zeros(len(snapshot.drivers))
    pair_count = 0
    for i, a in enumerate(snapshot.drivers):
        for j in range(i + 1, len(snapshot.drivers)):
            b = snapshot.drivers[j]
            if a.id in excluded or b.id in excluded:
                continue
            probability = 1 / (1 + np.exp(-np.clip(strength[i] - strength[j], -30, 30)))
            error = int(actual[a.id] < actual[b.id]) - probability
            gradient += error * (x[i] - x[j])
            residual[i] += error
            residual[j] -= error
            counts[i] += 1
            counts[j] += 1
            pair_count += 1
    stage = snapshot.session.value
    weights = np.array(state.weights[stage])
    if pair_count:
        weights += 0.12 * gradient / pair_count - 0.002 * (weights - np.array(INITIAL[stage]))
    result.weights[stage] = np.clip(weights, 0, 5).tolist()
    team_residuals: dict[str, list[float]] = {}
    for i, driver in enumerate(snapshot.drivers):
        if counts[i]:
            delta = float(residual[i] / counts[i])
            key = f"{stage}:{driver.id}"
            result.driver_effects[key] = float(
                np.clip(0.995 * state.driver_effects.get(key, 0) + 0.08 * delta, -0.75, 0.75)
            )
            team_residuals.setdefault(driver.team_id, []).append(delta)
    for team, residuals in team_residuals.items():
        key = f"{stage}:{team}"
        result.team_effects[key] = float(
            np.clip(0.995 * state.team_effects.get(key, 0) + 0.04 * np.mean(residuals), -0.75, 0.75)
        )
    if snapshot.session == Session.RACE:
        mechanical_ids = {e.driver_id for e in feedback.events if e.kind == "mechanical"}
        for driver in snapshot.drivers:
            failed, starts = result.reliability.get(driver.team_id, [0, 0])
            result.reliability[driver.team_id] = [
                failed + int(driver.id in mechanical_ids),
                starts + 1,
            ]
    rain_observed = int(feedback.actual_weather.wet_fraction > 0)
    result.rain_bias = float(
        np.clip(
            0.95 * state.rain_bias + 0.05 * (rain_observed - snapshot.weather.rain_probability),
            -0.25,
            0.25,
        )
    )
    result.version += 1
    result.trained_through = feedback.available_at.isoformat()
    return result

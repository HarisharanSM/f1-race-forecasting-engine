"""Bounded, explicitly heuristic adapters for optional pre-session measurements."""

import numpy as np

from .models import Car, Driver, RaceDynamics, Snapshot


def blend(base, estimate, weight):
    return float(np.clip(base + np.clip(weight * (estimate - base), -0.2, 0.2), 0, 1))


def effective_car(car: Car) -> Car:
    evidence = car.performance
    if evidence is None or evidence.weight == 0:
        return car
    updates = {}
    for stage in ("qualifying", "race"):
        gap = getattr(evidence, f"{stage}_gap_pct")
        if gap is not None:
            estimate = np.clip(1 - gap / 5, 0, 1)
            name = f"{stage}_pace"
            updates[name] = blend(getattr(car, name), estimate, evidence.weight)
    # Average overlapping measurements so several aero ratings do not stack bonuses.
    for name, fields in {
        "straight_speed": ("power_delivery", "aerodynamic_efficiency"),
        "high_speed_cornering": ("aerodynamics", "downforce", "medium_speed_cornering"),
        "low_speed_cornering": ("braking", "handling", "medium_speed_cornering"),
    }.items():
        values = [getattr(evidence, f) for f in fields if getattr(evidence, f) is not None]
        if values:
            updates[name] = blend(getattr(car, name), float(np.mean(values)), evidence.weight)
    if evidence.tyre_degradation_s_per_lap is not None:
        estimate = np.clip(1 - evidence.tyre_degradation_s_per_lap / 0.2, 0, 1)
        updates["tyre_management"] = blend(car.tyre_management, estimate, evidence.weight)
    return car.model_copy(update=updates) if updates else car


def effective_driver(driver: Driver) -> Driver:
    evidence = driver.performance
    if evidence is None or evidence.weight == 0:
        return driver
    updates = {}
    for stage in ("qualifying", "race"):
        delta = getattr(evidence, f"{stage}_teammate_delta_pct")
        if delta is not None:
            estimate = np.clip(0.5 - delta / 2, 0, 1)
            name = f"{stage}_skill"
            updates[name] = blend(getattr(driver, name), estimate, evidence.weight)
    if evidence.clean_lap_variability_pct is not None:
        estimate = np.clip(1 - evidence.clean_lap_variability_pct / 2, 0, 1)
        updates["consistency"] = blend(driver.consistency, estimate, evidence.weight)
    return driver.model_copy(update=updates) if updates else driver


def tyre_rating(car: Car, driver: Driver) -> float:
    evidence = driver.performance
    if evidence is not None and evidence.tyre_management is not None:
        return blend(car.tyre_management, evidence.tyre_management, evidence.weight)
    return car.tyre_management


def explicit_dynamics(dynamics: RaceDynamics | None) -> bool:
    return dynamics is not None and any(
        (
            dynamics.safety_car_probability,
            dynamics.virtual_safety_car_probability,
            dynamics.red_flag_probability,
        )
    )


def race_event_effects(latent, snapshot: Snapshot, rng):
    """One sampled period per event type; latent score compression approximates gap loss."""
    d = snapshot.race_dynamics
    if d is not None and d.history:
        from .interruptions import simulate

        result = simulate(latent, snapshot, rng)
        if result is not None:
            return result
    if not explicit_dynamics(d):
        return latent, {}
    count, drivers = latent.shape
    sc = rng.random(count) < d.safety_car_probability
    vsc = rng.random(count) < d.virtual_safety_car_probability
    red = rng.random(count) < d.red_flag_probability
    active = sc | vsc | red
    timing = (
        rng.uniform(0.1, 0.9, count)
        if d.event_lap_fraction is None
        else np.full(count, d.event_lap_fraction)
    )
    duration = np.minimum(d.neutralized_fraction, 1 - timing)
    # Late interruptions leave less racing time to rebuild the accumulated advantage.
    compression = np.where(sc | red, 0.65 * timing + duration, 0)
    compression = np.clip(compression, 0, 0.9)
    center = latent.mean(axis=1, keepdims=True)
    adjusted = center + (latent - center) * (1 - compression[:, None])
    teams = {t.id: t for t in snapshot.teams}
    execution = np.array(
        [
            (teams[driver.team_id].strategy + teams[driver.team_id].pit_crew) / 2
            for driver in snapshot.drivers
        ]
    )
    pit = (rng.random((count, drivers)) < d.pit_opportunity_probability) & active[:, None]
    # Red flags offer a shared reset in this approximation, not a random cheap-stop gain.
    adjusted += pit * (~red[:, None]) * (0.1 + 0.2 * execution) * (1 - timing[:, None])
    restart = (sc | red) & (timing + duration < 1)
    adjusted += rng.normal(size=(count, drivers)) * restart[:, None] * d.restart_variability
    return adjusted, {
        "safety_car": float(sc.mean()),
        "virtual_safety_car": float(vsc.mean()),
        "red_flag": float(red.mean()),
    }

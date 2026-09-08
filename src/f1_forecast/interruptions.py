"""Earlier complete-race episode resampling; conditional score-space gap approximation."""

import numpy as np

KINDS = ("incident", "safety_car", "virtual_safety_car", "red_flag")


def eligible(snapshot):
    dynamics = snapshot.race_dynamics
    if dynamics is None:
        return []
    return [
        r
        for r in dynamics.history
        if r.complete
        and r.race_format == snapshot.race_format
        and r.available_at < snapshot.as_of
        and (r.season, r.round) != (snapshot.season, snapshot.round)
    ]


def distribution(snapshot):
    rows = eligible(snapshot)
    if len(rows) < 8:
        return rows, None
    # Same-circuit evidence is shrunk toward the format-wide empirical distribution.
    local = np.array([r.circuit_id == snapshot.circuit.id for r in rows], dtype=float)
    weights = np.full(len(rows), 1 / len(rows))
    if local.sum():
        fraction = local.sum() / (local.sum() + 8)
        weights = (1 - fraction) * weights + fraction * local / local.sum()
    return rows, weights


def estimate(snapshot):
    rows, weights = distribution(snapshot)
    if weights is None:
        return {"status": "insufficient_history", "complete_races": len(rows)}
    return {
        "status": "empirical_episode_resampling",
        "complete_races": len(rows),
        "circuit_races": sum(r.circuit_id == snapshot.circuit.id for r in rows),
        "probabilities": {
            kind: float(
                sum(
                    w
                    for r, w in zip(rows, weights, strict=True)
                    if any(e.kind == kind for e in r.episodes)
                )
            )
            for kind in KINDS
        },
        "limitations": "Complete race logs including event-free races only. Format-specific empirical "
        "probabilities, circuit-shrunk; not calibrated probabilities. Whole-race sampling preserves "
        "observed event timing and co-occurrence. Score-to-seconds conversion, incident delays, pit "
        "losses and restart spread are explicit assumptions, not learned telemetry. Affected cars are "
        "sampled from earlier involvement rates; no exact crash is predicted. No extra DNF risk is "
        "added, avoiding duplication of the existing retirement model. No wet/dry stratification.",
    }


def simulate(latent, snapshot, rng):
    rows, weights = distribution(snapshot)
    if weights is None:
        return None
    d = snapshot.race_dynamics
    ids = [r.id for r in snapshot.drivers]
    risk = np.array(
        [
            (
                1
                + sum(
                    any(e.kind == "incident" and driver in e.affected_drivers for e in row.episodes)
                    for row in rows
                )
            )
            / (4 + sum(driver in row.entrants for row in rows))
            for driver in ids
        ]
    )
    risk /= risk.sum()
    team = {t.id: t for t in snapshot.teams}
    pit_skill = np.array(
        [(team[r.team_id].pit_crew + team[r.team_id].strategy) / 2 for r in snapshot.drivers]
    )
    adjusted = latent.copy()
    counts = {kind: 0 for kind in KINDS}
    for i, index in enumerate(rng.choice(len(rows), size=len(latent), p=weights)):
        episodes = sorted(rows[index].episodes, key=lambda e: e.start_fraction)
        counts_seen = set()
        pit_used = np.zeros(len(ids), dtype=bool)
        restart_at = []
        compressed_through = -1.0
        for e in episodes:
            counts_seen.add(e.kind)
            end = e.start_fraction + e.duration_fraction
            if e.kind == "incident":
                affected = rng.choice(
                    len(ids), size=min(len(ids), len(e.affected_drivers)), replace=False, p=risk
                )
                adjusted[i, affected] -= d.incident_delay_s / d.seconds_per_score
                continue
            if e.kind in ("safety_car", "red_flag") and e.start_fraction >= compressed_through:
                # Only the accumulated part of the projected gap is compressed.
                leader = adjusted[i].max()
                gaps_s = (leader - adjusted[i]) * d.seconds_per_score
                adjusted[i] = leader - gaps_s * (1 - 0.85 * e.start_fraction) / d.seconds_per_score
                compressed_through = end
                restart_at.append(end)
            elif e.kind in ("safety_car", "red_flag"):
                compressed_through = max(compressed_through, end)
                restart_at[-1] = compressed_through
            # VSC does not bunch the field; red flags do not award random cheap-stop bonuses.
            if e.kind != "red_flag" and e.start_fraction < 1:
                pit = (rng.random(len(ids)) < d.pit_opportunity_probability * pit_skill) & ~pit_used
                saving = max(0, d.green_pit_loss_s - d.neutralized_pit_loss_s)
                adjusted[i] += pit * saving / d.seconds_per_score
                pit_used |= pit
        for end in sorted(set(restart_at)):
            if end < 1:
                adjusted[i] += rng.normal(size=len(ids)) * d.restart_variability * (1 - end)
        for kind in counts_seen:
            counts[kind] += 1
    return adjusted, {k: v / len(latent) for k, v in counts.items()}

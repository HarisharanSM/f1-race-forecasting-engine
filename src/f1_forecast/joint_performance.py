"""Joint, time-aware driver and team effects from pre-cutoff comparable timing runs."""

from collections import defaultdict
from datetime import timedelta

import numpy as np

from .models import (
    CarPerformance,
    CarUpgrade,
    DriverPerformance,
    JointEffectEstimate,
    Source,
    TeamUpgrade,
)
from .performance import has_tyres
from .performance_collection import utc
from .performance_enrichment import evidence_availability

METHOD = "joint-driver-team-v1"


def comparable_runs(sessions):
    """Retain a cohort's driver medians, so teammate and cross-team contrasts share one likelihood."""
    result = []
    for data in sessions:
        cohorts = defaultdict(lambda: defaultdict(list))
        for lap in data.get("conditioned_laps", []):
            if not lap["pace_usable"] or not has_tyres(lap):
                continue
            if lap.get("rainfall") is None or lap.get("track_temperature_c") is None:
                continue
            key = (
                lap["compound"],
                int(lap["tyre_age_laps"] // 3),
                int(lap["start_s"] // 900),
                lap.get("phase"),
                lap["rainfall"],
                int(lap["track_temperature_c"] // 5),
            )
            cohorts[key][lap["driver_number"]].append(lap["duration_s"])
        groups = []
        for key, drivers in sorted(cohorts.items(), key=lambda item: str(item[0])):
            rows = []
            for number, times in sorted(drivers.items()):
                identity = data["drivers"].get(number, {})
                if not identity.get("driver_id") or not identity.get("team_id"):
                    continue
                rows.append(
                    {
                        "driver": identity["driver_id"],
                        "team": identity["team_id"],
                        "log_time_pct": float(100 * np.log(np.median(times))),
                        "samples": len(times),
                    }
                )
            teams = {r["team"] for r in rows}
            if len(rows) >= 4 and len(teams) >= 3:
                groups.append({"conditions": list(key), "rows": rows})
            else:
                # A teammate-only cohort supplies driver contrasts without an unrelated team reference.
                for team in sorted(teams):
                    peers = [r for r in rows if r["team"] == team]
                    if len(peers) >= 2:
                        groups.append({"conditions": list(key), "rows": peers})
        if groups:
            result.append(
                {
                    k: data[k]
                    for k in ("meta", "ended_at", "retrieved_at", "source_url", "source_note")
                }
                | {"groups": groups}
            )
    return result


def eligible_upgrades(snapshot, catalog):
    records = {}
    for team in snapshot.teams:
        for upgrade in team.car.upgrades:
            records[team.id, upgrade.id] = TeamUpgrade(team_id=team.id, **upgrade.model_dump())
    for raw in catalog:
        upgrade = TeamUpgrade.model_validate(raw)
        key = (upgrade.team_id, upgrade.id)
        if key in records and records[key] != upgrade:
            raise ValueError("Conflicting records for the same team upgrade")
        records[key] = upgrade
    return sorted(
        [u for u in records.values() if u.available_at <= snapshot.as_of],
        key=lambda u: (u.introduced_at, u.team_id, u.id),
    )


def fit_joint_effects(snapshot, runs, upgrades=()):
    """Ridge state model: relative log time = driver-season effect + evolving team effect."""
    stage = snapshot.session.value
    codes = {"FP1", "FP2", "FP3", *(["R", "S"] if stage == "race" else ["Q", "SQ"])}
    data = [
        d
        for d in runs
        if snapshot.season - 1 <= d["meta"]["season"] <= snapshot.season
        and utc(d["ended_at"]) < snapshot.as_of
        and snapshot.as_of - utc(d["ended_at"]) <= timedelta(days=730)
        and evidence_availability(d, snapshot) <= snapshot.as_of
        and d["meta"]["code"] in codes
    ]
    data.sort(key=lambda d: (utc(d["ended_at"]), d["meta"]["code"]))
    known = eligible_upgrades(snapshot, upgrades)
    audit = {
        "method": METHOD,
        "stage": stage,
        "cutoff": snapshot.as_of.isoformat(),
        "source_sessions": [],
        "known_upgrades": [u.model_dump(mode="json") for u in known],
        "limitations": "Effects are field-relative percentage log-lap-time contributions, lower is faster. "
        "They depend on zero-centered priors: driver and car effects are not fully identifiable from "
        "fixed teammate lineups. Transfer evidence helps connect teams. Fuel, traffic, setup and "
        "circuit-specific effects remain confounded. Uncertainty is a conditional Gaussian approximation, "
        "not calibrated forecast confidence. Upgrade dates indicate a possible change, not its benefit.",
    }
    if not data:
        return {"drivers": {}, "teams": {}, "combined": {}, "audit": audit}

    def phase(team, season, when):
        return sum(
            u.team_id == team and u.introduced_at.year == season and u.introduced_at <= when
            for u in known
        )

    def team_key(team, meta, when):
        return ("team", team, meta["season"], meta["round"], phase(team, meta["season"], when))

    nodes, node_times, observations = set(), {}, []
    driver_support, team_support = defaultdict(list), defaultdict(list)
    for session_index, session in enumerate(data):
        meta, when = session["meta"], utc(session["meta"]["scheduled_start"])
        for group in session["groups"]:
            obs = []
            cross_team = len({row["team"] for row in group["rows"]}) > 1
            for row in group["rows"]:
                dk, tk = (
                    ("driver", row["driver"], meta["season"]),
                    team_key(row["team"], meta, when),
                )
                nodes.update((dk, tk))
                node_times[tk] = min(node_times.get(tk, when), when)
                obs.append((row, dk, tk))
                driver_support[row["driver"]].append((session_index, row, dk))
                if cross_team:
                    team_support[row["team"]].append((session_index, row, tk))
            observations.append((session_index, obs))
    target_team = {}
    for team in snapshot.teams:
        tk = team_key(
            team.id, {"season": snapshot.season, "round": snapshot.round}, snapshot.session_start
        )
        target_team[team.id] = tk
        nodes.add(tk)
        node_times.setdefault(tk, snapshot.session_start)
    for driver in snapshot.drivers:
        nodes.add(("driver", driver.id, snapshot.season))
    keys = sorted(nodes)
    index = {key: i for i, key in enumerate(keys)}
    p = len(keys)
    prior = np.zeros((p, p))

    def link(key, sd, previous=None, retention=1.0):
        row = np.zeros(p)
        row[index[key]] = 1
        if previous is not None:
            row[index[previous]] = -retention
        prior[:] += np.outer(row, row) / sd**2

    chains = defaultdict(list)
    for key in keys:
        chains[key[:2]].append(key)
    transitions = []
    for (kind, subject), chain in chains.items():
        chain.sort(key=(lambda key: key[2]) if kind == "driver" else lambda key: node_times[key])
        previous = None
        for key in chain:
            if previous is None:
                sd, retention = (1.0 if kind == "driver" else 2.0), 0.0
            elif key[2] != previous[2]:
                gap = key[2] - previous[2]
                sd = (0.35 if kind == "driver" else 1.2) * np.sqrt(gap)
                retention = (0.8 if kind == "driver" else 0.2) ** gap
            else:
                weeks = max(
                    1, (node_times[key] - node_times[previous]).total_seconds() / (7 * 86400)
                )
                changed = max(0, key[4] - previous[4])
                sd, retention = np.sqrt(0.12**2 * weeks + 0.8**2 * changed), 1.0
            link(key, sd, previous, retention)
            transitions.append(
                {
                    "state": list(key),
                    "previous": list(previous) if previous else None,
                    "retention": retention,
                    "innovation_std_pct": float(sd),
                }
            )
            previous = key

    xs, ys, weights = [], [], []
    for session_index, rows in observations:
        session = data[session_index]
        x = np.zeros((len(rows), p))
        y, counts = [], []
        for i, (row, dk, tk) in enumerate(rows):
            x[i, index[dk]], x[i, index[tk]] = 1, 1
            y.append(row["log_time_pct"])
            counts.append(min(6, row["samples"]))
        w = np.asarray(counts, dtype=float)
        w /= w.sum()
        x -= w @ x
        y = np.array(y) - w @ y
        age = (snapshot.as_of - utc(session["ended_at"])).total_seconds() / 86400
        xs.append(x)
        ys.append(y)
        weights.append(w * 2 ** (-age / 90))
    x, y, base_weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    robust = np.ones(len(y))
    noise = 0.5
    for _ in range(4):
        weight = base_weight * robust / noise**2
        precision = prior + (x.T * weight) @ x
        mean = np.linalg.solve(precision, x.T @ (weight * y))
        residual = y - x @ mean
        noise = max(0.25, float(1.4826 * np.median(np.abs(residual - np.median(residual)))))
        robust = np.minimum(1, 1.5 * noise / np.maximum(np.abs(residual), 1e-9))
    weight = base_weight * robust / noise**2
    precision = prior + (x.T * weight) @ x
    covariance = np.linalg.solve(precision, np.eye(p))
    mean = covariance @ (x.T @ (weight * y))
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Nonfinite joint driver/team estimate")

    available_at = max(
        [evidence_availability(d, snapshot) for d in data]
        + [u.available_at for u in known if u.introduced_at <= snapshot.session_start]
    )

    def estimate(subject, key, support, upgrades_count=0):
        evidence = support.get(subject, [])
        sample_count = sum(row["samples"] for _, row, _ in evidence)
        ids = sorted({i for i, _, _ in evidence})
        if sample_count < 6 or not ids:
            return None
        i = index[key]
        if abs(mean[i]) > 20:
            return None
        # An estimate can borrow from previous seasons, but its age remains explicit.
        return JointEffectEstimate(
            mean_pct=float(mean[i]),
            std_pct=float(np.sqrt(max(0, covariance[i, i]))),
            available_at=available_at,
            observed_at=max(utc(data[j]["ended_at"]) for j in ids),
            season=snapshot.season,
            samples=sample_count,
            sessions=len(ids),
            upgrade_count=upgrades_count,
            current_season_observed=any(data[j]["meta"]["season"] == snapshot.season for j in ids),
        ).model_dump(mode="json")

    drivers, teams, combined = {}, {}, {}
    for driver in snapshot.drivers:
        dk, tk = ("driver", driver.id, snapshot.season), target_team[driver.team_id]
        value = estimate(driver.id, dk, driver_support)
        if value is not None:
            drivers[driver.id] = value
            i, j = index[dk], index[tk]
            combined[driver.id] = {
                "mean_pct": float(mean[i] + mean[j]),
                "std_pct": float(
                    np.sqrt(max(0, covariance[i, i] + covariance[j, j] + 2 * covariance[i, j]))
                ),
                "driver_team_covariance": float(covariance[i, j]),
            }
    for team in snapshot.teams:
        tk = target_team[team.id]
        value = estimate(team.id, tk, team_support, tk[4])
        if value is not None:
            teams[team.id] = value
    audit.update(
        source_sessions=[
            {
                "season": d["meta"]["season"],
                "round": d["meta"]["round"],
                "code": d["meta"]["code"],
                "available_at": evidence_availability(d, snapshot).isoformat(),
                "observed_at": d["ended_at"],
                "source_url": d["source_url"],
                "retrieved_at": d["retrieved_at"],
                "source_note": d["source_note"],
            }
            for d in data
        ],
        cohorts=len(observations),
        runs=len(y),
        residual_scale_pct=noise,
        transitions=transitions,
        data_rank=int(np.linalg.matrix_rank(x.T @ x)),
        state_parameters=p,
        transfer_drivers=sorted(
            d for d, rows in driver_support.items() if len({r["team"] for _, r, _ in rows}) > 1
        ),
        assumptions={
            "evidence_half_life_days": 90,
            "driver_season_retention": 0.8,
            "team_season_retention": 0.2,
            "team_weekly_drift_std_pct": 0.12,
            "upgrade_innovation_std_pct": 0.8,
            "priors_fitted_from_outcomes": False,
        },
    )
    return {"drivers": drivers, "teams": teams, "combined": combined, "audit": audit}


def apply_joint_effects(snapshot, result, catalog=()):
    updated = snapshot.model_copy(deep=True)
    stage = snapshot.session.value
    for driver in updated.drivers:
        estimate = result["drivers"].get(driver.id)
        if estimate is not None:
            driver.performance = driver.performance or DriverPerformance()
            driver.performance.joint_effects.setdefault(
                stage, JointEffectEstimate.model_validate(estimate)
            )
    for team in updated.teams:
        estimate = result["teams"].get(team.id)
        if estimate is not None:
            team.car.performance = team.car.performance or CarPerformance()
            team.car.performance.joint_effects.setdefault(
                stage, JointEffectEstimate.model_validate(estimate)
            )
        existing = {u.id for u in team.car.upgrades}
        for upgrade in eligible_upgrades(snapshot, catalog):
            if upgrade.team_id == team.id and upgrade.id not in existing:
                team.car.upgrades.append(
                    CarUpgrade.model_validate(upgrade.model_dump(exclude={"team_id"}))
                )
                existing.add(upgrade.id)
    if result["drivers"] or result["teams"]:
        existing_sources = {(s.url, s.available_at) for s in updated.sources}
        for source in result["audit"]["source_sessions"]:
            available = utc(source["available_at"])
            if (source["source_url"], available) not in existing_sources:
                updated.sources.append(
                    Source(
                        url=source["source_url"],
                        available_at=available,
                        retrieved_at=utc(source["retrieved_at"]),
                        note=source["source_note"],
                        availability_basis=(
                            "Assumed six hours after reconstructed session end; archive may contain later revisions."
                            if snapshot.data_mode == "historical_reconstruction"
                            else "Local archived collection available before forecast cutoff"
                        ),
                    )
                )
                existing_sources.add((source["source_url"], available))
        updated.notes.append(
            "Joint driver-season and evolving team effects from earlier comparable runs. "
            "Separation depends on shrinkage priors; dated upgrades increase flexibility "
            "and uncertainty without assuming a performance gain."
        )
    return type(snapshot).model_validate(updated.model_dump())

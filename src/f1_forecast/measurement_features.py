"""Separate measured evidence and its reliability; never modify the original ratings."""

import numpy as np

FIELDS = (
    ("driver", "teammate_delta_pct", 2.0),
    ("driver", "clean_lap_variability_pct", 2.0),
    ("driver", "tyre_management", 1.0),
    ("car", "gap_pct", 5.0),
    ("car", "braking", 1.0),
    ("car", "aerodynamics", 1.0),
    ("car", "handling", 1.0),
    ("car", "power_delivery", 1.0),
    ("car", "medium_speed_cornering", 1.0),
    ("car", "downforce", 1.0),
    ("car", "aerodynamic_efficiency", 1.0),
    ("car", "tyre_degradation_s_per_lap", 0.2),
)
CHANNELS = (
    "value",
    "present",
    "quality_known",
    "samples",
    "spread",
    "usable_fraction",
    "age",
    "same_event",
    "practice",
    "wet_interaction",
    "circuit_interaction",
)
LEGACY_MEASUREMENT_NAMES = tuple(
    f"{owner}_{field}_{channel}" for owner, field, _ in FIELDS for channel in CHANNELS
)
JOINT_CHANNELS = (
    "value",
    "present",
    "std",
    "samples",
    "sessions",
    "age",
    "current_season",
    "upgrades",
)
MEASUREMENT_NAMES = LEGACY_MEASUREMENT_NAMES + tuple(
    f"{owner}_joint_{channel}" for owner in ("driver", "car") for channel in JOINT_CHANNELS
)


def without_performance(snapshot):
    result = snapshot.model_copy(deep=True)
    for driver in result.drivers:
        driver.performance = None
    for team in result.teams:
        team.car.performance = None
        team.car.upgrades = []
    return result


def measurement_matrix(snapshot, wet, feature_names=None):
    names = tuple(feature_names) if feature_names is not None else MEASUREMENT_NAMES
    if names not in (LEGACY_MEASUREMENT_NAMES, MEASUREMENT_NAMES):
        raise ValueError("Unsupported optional measurement schema")
    teams = {t.id: t for t in snapshot.teams}
    stage = snapshot.session.value
    rows = []
    for driver in snapshot.drivers:
        row = []
        for owner, field, scale in FIELDS:
            evidence = (
                driver.performance if owner == "driver" else teams[driver.team_id].car.performance
            )
            name = f"{stage}_{field}" if field in {"teammate_delta_pct", "gap_pct"} else field
            value = getattr(evidence, name, None)
            if value is None or evidence.weight == 0:
                row.extend([0.0] * len(CHANNELS))
                continue
            quality = evidence.quality.get(name)
            observed = quality.observed_at if quality else evidence.available_at
            if observed is not None and observed > snapshot.as_of:
                raise ValueError("Optional measurement is newer than forecast cutoff")
            age = (
                min(1, (snapshot.as_of - observed).total_seconds() / (30 * 86400))
                if observed
                else 1.0
            )
            scaled = float(np.clip(value / scale, -2, 2))
            row.extend(
                [
                    scaled,
                    1.0,
                    float(quality is not None),
                    min(1, quality.samples / 30) if quality else 0,
                    min(2, quality.spread_pct / scale)
                    if quality and quality.spread_pct is not None
                    else 0,
                    quality.usable_fraction if quality else 0,
                    age,
                    float(
                        quality is not None
                        and quality.source_event
                        in {snapshot.event_id, f"{snapshot.season}-{snapshot.round:02d}"}
                    ),
                    float(quality is not None and quality.source_session.startswith("FP")),
                    scaled * wet,
                    scaled * snapshot.circuit.tyre_stress,
                ]
            )
        for evidence in (driver.performance, teams[driver.team_id].car.performance):
            joint = evidence.joint_effects.get(stage) if evidence and evidence.weight > 0 else None
            if joint is None:
                row.extend([0.0] * len(JOINT_CHANNELS))
                continue
            if max(joint.available_at, joint.observed_at) > snapshot.as_of:
                raise ValueError("Joint effect is newer than forecast cutoff")
            row.extend(
                [
                    float(np.clip(joint.mean_pct / 2, -2, 2)),
                    1.0,
                    min(2, joint.std_pct / 2),
                    min(1, joint.samples / 300),
                    min(1, joint.sessions / 20),
                    min(1, (snapshot.as_of - joint.observed_at).total_seconds() / (365 * 86400)),
                    float(joint.current_season_observed),
                    min(1, joint.upgrade_count / 5),
                ]
            )
        rows.append(row[: len(names)])
    return np.asarray(rows, dtype=np.float32)

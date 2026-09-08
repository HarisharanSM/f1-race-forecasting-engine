"""Quality gates and comparable-lap evidence, independent of provider clients."""

from collections import Counter, defaultdict

import numpy as np

COMPOUNDS = {
    "SOFT",
    "MEDIUM",
    "HARD",
    "INTERMEDIATE",
    "WET",
    "SUPERSOFT",
    "ULTRASOFT",
    "HYPERSOFT",
    "SUPERHARD",
}


def has_tyres(lap):
    age = number(lap.get("tyre_age_laps"))
    return lap.get("compound") in COMPOUNDS and age is not None and age >= 1


def number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def assess_laps(laps, slow_lap_factor=1.07):
    """Keep every row; exclusions apply only to clean-pace feature extraction."""
    if not 1.01 <= slow_lap_factor <= 1.5:
        raise ValueError("slow_lap_factor must be between 1.01 and 1.5")
    counts = Counter((r["driver_number"], r["lap_number"]) for r in laps)
    result, groups = [], defaultdict(list)
    for lap in laps:
        row, flags = dict(lap), []
        duration = number(row.get("duration_s"))
        sectors = [number(row.get(f"sector_{i}_s")) for i in (1, 2, 3)]
        if counts[row["driver_number"], row["lap_number"]] > 1:
            flags.append("duplicate_lap")
        if duration is None or duration <= 0:
            flags.append("missing_lap_time")
        if any(s is None or s <= 0 for s in sectors):
            flags.append("missing_sector_time")
        elif duration is not None and abs(sum(sectors) - duration) > 0.05:
            flags.append("sector_sum_mismatch")
        if row.get("pit_in") or row.get("pit_out"):
            flags.append("pit_lap")
        if row.get("deleted") is not False:
            flags.append("deleted_lap" if row.get("deleted") else "deletion_status_unknown")
        if row.get("timing_accurate") is not True:
            flags.append("timing_unverified")
        if row.get("generated"):
            flags.append("generated_lap")
        if row.get("track_status") != "1":
            flags.append(
                "track_interruption" if row.get("track_status") else "track_status_unknown"
            )
        if number(row.get("start_s")) is None:
            flags.append("missing_start_time")
        row["exclusions"] = flags
        result.append(row)
        if not flags:
            groups[row["driver_number"], row.get("compound")].append(row)
    for group in groups.values():
        if len(group) >= 3:
            reference = float(np.quantile([r["duration_s"] for r in group], 0.1))
            for row in group:
                if row["duration_s"] > reference * slow_lap_factor:
                    row["exclusions"].append("slow_lap_candidate")
    for row in result:
        row["pace_usable"] = not row["exclusions"]
    return result


def summarize_session(laps, drivers, *, strict=False):
    """Measurements retain their sample count and conditions; unknown physics stays unknown."""
    by_driver, cohorts, stint_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for lap in laps:
        by_driver[lap["driver_number"]].append(lap)
        if not lap["pace_usable"]:
            continue
        if strict and (lap.get("rainfall") is None or lap.get("track_temperature_c") is None):
            continue
        age = number(lap.get("tyre_age_laps"))
        if has_tyres(lap):
            key = (lap["compound"], int(age // 3), int(lap["start_s"] // 900), lap.get("phase"))
            if strict:
                key += (lap["rainfall"], int(lap["track_temperature_c"] // 5))
            cohorts[key].append(lap)
            if lap.get("stint") is not None:
                conditions = (
                    (lap["rainfall"], int(lap["track_temperature_c"] // 5))
                    if strict
                    else (None, None)
                )
                stint_groups[
                    lap["driver_number"], lap["stint"], lap["compound"], *conditions
                ].append(lap)
    evidence = {d: {"pace_gaps": [], "teammate_deltas": [], "stints": []} for d in drivers}
    for group in cohorts.values():
        per_driver = defaultdict(list)
        for lap in group:
            if lap["driver_number"] in drivers:
                per_driver[lap["driver_number"]].append(lap["duration_s"])
        medians = {d: float(np.median(times)) for d, times in per_driver.items()}
        teams = {drivers[d].get("team_id") or drivers[d].get("team_name") for d in medians}
        if len(medians) >= 4 and len(teams - {None, ""}) >= 3:
            reference = min(medians.values())
            for d, pace in medians.items():
                evidence[d]["pace_gaps"].append((100 * (pace / reference - 1), len(per_driver[d])))
        for d, pace in medians.items():
            team = drivers[d].get("team_id") or drivers[d].get("team_name")
            peers = [
                p
                for other, p in medians.items()
                if other != d
                and team
                and (drivers[other].get("team_id") or drivers[other].get("team_name")) == team
            ]
            if peers:
                reference = float(np.median(peers))
                count = len(per_driver[d])
                if strict:
                    peer_counts = [
                        len(per_driver[other])
                        for other in medians
                        if other != d
                        and (drivers[other].get("team_id") or drivers[other].get("team_name"))
                        == team
                    ]
                    count = min(count, min(peer_counts))
                evidence[d]["teammate_deltas"].append((100 * (pace / reference - 1), count))
    for (driver, stint, compound, _, _), group in stint_groups.items():
        x = np.array([r["tyre_age_laps"] for r in group])
        y = np.array([r["duration_s"] for r in group])
        if driver not in evidence or len(x) < 6 or np.ptp(x) < 4:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (slope * x + intercept)
        evidence[driver]["stints"].append(
            {
                "stint": stint,
                "compound": compound,
                "laps": len(group),
                "observed_pace_slope_s_per_lap": float(slope),
                "detrended_variability_pct": float(100 * np.std(residuals, ddof=2) / np.mean(y)),
                "limitation": "Observed slope is not fuel/traffic-corrected tyre degradation.",
            }
        )
    summaries = []
    for d, info in drivers.items():
        group, values = by_driver[d], evidence[d]
        summary = {
            **info,
            "driver_number": d,
            "laps": len(group),
            "usable_laps": sum(r["pace_usable"] for r in group),
            "laps_with_tyres": sum(has_tyres(r) for r in group),
            "exclusions": dict(Counter(f for r in group for f in r["exclusions"])),
            "stints": values["stints"],
        }
        for source, field in (
            ("pace_gaps", "pace_gap_pct"),
            ("teammate_deltas", "teammate_delta_pct"),
        ):
            points = values[source]
            count = sum(p[1] for p in points)
            summary[field] = (
                float(np.median([p[0] for p in points])) if count >= (6 if strict else 3) else None
            )
            summary[field + "_samples"] = count
            summary[field + "_cohorts"] = len(points)
            summary[field + "_spread_pct"] = (
                float(np.median(np.abs(np.array([p[0] for p in points]) - summary[field])))
                if summary[field] is not None
                else None
            )
        stints = values["stints"]
        summary["clean_lap_variability_pct"] = (
            float(np.median([r["detrended_variability_pct"] for r in stints])) if stints else None
        )
        summaries.append(summary)
    return {
        "laps": len(laps),
        "usable_laps": sum(r["pace_usable"] for r in laps),
        "laps_with_tyres": sum(has_tyres(r) for r in laps),
        "drivers": summaries,
        "exclusions": dict(Counter(f for r in laps for f in r["exclusions"])),
    }

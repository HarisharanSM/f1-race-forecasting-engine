"""Populate optional measurements only from earlier, identity-matched session evidence."""

import json
from bisect import bisect_right
from datetime import timedelta
from pathlib import Path

import numpy as np

from .models import CarPerformance, DriverPerformance, MeasurementQuality, Session, Snapshot, Source
from .performance import number, summarize_session
from .performance_collection import VERSION, atomic_json, digest, utc


def load_collection(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["settings"]["version"] != VERSION:
        raise ValueError("Unsupported performance collection version")
    sessions = []
    for entry in manifest["sessions"].values():
        if entry["status"] != "collected":
            continue
        path = (directory / entry["file"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Collection file must be inside its directory")
        data = json.loads(path.read_text())
        if digest(data) != entry["sha256"]:
            raise ValueError(f"Collection checksum mismatch: {entry['file']}")
        sessions.append(data)
    return sessions


def evidence_availability(data, snapshot):
    if snapshot.data_mode == "historical_reconstruction":
        return utc(data["ended_at"]) + timedelta(hours=6)
    return max(utc(data["retrieved_at"]), utc(data["ended_at"]))


def quality_sessions(sessions):
    """Recompute stronger cohort summaries without overwriting the archived collection."""
    import pandas as pd

    result = []
    for data in sessions:
        weather = []
        for row in data.get("channels", {}).get("weather_data", []):
            if row.get("Time") is None:
                continue
            time = pd.Timedelta(row["Time"]).total_seconds()
            temperature = number(row.get("TrackTemp"))
            rain = row.get("Rainfall")
            if np.isfinite(time) and temperature is not None and isinstance(rain, bool):
                weather.append((time, temperature, rain))
        weather.sort()
        times = [r[0] for r in weather]
        laps = []
        for original in data["laps"]:
            lap = dict(original)
            start = number(lap.get("start_s"))
            index = bisect_right(times, start) - 1 if start is not None else -1
            if index >= 0 and start - times[index] <= 180:
                lap.update(track_temperature_c=weather[index][1], rainfall=weather[index][2])
            laps.append(lap)
        result.append(
            {
                **data,
                "conditioned_laps": laps,
                "quality": summarize_session(laps, data["drivers"], strict=True),
            }
        )
    return result


def enrich_snapshot(snapshot, sessions, *, quality_aware=False):
    if snapshot.synthetic:
        raise ValueError("Real collected measurements cannot enrich synthetic snapshots")
    stage = "race" if snapshot.session == Session.RACE else "qualifying"
    codes = {"FP1", "FP2", "FP3", *(["R", "S"] if stage == "race" else ["Q", "SQ"])}
    candidates = sorted(
        [
            data
            for data in sessions
            if data["meta"]["season"] == snapshot.season
            and data["meta"]["code"] in codes
            and utc(data["ended_at"]) < snapshot.as_of
            and snapshot.as_of - utc(data["ended_at"]) <= timedelta(days=30)
            and evidence_availability(data, snapshot) <= snapshot.as_of
        ],
        key=lambda d: utc(d["ended_at"]),
        reverse=True,
    )
    updated = snapshot.model_copy(deep=True)
    audit, used = [], {}

    def quality(data, rows, field, samples, spread=None, cohorts=1):
        return MeasurementQuality(
            samples=samples,
            cohorts=max(1, cohorts),
            spread_pct=spread,
            usable_fraction=sum(r["usable_laps"] for r in rows)
            / max(1, sum(r["laps"] for r in rows)),
            observed_at=utc(data["ended_at"]),
            source_session=data["meta"]["code"],
            source_event=f"{data['meta']['season']}-{data['meta']['round']:02d}",
        )

    def remember(data, subject, values, sample_counts):
        key = (data["meta"]["season"], data["meta"]["round"], data["meta"]["code"])
        used[key] = data
        audit.append(
            {
                "subject": subject,
                "session": list(key),
                "measurements": values,
                "samples": sample_counts,
                "available_at": evidence_availability(data, snapshot).isoformat(),
                "limitation": "Compound/tyre-age/time-window matched pace proxy; fuel and traffic remain uncertain.",
            }
        )

    for driver in updated.drivers:
        if driver.performance is not None:
            continue
        for data in candidates:
            # IDs and team membership must both match; numbers alone are not identities.
            matching = [
                r
                for r in data["quality"]["drivers"]
                if r["driver_id"] == driver.id and r["team_id"] == driver.team_id
            ]
            if len(matching) != 1:
                continue
            row, values = matching[0], {}
            delta = row["teammate_delta_pct"]
            if delta is not None and -20 <= delta <= 20:
                values[f"{stage}_teammate_delta_pct"] = delta
            variability = row["clean_lap_variability_pct"]
            if variability is not None and 0 <= variability <= 20:
                values["clean_lap_variability_pct"] = variability
            if not values:
                continue
            qualities = {}
            if quality_aware:
                for name in values:
                    if name.endswith("teammate_delta_pct"):
                        qualities[name] = quality(
                            data,
                            [row],
                            name,
                            row["teammate_delta_pct_samples"],
                            row["teammate_delta_pct_spread_pct"],
                            row["teammate_delta_pct_cohorts"],
                        )
                    else:
                        samples = sum(s["laps"] for s in row["stints"])
                        qualities[name] = quality(
                            data, [row], name, samples, cohorts=len(row["stints"])
                        )
            driver.performance = DriverPerformance(
                **values,
                weight=0.25,
                available_at=evidence_availability(data, snapshot),
                quality=qualities,
            )
            remember(
                data,
                driver.id,
                values,
                {
                    "matched_laps": row["teammate_delta_pct_samples"],
                    "variability_laps": sum(s["laps"] for s in row["stints"]),
                },
            )
            break
    for team in updated.teams:
        if team.car.performance is not None:
            continue
        for data in candidates:
            members = [
                r
                for r in data["quality"]["drivers"]
                if r["team_id"] == team.id and r["pace_gap_pct"] is not None
            ]
            if len(members) < 2:
                continue
            value = float(np.median([r["pace_gap_pct"] for r in members]))
            if not -20 <= value <= 20:
                continue
            values = {f"{stage}_gap_pct": value}
            qualities = {}
            if quality_aware:
                qualities[f"{stage}_gap_pct"] = quality(
                    data,
                    members,
                    f"{stage}_gap_pct",
                    min(r["pace_gap_pct_samples"] for r in members),
                    max(r["pace_gap_pct_spread_pct"] for r in members),
                    min(r["pace_gap_pct_cohorts"] for r in members),
                )
            team.car.performance = CarPerformance(
                **values,
                weight=0.25,
                available_at=evidence_availability(data, snapshot),
                quality=qualities,
            )
            remember(
                data,
                team.id,
                values,
                {"matched_laps": sum(r["pace_gap_pct_samples"] for r in members)},
            )
            break
    for data in used.values():
        reconstructed = snapshot.data_mode == "historical_reconstruction"
        updated.sources.append(
            Source(
                url=data["source_url"],
                retrieved_at=utc(data["retrieved_at"]),
                available_at=evidence_availability(data, snapshot),
                availability_basis=(
                    "Assumed six hours after reconstructed session end; archive may contain later revisions."
                    if reconstructed
                    else "Local archived collection available before forecast cutoff"
                ),
                note=data["source_note"],
            )
        )
    if audit:
        updated.notes.append(
            "Optional measured pace proxies from prior sessions within 30 days; compound, tyre age "
            "and 15-minute windows matched. Weight 0.25 is heuristic. Fuel, traffic and setup are "
            "not fully controlled; observed stint slopes are not used as physical tyre degradation."
        )
        if quality_aware:
            updated.notes.append(
                "Quality-aware evidence: at least six matched pace laps; rain and "
                "5C track-temperature cohorts. Quality statistics are evidence "
                "descriptors, not calibrated confidence. Fuel load remains unknown."
            )
    return Snapshot.model_validate(updated.model_dump()), audit


def enrich_file(
    input_path,
    collection,
    output,
    report,
    *,
    quality_aware=False,
    joint_effects=False,
    upgrades=None,
):
    from .joint_performance import apply_joint_effects, comparable_runs, fit_joint_effects
    from .models import TeamUpgrade

    if upgrades is not None and not joint_effects:
        raise ValueError("Upgrade records require --joint-effects")
    catalog = (
        [TeamUpgrade.model_validate(u) for u in json.loads(Path(upgrades).read_text())]
        if upgrades
        else []
    )
    quality_aware = quality_aware or joint_effects
    paths = [Path(p).resolve() for p in (input_path, output, report)]
    if len(set(paths)) != 3 or any(p.exists() for p in paths[1:]):
        raise ValueError("Choose distinct, new output and evidence-report paths")
    data = json.loads(paths[0].read_text())
    sessions = load_collection(collection)
    if quality_aware:
        sessions = quality_sessions(sessions)
    runs = comparable_runs(sessions) if joint_effects else []
    rows = data if isinstance(data, list) else [{"snapshot": data}]
    changed, audits = [], []
    for row in rows:
        snapshot, evidence = enrich_snapshot(
            Snapshot.model_validate(row["snapshot"]), sessions, quality_aware=quality_aware
        )
        joint = None
        if joint_effects:
            joint = fit_joint_effects(snapshot, runs, catalog)
            snapshot = apply_joint_effects(snapshot, joint, catalog)
        changed.append({**row, "snapshot": snapshot.model_dump(mode="json")})
        audits.append(
            {
                "event_id": snapshot.event_id,
                "session": snapshot.session.value,
                "evidence": evidence,
                **({"joint": joint} if joint_effects else {}),
            }
        )
    result = changed if isinstance(data, list) else changed[0]["snapshot"]
    summary = {
        "snapshots": len(rows),
        "snapshots_enriched": sum(bool(a["evidence"]) for a in audits),
        "input_sha256": digest(data),
        "output_sha256": digest(result),
        "collection": str(Path(collection).resolve()),
        "quality_aware": quality_aware,
        "joint_effects": joint_effects,
        "snapshots_joint_estimated": sum(
            bool(a.get("joint", {}).get("drivers") or a.get("joint", {}).get("teams"))
            for a in audits
        ),
        "results": audits,
        "limitations": "Measured timing proxies, not isolated driver/car skill or calibrated probabilities. "
        "Historical publication times remain reconstructed. Existing optional objects are preserved.",
    }
    atomic_json(paths[1], result)
    atomic_json(paths[2], summary)
    return {k: v for k, v in summary.items() if k != "results"}

"""Resumable session/lap/stint collection with explicit quality and source provenance."""

import csv
import hashlib
import html
import io
import json
import logging
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .data import DataUnavailable
from .historical import OPENF1, Archive, season_table
from .models import utcnow
from .performance import assess_laps, number, summarize_session

SESSION_CODES = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Race": "R",
    "Sprint": "S",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SQ",
}
ALL_SESSIONS = ("FP1", "FP2", "FP3", "Q", "SQ", "S", "R")
VERSION = 1


class ProviderWarnings(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def utc(value):
    parsed = datetime.fromisoformat(str(value))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def session_id(meta):
    return f"{meta['season']}-{meta['round']:02d}-{meta['code']}"


def elapsed(value):
    return number(value.total_seconds()) if hasattr(value, "total_seconds") else None


def fastf1_catalog(season, cache):
    import fastf1
    import pandas as pd

    Path(cache).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    result = []
    for _, event in schedule.iterrows():
        for i in range(1, 6):
            name = event[f"Session{i}"]
            date = event[f"Session{i}DateUtc"]
            if name not in SESSION_CODES or pd.isna(date):
                continue
            code = SESSION_CODES[name]
            if season <= 2022 and name == "Sprint Qualifying":
                code = "S"
            result.append(
                {
                    "season": season,
                    "round": int(event["RoundNumber"]),
                    "event_name": str(event["EventName"]),
                    "code": code,
                    "session_name": name,
                    "scheduled_start": utc(date).isoformat(),
                }
            )
    return result


def openf1_catalog(season, archive):
    sessions, _ = archive.get(f"{OPENF1}/sessions", {"year": season})
    races, _ = season_table(archive, season)
    if not isinstance(sessions, list):
        raise DataUnavailable("OpenF1 returned a non-list session catalog")
    result = []
    for rnd, event in races.items():
        start = utc(event["date"] + "T" + event["time"])
        matches = [
            s
            for s in sessions
            if s.get("session_name") == "Race"
            and abs((utc(s["date_start"]) - start).total_seconds()) < 30 * 3600
        ]
        if len(matches) != 1:
            continue
        for s in sessions:
            if (
                s.get("meeting_key") != matches[0]["meeting_key"]
                or s.get("session_name") not in SESSION_CODES
            ):
                continue
            result.append(
                {
                    "season": season,
                    "round": rnd,
                    "event_name": event["raceName"],
                    "code": SESSION_CODES[s["session_name"]],
                    "session_name": s["session_name"],
                    "scheduled_start": s["date_start"],
                    "session_key": s["session_key"],
                }
            )
    return result


def fastf1_session(meta, cache, telemetry=False):
    import fastf1
    import pandas as pd

    fastf1.Cache.enable_cache(str(cache))
    session = fastf1.get_session(meta["season"], meta["round"], meta["session_name"])
    captured = ProviderWarnings()
    logger = logging.getLogger("fastf1")
    logger.addHandler(captured)
    try:
        session.load(laps=True, telemetry=telemetry, weather=True, messages=True)
    finally:
        logger.removeHandler(captured)
    if session.laps.empty:
        raise DataUnavailable("FastF1 returned no laps")
    drivers = {}
    for _, r in session.results.iterrows():
        drivers[str(r["DriverNumber"])] = {
            "driver_id": str(r["DriverId"]) or None,
            "name": str(r["FullName"]),
            "team_id": str(r["TeamId"]) or None,
            "team_name": str(r["TeamName"]),
        }
    phases = {}
    if meta["code"] in {"Q", "SQ"}:
        for phase, frame in enumerate(session.laps.split_qualifying_sessions(), 1):
            if frame is not None:
                phases.update(
                    {
                        (str(r["DriverNumber"]), int(r["LapNumber"])): phase
                        for _, r in frame.iterrows()
                        if pd.notna(r["LapNumber"])
                    }
                )
    laps, rejected = [], []
    for _, r in session.laps.iterrows():
        d, lap = str(r["DriverNumber"]), number(r["LapNumber"])
        if d not in drivers or lap is None or lap < 1 or not lap.is_integer():
            rejected.append("Missing/unknown driver or invalid lap number")
            continue
        flag = lambda key, row=r: bool(row[key]) if pd.notna(row.get(key)) else None
        text = lambda key, row=r: (
            str(row[key]) if pd.notna(row.get(key)) and row[key] != "" else None
        )
        laps.append(
            {
                "driver_number": d,
                "lap_number": int(lap),
                "duration_s": elapsed(r["LapTime"]),
                **{f"sector_{i}_s": elapsed(r[f"Sector{i}Time"]) for i in (1, 2, 3)},
                "start_s": elapsed(r["LapStartTime"]),
                "end_s": elapsed(r["Time"]),
                "stint": number(r["Stint"]),
                "compound": text("Compound"),
                "tyre_age_laps": number(r["TyreLife"]),
                "pit_in": pd.notna(r["PitInTime"]),
                "pit_out": pd.notna(r["PitOutTime"]),
                "deleted": flag("Deleted"),
                "timing_accurate": flag("IsAccurate"),
                "generated": flag("FastF1Generated"),
                "track_status": text("TrackStatus"),
                "phase": phases.get((d, int(lap))),
                "speed_trap_kmh": number(r["SpeedST"]),
            }
        )
    status = session.session_status
    ends = [
        elapsed(r["Time"])
        for _, r in status.iterrows()
        if r["Status"] in {"Finished", "Finalised", "Ends"}
    ]
    starts = [elapsed(r["Time"]) for _, r in status.iterrows() if r["Status"] == "Started"]
    if not ends or not starts or any(t is None for t in [*starts, *ends]):
        raise DataUnavailable("No completed session timing window")
    duration = max(ends) - min(starts)
    if not 0 < duration <= 12 * 3600:
        raise DataUnavailable("Invalid completed session duration")
    ended_at = utc(meta["scheduled_start"]) + timedelta(seconds=duration)
    if ended_at > utcnow():
        raise DataUnavailable("Session is not yet completed")
    channels, issues = {}, list(dict.fromkeys(captured.messages))
    for name in ("weather_data", "race_control_messages", "track_status", "session_status"):
        try:
            frame = getattr(session, name)
            channels[name] = json.loads(frame.to_json(orient="records", date_format="iso"))
            if frame.empty:
                issues.append(f"{name}: empty")
        except fastf1.exceptions.DataNotLoadedError:
            channels[name] = []
            issues.append(f"{name}: unavailable")
    telemetry_summary = {}
    if telemetry:
        try:
            for driver, frame in session.car_data.items():
                speeds = [v for x in frame["Speed"] if (v := number(x)) is not None]
                telemetry_summary[str(driver)] = {
                    "samples": len(frame),
                    "speed_p95_kmh": float(pd.Series(speeds).quantile(0.95)) if speeds else None,
                    "note": "Session-wide descriptive statistic, not a car capability rating.",
                }
        except fastf1.exceptions.DataNotLoadedError:
            issues.append("telemetry: unavailable")
    return {
        "provider": "fastf1",
        "provider_version": fastf1.__version__,
        "source_url": "https://livetiming.formula1.com" + session.api_path,
        "source_note": "FastF1 may use its public mirror; cached HTTP responses retain request URLs. "
        "retrieved_at below is local normalization time; cached upstream responses may be older.",
        "cache": str(Path(cache).resolve()),
        "retrieved_at": utcnow().isoformat(),
        "ended_at": ended_at.isoformat(),
        "end_time_basis": "Scheduled start plus archived session-status elapsed duration; reconstructed.",
        "drivers": drivers,
        "laps": laps,
        "channels": channels,
        "telemetry": telemetry_summary,
        "issues": issues,
        "rejected_rows": rejected,
    }


def openf1_session(meta, archive, telemetry=False):
    params = (
        {"session_key": meta["session_key"]}
        if meta.get("session_key")
        else {"year": meta["season"]}
    )
    sessions, source = archive.get(f"{OPENF1}/sessions", params)
    matches = [
        s
        for s in sessions
        if s.get("session_name") == meta["session_name"]
        and abs((utc(s["date_start"]) - utc(meta["scheduled_start"])).total_seconds()) < 10800
    ]
    if len(matches) != 1:
        raise DataUnavailable("Missing/ambiguous OpenF1 session")
    session = matches[0]
    if utc(session["date_end"]) > utcnow():
        raise DataUnavailable("OpenF1 session is not completed")
    key, channels, issues, sources = (
        session["session_key"],
        {},
        [],
        [source.model_dump(mode="json")],
    )
    for endpoint in ("drivers", "laps", "stints", "pit", "weather", "race_control", "intervals"):
        try:
            rows, origin = archive.get(f"{OPENF1}/{endpoint}", {"session_key": key})
            if not isinstance(rows, list) or any(r.get("session_key") != key for r in rows):
                raise DataUnavailable("Response contains different session or invalid records")
            channels[endpoint] = rows
            sources.append(origin.model_dump(mode="json"))
        except DataUnavailable as exc:
            if endpoint in {"drivers", "laps"}:
                raise
            channels[endpoint] = []
            issues.append(f"{endpoint}: {exc}")
    if not channels["laps"]:
        raise DataUnavailable("OpenF1 returned no laps")
    drivers = {
        str(r["driver_number"]): {
            "driver_id": None,
            "name": r.get("full_name"),
            "team_id": None,
            "team_name": r.get("team_name"),
        }
        for r in channels["drivers"]
    }
    laps, rejected = [], []
    pits = {(str(r["driver_number"]), r.get("lap_number")) for r in channels["pit"]}
    for r in channels["laps"]:
        driver, lap = str(r.get("driver_number")), number(r.get("lap_number"))
        if driver not in drivers or lap is None or lap < 1 or not lap.is_integer():
            rejected.append("Missing/unknown driver or invalid lap number")
            continue
        stint = [
            s
            for s in channels["stints"]
            if str(s.get("driver_number")) == driver
            and number(s.get("lap_start")) is not None
            and number(s.get("lap_end")) is not None
            and s["lap_start"] <= lap <= s["lap_end"]
        ]
        stint = stint[0] if len(stint) == 1 else {}
        start = (
            (utc(r["date_start"]) - utc(session["date_start"])).total_seconds()
            if r.get("date_start")
            else None
        )
        duration = number(r.get("lap_duration"))
        age = number(stint.get("tyre_age_at_start"))
        laps.append(
            {
                "driver_number": driver,
                "lap_number": int(lap),
                "duration_s": duration,
                **{f"sector_{i}_s": number(r.get(f"duration_sector_{i}")) for i in (1, 2, 3)},
                "start_s": start,
                "end_s": start + duration if start is not None and duration is not None else None,
                "stint": stint.get("stint_number"),
                "compound": stint.get("compound"),
                "tyre_age_laps": age + lap - stint["lap_start"] + 1 if age is not None else None,
                "pit_in": (driver, int(lap)) in pits,
                "pit_out": r.get("is_pit_out_lap"),
                "deleted": None,
                "timing_accurate": None,
                "generated": False,
                "track_status": None,
                "phase": None,
                "speed_trap_kmh": number(r.get("st_speed")),
            }
        )
    issues.append(
        "OpenF1 laps lack equivalent verified deletion/accuracy/track-status flags; retained as provisional, not clean-pace evidence."
    )
    if telemetry:
        issues.append("Raw high-volume telemetry collection is supported through FastF1 only.")
    return {
        "provider": "openf1",
        "source_url": source.url,
        "sources": sources,
        "source_note": "Immutable cached OpenF1 responses with original retrieval timestamps.",
        "retrieved_at": max(s["retrieved_at"] for s in sources),
        "ended_at": session["date_end"],
        "end_time_basis": "OpenF1 session metadata",
        "drivers": drivers,
        "laps": laps,
        "channels": channels,
        "telemetry": {},
        "issues": issues,
        "rejected_rows": rejected,
    }


def collect_performance(
    seasons,
    output,
    cache="data/raw/performance",
    *,
    rounds=None,
    sessions=ALL_SESSIONS,
    provider="auto",
    telemetry=False,
    resume=False,
    slow_lap_factor=1.07,
    max_sessions=20,
    settle_hours=6,
    progress=None,
):
    if not seasons or any(y < 2018 or y > utcnow().year for y in seasons):
        raise ValueError("Choose seasons from 2018 through the current year")
    if rounds is not None and (not rounds or any(r < 1 or r > 40 for r in rounds)):
        raise ValueError("Rounds must be between 1 and 40")
    if (
        provider not in {"auto", "fastf1", "openf1"}
        or not sessions
        or set(sessions) - set(ALL_SESSIONS)
    ):
        raise ValueError("Invalid provider or session selection")
    assess_laps([], slow_lap_factor)
    if not 1 <= max_sessions <= 40:
        raise ValueError(
            "max_sessions must be between 1 and 40; resume later for larger collections"
        )
    if not 0 <= settle_hours <= 24:
        raise ValueError("settle_hours must be between 0 and 24")
    output, cache = Path(output), Path(cache)
    settings = {
        "version": VERSION,
        "provider": provider,
        "telemetry": telemetry,
        "slow_lap_factor": slow_lap_factor,
    }
    manifest_path = output / "manifest.json"
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Output directory exists; use --resume or a new directory")
    if output.exists() and any(output.iterdir()) and not manifest_path.exists():
        raise ValueError("Cannot resume a nonempty directory without a collection manifest")
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "settings": settings,
            "sessions": {},
            "catalog_errors": [],
            "created_at": utcnow().isoformat(),
        }
    )
    manifest["paused"] = None
    if manifest["settings"] != settings:
        raise ValueError("Resume settings differ; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    (cache / "fastf1").mkdir(parents=True, exist_ok=True)
    archive = Archive(cache / "openf1")
    attempted, blocked = 0, set()
    try:
        for season in sorted(set(seasons)):
            catalog = None
            for backend in ["fastf1", "openf1"] if provider == "auto" else [provider]:
                try:
                    catalog = (
                        fastf1_catalog(season, cache / "fastf1")
                        if backend == "fastf1"
                        else openf1_catalog(season, archive)
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - isolate third-party catalog failures
                    manifest["catalog_errors"].append(
                        {
                            "season": season,
                            "provider": backend,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            if catalog is None:
                atomic_json(manifest_path, manifest)
                continue
            for meta in sorted(catalog, key=lambda m: m["scheduled_start"]):
                if (rounds is not None and meta["round"] not in rounds) or meta[
                    "code"
                ] not in sessions:
                    continue
                sid = session_id(meta)
                existing = manifest["sessions"].get(sid)
                if existing and existing["status"] == "collected":
                    path = (output / existing["file"]).resolve()
                    if not path.is_relative_to(output.resolve()):
                        raise ValueError("Collection file must be inside its directory")
                    saved = json.loads(path.read_text())
                    if digest(saved) != existing["sha256"]:
                        raise ValueError(f"Session checksum mismatch: {sid}")
                    if progress:
                        progress(f"{sid}: verified cached session")
                    continue
                entry = {
                    **meta,
                    "status": "failed",
                    "attempts": list((existing or {}).get("attempts", [])),
                }
                if attempted >= max_sessions or manifest["paused"]:
                    entry.update(
                        status="pending",
                        reason=manifest["paused"] or "Batch limit reached; resume collection",
                    )
                elif utc(meta["scheduled_start"]) + timedelta(hours=settle_hours) > utcnow():
                    entry.update(
                        status="pending",
                        reason=f"Awaiting completed archive ({settle_hours:g}-hour scheduling margin)",
                    )
                else:
                    attempted += 1
                    data = None
                    for backend in ["fastf1", "openf1"] if provider == "auto" else [provider]:
                        if backend in blocked:
                            entry["attempts"].append(
                                {
                                    "provider": backend,
                                    "error": "Provider unavailable for this batch",
                                }
                            )
                            continue
                        try:
                            data = (
                                fastf1_session(meta, cache / "fastf1", telemetry)
                                if backend == "fastf1"
                                else openf1_session(meta, archive, telemetry)
                            )
                            break
                        except Exception as exc:  # noqa: BLE001 - isolate third-party session failures
                            entry["attempts"].append(
                                {"provider": backend, "error": f"{type(exc).__name__}: {exc}"}
                            )
                            if type(exc).__name__ == "RateLimitExceededError" or "429" in str(exc):
                                manifest["paused"] = (
                                    "Provider rate limit reached; resume after the provider limit resets"
                                )
                                entry.update(status="pending", reason=manifest["paused"])
                                break
                            if (
                                isinstance(exc, ImportError)
                                or "401" in str(exc)
                                or "403" in str(exc)
                            ):
                                blocked.add(backend)
                    if data is not None:
                        data["laps"] = assess_laps(data["laps"], slow_lap_factor)
                        data["quality"] = summarize_session(data["laps"], data["drivers"])
                        data.update(meta=meta, collection_version=VERSION)
                        filename = f"sessions/{sid}.json"
                        atomic_json(output / filename, data)
                        entry.update(
                            status="collected",
                            provider=data["provider"],
                            file=filename,
                            sha256=digest(data),
                            quality=data["quality"],
                            issues=data["issues"],
                        )
                manifest["sessions"][sid] = entry
                manifest["updated_at"] = utcnow().isoformat()
                atomic_json(manifest_path, manifest)
                if progress:
                    q = entry.get("quality", {})
                    progress(
                        f"{sid}: {entry['status']}; {q.get('usable_laps', 0)}/{q.get('laps', 0)} usable laps"
                    )
    finally:
        archive.close()
    atomic_json(manifest_path, manifest)
    return render_quality_report(output, manifest)


def render_quality_report(output, manifest):
    output = Path(output)
    entries = list(manifest["sessions"].values())
    collected = [e for e in entries if e["status"] == "collected"]
    summary = {
        "sessions_requested": len(entries),
        "sessions_collected": len(collected),
        "sessions_failed": sum(e["status"] == "failed" for e in entries),
        "sessions_pending": sum(e["status"] == "pending" for e in entries),
        "catalog_errors": len(manifest["catalog_errors"]),
        "laps": sum(e["quality"]["laps"] for e in collected),
        "usable_laps": sum(e["quality"]["usable_laps"] for e in collected),
        "laps_with_tyres": sum(e["quality"]["laps_with_tyres"] for e in collected),
    }
    drivers = [d for e in collected for d in e["quality"]["drivers"]]
    summary.update(
        driver_sessions=len(drivers),
        driver_sessions_with_pace_gap=sum(d["pace_gap_pct"] is not None for d in drivers),
        driver_sessions_with_teammate_delta=sum(
            d["teammate_delta_pct"] is not None for d in drivers
        ),
        driver_sessions_with_variability=sum(
            d["clean_lap_variability_pct"] is not None for d in drivers
        ),
    )
    rows = []
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "Session",
            "Driver",
            "Provider",
            "Laps",
            "Usable laps",
            "Laps with tyres",
            "Pace gap %",
            "Teammate delta %",
            "Exclusions",
        ]
    )
    for e in sorted(entries, key=lambda x: x["scheduled_start"]):
        q = e.get("quality", {})
        detail = "; ".join(
            [
                *e.get("issues", []),
                *([e["reason"]] if e.get("reason") else []),
                *(a["error"] for a in e["attempts"]),
            ]
        )
        cells = [
            session_id(e),
            e["event_name"],
            e["status"],
            e.get("provider", ""),
            q.get("laps", 0),
            q.get("usable_laps", 0),
            q.get("laps_with_tyres", 0),
            detail,
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in cells) + "</tr>")
        for d in q.get("drivers", []):
            writer.writerow(
                [
                    session_id(e),
                    d["name"],
                    e["provider"],
                    d["laps"],
                    d["usable_laps"],
                    d["laps_with_tyres"],
                    d["pace_gap_pct"],
                    d["teammate_delta_pct"],
                    json.dumps(d["exclusions"], sort_keys=True),
                ]
            )
    title = "F1 collection quality"
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px Georgia,serif;color:#202b32;background:#f5f2e9;margin:3vw}}
h1{{font-size:2.4rem}}table{{border-collapse:collapse;width:100%;background:white}}
td,th{{padding:.7rem;border-bottom:1px solid #ddd;text-align:left}}td:last-child{{max-width:35rem}}
.scroll{{overflow-x:auto}}.stats{{font-size:1.2rem;color:#135c54}}a{{color:#135c54}}</style>
<h1>{title}</h1><p class="stats">{summary["sessions_collected"]} sessions collected &middot;
{summary["laps"]:,} laps retained &middot; {summary["usable_laps"]:,} usable pace laps</p>
<p>Rejected pace laps remain in the dataset for audit. Usable laps are timing-screened;
traffic and fuel effects are not fully observable. Stint slopes are not corrected tyre degradation.
Source availability reconstructed from old archives is not proof of what was published at the time.</p>
<p><a href="quality-drivers.csv">Driver detail CSV</a> &middot;
<a href="manifest.json">Sources, checksums and failures</a></p>
<p>{summary["sessions_failed"]} failed sessions; {summary["sessions_pending"]} pending.
Tyre compound and age available for {summary["laps_with_tyres"]:,} laps.</p>
<p>Of {summary["driver_sessions"]} driver-session records:
{summary["driver_sessions_with_pace_gap"]} have comparable pace gaps,
{summary["driver_sessions_with_teammate_delta"]} have teammate comparisons, and
{summary["driver_sessions_with_variability"]} have stint variability estimates.</p>
<p>Braking, downforce, aero efficiency and corrected tyre-degradation ratings remain unknown;
these are not inferred from finishing positions or raw stint slopes.</p>
<div class="scroll"><table><thead><tr>{"".join(f"<th>{h}</th>" for h in ["Session", "Event", "Status", "Provider", "Laps", "Usable", "With tyres", "Data gaps"])}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p>Catalog failures: {html.escape(json.dumps(manifest["catalog_errors"]))}</p></html>"""
    (output / "quality-report.html").write_text(page, encoding="utf-8")
    (output / "quality-drivers.csv").write_text(buffer.getvalue(), encoding="utf-8-sig")
    atomic_json(output / "quality-summary.json", summary)
    return {**summary, "report": str((output / "quality-report.html").resolve())}

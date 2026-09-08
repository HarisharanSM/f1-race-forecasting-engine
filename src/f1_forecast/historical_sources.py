"""Resumable historical source acquisition without relaxing forecast input requirements."""

import html
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .data import DataUnavailable
from .historical import JOLPICA, season_table
from .legacy_history import BASE, TimingArchive
from .models import utcnow
from .performance_collection import atomic_json, collect_performance, digest


def paged_race_data(archive, year, rnd, endpoint):
    pages, sources, offset = [], [], 0
    while True:
        body, source = archive.get(
            f"{JOLPICA}/{year}/{rnd}/{endpoint}/", {"limit": 100, "offset": offset}
        )
        mr = body["MRData"]
        if any(
            int(r["season"]) != year or int(r["round"]) != rnd for r in mr["RaceTable"]["Races"]
        ):
            raise DataUnavailable("Historical page returned a different race")
        total, limit = int(mr["total"]), int(mr["limit"])
        if limit <= 0 or (not mr["RaceTable"]["Races"] and offset < total):
            raise DataUnavailable("Incomplete historical pagination")
        pages.append(body)
        sources.append(source.model_dump(mode="json"))
        offset += limit
        if offset >= total:
            return {"pages": pages, "sources": sources, "rows": total}


def collect_sources(
    years,
    output,
    *,
    max_jobs=20,
    details=False,
    performance=False,
    max_sessions=5,
    archive=None,
    cache="data/raw/full-history-2010-2026",
):
    years = sorted(set(years))
    if not years or min(years) < 1950 or max(years) > utcnow().year or not 1 <= max_jobs <= 1000:
        raise ValueError("Invalid collection years or job budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"version": 1, "years": years, "jobs": {}, "ready_for_full_backtest": False}
    )
    if manifest["years"] != years or manifest["version"] != 1:
        raise ValueError("Resume must retain the same collection years")
    owned = archive is None
    archive = archive or TimingArchive(cache)
    jobs = [
        (year, endpoint, None)
        for year in years
        for endpoint in ("schedule", "results", "qualifying", "timing_index")
    ]
    attempted = 0
    manifest.pop("paused", None)
    manifest["unresolved_full_inputs"] = {
        "pre_2018_detailed_timing": "Needs an alternative provider; current timing feed unsupported",
        "archived_pre_session_weather": "Not supplied by observed weather streams",
        "fuel_corrected_tyre_curves": "Raw lap/stint data alone does not isolate fuel burn or traffic",
        "complete_incident_logs": "Raw race-control messages require completeness verification",
        "verified_car_upgrade_history": "Needs dated source evidence",
        "full_snapshot_assembly": "Source material must pass existing validation before backtesting",
    }
    try:
        for year, endpoint, rnd in jobs:
            key = f"{year}:{endpoint}" + (f":{rnd}" if rnd is not None else "")
            previous = manifest["jobs"].get(key, {})
            if previous.get("status") == "collected":
                path = output / previous["file"]
                if digest(json.loads(path.read_text())) != previous["sha256"]:
                    raise ValueError(f"Historical source checksum mismatch: {key}")
                if endpoint == "schedule" and details:
                    append_details(jobs, year, json.loads(path.read_text())["races"])
                continue
            if previous.get("status") == "unsupported_by_provider":
                continue
            if endpoint == "timing_index" and year < 2018:
                manifest["jobs"][key] = {
                    "status": "unsupported_by_provider",
                    "reason": "Current F1/FastF1 detailed timing source starts in 2018; alternative source required",
                }
                continue
            if attempted >= max_jobs:
                manifest["jobs"].setdefault(key, {"status": "pending"})
                continue
            attempted += 1
            try:
                if rnd is not None:
                    payload = paged_race_data(archive, year, rnd, endpoint)
                elif endpoint == "timing_index":
                    body, source = archive.get(f"{BASE}{year}/Index.json")
                    if not body.get("Meetings"):
                        raise DataUnavailable("Timing index is empty or unsupported")
                    payload = {"index": body, "sources": [source.model_dump(mode="json")]}
                else:
                    table, sources = season_table(
                        archive, year, "" if endpoint == "schedule" else endpoint
                    )
                    payload = {
                        "races": {
                            str(k): {a: b for a, b in v.items() if a != "_sources"}
                            for k, v in table.items()
                        },
                        "sources": [s.model_dump(mode="json") for s in sources],
                    }
                path = f"sources/{key.replace(':', '-')}.json"
                atomic_json(output / path, payload)
                manifest["jobs"][key] = {
                    "status": "collected",
                    "file": path,
                    "sha256": digest(payload),
                    "retrieved_at": utcnow().isoformat(),
                }
                if endpoint == "schedule" and details:
                    append_details(jobs, year, payload["races"])
                if payload.get("rows") == 0:
                    manifest["jobs"][key]["reason"] = (
                        "Provider returned zero records; no historical observations available from this response"
                    )
            except DataUnavailable as exc:
                manifest["jobs"][key] = {"status": "pending", "last_error": str(exc)}
                if "429" in str(exc):
                    manifest["paused"] = "Provider rate limit; resume later"
                    break
            atomic_json(manifest_path, manifest)
    finally:
        if owned:
            archive.close()
        for year, endpoint, rnd in jobs:
            key = f"{year}:{endpoint}" + (f":{rnd}" if rnd is not None else "")
            manifest["jobs"].setdefault(key, {"status": "pending"})
        manifest["limitations"] = (
            "Source acquisition only; not a reduced-input dataset or a backtest. Full input requirements are unchanged. "
            "Collected means downloaded, not complete coverage. Current-season cached responses are a fixed vintage; "
            "use a new raw cache/collection for a later vintage. Race lap times do not supply tyre compounds, fuel "
            "corrections or telemetry. Observed weather is not an archived pre-session forecast. Race-control "
            "messages do not establish complete incident-log coverage. Pre-2018 detailed timing, archived forecasts, "
            "fuel corrections and verified upgrades may need other sources. Missing inputs are not invented."
        )
        atomic_json(manifest_path, manifest)
    if performance and not manifest.get("paused"):
        supported = [y for y in years if y >= 2018]
        if supported:
            manifest["performance"] = collect_performance(
                supported,
                output / "performance",
                "data/raw/performance",
                provider="fastf1",
                telemetry=True,
                resume=(output / "performance" / "manifest.json").exists(),
                max_sessions=max_sessions,
            )
            atomic_json(manifest_path, manifest)
    render_coverage(output, manifest)
    return {
        "attempted_jobs": attempted,
        "collected_jobs": sum(j["status"] == "collected" for j in manifest["jobs"].values()),
        "ready_for_full_backtest": False,
        "report": str((output / "coverage.html").resolve()),
    }


def append_details(jobs, year, races):
    for rnd, event in races.items():
        # Avoid requesting future race results or freshly unfinished sessions.
        available = datetime.fromisoformat(event["date"]).replace(tzinfo=UTC) + timedelta(days=2)
        if available <= utcnow():
            for endpoint in ("laps", "pitstops"):
                job = (year, endpoint, int(rnd))
                if job not in jobs:
                    jobs.append(job)


def render_coverage(output, manifest):
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(v['status'])}</td>"
        f"<td>{html.escape(v.get('reason') or v.get('last_error') or '')}</td></tr>"
        for k, v in sorted(manifest["jobs"].items())
    )
    page = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Historical input coverage</title><style>body{font:17px/1.6 Georgia;background:#f4f0e6;color:#16343c;max-width:1100px;margin:auto;padding:24px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccc}table{width:100%;border-collapse:collapse}.scroll{overflow:auto}</style><h1>Full historical input collection</h1>"
    page += f"<p><strong>Not ready for a full-input backtest.</strong> No reduced-input backtest has been run.</p><p>{html.escape(manifest['limitations'])}</p>"
    if manifest.get("performance"):
        stats = manifest["performance"]
        page += f"<p>Detailed sessions: {stats['sessions_collected']}; laps: {stats['laps']}; with tyre information: {stats['laps_with_tyres']}. <a href='performance/quality-report.html'>Detailed timing coverage</a></p>"
    page += f"<div class='scroll'><table><tr><th>Source job</th><th>Status</th><th>Gap or error</th></tr>{rows}</table></div><p><a href='manifest.json'>Exact manifest and checksums</a></p></html>"
    (output / "coverage.html").write_text(page)

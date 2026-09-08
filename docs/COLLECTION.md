# Expanded lap, stint and telemetry collection

The optional collection extra adds FastF1. Core forecasting remains usable without it.
`fetch-performance` collects real session observations independently of the existing
classification importer, so a missing qualifying classification does not automatically
discard the weekend's practice or race laps.

```sh
uv sync --all-extras
source .venv/bin/activate

f1-forecast fetch-performance --seasons 2024 2025 --rounds 1 2 \
  --provider fastf1 --output data/processed/performance-expanded
```

Choose a new output directory or explicitly use `--resume`. Completed files are
checked against their stored hashes before reuse. Failed sessions can be retried;
earlier failures remain in the audit. Changing provider, telemetry or cleaning
settings requires a new output directory. Reuse the same cache to avoid downloading
successful raw responses again; use a new cache for a new archive vintage.

## Coverage and batching

The session selection defaults to `FP1 FP2 FP3 Q SQ S R`, taking only the sessions
actually scheduled on each weekend. Testing is excluded. Sprint and Grand Prix
observations retain distinct codes; older sprint naming is normalized through
FastF1's schedule. Sessions use year, round and session code as their identity,
so repeated visits to a circuit are distinct.

Each invocation attempts up to 20 new sessions. `--max-sessions` accepts 1..40.
Remaining sessions are listed as pending in the report. A provider rate-limit
failure pauses the batch; resume only after its limit resets. Authentication
failures disable that provider for the remaining session attempts in the batch.
This is an on-demand collector, not a background scheduler.

For example, expand the recorded first batch to additional weekends and seasons:

```sh
f1-forecast fetch-performance --seasons 2024 2025 2026 \
  --provider fastf1 --resume --output data/processed/performance-expanded
```

Default collection waits until six hours after the scheduled start before trying
a session, then verifies the archive has a completed timing window. Use
`--settle-hours 0` to check a just-completed practice session sooner. Unfinished
sessions are never accepted. For recent sessions, a new cache directory avoids
reusing partial upstream archives fetched while the session was still underway.
Future sessions remain pending. Supported season arguments start at 2018, but
actual provider and channel coverage varies by year.

## Providers and retained observations

`--provider auto` tries FastF1 first and falls back to OpenF1 when session loading
fails. Explicit `fastf1` and `openf1` modes are also available. FastF1's cache handles
its upstream timing requests and public mirror support. OpenF1 uses the project's
immutable raw-response archive with bounded retries and request pacing.

FastF1 records include lap/sector durations, stint and compound, tyre age, pit-lap
flags, deletion and timing-accuracy flags, track status, qualifying phase when
available, speed-trap values, weather and race-control/status messages. Provider
warnings and corrections reported during loading are retained in the quality audit.
No final finishing-order labels are invented from missing classifications.

OpenF1 fallback retains laps, stints, pit stops, weather, race control and intervals.
Its laps currently remain provisional because the adapter cannot establish the
same deletion, accuracy and track-status flags. They do not silently qualify as
clean pace measurements. That conservative limitation is shown in the report.

Optional full telemetry is larger and is fetched through FastF1:

```sh
f1-forecast fetch-performance --seasons 2025 --rounds 1 --sessions FP1 \
  --provider fastf1 --telemetry --max-sessions 1 \
  --output data/processed/performance-telemetry-sample
```

The FastF1 cache retains car and position traces; session JSON includes sample
counts and descriptive speed statistics. Telemetry does not automatically become
a braking/downforce rating. No paid account or API key is configured by this command.
Provider access restrictions remain visible failures rather than fabricated data.

## Quality rules

Every identifiable lap is retained, including laps excluded from pace estimation.
Rows with missing/unknown driver identity or invalid lap number are counted in
`rejected_rows`. Pace exclusions cover duplicate driver/lap keys, missing or
inconsistent sector timing, pit laps, deleted laps, unknown deletion/accuracy
status, generated laps, track interruptions and missing lap-start timing.

After these checks, laps slower than 1.07 times the driver's compound-specific
10th-percentile time are flagged as slow-lap candidates when at least three laps
exist. This is a heuristic cooldown/traffic filter; `--slow-lap-factor` controls it.
It does not prove the remaining laps are free of traffic. Missing, unknown or
invalid tyre information cannot enter comparable-lap cohorts.

Pace comparisons group the same compound, three-lap tyre-age bins, 15-minute
session-time windows and qualifying phase. Field-relative gaps require at least
four drivers from three known teams in a cohort. Teammate comparisons require a
matching team. A reported gap requires at least three supporting driver laps;
sample counts and cohort dispersion accompany the estimate.

Stints with at least six usable laps across four tyre-age increments receive a
linear observed pace slope and detrended variability estimate. The slope can be
negative because fuel and conditions affect lap times. It is **not** assigned to
the optional physical tyre-degradation input. Aero, braking, downforce, corrected
degradation and event-risk ratings remain unknown unless supplied separately.

## Optional forecast enrichment

```sh
f1-forecast enrich-performance data/processed/real-history-2024-2025.json \
  --collection data/processed/performance-expanded \
  --output data/processed/real-history-2024-2025-performance.json \
  --report artifacts/performance-enrichment.json
```

The input may be one Snapshot or a list of historical snapshot/feedback pairs.
Both output paths must be new. Original inputs and feedback labels are preserved.

Only same-season sessions ending within the preceding 30 days and available by
the forecast cutoff are eligible. Qualifying uses prior qualifying/practice;
races use prior races/practice. Driver and team IDs must match; driver numbers
alone are insufficient. Current manually supplied optional objects are preserved.

The most recent eligible evidence fills teammate pace differences, detrended lap
variability and team pace gaps. Team pace requires measurements for at least two
drivers. Evidence uses the existing optional-input weight 0.25 as an explicit
heuristic. Practice pace remains a proxy with uncertain fuel, traffic and setup.
No forecast probability calibration is implied by measurement availability.

For live snapshots, actual local collection availability must precede the cutoff:
collect first, then create the forecast snapshot. Historical reconstructions use
an explicitly assumed six-hour delay after the archived session end. FastF1 session
end is reconstructed from scheduled start and session-status elapsed duration.
Local normalization/retrieval time remains separately recorded, and may be later
than the original upstream cache response. Archive revisions mean this is not an
immutable point-in-time dataset. Target-session actuals never enter their own
prediction inputs.

## Files and recorded initial run

The collection directory contains:

- `sessions/*.json`: observations, source details, provider warnings and measurements.
- `manifest.json`: session selection, outcomes, hashes, failed attempts and settings.
- `quality-report.html`: readable coverage and data gaps, ready to open directly.
- `quality-drivers.csv`: per-driver coverage, comparable gaps and exclusions.
- `quality-summary.json`: machine-readable collection totals.

On 5 September 2026 the first two weekends of each of 2024 and 2025 produced:

| Item | Recorded value |
| --- | ---: |
| Completed sessions | 20 / 20 |
| Laps retained | 10,147 |
| Usable pace laps | 5,842 |
| Laps with known compound and valid tyre age | 10,139 |
| Driver-session records | 400 |
| Driver-session records with comparable pace gaps | 345 |
| Driver-session records with teammate comparisons | 284 |
| Driver-session records with stint variability | 177 |
| Historical snapshots enriched | 18 / 86 |
| Driver/team optional evidence objects populated | 523 |

An additional telemetry run collected 362,700 car-data observations across 20
drivers in the already-counted 2025 Australian FP1 session; it is not an extra
independent session. OpenF1 returned an authentication
restriction during a live session, so the real batch used FastF1. OpenF1 parsing and
fallback paths are covered with mocked responses, not a successful live download.

The new files provide richer observations, not additional independent race labels.
The matched chronological evaluation is now recorded in
[Expanded collection backtest](EXPANDED_COLLECTION_BACKTEST.md). It found no
overall improvement. Retain a later untouched period if these results inform
model development.

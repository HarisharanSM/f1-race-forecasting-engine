# Separating driver and car performance

The expanded [2020-2026 year-to-date evaluation](JOINT_BACKTEST_2020_2026.md)
preserves earlier forecasts and adds verified completed 2026 sessions.

The optional joint estimator fits driver-season and evolving team effects in one
model, rather than treating raw team pace as car capability. Existing skill and
car ratings are never overwritten. These are statistical estimates, not direct
physical measurements or a causal decomposition of car and driver performance.

## Inputs and matching

Use `enrich-performance --joint-effects`. This also enables quality-aware
enrichment. The current and previous season's available timing evidence is used,
with a 730-day maximum age and a 90-day evidence half-life. Race and qualifying
estimates are separate: both can use earlier practice, but race timing is not
used for qualifying effects, nor qualifying timing for race effects.

Clean laps are matched within a session by compound, tyre-age groups of three
laps, 15-minute windows, qualifying phase, wet/dry status and 5 C track-temperature
groups. Weather observations must precede lap start and be no more than three
minutes old. Existing pit, deletion, interruption, accuracy, generated-lap and
slow-lap filters still apply. Fuel, traffic and setup cannot be fully controlled.

Each driver's median in a cohort becomes one comparable-run observation. A
cross-team cohort requires at least four drivers across three teams. Otherwise,
only same-team pairs/groups contribute. Driver identities, not racing numbers,
link observations across sessions and team transfers. A returned effect requires
at least six supporting laps in total. This joint threshold is across cohorts,
not a claim that every matched pair has six laps. Team effects additionally need
cross-team evidence; teammate-only evidence does not produce a measured car effect.

## Joint model

`100 * log(lap time) = cohort intercept + driver-season effect + team-state effect`

The fit removes each cohort's intercept by weighted centering. Teammate contrasts
therefore cancel their shared car contribution naturally, without counting an
additional duplicate pair likelihood. Cross-team runs identify combined pace.
Transfers help connect driver and team identities. Run weights cap lap support
at six per driver, normalize within each cohort and decay with age. Robust
residual weighting reduces outlier influence.

The regularized state model uses explicit, fixed assumptions:

- Initial driver standard deviation: 1 percentage point; initial team: 2.
- Driver season transition: retain 80% of the preceding effect; innovation SD 0.35.
- Team season transition: retain 20%; innovation SD 1.2. Cars reset more strongly.
- Team state evolves between weekends, allowing ordinary drift without a known
  upgrade. Innovation variance is `0.12^2 * elapsed weeks` (minimum one week).
- Each dated, applicable upgrade adds `0.8^2` to transition variance. The inherited
  mean does not improve automatically. Subsequent common pace changes can update
  the car estimate more readily.

Team states are keyed by season, round and upgrade phase. A known upgrade between
sessions can create a new state within a weekend. Drivers retain the same
season-level state, so short-term shared improvements are preferentially
attributed to the evolving car. Driver skill does not currently drift within a
season. Team IDs must be stable; team renames are not inferred automatically.

These priors are not fitted from race outcomes or tuned on the test set. Lineups
that never change cannot identify unique absolute driver/car effects. All
estimates are labeled `prior_dependent`; the audit includes likelihood rank,
state count, transition assumptions and transfer evidence.

## Outputs and forecast use

`driver.performance.joint_effects[session]` and
`team.car.performance.joint_effects[session]` contain relative mean log-time
contributions, conditional posterior standard deviations, evidence timestamps,
lap/session support, target season and current-season evidence availability.
Lower values mean faster relative pace. The evidence audit includes combined
driver-plus-car uncertainty with covariance, all timing sources and known upgrades.
Do not add individual standard deviations or interpret them as calibrated
position confidence.

Newly trained `--optional-mode learned` models receive 16 additional optional
features: driver and car mean, presence, uncertainty, samples, sessions, evidence
age, current-season observation flag and upgrade count. Existing measurement
features remain available; regularization and earlier validation select whether
to use a correction. Six earlier training weekends per feature and two measured
validation weekends are still required. Both probability scores must improve
on validation before selecting the correction. This does not guarantee test gains.

Missing effects contribute zero optional features. `weight=0` disables evidence.
The legacy heuristic and ignore modes do not use joint effects. Existing
132-feature measurement checkpoints load with unchanged feature ordering and
behavior; new checkpoints support 148. An old checkpoint must be retrained to
use the new joint features. No deployed model is automatically replaced.

## Dated upgrades

Supply an optional JSON list with `--upgrades path.json`. Each record requires
`team_id`, a team-unique `id`, `introduced_at`, `available_at`, and `source_url`;
`description` is optional. Timestamps must include a timezone. The same fields
except `team_id` can be supplied in `team.car.upgrades` within a snapshot.

`available_at` must describe when the information became available, not a
backdated retrieval timestamp. External catalog entries after a forecast cutoff
are ignored; conflicting IDs are rejected. An introduction affects only states
at or after its date. Known future introductions can affect the target-session
uncertainty, never grant an assumed pace benefit. Upgrades are currently team-wide;
driver-specific deployments must not be represented as a shared team upgrade.

No verified real upgrade chronology is present in the current collection. The
real run uses no upgrade records; dated-upgrade behavior is tested with synthetic
fixtures. Zero recorded upgrades does not mean a team made none.

## Reproduce

```sh
uv run f1-forecast enrich-performance data/processed/real-history-2024-2025.json \
  --collection data/processed/performance-full-2024-2025 --joint-effects \
  --output data/processed/real-history-joint-effects-new.json \
  --report artifacts/joint-effects-new/enrichment.json

uv run python scripts/backtest_joint_effects.py \
  --joint data/processed/real-history-joint-effects-new.json \
  --output artifacts/joint-effects-backtest-new
```

Outputs must be new paths; original datasets and reports are preserved. The
backtest uses matching 2025 Grand Prix qualifying/race targets with 2024 development
history, not a rerun of the full 2020-2025 GP/sprint evaluation. It compares the
previous quality-aware learned/calibrated model with added joint inputs, using
the same training configuration, 1,000 simulations per scenario and seed 42.
Reports retain the year/circuit/session/scenario workflow and show joint effects,
their uncertainty and evidence ages alongside the forecast.

The initial run produces estimates for 86 snapshots from the available 75
collected sessions. This is not 86 snapshots with fresh timing. Later snapshots
carry stale evidence forward and have wider car uncertainty; the timing archive
is incomplete. Historical source availability is reconstructed as six hours
after session end, with source provenance retained. Later archive corrections
cannot be ruled out. No target-session outcomes are inputs to the estimator.

## Initial matched backtest

Saved in `artifacts/joint-effects-backtest/`: 45 held-out 2025 Grand Prix sessions
across 23 weekends, using the same 2024 development history in both arms.

| Metric | Previous learned + calibrated | With joint effects |
| --- | ---: | ---: |
| Average position error (lower better) | 3.302956 | 3.300283 |
| Pairwise accuracy | 76.004% | 75.899% |
| Correct pole/winner predictions | 15 / 45 | 15 / 45 |
| Winner Brier (lower better) | 0.794131 | 0.792022 |
| Position log loss (lower better) | 2.696996 | 2.696054 |
| Coverage of nominal 80% position intervals | 87.170% | 86.947% |
| Average interval width (positions) | 10.769 | 10.677 |
| Weekend folds selecting any optional correction | 2 / 23 | 9 / 23 |

Average position error improves only 0.081%; the probability-score gains are also
small, while pairwise accuracy declines. This is mixed evidence, not a material
or established accuracy/confidence improvement. More correction selections do
not themselves prove usefulness. The collection and upgrade chronology gaps
remain. That initial experiment did not rerun the full 2020-2025 report; the
subsequent extended evaluation is linked above. Existing reports remain preserved.

A paired calendar-weekend bootstrap (5,000 resamples, seed 42) gives a 95%
interval of -0.01255 to +0.00526 positions for joint-minus-previous MAE. It includes
zero and does not model all temporal dependence or selection bias. The final
experimental checkpoint's validation did not select an optional correction;
some earlier backtest folds did. It therefore falls back to the calibrated base
model for inference. The feature path is implemented, not forced into production.

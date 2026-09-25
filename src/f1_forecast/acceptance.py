"""Paired, chronological, multi-metric screening. No automatic model promotion."""

from dataclasses import asdict, dataclass

import numpy as np

from .engine import evaluate

# All deltas are oriented so positive means worse.
DIRECTIONS = {
    "expected_position_mae": 1,
    "winner_brier": 1,
    "position_log_loss": 1,
    "pairwise_accuracy": -1,
    "winner_correct": -1,
    "position_interval_coverage": -1,
    "position_interval_width": 1,
    "position_interval_score": 1,
}


@dataclass(frozen=True)
class AcceptancePolicy:
    windows: int = 3
    weekends_per_window: int = 6
    training_seeds: int = 3
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 42
    mae_improvement: float = 0.001
    brier_improvement: float = 0.0001
    coverage_mode: str = "relative"
    minimum_coverage: float = 0.8
    pairwise_tolerance: float = 0.0
    winner_tolerance: float = 0.0

    def __post_init__(self):
        if self.windows < 3 or self.weekends_per_window < 6 or self.training_seeds < 3:
            raise ValueError("Require three windows, six weekends per window, three training seeds")
        if self.bootstrap_draws < 100 or self.bootstrap_seed < 0:
            raise ValueError("Invalid bootstrap settings")
        if not all(
            np.isfinite(x) and x > 0 for x in (self.mae_improvement, self.brier_improvement)
        ):
            raise ValueError("Improvement thresholds must be positive and finite")
        if self.coverage_mode not in {"relative", "nominal"}:
            raise ValueError("Unknown coverage acceptance mode")
        if not np.isfinite(self.minimum_coverage) or not 0.8 <= self.minimum_coverage <= 1:
            raise ValueError("Coverage floor must protect the nominal 80% interval")
        if not all(
            np.isfinite(x) and 0 <= x <= 0.02
            for x in (self.pairwise_tolerance, self.winner_tolerance)
        ):
            raise ValueError("Accuracy tolerances must be finite and between zero and 0.02")


def interval_quality_policy():
    # Declared before experiments: keep primary score protections, allow tiny accuracy changes.
    return AcceptancePolicy(
        coverage_mode="nominal",
        minimum_coverage=0.8,
        pairwise_tolerance=0.002,
        winner_tolerance=0.02,
    )


def protected_metrics(forecast, feedback):
    metrics = evaluate(forecast, feedback)
    actual = {driver: i + 1 for i, driver in enumerate(feedback.finishing_order)}
    # Central 80% interval: width + (2 / alpha) times distance of a missed outcome.
    metrics["position_interval_score"] = float(
        np.mean(
            [
                s.p90_position
                - s.p10_position
                + 10 * max(s.p10_position - actual[s.driver_id], 0)
                + 10 * max(actual[s.driver_id] - s.p90_position, 0)
                for s in forecast.standings
            ]
        )
    )
    return metrics


def mean_scores(rows, key):
    return {m: float(np.mean([r[key][m] for r in rows])) for m in DIRECTIONS}


def paired_summary(rows, policy, *, bootstrap=True):
    if not rows:
        return {"sessions": 0, "weekends": 0}
    base, candidate = mean_scores(rows, "baseline"), mean_scores(rows, "candidate")
    result = {
        "sessions": len({(r["weekend"], r["format"], r["session"]) for r in rows}),
        "forecast_pairs": len(rows),
        "weekends": len({r["weekend"] for r in rows}),
        "baseline": base,
        "candidate": candidate,
        "delta": {m: candidate[m] - base[m] for m in DIRECTIONS},
    }
    if bootstrap:
        # Resample entire calendar weekends, retaining both formats, sessions and seeds.
        # Preserve the report's equal-session weighting even for uneven weekend sizes.
        grouped, coverage = {}, {}
        for r in rows:
            grouped.setdefault(r["weekend"], []).append(
                [r["candidate"][m] - r["baseline"][m] for m in DIRECTIONS]
            )
            coverage.setdefault(r["weekend"], []).append(
                r["candidate"]["position_interval_coverage"]
            )
        sums = np.array([np.sum(v, axis=0) for v in grouped.values()])
        counts = np.array([len(v) for v in grouped.values()])
        rng = np.random.default_rng(policy.bootstrap_seed)
        coverage_sums = np.array([sum(v) for v in coverage.values()])
        samples, coverage_samples = [], []
        for _ in range(policy.bootstrap_draws):
            indexes = rng.integers(0, len(sums), len(sums))
            samples.append(sums[indexes].sum(axis=0) / counts[indexes].sum())
            coverage_samples.append(coverage_sums[indexes].sum() / counts[indexes].sum())
        bounds = np.quantile(samples, [0.025, 0.975], axis=0)
        result["paired_weekend_95pct_delta"] = {
            m: bounds[:, i].tolist() for i, m in enumerate(DIRECTIONS)
        }
        result["candidate_coverage_weekend_95pct"] = np.quantile(
            coverage_samples, [0.025, 0.975]
        ).tolist()
    return result


def breakdown(rows, policy=None):
    policy = policy or AcceptancePolicy()
    return {
        "overall": paired_summary(rows, policy),
        **{
            f"by_{dimension}": {
                str(value): paired_summary([r for r in rows if r[dimension] == value], policy)
                for value in sorted({r[dimension] for r in rows}, key=str)
            }
            for dimension in ("format", "session", "quality", "season", "retirement", "seed")
        },
    }


def acceptance_gate(rows, policy=None):
    """Rows must be validation-only matched pairs with explicit calendar/seed metadata.

    The caller owns the disjoint training/validation/test split; this function
    additionally checks duplicates, complete seed panels and chronological windows.
    """
    policy = policy or AcceptancePolicy()
    reasons = []
    if not rows:
        return {"accepted": False, "reasons": ["No validation evidence"], "policy": asdict(policy)}
    keys = [(r["seed"], r["weekend"], r["format"], r["session"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate validation forecast pair")
    for r in rows:
        if any(
            not np.isfinite(r[k].get(m, np.nan))
            for k in ("baseline", "candidate")
            for m in DIRECTIONS
        ):
            raise ValueError("Missing or nonfinite protected metric")
    seeds = sorted({r["seed"] for r in rows})
    panels = [{key[1:] for key in keys if key[0] == seed} for seed in seeds]
    if any(panel != panels[0] for panel in panels):
        reasons.append("Training seeds do not cover identical validation sessions")
    if len(seeds) < policy.training_seeds:
        reasons.append(
            f"Need {policy.training_seeds} independent training seeds; found {len(seeds)}"
        )
    weekends = sorted({r["weekend"] for r in rows})
    if len(weekends) < policy.windows * policy.weekends_per_window:
        reasons.append("Insufficient weekends for chronological validation windows")
    windows = [list(w) for w in np.array_split(weekends, policy.windows)]
    aggregate = paired_summary(rows, policy)

    def check(group, label, *, improvement=False, confidence=False):
        scores = paired_summary(group, policy, bootstrap=confidence)
        if not group:
            reasons.append(f"{label}: empty validation group")
            return scores
        for metric, direction in DIRECTIONS.items():
            if metric == "position_interval_coverage" and policy.coverage_mode == "nominal":
                # Integer quantiles can overcover; do not preserve incumbent excess coverage.
                if scores["candidate"][metric] + 1e-12 < policy.minimum_coverage:
                    reasons.append(f"{label}: coverage below nominal floor")
                if (
                    confidence
                    and scores["candidate_coverage_weekend_95pct"][0] + 1e-12
                    < policy.minimum_coverage
                ):
                    reasons.append(
                        f"{label}: coverage uncertainty does not establish nominal floor"
                    )
                continue
            limit = {
                "pairwise_accuracy": policy.pairwise_tolerance,
                "winner_correct": policy.winner_tolerance,
            }.get(metric, 0.0)
            if improvement and metric == "expected_position_mae":
                limit = -policy.mae_improvement
            if improvement and metric == "winner_brier":
                limit = -policy.brier_improvement
            if direction * scores["delta"][metric] > limit + 1e-12:
                reasons.append(f"{label}: {metric} failed")
            if confidence:
                low, high = scores["paired_weekend_95pct_delta"][metric]
                worse_bound = high if direction == 1 else -low
                if worse_bound > limit + 1e-12:
                    reasons.append(f"{label}: {metric} uncertainty does not exclude regression")
        return scores

    check(rows, "overall", improvement=True, confidence=True)
    audits = []
    for seed in seeds:
        seeded = [r for r in rows if r["seed"] == seed]
        check(seeded, f"seed {seed}", improvement=True)
        for i, window in enumerate(windows):
            part = [r for r in seeded if r["weekend"] in window]
            label = f"seed {seed}, window {i + 1}"
            scores = check(part, label)
            for kind, session in sorted({(r["format"], r["session"]) for r in rows}):
                subgroup = [r for r in part if (r["format"], r["session"]) == (kind, session)]
                if len({r["weekend"] for r in subgroup}) < policy.weekends_per_window:
                    reasons.append(f"{label}, {kind}/{session}: insufficient weekend support")
                check(subgroup, f"{label}, {kind}/{session}")
            audits.append({"seed": seed, "window": i + 1, "weekends": window, "scores": scores})
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "policy": asdict(policy),
        "overall": aggregate,
        "windows": audits,
        "limitations": "Retrospective screening; marginal bootstrap intervals are not simultaneous guarantees. "
        "Whole-weekend resampling does not capture all temporal dependence or development bias. "
        "Future performance is not guaranteed; freeze before prospective evaluation.",
    }


def pair_record(record, baseline, candidate, seed):
    s = record.snapshot
    return {
        "weekend": f"{s.season:04d}-{s.round:02d}",
        "format": s.race_format,
        "session": s.session.value,
        "season": s.season,
        "quality": "reduced"
        if any(n.startswith("REDUCED INPUT:") for n in s.notes)
        else "richer_existing",
        "retirement": "with_retirement"
        if record.feedback.retired_drivers
        else "without_retirement",
        "seed": seed,
        "baseline": protected_metrics(baseline, record.feedback),
        "candidate": protected_metrics(candidate, record.feedback),
    }


def selection_split(records, min_development_events, policy=None):
    """Reserve enough chronological weekends for each session, even with partial rosters.

    Only dates/session availability determine window size; outcomes and metrics do not.
    """
    from .ml_data import chronological_split, event_groups

    policy = policy or AcceptancePolicy()
    sessions = {r.snapshot.session for r in records}
    maximum = len(event_groups(records)) - min_development_events
    for count in range(policy.windows * policy.weekends_per_window, maximum + 1):
        try:
            development, selection = chronological_split(records, min_development_events, count)
        except ValueError:
            continue
        groups = event_groups(selection)
        windows = np.array_split(np.arange(len(groups)), policy.windows)
        if all(
            sum(any(r.snapshot.session == session for r in groups[i]) for i in window)
            >= policy.weekends_per_window
            for window in windows
            for session in sessions
        ):
            return development, selection
    raise ValueError("Insufficient session coverage for three chronological selection windows")

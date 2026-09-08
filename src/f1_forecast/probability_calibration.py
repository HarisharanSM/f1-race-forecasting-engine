"""Chronological calibration of coherent finishing-position probabilities."""

import json
from itertools import pairwise
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import AwareDatetime, Field

from .models import Forecast, Model

TEMPERATURES = (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0)
MIN_FIT_WEEKENDS = 12
MIN_SELECT_WEEKENDS = 6


def stratum(snapshot):
    return f"{snapshot.race_format}:{snapshot.session.value}"


def weekend_key(snapshot):
    return f"{snapshot.season}-{snapshot.round:02d}"


def check_example(snapshot, forecast, actual=None):
    if (forecast.event_id, forecast.session, forecast.race_format) != (
        snapshot.event_id,
        snapshot.session,
        snapshot.race_format,
    ):
        raise ValueError("Forecast and snapshot identities differ")
    ids = {d.id for d in snapshot.drivers}
    if any(
        len(s.standings) != len(ids) or {r.driver_id for r in s.standings} != ids
        for s in [forecast, *forecast.scenarios]
    ):
        raise ValueError("Forecast roster differs from snapshot")
    if forecast.ml_training_cutoff and forecast.ml_training_cutoff >= snapshot.as_of:
        raise ValueError("Base forecast includes future feedback")
    if actual is not None and (
        (actual.event_id, actual.session) != (snapshot.event_id, snapshot.session)
        or set(actual.finishing_order) != ids
        or actual.available_at <= snapshot.as_of
    ):
        raise ValueError("Feedback identity, roster or chronology differs")


def temperature_matrix(matrix, temperature):
    """Power transform with row/column balancing: each driver and position sums to one."""
    p = np.asarray(matrix, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[0] < 2 or not np.isfinite(p).all():
        raise ValueError("Expected a finite square probability matrix")
    if (
        np.any(p < 0)
        or not np.allclose(p.sum(0), 1, atol=1e-6)
        or not np.allclose(p.sum(1), 1, atol=1e-6)
    ):
        raise ValueError("Probability matrix must be doubly stochastic")
    if not 0.5 <= temperature <= 2:
        raise ValueError("Unsupported calibration temperature")
    if temperature == 1:
        return p.copy()
    q = np.maximum(p, 1e-8) ** (1 / temperature)
    for _ in range(1000):
        q /= q.sum(1, keepdims=True)
        q /= q.sum(0, keepdims=True)
        if np.max(np.abs(q.sum(1) - 1)) < 1e-10:
            return np.clip(q, 0, 1)
    raise ValueError("Probability balancing did not converge")


def transform(forecast, temperature, blend=1.0):
    from .engine import summarize

    if not np.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError("Calibration blend must be between zero and one")
    if temperature == 1 or blend == 0:
        return forecast.model_copy(deep=True)
    updated = forecast.model_copy(deep=True)
    ids = [r.driver_id for r in forecast.standings]
    mixture = np.zeros((len(ids), len(ids)))
    dnfs = np.array([r.dnf_probability for r in forecast.standings])
    if not forecast.scenarios:
        raise ValueError("Calibration requires scenario distributions")
    for scenario in updated.scenarios:
        by_id = {r.driver_id: r for r in scenario.standings}
        original = np.array([by_id[d].position_probabilities for d in ids])
        matrix = (1 - blend) * original + blend * temperature_matrix(original, temperature)
        risks = np.array([by_id[d].dnf_probability for d in ids])
        scenario.standings = summarize(ids, matrix, risks)
        scenario.winner = max(scenario.standings, key=lambda r: r.win_probability).driver_id
        mixture += scenario.scenario.probability * matrix
    updated.standings = summarize(ids, np.clip(mixture, 0, 1), np.clip(dnfs, 0, 1))
    updated.winner = max(updated.standings, key=lambda r: r.win_probability).driver_id
    return Forecast.model_validate(updated.model_dump())


class ProbabilityCalibrator(Model):
    version: Literal[1] = 1
    method: Literal["balanced_scenario_temperature"] = "balanced_scenario_temperature"
    fitted_through: AwareDatetime
    forecast_model: str
    synthetic: bool
    development_weekends: list[str]
    temperatures: dict[str, float]
    blends: dict[str, float] = Field(default_factory=dict)
    audit: dict = Field(default_factory=dict)

    def apply(self, snapshot, forecast):
        check_example(snapshot, forecast)
        if any(w.startswith("Probability calibration:") for w in forecast.warnings):
            raise ValueError("Forecast has already been probability-calibrated")
        if (
            self.fitted_through >= snapshot.as_of
            or weekend_key(snapshot) in self.development_weekends
        ):
            raise ValueError("Calibration contains target-weekend or future feedback")
        if snapshot.synthetic != self.synthetic or forecast.forecast_model != self.forecast_model:
            raise ValueError("Calibration model family or data mode differs")
        key = stratum(snapshot)
        if key not in self.temperatures:
            raise ValueError("Calibrator has no entry for this format/session")
        blend = self.blends.get(key, 1.0)  # Older artifacts retain their original behavior.
        result = transform(forecast, self.temperatures[key], blend)
        result.id = str(uuid4())
        result.warnings.append(
            f"Probability calibration: {self.method}, temperature {self.temperatures[key]:g}; "
            f"blend {blend:g}; "
            f"earlier feedback through {self.fitted_through.isoformat()}. "
            "Identity is used when evidence is insufficient or selection fails. "
            "DNF and weather probabilities are unchanged. Future calibration is not guaranteed."
        )
        return result

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise ValueError("Choose a new calibration artifact path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path):
        result = cls.model_validate(json.loads(Path(path).read_text()))
        if not result.temperatures or any(
            not np.isfinite(t) or not 0.5 <= t <= 2 for t in result.temperatures.values()
        ):
            raise ValueError("Invalid calibration temperature")
        if result.blends and (
            set(result.blends) != set(result.temperatures)
            or any(not np.isfinite(b) or not 0 <= b <= 1 for b in result.blends.values())
        ):
            raise ValueError("Invalid calibration blends")
        return result


def log_loss(examples, temperature):
    values = []
    for _, forecast, actual in examples:
        revised = transform(forecast, temperature)
        ranks = {d: i for i, d in enumerate(actual.finishing_order)}
        values.append(
            np.mean(
                [
                    -np.log(max(1e-8, s.position_probabilities[ranks[s.driver_id]]))
                    for s in revised.standings
                ]
            )
        )
    return float(np.mean(values))


def fit_calibrator(fit, selection):
    """Fit on earlier out-of-fold predictions; select identity versus fitted map on later validation."""
    if not fit or not selection:
        raise ValueError("Require separate calibration-fit and selection examples")
    examples = fit + selection
    for example in examples:
        check_example(*example)
    fit_events = {weekend_key(s) for s, _, _ in fit}
    selection_events = {weekend_key(s) for s, _, _ in selection}
    if fit_events & selection_events or max(a.available_at for _, _, a in fit) >= min(
        s.as_of for s, _, _ in selection
    ):
        raise ValueError(
            "Calibration fit and selection must be strictly chronological, disjoint weekends"
        )
    if (
        len({s.synthetic for s, _, _ in examples}) != 1
        or len({f.forecast_model for _, f, _ in examples}) != 1
    ):
        raise ValueError("Mixed forecast families or synthetic/real data")
    temperatures, audit = {}, {}
    for key in sorted({stratum(s) for s, _, _ in examples}):
        train = [r for r in fit if stratum(r[0]) == key]
        valid = [r for r in selection if stratum(r[0]) == key]
        counts = [len({weekend_key(s) for s, _, _ in group}) for group in (train, valid)]
        chosen, trials, selected = 1.0, [], False
        if counts[0] >= MIN_FIT_WEEKENDS and counts[1] >= MIN_SELECT_WEEKENDS:
            trials = [{"temperature": t, "fit_log_loss": log_loss(train, t)} for t in TEMPERATURES]
            chosen = min(
                trials,
                key=lambda r: (
                    r["fit_log_loss"] + 0.002 * np.log(r["temperature"]) ** 2,
                    abs(np.log(r["temperature"])),
                ),
            )["temperature"]
            before, after = log_loss(valid, 1), log_loss(valid, chosen)
            selected = chosen != 1 and after < before - 0.001
            audit[key] = {
                "selection_before_log_loss": before,
                "selection_candidate_log_loss": after,
            }
        temperatures[key] = chosen if selected else 1.0
        audit.setdefault(key, {}).update(
            fit_weekends=counts[0],
            selection_weekends=counts[1],
            candidate_temperature=chosen,
            selected=selected,
            trials=trials,
        )
    return ProbabilityCalibrator(
        fitted_through=max(a.available_at for _, _, a in selection),
        forecast_model=fit[0][1].forecast_model,
        synthetic=fit[0][0].synthetic,
        development_weekends=sorted(fit_events | selection_events),
        temperatures=temperatures,
        audit={
            "strata": audit,
            "rule": "Fixed temperature grid fitted on earlier OOF validation predictions; "
            "12 fit and 6 selection weekends per format/session, selection log loss improvement >0.001. "
            "No later test outcomes are supplied to fitting or selection.",
        },
    )


def fit_conservative_calibrator(fit, selection):
    """Freeze a fit-only temperature; require a small blend to pass every validation season."""
    from .engine import evaluate

    calibrator = fit_calibrator(fit, selection)
    keys = [(weekend_key(s), stratum(s)) for s, _, _ in fit + selection]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate calibration sessions")
    years = sorted({s.season for s, _, _ in selection})
    for earlier, later in pairwise(years):
        if max(a.available_at for s, _, a in selection if s.season == earlier) >= min(
            s.as_of for s, _, _ in selection if s.season == later
        ):
            raise ValueError("Validation seasons must be strictly chronological")

    def scores(rows, temperature, blend):
        values = [evaluate(transform(f, temperature, blend), a) for _, f, a in rows]
        return {k: float(np.mean([v[k] for v in values])) for k in values[0]}

    def passes(before, after):
        return (
            after["position_log_loss"] < before["position_log_loss"] - 0.001
            and after["winner_brier"] < before["winner_brier"] - 0.0001
            and after["expected_position_mae"] <= before["expected_position_mae"] + 0.01
            and after["winner_correct"] >= before["winner_correct"] - 1e-12
            and after["position_interval_width"] <= before["position_interval_width"] + 0.25
            and after["position_interval_coverage"] >= before["position_interval_coverage"] - 0.01
        )

    for key, audit in calibrator.audit["strata"].items():
        audit.pop("selection_before_log_loss", None)
        audit.pop("selection_candidate_log_loss", None)
        windows = [
            [r for r in selection if stratum(r[0]) == key and r[0].season == y] for y in years
        ]
        counts = [len({weekend_key(s) for s, _, _ in rows}) for rows in windows]
        supported = (
            len(years) >= 3
            and audit["fit_weekends"] >= MIN_FIT_WEEKENDS
            and all(n >= MIN_SELECT_WEEKENDS for n in counts)
        )
        temperature = audit["candidate_temperature"]
        trials = []
        if supported and temperature != 1:
            baseline = [scores(rows, 1, 0) for rows in windows]
            for blend in (0.1, 0.25, 0.5):
                results = [scores(rows, temperature, blend) for rows in windows]
                trials.append(
                    {
                        "blend": blend,
                        "accepted": all(
                            passes(b, a) for b, a in zip(baseline, results, strict=True)
                        ),
                        "windows": [
                            {"season": y, "before": b, "after": a}
                            for y, b, a in zip(years, baseline, results, strict=True)
                        ],
                    }
                )
        # Prefer the smallest correction that demonstrates benefit in every period.
        blend = next((r["blend"] for r in trials if r["accepted"]), 0.0)
        calibrator.temperatures[key] = temperature if blend else 1.0
        calibrator.blends[key] = blend
        audit.update(
            selected=bool(blend),
            validation_seasons=years,
            validation_weekends=counts,
            blend_trials=trials,
            supported=supported,
            fallback_reason=None
            if blend
            else "Insufficient multi-period support or no blend passed all gates",
        )
    calibrator.audit["rule"] = (
        "Fit temperature only on earlier data; select smallest blend from 0.1, 0.25, 0.5 passing "
        "every validation season (minimum 3 seasons, 6 weekends each, 12 fit weekends). "
        "Per season: log loss improves >0.001, winner Brier >0.0001, MAE increase <=0.01, "
        "winner accuracy does not decline, interval width increase <=0.25, coverage decline <=0.01. "
        "Otherwise exact identity. Later outcomes never enter parameter fitting."
    )
    return calibrator


def reliability(examples):
    """Equal-session-weight reliability bins; driver observations are not independent trials."""
    observations = {k: [] for k in ("winner", "podium", "displayed_position", "dnf")}
    losses = []
    for s, f, a in examples:
        check_example(s, f, a)
        ranks = {d: i + 1 for i, d in enumerate(a.finishing_order)}
        n = len(f.standings)
        loss, brier = [], []
        for r in f.standings:
            k = ranks[r.driver_id]
            probs = np.asarray(r.position_probabilities)
            loss.append(-np.log(max(1e-8, probs[k - 1])))
            brier.append(float(np.sum((probs - (np.arange(n) == k - 1)) ** 2)))
            for name, probability, outcome in (
                ("winner", r.win_probability, k == 1),
                ("podium", r.podium_probability, k <= 3),
                ("displayed_position", r.position_probabilities[r.position - 1], k == r.position),
                ("dnf", r.dnf_probability, r.driver_id in a.retired_drivers),
            ):
                if name != "dnf" or s.session.value == "race":
                    observations[name].append((probability, float(outcome), 1 / n, weekend_key(s)))
        losses.append((np.mean(loss), np.mean(brier)))
    output = {
        "sessions": len(examples),
        "weekends": len({weekend_key(s) for s, _, _ in examples}),
        "position_log_loss": float(np.mean([x[0] for x in losses])) if losses else None,
        "position_brier": float(np.mean([x[1] for x in losses])) if losses else None,
        "events": {},
    }
    for name, values in observations.items():
        bins = []
        total_weight = sum(v[2] for v in values)
        for i in range(10):
            rows = [v for v in values if min(9, int(v[0] * 10)) == i]
            weight = sum(r[2] for r in rows)
            mean = sum(r[0] * r[2] for r in rows) / weight if rows else None
            observed = sum(r[1] * r[2] for r in rows) / weight if rows else None
            weekends = len({r[3] for r in rows})
            bins.append(
                {
                    "low": i / 10,
                    "high": (i + 1) / 10,
                    "count": len(rows),
                    "weekends": weekends,
                    "weight": weight,
                    "predicted": mean,
                    "observed": observed,
                    "sparse": len(rows) < 30 or weekends < 10,
                }
            )
        output["events"][name] = {
            "bins": bins,
            "ece": sum(
                b["weight"] * abs(b["predicted"] - b["observed"]) for b in bins if b["count"]
            )
            / total_weight
            if values
            else None,
            "brier": sum(w * (p - y) ** 2 for p, y, w, _ in values) / total_weight
            if values
            else None,
        }
    return output

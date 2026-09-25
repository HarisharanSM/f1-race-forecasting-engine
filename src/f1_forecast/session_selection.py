"""Small session-specific probability corrections with disjoint chronological roles."""

import numpy as np

from .acceptance import acceptance_gate, interval_quality_policy, protected_metrics, selection_split
from .ml_data import chronological_split, event_groups
from .model_comparison import blend_forecasts
from .probability_calibration import check_example, transform

SHARES = (0.0, 0.05, 0.10, 0.20)
TEMPERATURES = (1.0, 0.8, 1.25)
CALIBRATION_BLEND = 0.1


def calibration_split(records, min_development_events, minimum=12):
    stages = {r.snapshot.session for r in records}
    for count in range(minimum, len(event_groups(records)) - min_development_events + 1):
        try:
            development, calibration = chronological_split(records, min_development_events, count)
        except ValueError:
            continue
        if all(
            len({r.snapshot.event_id for r in calibration if r.snapshot.session == stage})
            >= minimum
            for stage in stages
        ):
            return development, calibration
    raise ValueError("Insufficient disjoint calibration-fitting weekends")


def rolling_splits(records, config, *, folds=3, test_weekends=6, policy=None):
    policy = policy or interval_quality_policy()
    groups = event_groups(records)
    first = len(groups) - folds * test_weekends
    if first <= 0:
        raise ValueError("Insufficient later weekends")
    result = []
    for start in range(first, len(groups), test_weekends):
        later = [r for group in groups[start : start + test_weekends] for r in group]
        cutoff = min(r.snapshot.as_of for r in later)
        earlier = [r for group in groups[:start] for r in group if r.feedback.available_at < cutoff]
        before, selection = selection_split(
            earlier,
            config.min_train_events + config.validation_events + 12,
            policy,
        )
        development, calibration = calibration_split(
            before, config.min_train_events + config.validation_events
        )
        result.append(
            {
                "development": development,
                "calibration": calibration,
                "selection": selection,
                "later": later,
            }
        )
    return result


def revised_forecast(forecast, heuristic, share, temperature=1.0):
    if share not in SHARES or temperature not in TEMPERATURES:
        raise ValueError("Correction is outside the frozen candidate grid")
    mixed = forecast if share == 0 else blend_forecasts([forecast, heuristic], [1 - share, share])
    return mixed if temperature == 1 else transform(mixed, temperature, CALIBRATION_BLEND)


def fit_temperature(examples):
    """Earlier held-out calibration examples only, each (record, seed, mixed forecast)."""
    weekends = {f"{r.snapshot.season}-{r.snapshot.round:02d}" for r, _, _ in examples}
    if len(weekends) < 12:
        return 1.0, {"weekends": len(weekends), "reason": "Insufficient calibration support"}
    if len({(r.snapshot.race_format, r.snapshot.session) for r, _, _ in examples}) != 1:
        raise ValueError("Fit calibration separately by format/session")
    for r, _, forecast in examples:
        check_example(r.snapshot, forecast, r.feedback)
        if forecast.ml_training_cutoff is None or forecast.ml_training_cutoff >= r.snapshot.as_of:
            raise ValueError("Calibration forecast includes fitting outcomes or future feedback")
    scores = {}
    for t in TEMPERATURES:
        metrics = [
            protected_metrics(transform(f, t, CALIBRATION_BLEND), r.feedback)
            for r, _, f in examples
        ]
        scores[t] = (
            float(np.mean([m["position_log_loss"] + m["winner_brier"] for m in metrics]))
            + 0.002 * np.log(t) ** 2
        )
    best = min(TEMPERATURES, key=lambda t: scores[t])
    selected = best if scores[best] < scores[1.0] - 0.001 else 1.0
    return selected, {
        "weekends": len(weekends),
        "scores": scores,
        "selected": selected,
        "objective": "mean(position log loss + winner Brier) + 0.002 log(temperature)^2",
    }


def select_session_candidates(candidates, policy=None):
    """Candidates contain validation paired rows, never later outcomes."""
    policy = policy or interval_quality_policy()
    if not candidates:
        return "incumbent", {}
    stages = {(r["format"], r["session"]) for rows in candidates.values() for r in rows}
    if len(stages) != 1:
        raise ValueError("Select each format/session separately")
    panels = [
        {(r["seed"], r["weekend"], r["format"], r["session"]) for r in rows}
        for rows in candidates.values()
    ]
    if any(panel != panels[0] for panel in panels):
        raise ValueError("Candidates must share identical validation sessions and seeds")
    first = next(iter(candidates.values()))
    incumbents = {
        (r["seed"], r["weekend"], r["format"], r["session"]): r["baseline"] for r in first
    }
    if any(
        r["baseline"] != incumbents[(r["seed"], r["weekend"], r["format"], r["session"])]
        for rows in candidates.values()
        for r in rows
    ):
        raise ValueError("Candidates must share the identical incumbent forecasts")
    gates = {name: acceptance_gate(rows, policy) for name, rows in candidates.items()}
    accepted = [name for name in candidates if gates[name]["accepted"]]
    chosen = (
        min(accepted, key=lambda n: (gates[n]["overall"]["candidate"]["winner_brier"], n))
        if accepted
        else "incumbent"
    )
    return chosen, gates


def match_records(reference, enriched):
    """Enriched rows retain exact feedback, session timing, entrants and non-measurement inputs."""
    if len(reference) != len(enriched):
        raise ValueError("Enrichment must preserve all sessions")
    lookup = {(r.snapshot.event_id, r.snapshot.session): r for r in enriched}
    result = {}
    for r in reference:
        key = (r.snapshot.event_id, r.snapshot.session)
        other = lookup.get(key)
        if other is None or other.feedback != r.feedback:
            raise ValueError("Enrichment changed feedback or sessions")
        a, b = r.snapshot.model_dump(), other.snapshot.model_dump()
        for name in ("sources", "notes"):
            if b[name][: len(a[name])] != a[name]:
                raise ValueError("Enrichment replaced original provenance")
            a.pop(name)
            b.pop(name)
        for old, new in zip(a["drivers"], b["drivers"], strict=True):
            _check_added_fields(old.pop("performance"), new.pop("performance"))
        for old, new in zip(a["teams"], b["teams"], strict=True):
            _check_added_fields(old["car"].pop("performance"), new["car"].pop("performance"))
        if a != b:
            raise ValueError("Enrichment changed non-measurement inputs")
        result[key] = other
    return result


def _check_added_fields(old, new):
    if new is None and old is None:
        return
    if new is None:
        raise ValueError("Enrichment removed existing evidence")
    allowed = {
        "qualifying_teammate_delta_pct",
        "race_teammate_delta_pct",
        "clean_lap_variability_pct",
        "qualifying_gap_pct",
        "race_gap_pct",
    }
    if old is None:
        if any(
            value is not None and value != {}
            for field, value in new.items()
            if field not in allowed | {"available_at", "weight", "quality"}
        ):
            raise ValueError("Enrichment introduced an unsupported physical measurement")
        return
    for key, value in old.items():
        if key == "available_at":
            if value is not None and (new[key] is None or new[key] < value):
                raise ValueError("Enrichment backdated evidence")
        elif key == "quality":
            if any(new[key].get(k) != v for k, v in value.items()):
                raise ValueError("Enrichment replaced existing quality")
        elif value is not None and new[key] != value:
            raise ValueError("Enrichment replaced an existing measurement")
        elif value is None and new[key] is not None and key not in allowed:
            raise ValueError("Enrichment introduced an unsupported physical measurement")

"""Comparable-stint curves and conditional dry-race strategy costs, not ranking bonuses."""

from collections import defaultdict
from itertools import combinations, permutations

import numpy as np


def learn_curves(evidence):
    groups = defaultdict(list)
    for lap in evidence.laps:
        if (
            lap.clean
            and lap.dry
            and lap.circuit_id == evidence.circuit_id
            and lap.car_spec == evidence.car_spec
            and abs(lap.track_temperature_c - evidence.track_temperature_c) <= 5
        ):
            groups[lap.compound, lap.stint_id].append(lap)
    compounds = defaultdict(list)
    for (compound, _), rows in groups.items():
        if len(rows) >= 6 and np.ptp([r.age for r in rows]) >= 5:
            compounds[compound].append(rows)
    curves = {}
    for compound, stints in compounds.items():
        if len(stints) < 3:
            continue
        xs, ys = [], []
        for rows in stints:
            age = np.array([r.age for r in rows], dtype=float)
            x = np.column_stack((age, age * age))
            y = np.array([r.duration_s + r.fuel_correction_s for r in rows])
            # Remove each stint's pace intercept; give each stint equal weight.
            xs.append((x - x.mean(0)) / np.sqrt(len(rows)))
            ys.append((y - y.mean()) / np.sqrt(len(rows)))
        x, y = np.vstack(xs), np.concatenate(ys)
        if np.linalg.matrix_rank(x) < 2:
            continue
        candidates = [np.zeros(2)]
        for active in ((0,), (1,), (0, 1)):
            beta = np.linalg.lstsq(x[:, active], y, rcond=None)[0]
            if np.all(beta >= 0):
                full = np.zeros(2)
                full[list(active)] = beta
                candidates.append(full)
        beta = min(candidates, key=lambda b: np.sum((y - x @ b) ** 2))
        # Restrict strategy evaluation to ages supported by every retained stint.
        curves[compound] = {
            "linear_s_per_lap": float(beta[0]),
            "quadratic_s_per_lap2": float(beta[1]),
            "min_age": max(min(r.age for r in rows) for rows in stints),
            "max_age": min(max(r.age for r in rows) for rows in stints),
            "stints": len(stints),
            "laps": sum(map(len, stints)),
            "within_stint_rmse_s": float(np.sqrt(np.sum((y - x @ beta) ** 2) / len(stints))),
        }
    return curves


def analyze_strategy(evidence):
    curves = learn_curves(evidence)
    results = []
    # Exhaustive set orders, but pit timing uses an explicitly coarse five-lap grid.
    pit_grid = range(5, evidence.race_laps, 5)
    for stops in range(3):
        for sets in permutations(evidence.sets, stops + 1):
            if evidence.require_two_compounds and len({s.compound for s in sets}) < 2:
                continue
            if any(s.compound not in curves for s in sets):
                continue
            for pits in combinations(pit_grid, stops):
                boundaries = (0, *pits, evidence.race_laps)
                lengths = np.diff(boundaries)
                cost, valid = 0.0, True
                for tyre, length in zip(sets, lengths, strict=True):
                    curve = curves[tyre.compound]
                    ages = np.arange(tyre.initial_age + 1, tyre.initial_age + length + 1)
                    if ages[0] < curve["min_age"] or ages[-1] > curve["max_age"]:
                        valid = False
                        break
                    cost += float(
                        np.sum(
                            tyre.fresh_pace_offset_s
                            + curve["linear_s_per_lap"] * ages
                            + curve["quadratic_s_per_lap2"] * ages**2
                        )
                    )
                if valid:
                    results.append(
                        {
                            "sets": [s.id for s in sets],
                            "compounds": [s.compound for s in sets],
                            "stint_laps": lengths.tolist(),
                            "pit_laps": list(pits),
                            "stops": stops,
                            "relative_time_s": cost + stops * evidence.green_pit_loss_s,
                            "one_neutralized_stop_time_s": cost
                            + stops * evidence.green_pit_loss_s
                            - (evidence.green_pit_loss_s - evidence.neutralized_pit_loss_s)
                            if stops and evidence.neutralized_pit_loss_s is not None
                            else None,
                        }
                    )
    results.sort(key=lambda r: r["relative_time_s"])
    best_by_stops = [
        next(r for r in results if r["stops"] == n) for n in sorted({r["stops"] for r in results})
    ]
    return {
        "status": "conditional_estimates" if results else "insufficient_comparable_data_or_sets",
        "curves": curves,
        "alternatives": results[:10],
        "best_by_stops": best_by_stops,
        "strategies_evaluated": len(results),
        "limitations": "Dry-race conditional relative times, not absolute race times or win probabilities. "
        "Requires supplied fuel corrections and fresh-compound pace offsets; no fuel correction is inferred. "
        "Clean laps must exclude traffic, pit laps and neutralizations. At least three comparable stints "
        "per compound; no extrapolation beyond common observed ages. Maximum two stops, five-lap pit grid, "
        "no tyre-set reuse. Neutralized-stop time is a sensitivity case, not an incident prediction. "
        "Traffic, undercuts, warm-up, weather transitions and stochastic curve uncertainty are not simulated. "
        "The fitted curve is a proxy, not an isolated physical tyre measurement; rankings remain unchanged.",
    }

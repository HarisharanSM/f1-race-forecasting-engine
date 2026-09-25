"""Fill verified missing pace fields and collect a bounded set of recent practice archives."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from collect_full_performance import seed_collection

from f1_forecast.ml_data import QUALITY_FEATURE_NAMES, quality_matrix, records_from_json
from f1_forecast.models import utcnow
from f1_forecast.performance_collection import (
    atomic_json,
    collect_performance,
    digest,
    fastf1_catalog,
    utc,
)
from f1_forecast.performance_enrichment import enrich_snapshot, load_collection, quality_sessions


def run(source, collection, output, max_new_sessions=12):
    if output.exists():
        raise ValueError("Choose a new preparation directory")
    if not 0 <= max_new_sessions <= 12:
        raise ValueError("Collect at most twelve new practice archives")
    raw = {
        kind: json.loads((source / f"{kind}-records.json").read_text())
        for kind in ("grand_prix", "sprint")
    }
    for kind, rows in raw.items():
        reference = json.loads((source / f"{kind}-backtest.json").read_text())
        if digest(rows) != reference["dataset_sha256"]:
            raise ValueError("Forecast source checksum mismatch")
    manifest = json.loads((collection / "manifest.json").read_text())
    load_collection(collection)  # Verify immutable sources before copying or collecting.
    cache = Path("data/raw/performance/fastf1")
    cache.mkdir(parents=True, exist_ok=True)
    targets = []
    if max_new_sessions:
        # Select by date and missing archives only, never historical forecast errors.
        catalog = fastf1_catalog(2025, cache)
        targets = sorted(
            [
                m
                for m in catalog
                if m["code"] == "FP2"
                and utc(m["scheduled_start"]) < utcnow()
                and manifest["sessions"].get(f"2025-{m['round']:02d}-FP2", {}).get("status")
                != "collected"
            ],
            key=lambda m: m["scheduled_start"],
            reverse=True,
        )[:max_new_sessions]
    output.mkdir(parents=True)
    atomic_json(
        output / "preparation-plan.json",
        {
            "created_at": utcnow().isoformat(),
            "practice_targets": targets,
            "maximum_new_sessions": max_new_sessions,
            "source_hashes": {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [collection / "manifest.json", *source.glob("*-records.json")]
            },
            "rules": "Only missing fields; preserve existing values/weights and feedback. Strict quality cohorts. "
            "Identity/team match, same season, at most 30 days old, historical availability six hours after end. "
            "Use practice only. No target qualifying/race laps, race grids inferred from results, or invented tyre curves.",
        },
    )
    destination = output / "collection"
    seed_collection(destination, collection)
    if targets:
        print("Collecting bounded 2025 practice targets", flush=True)
        collect_performance(
            [2025],
            destination,
            "data/raw/performance",
            rounds=[m["round"] for m in targets],
            sessions=["FP2"],
            provider="fastf1",
            resume=True,
            max_sessions=max_new_sessions,
            progress=print,
        )
    sessions = quality_sessions(
        [s for s in load_collection(destination) if s["meta"]["code"] in {"FP1", "FP2", "FP3"}]
    )
    report = {}
    for kind, rows in raw.items():
        original = records_from_json(rows)
        enriched, audit = [], []
        for r in original:
            if r.snapshot.season >= 2023:
                snapshot, evidence = enrich_snapshot(
                    r.snapshot, sessions, quality_aware=True, fill_missing=True
                )
            else:
                snapshot, evidence = r.snapshot, []
            enriched.append(
                {
                    "snapshot": snapshot.model_dump(mode="json"),
                    "feedback": r.feedback.model_dump(mode="json"),
                }
            )
            if evidence:
                audit.append(
                    {
                        "event_id": snapshot.event_id,
                        "session": snapshot.session.value,
                        "additions": evidence,
                    }
                )
        checked = records_from_json(enriched)
        if any(a.feedback != b.feedback for a, b in zip(original, checked, strict=True)):
            raise ValueError("Enrichment changed target feedback")

        def indicators(records):
            values = np.concatenate(
                [quality_matrix(r.snapshot) for r in records if r.snapshot.season >= 2023]
            )
            return dict(zip(QUALITY_FEATURE_NAMES, values.mean(0).tolist(), strict=True))

        report[kind] = {
            "before": indicators(original),
            "after": indicators(checked),
            "changed_sessions": len(audit),
            "audit": audit,
            "input_sha256": digest(rows),
            "output_sha256": digest(enriched),
        }
        atomic_json(output / f"{kind}-records.json", enriched)
    collected = json.loads((destination / "manifest.json").read_text())
    report["collection"] = {
        "planned": len(targets),
        "paused": collected.get("paused"),
        "targets": {
            f"2025-{m['round']:02d}-FP2": collected["sessions"].get(
                f"2025-{m['round']:02d}-FP2", {}
            )
            for m in targets
        },
    }
    atomic_json(output / "input-audit.json", report)
    print(
        json.dumps(
            {
                k: {f: v[f] for f in ("before", "after", "changed_sessions")}
                for k, v in report.items()
                if k != "collection"
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/mixed-input-backtest-2010-2026")
    )
    parser.add_argument(
        "--collection", type=Path, default=Path("data/processed/performance-with-2026")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-sessions", type=int, default=12)
    args = parser.parse_args()
    run(args.source, args.collection, args.output, args.max_new_sessions)

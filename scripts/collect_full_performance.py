"""Resume bounded collection batches, stopping on provider limits or no progress."""

import argparse
import json
from pathlib import Path

from f1_forecast.performance_collection import (
    atomic_json,
    collect_performance,
    render_quality_report,
)
from f1_forecast.performance_enrichment import load_collection


def seed_collection(output, source):
    """Import verified normalized archives without issuing new provider requests."""
    if source.resolve() == output.resolve():
        raise ValueError("Choose a different seed collection")
    load_collection(source)
    origin = json.loads((source / "manifest.json").read_text())
    path = output / "manifest.json"
    target = json.loads(path.read_text()) if path.exists() else {**origin, "sessions": {}}
    if target["settings"] != origin["settings"]:
        raise ValueError("Seed collection settings differ")
    for key, entry in origin["sessions"].items():
        if (
            entry["status"] != "collected"
            or target["sessions"].get(key, {}).get("status") == "collected"
        ):
            continue
        data = json.loads((source / entry["file"]).read_text())
        atomic_json(output / entry["file"], data)
        target["sessions"][key] = {**entry, "imported_from": str(source.resolve())}
    atomic_json(path, target)
    return render_quality_report(output, target)


def run(args):
    if args.seed_collection:
        print(json.dumps(seed_collection(args.output, args.seed_collection)), flush=True)
    previous = -1
    for _ in range(args.batches):
        summary = collect_performance(
            args.seasons,
            args.output,
            "data/raw/performance",
            provider="fastf1",
            resume=args.output.exists(),
            max_sessions=40,
            progress=lambda message: (
                print(message, flush=True)
                if "verified cached" not in message and "pending" not in message
                else None
            ),
        )
        print(json.dumps(summary), flush=True)
        manifest = json.loads((args.output / "manifest.json").read_text())
        if manifest.get("paused") or not summary["sessions_pending"]:
            break
        if summary["sessions_collected"] <= previous:
            break
        previous = summary["sessions_collected"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-collection", type=Path)
    run(parser.parse_args())

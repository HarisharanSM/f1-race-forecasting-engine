"""Acquire full-model historical source material without manufacturing missing inputs."""

import argparse
import json
from pathlib import Path

from f1_forecast.historical_sources import collect_sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2010)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/full-history-2010-2026")
    )
    parser.add_argument("--max-jobs", type=int, default=20)
    parser.add_argument("--cache", default="data/raw/full-history-2010-2026")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Queue paginated lap and pit-stop sources per completed race",
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Also collect detailed timing/telemetry using supported years",
    )
    parser.add_argument("--max-sessions", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            collect_sources(
                range(args.start_season, args.end_season + 1),
                args.output,
                max_jobs=args.max_jobs,
                details=args.details,
                performance=args.performance,
                max_sessions=args.max_sessions,
                cache=args.cache,
            ),
            indent=2,
        )
    )

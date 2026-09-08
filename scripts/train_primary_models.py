"""Install a new primary Transformer bundle and gated alternatives for each race format."""

import argparse
import json
from pathlib import Path

import torch

from f1_forecast.neural import TrainingConfig
from f1_forecast.performance_collection import digest
from f1_forecast.primary_models import train_bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/incident-era-backtest-2020-2026")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/primary-models"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new primary output directory; existing models are preserved")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for kind in ("grand_prix", "sprint"):
        rows = json.loads((args.source / f"{kind}-enriched-records.json").read_text())
        reference = json.loads((args.source / f"{kind}-revised.json").read_text())
        if digest(rows) != reference["dataset_sha256"]:
            raise ValueError("Primary training source checksum mismatch")
        result = train_bundle(
            rows, TrainingConfig(**reference["reproduction_config"]), args.output / kind
        )
        print(
            json.dumps(
                {
                    "format": kind,
                    "weights": result["weights"],
                    "selected_through": result["selected_through"],
                    "rule": result["rule"],
                },
                indent=2,
            ),
            flush=True,
        )

"""Render optional tyre strategy evidence from a validated pre-race snapshot."""

import argparse
import html
import json
from pathlib import Path

from f1_forecast.models import Snapshot
from f1_forecast.tyre_strategy import analyze_strategy


def run(snapshot_path, output):
    snapshot = Snapshot.model_validate_json(snapshot_path.read_text())
    if snapshot.tyre_strategy is None:
        raise ValueError("Snapshot has no optional tyre_strategy evidence")
    result = analyze_strategy(snapshot.tyre_strategy)
    output.mkdir(parents=True, exist_ok=False)
    (output / "strategy.json").write_text(json.dumps(result, indent=2))
    rows = "".join(
        f"<tr><td>{r['stops']}</td><td>{html.escape(' / '.join(r['sets']))}</td>"
        f"<td>{html.escape(str(r['stint_laps']))}</td><td>{r['relative_time_s']:.2f}</td>"
        f"<td>{r['one_neutralized_stop_time_s']}</td></tr>"
        for r in result["alternatives"]
    )
    (output / "strategy.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>Tyre strategy analysis</title><style>body{font:18px/1.5 Georgia;background:#f5f5ef;"
        "color:#183d33;max-width:1000px;margin:auto;padding:24px}table{border-collapse:collapse}"
        "td,th{padding:12px;border-bottom:1px solid #bbb;text-align:left}.scroll{overflow:auto}</style>"
        f"<h1>Tyre strategy</h1><p>{html.escape(result['status'])}</p>"
        f"<p>{html.escape(result['limitations'])}</p><div class='scroll'><table><tr><th>Stops</th>"
        "<th>Sets</th><th>Stint laps</th><th>Relative seconds</th><th>One neutralized stop</th></tr>"
        f"{rows}</table></div><p><a href='strategy.json'>Curves and exact results</a></p></html>"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.snapshot, args.output)

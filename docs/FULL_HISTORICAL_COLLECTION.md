# Full Historical Source Collection

The full forecast input requirements remain unchanged. No reduced-input backtest was run. The abandoned reduced-input runner was removed; the earlier downloaded raw classifications remain preserved.

## Collection Command

```sh
.venv/bin/python scripts/collect_historical_sources.py \
  --start-season 2010 --end-season 2026 --details --performance \
  --max-jobs 20 --max-sessions 5
```

Re-run the same command to resume. Source jobs have checksums and are verified before reuse. Each job can involve multiple paginated requests. Provider rate limits pause collection rather than generating replacement observations. No background collection is scheduled. The source manifest and `coverage.html` live in `data/processed/full-history-2010-2026`.

The collector retrieves schedules, classifications and qualifying for all requested seasons. The details flag queues completed-race lap and pit-stop pages, preserving each page and its actual retrieval metadata. The performance flag invokes the existing FastF1 pipeline with telemetry enabled for supported years (2018 onward), including practice/qualifying/race timing, tyre stints, weather observations and race-control information. FastF1 cache and per-session provenance are retained. A timing index alone is not marked as a downloaded telemetry session.

The historical source cache is immutable. Current-year results are therefore a fixed collection vintage, not automatically refreshed records. To collect a newer vintage use both a new `--cache` and a new `--output` directory. No existing experiment sources or reports are overwritten.

## Remaining Data Requirements

Pre-2018 detailed timing is unsupported by the current feed; an alternative dataset/provider is needed. Lap times alone cannot reconstruct compounds, telemetry or fuel-corrected tyre degradation. Observed weather cannot be relabelled an archived pre-session forecast. Race-control messages need completeness verification before estimating event-free incident exposure. Verified upgrade information needs dated evidence.

`ready_for_full_backtest` remains false: this command acquires and audits source material, not completed model-ready snapshots. The full-input importer must validate coverage and historical availability before a 2010-2026 backtest can be claimed. The existing model is not weakened to work around unavailable inputs. Optional physics inputs remain optional, but must not be invented or silently marked present.

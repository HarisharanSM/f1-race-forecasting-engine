"""Chronological LLM datasets and explicit fine-tuning job submission."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .models import OutcomeAdvice, Snapshot, validate_order
from .store import Store


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_training(store: Store, output: str | Path, validation_fraction: float = 0.2) -> dict:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    records = [
        r
        for r in store.training_records()
        if r["feedback"]["verified"] and not r["feedback"]["synthetic"]
    ]
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["snapshot"]["event_id"], []).append(record)
    ordered = sorted(
        groups, key=lambda e: min(datetime.fromisoformat(r["snapshot"]["as_of"]) for r in groups[e])
    )
    if len(ordered) < 3:
        raise ValueError("Collect at least three real event weekends before exporting a time split")
    split = min(len(ordered) - 1, max(1, int(len(ordered) * (1 - validation_fraction))))
    train_events, validation_events = ordered[:split], ordered[split:]
    validation = [r for event in validation_events for r in groups[event]]
    cutoff = min(datetime.fromisoformat(r["snapshot"]["as_of"]) for r in validation)
    # Purge delayed results that were unavailable before the held-out period.
    training = [
        r
        for event in train_events
        for r in groups[event]
        if datetime.fromisoformat(r["feedback"]["available_at"]) < cutoff
    ]
    if len(training) < 10:
        raise ValueError(
            "Need at least 10 verified real training sessions before the validation cutoff"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("train.jsonl", training), ("validation.jsonl", validation)):
        with (output / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                label = OutcomeAdvice(predicted_order=row["feedback"]["finishing_order"])
                handle.write(
                    json.dumps(
                        {
                            "messages": row["messages"]
                            + [{"role": "assistant", "content": label.model_dump_json()}]
                        }
                    )
                    + "\n"
                )
    manifest = {
        "purpose": "F1 outcome ordering head; excludes actual weather/events from inputs",
        "training_examples": len(training),
        "validation_examples": len(validation),
        "training_events": sorted({r["snapshot"]["event_id"] for r in training}),
        "validation_events": validation_events,
        "validation_cutoff": cutoff.isoformat(),
        "sha256": {name: digest(output / name) for name in ("train.jsonl", "validation.jsonl")},
        "note": "Evaluate the candidate on a separate later test period before adopting it.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_dataset(directory: str | Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    event_sets = []
    for filename, minimum in (("train.jsonl", 10), ("validation.jsonl", 1)):
        path = directory / filename
        if digest(path) != manifest["sha256"][filename]:
            raise ValueError("Dataset changed after export; regenerate its verified manifest")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) < minimum:
            raise ValueError(f"{filename} needs at least {minimum} examples")
        events = set()
        for row in rows:
            messages = row["messages"]
            if [m["role"] for m in messages] != ["system", "user", "assistant"]:
                raise ValueError("Unexpected training message format")
            snapshot = Snapshot.model_validate(json.loads(messages[1]["content"])["snapshot"])
            label = OutcomeAdvice.model_validate_json(messages[2]["content"])
            if snapshot.synthetic:
                raise ValueError("Synthetic sessions cannot be used for remote fine-tuning")
            validate_order(label.predicted_order, [d.id for d in snapshot.drivers])
            events.add(snapshot.event_id)
        event_sets.append(events)
    if event_sets[0] & event_sets[1]:
        raise ValueError("Event leakage between training and validation sets")
    return manifest


def start_finetune(directory: str | Path, base_model: str, client=None) -> dict:
    """Creates a billable remote job only when explicitly called; never auto-promotes it."""
    validate_dataset(directory)
    if not base_model.strip():
        raise ValueError("Specify a base model eligible for supervised fine-tuning")
    if client is None:
        from openai import OpenAI

        client = OpenAI(timeout=60, max_retries=2)
    directory = Path(directory)
    uploaded = []
    for filename in ("train.jsonl", "validation.jsonl"):
        with (directory / filename).open("rb") as file:
            uploaded.append(client.files.create(file=file, purpose="fine-tune").id)
    job = client.fine_tuning.jobs.create(
        model=base_model,
        training_file=uploaded[0],
        validation_file=uploaded[1],
        method={"type": "supervised"},
    )
    result = {
        "job_id": job.id,
        "status": job.status,
        "base_model": base_model,
        "training_file": uploaded[0],
        "validation_file": uploaded[1],
    }
    (directory / "job.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

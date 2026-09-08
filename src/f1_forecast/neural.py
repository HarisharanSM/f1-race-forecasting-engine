"""A small, permutation-equivariant Transformer for conditional pace and retirement risk."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError(
        "Install the ML extra: uv sync --extra ml (or pip install -e '.[ml]')"
    ) from exc

from .ml_data import (
    FEATURE_NAMES,
    Record,
    chronological_split,
    excluded_drivers,
    feature_matrix,
    history_summary,
    records_from_json,
)
from .models import Scenario, Session, Snapshot, utcnow


@dataclass
class TrainingConfig:
    epochs: int = 60
    patience: int = 8
    learning_rate: float = 0.002
    batch_size: int = 16
    hidden_size: int = 32
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    min_train_events: int = 3
    validation_events: int = 1
    seed: int = 42
    optional_mode: str = "legacy"
    calibrate: bool = False

    def __post_init__(self):
        if not 1 <= self.epochs <= 2000 or not 1 <= self.patience <= 2000:
            raise ValueError("epochs and patience must be between 1 and 2000")
        if not 1 <= self.batch_size <= 256 or not 8 <= self.hidden_size <= 256:
            raise ValueError("Invalid batch or hidden size")
        if self.heads < 1 or self.hidden_size % self.heads or not 1 <= self.layers <= 6:
            raise ValueError("hidden_size must be divisible by heads; layers must be 1..6")
        if not 0 <= self.dropout < 1 or not 0 < self.learning_rate <= 0.1 or self.seed < 0:
            raise ValueError("Invalid dropout, learning rate or seed")
        if self.min_train_events < 1 or self.validation_events < 1:
            raise ValueError("Require separate training and validation events")
        if self.optional_mode not in {"legacy", "ignore", "learned"}:
            raise ValueError("optional_mode must be legacy, ignore or learned")


class FieldTransformer(nn.Module):
    """Driver rows are a set, not a time sequence. Padding is excluded from attention."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.projection = nn.Linear(len(FEATURE_NAMES), config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.heads,
            dim_feedforward=config.hidden_size * 2,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, 3)
        )
        # Encoder layers otherwise start from copies of the same initialization.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, x, padding):
        encoded = self.encoder(self.projection(x), src_key_padding_mask=padding)
        values = self.output(encoded)
        race = x[:, :, 8] > 0.5
        pace = torch.where(race, values[:, :, 1], values[:, :, 0])
        return pace, values[:, :, 2]


class IndependentRanker(nn.Module):
    """Linear pairwise ranker or small per-driver MLP, with the same session/DNF heads."""

    def __init__(self, config, nonlinear=False):
        super().__init__()
        self.output = (
            nn.Sequential(
                nn.Linear(len(FEATURE_NAMES), config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 3),
            )
            if nonlinear
            else nn.Linear(len(FEATURE_NAMES), 3)
        )

    def forward(self, x, padding):
        values = self.output(x)
        return torch.where(x[:, :, 8] > 0.5, values[:, :, 1], values[:, :, 0]), values[:, :, 2]


def make_network(config, kind):
    if kind == "field_transformer":
        return FieldTransformer(config)
    if kind in {"linear_ranker", "mlp_ranker"}:
        return IndependentRanker(config, nonlinear=kind == "mlp_ranker")
    raise ValueError("Unsupported ranking architecture")


def pack(records: list[Record], history: list[dict], optional_mode="legacy"):
    """Actual weather is a training-only condition; forecast inference uses scenarios."""
    size = max(len(r.snapshot.drivers) for r in records)
    x = torch.zeros((len(records), size, len(FEATURE_NAMES)), dtype=torch.float32)
    padding = torch.ones((len(records), size), dtype=torch.bool)
    clean = torch.zeros_like(padding)
    ranks = torch.zeros((len(records), size))
    retired = torch.zeros_like(ranks)
    for index, record in enumerate(records):
        s, a = record.snapshot, record.feedback
        if optional_mode != "legacy":
            from .measurement_features import without_performance

            s = without_performance(s)
        n = len(s.drivers)
        x[index, :n] = torch.from_numpy(
            feature_matrix(
                s,
                history,
                a.actual_weather.wet_fraction,
                a.actual_weather.air_temperature_c,
                a.actual_weather.wind_speed_ms,
            )
        )
        padding[index, :n] = False
        excluded = excluded_drivers(a)
        for i, driver in enumerate(s.drivers):
            clean[index, i] = driver.id not in excluded
            ranks[index, i] = a.finishing_order.index(driver.id)
            retired[index, i] = float(driver.id in a.retired_drivers)
    return x, padding, clean, ranks, retired


def objective(network, batch):
    x, padding, clean, ranks, retired = batch
    pace, dnf_logits = network(x, padding)
    pairs = clean[:, :, None] & clean[:, None, :]
    pairs &= torch.triu(torch.ones_like(pairs, dtype=torch.bool), diagonal=1)
    target = (ranks[:, :, None] < ranks[:, None, :]).float()
    pair_losses = F.binary_cross_entropy_with_logits(
        pace[:, :, None] - pace[:, None, :], target, reduction="none"
    )
    counts = pairs.sum(dim=(1, 2))
    ranking = (pair_losses * pairs).sum(dim=(1, 2)) / counts.clamp_min(1)
    ranking = ranking[counts > 0].mean() if (counts > 0).any() else pace.sum() * 0
    race = (x[:, :, 8] > 0.5) & ~padding
    dnf = F.binary_cross_entropy_with_logits(dnf_logits, retired, reduction="none")
    dnf = (dnf * race).sum() / race.sum().clamp_min(1)
    return ranking + 0.3 * dnf


class NeuralForecaster:
    def __init__(self, network: FieldTransformer, metadata: dict, history: list[dict]):
        self.network = network.eval()
        self.metadata = metadata
        self.history = history

    @property
    def model_id(self):
        return self.metadata["model_id"]

    def check_snapshot(self, snapshot: Snapshot):
        if snapshot.race_format not in self.metadata.get("race_formats", ["grand_prix"]):
            raise ValueError(
                "ML checkpoint has not trained this race format; use its separate model"
            )
        if datetime.fromisoformat(self.metadata["trained_through"]) > snapshot.as_of:
            raise ValueError("ML model includes future training or validation feedback")
        if snapshot.event_id in self.metadata["development_events"]:
            raise ValueError("Cannot forecast a training/validation event with this ML checkpoint")
        if snapshot.synthetic != self.metadata["synthetic"]:
            raise ValueError("Synthetic ML checkpoints cannot be mixed with real sessions")
        if snapshot.session.value not in self.metadata["trained_sessions"]:
            raise ValueError("ML checkpoint has not trained this session type")

    def scenario_parameters(self, snapshot: Snapshot, scenario: Scenario):
        self.check_snapshot(snapshot)
        mode = self.metadata["config"].get("optional_mode", "legacy")
        base_snapshot = snapshot
        if mode != "legacy":
            from .measurement_features import without_performance

            base_snapshot = without_performance(snapshot)
        matrix = feature_matrix(base_snapshot, self.history, scenario.wet_fraction)
        with torch.inference_mode():
            pace, logits = self.network(
                torch.from_numpy(matrix).unsqueeze(0),
                torch.zeros((1, len(matrix)), dtype=torch.bool),
            )
        pace = pace[0].numpy().astype(float)
        adjustment = self.metadata.get("measurement_adjustment")
        if adjustment:
            from .measurement_features import measurement_matrix

            weights = adjustment["weights"][snapshot.session.value]
            delta = measurement_matrix(
                snapshot, scenario.wet_fraction, adjustment["feature_names"]
            ) @ np.asarray(weights)
            pace += np.clip(delta, -0.5, 0.5)
        pace /= self.metadata.get("pace_temperature", 1.0)
        risks = torch.sigmoid(logits[0]).numpy().astype(float)
        if snapshot.session == Session.QUALIFYING:
            risks = np.zeros(len(matrix))
        if not np.isfinite(pace).all() or not np.isfinite(risks).all():
            raise ValueError("ML model returned nonfinite predictions")
        return pace, risks

    def save(self, directory: str | Path):
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Model directory is not empty; choose a new checkpoint directory")
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), directory / "weights.pt")
        history_text = json.dumps(self.history, sort_keys=True)
        (directory / "history.json").write_text(history_text, encoding="utf-8")
        metadata = {
            **self.metadata,
            "weights_sha256": hashlib.sha256((directory / "weights.pt").read_bytes()).hexdigest(),
            "history_sha256": hashlib.sha256(history_text.encode()).hexdigest(),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["format_version"] != 1 or metadata["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Unsupported ML checkpoint or feature schema")
        for filename, key in (("weights.pt", "weights_sha256"), ("history.json", "history_sha256")):
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != metadata[key]:
                raise ValueError("ML checkpoint checksum mismatch")
        config = TrainingConfig(**metadata["config"])
        adjustment = metadata.get("measurement_adjustment")
        if adjustment:
            from .measurement_features import LEGACY_MEASUREMENT_NAMES, MEASUREMENT_NAMES

            if adjustment.get("feature_names") not in (
                list(LEGACY_MEASUREMENT_NAMES),
                list(MEASUREMENT_NAMES),
            ):
                raise ValueError("Unsupported optional measurement schema")
            for stage in ("race", "qualifying"):
                weights = np.asarray(adjustment.get("weights", {}).get(stage, []))
                if (
                    weights.shape != (len(adjustment["feature_names"]),)
                    or not np.isfinite(weights).all()
                ):
                    raise ValueError("Invalid optional measurement coefficients")
        temperature = metadata.get("pace_temperature", 1.0)
        if not np.isfinite(temperature) or not 0.5 <= temperature <= 2:
            raise ValueError("Invalid pace calibration temperature")
        with torch.random.fork_rng(devices=[]):
            network = make_network(config, metadata["architecture"])
        network.load_state_dict(
            torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        )
        history = json.loads((directory / "history.json").read_text())
        return cls(network, metadata, history)


def fit_records(
    records: list[Record], config: TrainingConfig, *, network_kind="field_transformer"
) -> NeuralForecaster:
    train, validation = chronological_split(
        records, config.min_train_events, config.validation_events
    )
    train_history = [history_summary(r) for r in train]
    if not any(
        len(set(r.feedback.finishing_order) - excluded_drivers(r.feedback)) >= 2 for r in train
    ):
        raise ValueError("Training needs incident-free driver pairs for the pace objective")
    train_batch = pack(train, train_history, config.optional_mode)
    validation_batch = pack(validation, train_history, config.optional_mode)
    # This local RNG context makes retraining reproducible without changing the caller's seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        network = make_network(config, network_kind)
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=config.learning_rate, weight_decay=0.01
        )
        rng = np.random.default_rng(config.seed)
        best_loss, best_weights, best_epoch, stale = float("inf"), None, 0, 0
        curve = []
        network.eval()
        with torch.inference_mode():
            initial_loss = float(objective(network, train_batch))
        for epoch in range(config.epochs):
            network.train()
            order = rng.permutation(len(train))
            epoch_losses = []
            for start in range(0, len(train), config.batch_size):
                indexes = order[start : start + config.batch_size]
                batch = tuple(t[indexes] for t in train_batch)
                optimizer.zero_grad()
                loss = objective(network, batch)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite ML training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            network.eval()
            with torch.inference_mode():
                validation_loss = float(objective(network, validation_batch))
            curve.append(
                {
                    "epoch": epoch + 1,
                    "training_loss": float(np.mean(epoch_losses)),
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1e-6:
                best_loss, best_epoch, stale = validation_loss, epoch + 1, 0
                best_weights = deepcopy(network.state_dict())
            else:
                stale += 1
            if stale >= config.patience:
                break
        network.load_state_dict(best_weights)
        with torch.inference_mode():
            selected_train_loss = float(objective(network, train_batch))
    development = train + validation
    metadata = {
        "format_version": 1,
        "architecture": network_kind,
        "model_id": str(uuid4()),
        "feature_names": list(FEATURE_NAMES),
        "config": asdict(config),
        "created_at": utcnow().isoformat(),
        "trained_through": max(r.feedback.available_at for r in development).isoformat(),
        "synthetic": records[0].snapshot.synthetic,
        "data_modes": sorted({r.snapshot.data_mode for r in records}),
        "race_formats": sorted({r.snapshot.race_format for r in records}),
        "development_data_sha256": hashlib.sha256(
            json.dumps(
                [
                    {
                        "snapshot": r.snapshot.model_dump(mode="json"),
                        "feedback": r.feedback.model_dump(mode="json"),
                    }
                    for r in development
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "trained_sessions": sorted({r.snapshot.session.value for r in train}),
        "development_events": sorted({r.snapshot.event_id for r in development}),
        "training_events": sorted({r.snapshot.event_id for r in train}),
        "validation_events": sorted({r.snapshot.event_id for r in validation}),
        "training_sessions": len(train),
        "validation_sessions": len(validation),
        "validation_cutoff": min(r.snapshot.as_of for r in validation).isoformat(),
        "training_feedback_max": max(r.feedback.available_at for r in train).isoformat(),
        "initial_training_loss": initial_loss,
        "selected_training_loss": selected_train_loss,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "learning_curve": curve,
        "probabilities_calibrated": False,
    }
    model = NeuralForecaster(network, metadata, [history_summary(r) for r in development])
    if config.optional_mode == "learned" or config.calibrate:
        from .measurement_learning import fit_adjustment

        fit_adjustment(model, train, validation, config)
    return model


def train_model(
    rows: list[dict],
    output: str | Path,
    config: TrainingConfig | None = None,
    as_of: datetime | None = None,
) -> dict:
    cutoff = as_of or utcnow()
    if cutoff.utcoffset() is None:
        raise ValueError("Training cutoff must include a timezone")
    if cutoff > utcnow():
        raise ValueError("Training cutoff cannot be in the future")
    records = [r for r in records_from_json(rows) if r.feedback.available_at <= cutoff]
    model = fit_records(records, config or TrainingConfig())
    model.save(output)
    return {**model.metadata, "model_directory": str(Path(output).resolve())}

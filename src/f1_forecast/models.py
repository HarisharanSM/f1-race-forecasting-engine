"""Validated contracts. Ratings are 0..1; higher always means better capability."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Unit = Annotated[float, Field(ge=0, le=1)]
Identifier = Annotated[str, Field(min_length=1, max_length=100)]


def utcnow() -> datetime:
    return datetime.now(UTC)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Session(StrEnum):
    QUALIFYING = "qualifying"
    RACE = "race"


class Source(Model):
    url: str
    retrieved_at: AwareDatetime
    published_at: AwareDatetime | None = None
    note: str = ""
    available_at: AwareDatetime | None = None
    availability_basis: str = ""


class Car(Model):
    name: str = "Unspecified"
    qualifying_pace: Unit = 0.5
    race_pace: Unit = 0.5
    straight_speed: Unit = 0.5
    high_speed_cornering: Unit = 0.5
    low_speed_cornering: Unit = 0.5
    tyre_management: Unit = 0.5
    wet_performance: Unit = 0.5
    cooling: Unit = 0.5
    reliability: Unit = 0.95
    capabilities: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class Team(Model):
    id: Identifier
    name: str
    car: Car
    strategy: Unit = 0.5
    pit_crew: Unit = 0.5


class Driver(Model):
    id: Identifier
    name: str
    team_id: Identifier
    number: int | None = Field(default=None, ge=0)
    qualifying_skill: Unit = 0.5
    race_skill: Unit = 0.5
    wet_skill: Unit = 0.5
    consistency: Unit = 0.5


class Circuit(Model):
    id: Identifier
    name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    laps: int | None = Field(default=None, ge=1, le=200)
    straight_weight: Unit = 0.33
    high_speed_weight: Unit = 0.34
    low_speed_weight: Unit = 0.33
    overtaking: Unit = 0.5
    tyre_stress: Unit = 0.5
    disruption_probability: Unit = 0.25

    @model_validator(mode="after")
    def valid_weights(self):
        if self.straight_weight + self.high_speed_weight + self.low_speed_weight <= 0:
            raise ValueError("Circuit capability weights must have a positive sum")
        return self


class Weather(Model):
    rain_probability: Unit
    air_temperature_c: float = Field(ge=-20, le=65)
    wind_speed_ms: float = Field(ge=0, le=100)
    issued_at: AwareDatetime
    valid_at: AwareDatetime
    source: str


class Snapshot(Model):
    event_id: Identifier
    season: int = Field(ge=1950, le=2200)
    round: int = Field(ge=1, le=40)
    session: Session
    race_format: Literal["grand_prix", "sprint"] = "grand_prix"
    session_start: AwareDatetime
    as_of: AwareDatetime
    drivers: list[Driver] = Field(min_length=2, max_length=30)
    teams: list[Team] = Field(min_length=1, max_length=15)
    circuit: Circuit
    weather: Weather
    qualifying_order: list[str] | None = None
    starting_grid: list[str] | None = None
    sources: list[Source] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    synthetic: bool = False
    data_mode: Literal["live", "historical_reconstruction"] = "live"

    @model_validator(mode="after")
    def coherent(self):
        drivers = [d.id for d in self.drivers]
        teams = [t.id for t in self.teams]
        if len(drivers) != len(set(drivers)) or len(teams) != len(set(teams)):
            raise ValueError("Driver and team IDs must be unique")
        if any(d.team_id not in teams for d in self.drivers):
            raise ValueError("Every driver must reference a supplied team")
        if self.as_of >= self.session_start:
            raise ValueError("Prediction cutoff must be before the session start")
        if self.weather.issued_at > self.as_of:
            raise ValueError("Weather forecast was issued after the prediction cutoff")
        if abs((self.weather.valid_at - self.session_start).total_seconds()) > 10800:
            raise ValueError("Weather must be valid within three hours of the session")
        for source in self.sources:
            check_source_time(source, self.as_of, self.data_mode)
        if self.session == Session.RACE and self.qualifying_order is None:
            raise ValueError("Race prediction requires actual qualifying_order")
        if self.session == Session.QUALIFYING and (
            self.qualifying_order is not None or self.starting_grid is not None
        ):
            raise ValueError("Qualifying predictions cannot contain qualifying results or grid")
        for order in (self.qualifying_order, self.starting_grid):
            if order is not None:
                validate_order(order, drivers)
        return self


def validate_order(order: list[str], drivers: list[str]) -> None:
    if len(order) != len(drivers) or set(order) != set(drivers):
        raise ValueError("Order must contain every entered driver exactly once")


class Scenario(Model):
    name: str
    probability: float = Field(gt=0, le=1)
    wet_fraction: Unit
    disruption_probability: Unit
    rationale: str


class ScenarioPlan(Model):
    scenarios: list[Scenario] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def probabilities(self):
        if abs(sum(s.probability for s in self.scenarios) - 1) > 1e-6:
            raise ValueError("Scenario probabilities must sum to one")
        if len({s.name for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("Scenario names must be unique")
        return self


class OutcomeAdvice(Model):
    predicted_order: list[str]


class Standing(Model):
    position: int
    driver_id: str
    expected_position: float
    p10_position: int
    p90_position: int
    win_probability: Unit
    podium_probability: Unit
    dnf_probability: Unit
    position_probabilities: list[Unit]


class ScenarioResult(Model):
    scenario: Scenario
    winner: str
    standings: list[Standing]


class Forecast(Model):
    id: str
    event_id: str
    session: Session
    race_format: Literal["grand_prix", "sprint"] = "grand_prix"
    created_at: AwareDatetime
    model_version: int
    forecast_model: str = "heuristic"
    ml_model_id: str | None = None
    ml_training_cutoff: AwareDatetime | None = None
    simulations_per_scenario: int
    seed: int
    winner: str
    standings: list[Standing]
    scenarios: list[ScenarioResult]
    llm_order: list[str] | None
    llm_model: str | None
    warnings: list[str]


class RaceEvent(Model):
    kind: Literal["mechanical", "collision", "penalty", "red_flag", "safety_car", "other"]
    driver_id: str | None = None
    description: str


class ActualWeather(Model):
    wet_fraction: Unit
    air_temperature_c: float = Field(ge=-20, le=65)
    wind_speed_ms: float = Field(ge=0, le=100)


class Feedback(Model):
    event_id: Identifier
    session: Session
    available_at: AwareDatetime
    finishing_order: list[str] = Field(min_length=2)
    retired_drivers: list[str] = Field(default_factory=list)
    events: list[RaceEvent] = Field(default_factory=list)
    actual_weather: ActualWeather
    sources: list[Source] = Field(min_length=1)
    verified: bool = False
    synthetic: bool = False
    data_mode: Literal["live", "historical_reconstruction"] = "live"

    @model_validator(mode="after")
    def valid_feedback(self):
        if len(set(self.finishing_order)) != len(self.finishing_order):
            raise ValueError("Actual results contain duplicate drivers")
        if len(set(self.retired_drivers)) != len(self.retired_drivers):
            raise ValueError("Duplicate retired drivers")
        if not set(self.retired_drivers) <= set(self.finishing_order):
            raise ValueError("Retired drivers must appear in the official classification")
        if any(e.driver_id and e.driver_id not in self.finishing_order for e in self.events):
            raise ValueError("Event references an unknown driver")
        for source in self.sources:
            check_source_time(source, self.available_at, self.data_mode)
        return self


def check_source_time(source: Source, cutoff: datetime, mode: str):
    """Keep actual retrieval times; reconstructed availability is explicit and scoped."""
    if mode == "historical_reconstruction":
        if source.available_at is None or not source.availability_basis:
            raise ValueError("Historical sources need an explicit availability time and basis")
        available = source.available_at
    else:
        available = max(source.retrieved_at, source.available_at or source.retrieved_at)
    if available > cutoff or (source.published_at and source.published_at > cutoff):
        raise ValueError(
            "Input source is newer than the prediction cutoff or feedback availability"
        )

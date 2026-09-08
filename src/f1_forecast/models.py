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


class MeasurementQuality(Model):
    samples: int = Field(ge=1)
    cohorts: int = Field(default=1, ge=1)
    spread_pct: float | None = Field(default=None, ge=0, le=100)
    usable_fraction: Unit
    observed_at: AwareDatetime
    source_session: Literal["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"]
    source_event: str


class JointEffectEstimate(Model):
    mean_pct: float = Field(ge=-20, le=20)
    std_pct: float = Field(ge=0, le=100)
    available_at: AwareDatetime
    observed_at: AwareDatetime
    season: int = Field(ge=1950, le=2200)
    samples: int = Field(ge=1)
    sessions: int = Field(ge=1)
    current_season_observed: bool
    prior_dependent: bool = True
    upgrade_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def coherent(self):
        if self.observed_at > self.available_at:
            raise ValueError("Joint evidence cannot be available before it was observed")
        return self


class CarUpgrade(Model):
    id: Identifier
    introduced_at: AwareDatetime
    available_at: AwareDatetime
    source_url: str = Field(min_length=1)
    description: str = ""


class TeamUpgrade(CarUpgrade):
    team_id: Identifier


class PerformanceEvidence(Model):
    available_at: AwareDatetime | None = None
    weight: Unit = 0.5
    quality: dict[str, MeasurementQuality] = Field(default_factory=dict)
    joint_effects: dict[Literal["qualifying", "race"], JointEffectEstimate] = Field(
        default_factory=dict
    )


class CarPerformance(PerformanceEvidence):
    qualifying_gap_pct: float | None = Field(default=None, ge=-20, le=20)
    race_gap_pct: float | None = Field(default=None, ge=-20, le=20)
    braking: Unit | None = None
    aerodynamics: Unit | None = None
    handling: Unit | None = None
    power_delivery: Unit | None = None
    medium_speed_cornering: Unit | None = None
    downforce: Unit | None = None
    aerodynamic_efficiency: Unit | None = None
    tyre_degradation_s_per_lap: float | None = Field(default=None, ge=0, le=2)


class DriverPerformance(PerformanceEvidence):
    qualifying_teammate_delta_pct: float | None = Field(default=None, ge=-20, le=20)
    race_teammate_delta_pct: float | None = Field(default=None, ge=-20, le=20)
    clean_lap_variability_pct: float | None = Field(default=None, ge=0, le=20)
    tyre_management: Unit | None = None
    # Total per-race DNF probability replaces, rather than adds to, existing DNF risk.
    retirement_probability: Unit | None = None


class InterruptionEpisode(Model):
    kind: Literal["incident", "safety_car", "virtual_safety_car", "red_flag"]
    start_fraction: Unit
    duration_fraction: Unit = 0
    affected_drivers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_episode(self):
        if self.start_fraction + self.duration_fraction > 1:
            raise ValueError("Interruption extends beyond race distance")
        if len(set(self.affected_drivers)) != len(self.affected_drivers):
            raise ValueError("Duplicate affected drivers")
        return self


class InterruptionObservation(Model):
    event_id: str
    season: int
    round: int
    circuit_id: str
    race_format: Literal["grand_prix", "sprint"]
    observed_at: AwareDatetime
    available_at: AwareDatetime
    source: str = Field(min_length=1)
    complete: bool
    entrants: list[str] = Field(min_length=2)
    episodes: list[InterruptionEpisode] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_observation(self):
        if self.observed_at > self.available_at:
            raise ValueError("Observation availability precedes observation")
        if len(set(self.entrants)) != len(self.entrants):
            raise ValueError("Duplicate incident-history entrants")
        if any(not set(e.affected_drivers) <= set(self.entrants) for e in self.episodes):
            raise ValueError("Unknown affected driver")
        return self


class RaceDynamics(Model):
    available_at: AwareDatetime | None = None
    safety_car_probability: Unit = 0
    virtual_safety_car_probability: Unit = 0
    red_flag_probability: Unit = 0
    event_lap_fraction: Unit | None = None
    neutralized_fraction: Unit = 0.1
    pit_opportunity_probability: Unit = 0.5
    restart_variability: Unit = 0.25
    history: list[InterruptionObservation] = Field(default_factory=list, max_length=1000)
    seconds_per_score: float = Field(default=20, gt=0, le=300)
    incident_delay_s: float = Field(default=10, ge=0, le=300)
    green_pit_loss_s: float = Field(default=22, ge=0, le=120)
    neutralized_pit_loss_s: float = Field(default=12, ge=0, le=120)


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
    performance: CarPerformance | None = None
    upgrades: list[CarUpgrade] = Field(default_factory=list)
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
    performance: DriverPerformance | None = None


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


class TyreLap(Model):
    stint_id: str
    lap: int = Field(ge=1)
    age: int = Field(ge=0, le=150)
    duration_s: float = Field(gt=0, le=600)
    fuel_correction_s: float = Field(ge=0, le=30)
    observed_at: AwareDatetime
    available_at: AwareDatetime
    circuit_id: str
    car_spec: str
    compound: Literal["SOFT", "MEDIUM", "HARD"]
    track_temperature_c: float = Field(ge=0, le=80)
    clean: bool
    dry: bool


class TyreSet(Model):
    id: str
    compound: Literal["SOFT", "MEDIUM", "HARD"]
    initial_age: int = Field(default=0, ge=0, le=150)
    fresh_pace_offset_s: float = Field(ge=-10, le=10)


class TyreStrategyInput(Model):
    available_at: AwareDatetime
    source: str = Field(min_length=1)
    circuit_id: str
    car_spec: str
    track_temperature_c: float = Field(ge=0, le=80)
    race_laps: int = Field(ge=5, le=100)
    green_pit_loss_s: float = Field(gt=0, le=120)
    neutralized_pit_loss_s: float | None = Field(default=None, gt=0, le=120)
    require_two_compounds: bool = True
    sets: list[TyreSet] = Field(min_length=1, max_length=6)
    laps: list[TyreLap] = Field(default_factory=list, max_length=20000)

    @model_validator(mode="after")
    def unique_evidence(self):
        if len({s.id for s in self.sets}) != len(self.sets):
            raise ValueError("Tyre set IDs must be unique")
        if len({(r.stint_id, r.lap) for r in self.laps}) != len(self.laps):
            raise ValueError("Duplicate stint laps")
        if any(r.observed_at > r.available_at for r in self.laps):
            raise ValueError("Tyre lap availability precedes observation")
        return self


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
    race_dynamics: RaceDynamics | None = None
    tyre_strategy: TyreStrategyInput | None = None
    sources: list[Source] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    synthetic: bool = False
    data_mode: Literal["live", "historical_reconstruction"] = "live"

    @model_validator(mode="after")
    def coherent(self):
        if self.race_dynamics is not None:
            history = self.race_dynamics.history
            if len({(h.season, h.round, h.race_format) for h in history}) != len(history):
                raise ValueError("Duplicate interruption history weekends")
            if any(
                h.available_at >= self.as_of or (h.season, h.round) == (self.season, self.round)
                for h in history
            ):
                raise ValueError("Interruption history includes target weekend or future evidence")
        if self.tyre_strategy is not None:
            t = self.tyre_strategy
            if self.session != Session.RACE:
                raise ValueError("Tyre strategy requires a race snapshot")
            if t.circuit_id != self.circuit.id:
                raise ValueError("Tyre strategy circuit differs from snapshot")
            if self.circuit.laps is not None and t.race_laps != self.circuit.laps:
                raise ValueError("Tyre strategy race length differs from snapshot")
            if t.available_at > self.as_of or any(r.available_at > self.as_of for r in t.laps):
                raise ValueError("Tyre strategy evidence is newer than prediction cutoff")
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
        evidence = [d.performance for d in self.drivers]
        evidence.extend(t.car.performance for t in self.teams)
        evidence.append(self.race_dynamics)
        if any(e and e.available_at and e.available_at > self.as_of for e in evidence):
            raise ValueError("Optional input evidence is newer than the prediction cutoff")
        for e in evidence:
            if isinstance(e, PerformanceEvidence) and any(
                j.season != self.season for j in e.joint_effects.values()
            ):
                raise ValueError("Joint effect must target the snapshot season")
            if isinstance(e, PerformanceEvidence) and any(
                j.available_at > self.as_of or j.observed_at > self.as_of
                for j in e.joint_effects.values()
            ):
                raise ValueError("Joint performance evidence is newer than prediction cutoff")
        for team in self.teams:
            if len({u.id for u in team.car.upgrades}) != len(team.car.upgrades):
                raise ValueError("Car upgrade IDs must be unique within a team")
            if any(u.available_at > self.as_of for u in team.car.upgrades):
                raise ValueError("Car upgrade information is newer than prediction cutoff")
        if any(
            q.observed_at > self.as_of
            for e in evidence
            if isinstance(e, PerformanceEvidence)
            for q in e.quality.values()
        ):
            raise ValueError("Optional measurement quality is newer than the prediction cutoff")
        if self.session == Session.QUALIFYING and self.race_dynamics is not None:
            raise ValueError("race_dynamics is only supported for race predictions")
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
    race_event_rates: dict[str, Unit] = Field(default_factory=dict)


class Forecast(Model):
    interruption_analysis: dict | None = None
    tyre_strategy_analysis: dict | None = None
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

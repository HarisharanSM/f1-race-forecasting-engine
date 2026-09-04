"""Python API that orchestrates prediction, memory retrieval, and persistence."""

from pathlib import Path

from .engine import predict
from .llm import LLM, outcome_messages
from .models import Feedback, Forecast, ScenarioPlan, Snapshot
from .store import Store


class ForecastService:
    def __init__(self, database="data/forecast.sqlite3", llm: LLM | None = None, ml_model=None):
        self.store = Store(database)
        self.llm = llm
        if isinstance(ml_model, (str, Path)):
            from .neural import NeuralForecaster

            ml_model = NeuralForecaster.load(ml_model)
        self.ml_model = ml_model

    def predict(
        self,
        snapshot: Snapshot,
        *,
        simulations=5000,
        seed=42,
        max_scenarios=5,
        plan: ScenarioPlan | None = None,
    ) -> Forecast:
        if not 1 <= max_scenarios <= 5 or not 100 <= simulations <= 100000 or seed < 0:
            raise ValueError(
                "Require 1..5 scenarios, 100..100000 simulations, and a nonnegative seed"
            )
        state = self.store.state_for(snapshot)
        if self.ml_model:
            self.ml_model.check_snapshot(snapshot)
        messages = outcome_messages(snapshot, self.store.memories(snapshot))
        advice = None
        if self.llm:
            plan = plan or self.llm.scenarios(snapshot, max_scenarios)
            advice = self.llm.outcome(snapshot, messages)
        forecast = predict(
            snapshot,
            state,
            simulations=simulations,
            seed=seed,
            max_scenarios=max_scenarios,
            plan=plan,
            llm_order=advice.predicted_order if advice else None,
            llm_model=self.llm.outcome_model if self.llm else None,
            ml_model=self.ml_model,
        )
        self.store.save(snapshot, forecast, messages)
        return forecast

    def feedback(self, forecast_id: str, actual: Feedback) -> dict:
        return self.store.feedback(forecast_id, actual)

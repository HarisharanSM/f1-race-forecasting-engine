"""Optional local HTTP API. Run: uvicorn f1_forecast.api:app --host 127.0.0.1."""

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import Field

from .llm import LLM
from .models import Feedback, Forecast, Model, ScenarioPlan, Snapshot
from .service import ForecastService


class PredictionRequest(Model):
    snapshot: Snapshot
    simulations: int = Field(default=5000, ge=100, le=100000)
    seed: int = Field(default=42, ge=0)
    max_scenarios: int = Field(default=5, ge=1, le=5)
    use_llm: bool = False
    use_ml: bool = True
    model_policy: Literal["primary", "ensemble"] = "primary"
    plan: ScenarioPlan | None = None


def create_app(database: str | None = None, ml_model=None) -> FastAPI:
    application = FastAPI(title="F1 Race Forecasting Engine", version="0.1.0")

    def service(use_llm=False, use_ml=True, model_policy="primary"):
        return ForecastService(
            database or os.getenv("F1_DATABASE", "data/forecast.sqlite3"),
            llm=LLM() if use_llm else None,
            ml_model=(ml_model or os.getenv("F1_ML_MODEL")) if use_ml else None,
            model_policy=model_policy if use_ml else "heuristic",
        )

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.post("/predictions", response_model=Forecast)
    def forecast(request: PredictionRequest):
        try:
            return service(request.use_llm, request.use_ml, request.model_policy).predict(
                request.snapshot,
                simulations=request.simulations,
                seed=request.seed,
                max_scenarios=request.max_scenarios,
                plan=request.plan,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/predictions/{forecast_id}", response_model=Forecast)
    def get_forecast(forecast_id: str):
        try:
            return service().store.get(forecast_id)[1]
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/predictions/{forecast_id}/feedback")
    def feedback(forecast_id: str, actual: Feedback):
        try:
            return service().feedback(forecast_id, actual)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/model")
    def model():
        state = service().store.state()
        return {"version": state.version, "trained_through": state.trained_through}

    return application


app = create_app()

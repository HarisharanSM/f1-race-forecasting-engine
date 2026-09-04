from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from f1_forecast.api import create_app
from f1_forecast.data import DataUnavailable, Fetcher, Jolpica, OpenMeteo, fetch_actual
from f1_forecast.demo import weekend
from f1_forecast.llm import LLM, Document, EvidenceClaim, EvidenceReport, outcome_messages
from f1_forecast.models import OutcomeAdvice, Session, Source, utcnow
from f1_forecast.service import ForecastService


class StubResponses:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.result)


def llm_stub(result):
    responses = StubResponses(result)
    return LLM(model="test-only-model", client=SimpleNamespace(responses=responses)), responses


def test_web_research_preserves_clickable_citations_and_refreshes_cutoff():
    q = weekend()[0]
    now = utcnow()
    data = q.model_dump()
    data.update(
        synthetic=False, as_of=now - timedelta(minutes=5), session_start=now + timedelta(days=1)
    )
    data["weather"].update(issued_at=now - timedelta(hours=1), valid_at=data["session_start"])
    from f1_forecast.models import Snapshot

    snapshot = Snapshot.model_validate(data)

    def create(**kwargs):
        assert kwargs["tools"][0]["type"] == "web_search"
        citation = SimpleNamespace(
            type="url_citation",
            title="Official report",
            url="https://www.formula1.com/example",
            start_index=6,
            end_index=11,
        )
        part = SimpleNamespace(text="Report[ref]", annotations=[citation])
        return SimpleNamespace(output=[SimpleNamespace(type="message", content=[part])])

    llm = LLM(model="test-only", client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    enriched, report = llm.research(snapshot)
    assert enriched.as_of >= now
    assert "[Official report](https://www.formula1.com/example)" in report["text"]
    assert report["verified"] is False
    assert enriched.drivers == snapshot.drivers
    with pytest.raises(ValueError, match="future session"):
        llm.research(q)


def test_llm_outcome_uses_structured_contract_and_checks_roster():
    q = weekend()[0]
    advice = OutcomeAdvice(predicted_order=[d.id for d in q.drivers])
    llm, responses = llm_stub(advice)
    assert llm.outcome(q, outcome_messages(q, [])) == advice
    assert responses.calls[0]["text_format"] is OutcomeAdvice
    assert responses.calls[0]["store"] is False
    llm, _ = llm_stub(OutcomeAdvice(predicted_order=["invented"]))
    with pytest.raises(ValueError):
        llm.outcome(q, [])


def test_refusal_is_not_silently_treated_as_prediction():
    llm, _ = llm_stub(None)
    with pytest.raises(ValueError, match="refusal"):
        llm.outcome(weekend()[0], [])


def test_evidence_requires_verbatim_quote_and_cannot_change_unsupported_traits():
    q = weekend()[0]
    document = Document(source=q.sources[0], text="Aurora has improved its straight line speed.")
    claim = EvidenceClaim(
        team_id="aurora",
        trait="straight_speed",
        value=1,
        confidence=0.8,
        quote="improved its straight line speed",
        source_index=0,
    )
    llm, _ = llm_stub(EvidenceReport(claims=[claim], unknowns=[]))
    enriched, _ = llm.extract(q, [document])
    assert enriched.teams[0].car.straight_speed == pytest.approx(0.92)
    assert enriched.teams[0].car.race_pace == q.teams[0].car.race_pace
    assert q.teams[0].car.straight_speed == 0.8
    bad = claim.model_copy(update={"quote": "invented supporting evidence"})
    llm, _ = llm_stub(EvidenceReport(claims=[bad], unknowns=[]))
    with pytest.raises(ValueError, match="absent"):
        llm.extract(q, [document])


def test_llm_actuals_cannot_certify_themselves():
    q, actual, _, _ = weekend()
    llm, _ = llm_stub(actual)
    doc = Document(source=actual.sources[0], text="An externally supplied session report")
    result = llm.extract_actual([doc], q.event_id, q.session)
    assert result.verified is False
    assert result.synthetic is False


def test_future_evidence_is_rejected_before_llm_call():
    q = weekend()[0]
    doc = Document(
        source=Source(url="https://example.org", retrieved_at=q.as_of + timedelta(hours=1)),
        text="Too late",
    )
    llm, calls = llm_stub(EvidenceReport(claims=[], unknowns=[]))
    with pytest.raises(ValueError, match="newer"):
        llm.extract(q, [doc])
    assert not calls.calls


def test_jolpica_sorts_numeric_positions_and_archives_raw_source(tmp_path):
    def handler(request):
        assert request.url.params["limit"] == "100"
        return httpx.Response(
            200,
            json={
                "MRData": {
                    "RaceTable": {
                        "Races": [
                            {
                                "season": "2025",
                                "round": "1",
                                "QualifyingResults": [
                                    {"position": "2", "Driver": {"driverId": "second"}},
                                    {"position": "1", "Driver": {"driverId": "first"}},
                                ],
                            }
                        ]
                    }
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetcher = Fetcher(client, tmp_path)
        rows, source = Jolpica(fetcher).results(2025, 1, Session.QUALIFYING)
    assert rows[0]["Driver"]["driverId"] == "first"
    assert source.url.startswith("https://api.jolpi.ca/")
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"MRData": {"RaceTable": {"Races": []}}},
        {},
        {"MRData": {"RaceTable": {"Races": [{"season": "2024", "round": "1"}]}}},
    ],
)
def test_missing_or_mismatched_results_fail_closed(payload):
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
        ) as client,
        pytest.raises(DataUnavailable),
    ):
        Jolpica(Fetcher(client, archive=None)).results(2025, 1, Session.RACE)


def test_weather_units_and_nearest_session_hour():
    target = (utcnow() + timedelta(days=1)).replace(minute=0, second=0, microsecond=0)

    def handler(request):
        assert request.url.params["wind_speed_unit"] == "ms"
        return httpx.Response(
            200,
            json={
                "hourly": {
                    "time": [target.strftime("%Y-%m-%dT%H:%M")],
                    "temperature_2m": [24],
                    "precipitation_probability": [60],
                    "wind_speed_10m": [4.2],
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        weather, _ = OpenMeteo(Fetcher(client, archive=None)).forecast(weekend()[0].circuit, target)
    assert weather.rain_probability == 0.6
    assert weather.wind_speed_ms == 4.2


def test_actual_collection_crosschecks_session_and_requires_observed_weather():
    snapshot = weekend()[2].model_copy(update={"synthetic": False})

    def handler(request):
        if request.url.host == "api.jolpi.ca":
            rows = [
                {"position": str(i + 1), "Driver": {"driverId": d.id}, "status": "Finished"}
                for i, d in enumerate(snapshot.drivers)
            ]
            return httpx.Response(
                200,
                json={
                    "MRData": {
                        "RaceTable": {
                            "Races": [
                                {
                                    "season": "2025",
                                    "round": "1",
                                    "Results": rows,
                                }
                            ]
                        }
                    }
                },
            )
        if request.url.path.endswith("sessions"):
            return httpx.Response(
                200,
                json=[
                    {
                        "session_name": "Race",
                        "year": 2025,
                        "date_start": snapshot.session_start.isoformat(),
                        "date_end": (snapshot.session_start + timedelta(hours=2)).isoformat(),
                    }
                ],
            )
        return httpx.Response(200, json=[])

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(DataUnavailable, match="Actual session weather"),
    ):
        fetch_actual(Fetcher(client, archive=None), snapshot, 123)


def test_http_prediction_feedback_and_validation(tmp_path):
    client = TestClient(create_app(str(tmp_path / "api.sqlite3")))
    q, actual, _, _ = weekend()
    response = client.post(
        "/predictions", json={"snapshot": q.model_dump(mode="json"), "simulations": 100}
    )
    assert response.status_code == 200
    forecast_id = response.json()["id"]
    assert client.get(f"/predictions/{forecast_id}").status_code == 200
    assert (
        client.post(
            f"/predictions/{forecast_id}/feedback", json=actual.model_dump(mode="json")
        ).status_code
        == 200
    )
    assert client.get("/model").json()["version"] == 1
    assert (
        client.post(
            "/predictions", json={"snapshot": q.model_dump(mode="json"), "max_scenarios": 6}
        ).status_code
        == 422
    )
    assert client.get("/predictions/missing").status_code == 404


def test_llm_and_numerical_paths_are_wired_together(tmp_path):
    q = weekend()[0]

    class AdviceProvider:
        outcome_model = "test-model"

        def scenarios(self, snapshot, maximum):
            from f1_forecast.engine import LearnerState, default_scenarios

            return default_scenarios(snapshot, LearnerState(), maximum)

        def outcome(self, snapshot, messages):
            return OutcomeAdvice(predicted_order=list(reversed([d.id for d in snapshot.drivers])))

    service = ForecastService(tmp_path / "llm.sqlite3", AdviceProvider())
    result = service.predict(q, simulations=100)
    assert result.llm_order[0] == q.drivers[-1].id
    assert result.llm_model == "test-model"
    assert len(result.scenarios) <= 5

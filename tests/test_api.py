"""Integration tests for the FastAPI inference service.

Uses FastAPI's TestClient (which wraps httpx) to call the endpoints without
needing a running uvicorn server. Tests cover:

- Health endpoint returns 'ok'
- Root endpoint returns project metadata
- /examples returns a list of demo engines
- /predict/xgb and /predict/lstm work end-to-end on a valid window
- Validation errors fire on malformed inputs (wrong shape, missing fields)
- The two models produce different predictions on the same input
  (regression test for the LSTM-hedges-at-the-cap finding from Week 5)
"""

import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture(scope="module")
def client():
    """Single TestClient shared across the module — avoids re-loading models per test."""
    return TestClient(app)


@pytest.fixture
def zeros_window():
    """A 30x14 zeros window — corresponds to 'perfectly average healthy' z-scored input."""
    return [[0.0] * 14 for _ in range(30)]


@pytest.fixture
def valid_request_payload(zeros_window):
    """A valid PredictionRequest payload as a dict."""
    return {
        "window": zeros_window,
        "is_normalized": True,
        "engine_id": 999,
    }


# ----- Basic endpoints -----


def test_root_returns_project_info(client):
    """GET / returns the expected project metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "predictive-maintenance-cmapss"
    assert "endpoints" in data
    assert data["alarm_threshold"] == 35


def test_health_returns_ok(client):
    """GET /health returns status 'ok'."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_examples_returns_six_demo_engines(client):
    """GET /examples returns the canned demo engines from app/example_data/."""
    response = client.get("/examples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 6
    # Each engine should have the expected fields
    for engine in data:
        assert "engine_id" in engine
        assert "actual_rul" in engine
        assert "regime" in engine
        assert "window" in engine
        assert len(engine["window"]) == 30
        assert len(engine["window"][0]) == 14


def test_examples_span_three_regimes(client):
    """The canned demo engines should cover all three operational regimes."""
    response = client.get("/examples")
    regimes = {e["regime"] for e in response.json()}
    assert regimes == {"Critical", "Watch", "Healthy"}


# ----- Prediction endpoints -----


def test_predict_xgb_zeros_window(client, valid_request_payload):
    """XGBoost predicts the cap (~125) for a zeros window (no degradation signal)."""
    response = client.post("/predict/xgb", json=valid_request_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "xgb"
    assert data["engine_id"] == 999
    assert data["alarm_threshold"] == 35
    # Zeros window → XGBoost commits to the cap
    assert data["predicted_rul"] >= 120
    assert data["alarm"] is False
    assert data["regime"] == "Healthy"


def test_predict_lstm_zeros_window(client, valid_request_payload):
    """LSTM predicts a sensible high value for a zeros window — but hedges below the cap.

    This is a regression test for the Week 5 finding that the LSTM doesn't fully
    commit to predicting 125 even when the input is maximally healthy.
    """
    response = client.post("/predict/lstm", json=valid_request_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "lstm"
    assert data["predicted_rul"] >= 90
    assert data["regime"] == "Healthy"
    assert data["alarm"] is False


def test_xgb_and_lstm_differ_on_zeros_window(client, valid_request_payload):
    """The two models should produce different predictions on the same input.

    This locks in the classical-vs-deep finding: XGBoost commits to 125,
    LSTM hedges around 110. They are genuinely different models with different
    behaviors.
    """
    xgb_response = client.post("/predict/xgb", json=valid_request_payload)
    lstm_response = client.post("/predict/lstm", json=valid_request_payload)
    xgb_rul = xgb_response.json()["predicted_rul"]
    lstm_rul = lstm_response.json()["predicted_rul"]
    # They should differ meaningfully — at least 5 cycles apart
    assert abs(xgb_rul - lstm_rul) > 5, (
        f"XGBoost ({xgb_rul}) and LSTM ({lstm_rul}) should produce different "
        f"predictions on the same input"
    )


def test_predict_xgb_on_critical_demo_engine(client):
    """XGBoost should predict low RUL for an engine genuinely near failure.

    Uses the first Critical demo engine (engine_id=34, actual RUL=7) from the
    canned data — the model should predict somewhere in the 0-30 range.
    """
    examples = client.get("/examples").json()
    critical_engine = next(e for e in examples if e["regime"] == "Critical")
    payload = {
        "window": critical_engine["window"],
        "is_normalized": True,
        "engine_id": critical_engine["engine_id"],
    }
    response = client.post("/predict/xgb", json=payload)
    assert response.status_code == 200
    data = response.json()
    # The Critical-regime calibration analysis (Task 2 of Week 6) showed bias <3
    # cycles, so prediction should be within ~10 of actual
    assert abs(data["predicted_rul"] - critical_engine["actual_rul"]) < 15
    # Alarm should fire (model says <= 35)
    assert data["alarm"] is True
    assert data["regime"] == "Critical"


# ----- Validation tests -----


def test_predict_rejects_wrong_window_length(client):
    """A window with too few timesteps returns a 422 validation error."""
    bad_payload = {
        "window": [[0.0] * 14 for _ in range(10)],  # only 10 timesteps
        "is_normalized": True,
    }
    response = client.post("/predict/xgb", json=bad_payload)
    assert response.status_code == 422  # Pydantic validation error
    assert "30 timesteps" in response.text


def test_predict_rejects_wrong_sensor_count(client):
    """A window with wrong number of sensors returns 422."""
    bad_payload = {
        "window": [[0.0] * 10 for _ in range(30)],  # 10 sensors instead of 14
        "is_normalized": True,
    }
    response = client.post("/predict/xgb", json=bad_payload)
    assert response.status_code == 422
    assert "14 sensor" in response.text


def test_predict_rejects_missing_window(client):
    """A request missing the window field returns 422."""
    bad_payload = {"is_normalized": True}
    response = client.post("/predict/xgb", json=bad_payload)
    assert response.status_code == 422

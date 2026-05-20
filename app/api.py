"""
FastAPI inference service for the predictive maintenance demo.

Endpoints:
- GET  /              : Root, returns project info.
- GET  /health        : Liveness check.
- GET  /examples      : List of canned demo engines for the dashboard dropdown.
- POST /predict/xgb   : Predict RUL using the honest-tuned XGBoost model.
- POST /predict/lstm  : Predict RUL using the LSTM-64x2 model.

Run locally:
    uvicorn app.api:app --reload --port 8000

Then visit http://localhost:8000/docs for the auto-generated Swagger UI.
"""

import json
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.inference import InferenceEngine, ALARM_THRESHOLD
from app.schemas import PredictionRequest, PredictionResponse, HealthResponse

# Set up the app with metadata
app = FastAPI(
    title="C-MAPSS FD001 RUL Prediction API",
    description=(
        "Inference service for the predictive-maintenance-cmapss project.\n\n"
        "Two models served:\n"
        "- **XGBoost** (recommended for deployment): Test RMSE 11.62, F1=0.98 at T=35.\n"
        "- **LSTM-64×2** (for comparison): Test RMSE 12.66.\n\n"
        "See [project README](https://github.com/beebzy-droid/predictive-maintenance-cmapss) "
        "for the full classical-vs-deep comparison."
    ),
    version="0.1.0",
)

# CORS — allow Streamlit (default port 8501) to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Singleton inference engine — loaded lazily on first prediction request
_engine = InferenceEngine()


# Load canned demo engines once at startup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA_PATH = PROJECT_ROOT / "app" / "example_data" / "demo_engines.json"

if DEMO_DATA_PATH.exists():
    with open(DEMO_DATA_PATH) as f:
        DEMO_ENGINES = json.load(f)
else:
    DEMO_ENGINES = []


@app.get("/")
def root() -> dict[str, Any]:
    """Project info."""
    return {
        "project": "predictive-maintenance-cmapss",
        "description": "C-MAPSS FD001 RUL prediction with classical vs deep comparison.",
        "endpoints": ["/health", "/examples", "/predict/xgb", "/predict/lstm", "/docs"],
        "alarm_threshold": ALARM_THRESHOLD,
        "demo_engines_available": len(DEMO_ENGINES),
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service liveness check. Returns 'ok' if the service is up.

    Does NOT pre-load models — they load lazily on first prediction request,
    so the service can boot quickly even on slow disks.
    """
    return HealthResponse(
        status="ok",
        models_loaded=_engine.loaded_models(),
    )


@app.get("/examples")
def list_examples() -> list[dict[str, Any]]:
    """Return the canned demo engines (used by the dashboard dropdown).

    Each entry has the engine_id, actual RUL, regime, and the full sensor window.
    """
    if not DEMO_ENGINES:
        raise HTTPException(
            status_code=503,
            detail=(
                "Demo engines file not found. Run `python app/extract_demo_engines.py` "
                "to generate it."
            ),
        )
    return DEMO_ENGINES


@app.post("/predict/xgb", response_model=PredictionResponse)
def predict_xgb(request: PredictionRequest) -> PredictionResponse:
    """Predict RUL using the honest-tuned XGBoost model.

    The window must be 30 timesteps × 14 sensors, in the order defined by
    `src.features.KEEP_SENSORS`. If `is_normalized=false`, the API will
    z-score using the saved scaler before predicting.
    """
    try:
        result = _engine.predict(
            window=request.window,
            model_name="xgb",
            is_normalized=request.is_normalized,
            engine_id=request.engine_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return PredictionResponse(**result)


@app.post("/predict/lstm", response_model=PredictionResponse)
def predict_lstm(request: PredictionRequest) -> PredictionResponse:
    """Predict RUL using the LSTM-64×2 model (kept for comparison purposes)."""
    try:
        result = _engine.predict(
            window=request.window,
            model_name="lstm",
            is_normalized=request.is_normalized,
            engine_id=request.engine_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return PredictionResponse(**result)

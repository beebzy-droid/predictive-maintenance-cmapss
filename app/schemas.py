"""
Pydantic schemas for FastAPI request/response models.

These define the shape of data that comes in (PredictionRequest) and
goes out (PredictionResponse) of the inference endpoints.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """A single window of sensor readings for one engine.

    Expected shape: 30 timesteps × 14 sensors.
    Sensor order matches src.features.KEEP_SENSORS:
    [sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9, sensor_11,
     sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21]

    Values should be z-scored using the same StandardScaler fit on the FD001
    training set (saved in data/processed/fd001_scaler.joblib). If you provide
    raw sensor values, set `is_normalized=False` and the API will scale them
    server-side.
    """

    window: List[List[float]] = Field(
        ...,
        description="Sensor window: list of 30 timesteps, each a list of 14 sensor values.",
    )
    is_normalized: bool = Field(
        default=True,
        description="If False, the API will z-score the window using the saved scaler.",
    )
    engine_id: int | None = Field(
        default=None,
        description="Optional engine identifier for logging / display.",
    )

    @field_validator("window")
    @classmethod
    def validate_window_shape(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 30:
            raise ValueError(f"window must have 30 timesteps, got {len(v)}")
        for i, row in enumerate(v):
            if len(row) != 14:
                raise ValueError(
                    f"window row {i} must have 14 sensor values, got {len(row)}"
                )
        return v


class PredictionResponse(BaseModel):
    """Response from a prediction endpoint."""

    predicted_rul: float = Field(
        ..., description="Predicted remaining useful life in cycles."
    )
    alarm: bool = Field(
        ..., description="True if predicted_rul ≤ recommended alarm threshold (35)."
    )
    alarm_threshold: int = Field(
        default=35, description="Threshold used for the alarm decision."
    )
    regime: Literal["Critical", "Watch", "Healthy"] = Field(
        ...,
        description="Operational regime based on the prediction's RUL value.",
    )
    model_name: str = Field(..., description="Which model produced this prediction.")
    engine_id: int | None = Field(default=None)


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: Literal["ok", "degraded"] = "ok"
    models_loaded: List[str] = Field(default_factory=list)
    project: str = "predictive-maintenance-cmapss"
